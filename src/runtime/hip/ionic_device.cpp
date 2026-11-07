#include "aboleth/Target/IonicAbi.h"
#include <hip/hip_runtime.h>

namespace {
using namespace aboleth::abi;
__device__ IonicEndpointDescriptor &endpoint(uint64_t p) {
  return *reinterpret_cast<IonicEndpointDescriptor *>(p);
}
__device__ IonicQueueState &state(IonicEndpointDescriptor &ep) {
  return *reinterpret_cast<IonicQueueState *>(ep.state);
}
__device__ void fail(IonicQueueState &s, uint64_t code) {
  __hip_atomic_store(&s.status, code, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  __builtin_trap();
}
__device__ void lock(IonicEndpointDescriptor &ep) {
  auto &s = state(ep);
  uint64_t start = wall_clock64();
  uint32_t expected = 0;
  while (!__hip_atomic_compare_exchange_strong(&s.lock, &expected, 1,
          __ATOMIC_ACQUIRE, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
    expected = 0;
    if (wall_clock64() - start > ep.timeout_ticks) fail(s, 1);
  }
}
__device__ void unlock(IonicQueueState &s) {
  __hip_atomic_store(&s.lock, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
// A CCQE continuously exposes the 24-bit cumulative message sequence number.
// Runtime requires CCQE (mask zero); ordinary color-ring CQs are refused.
__device__ void retire(IonicEndpointDescriptor &ep, uint64_t target) {
  auto &s = state(ep);
  auto *cq = reinterpret_cast<volatile IonicCqe *>(ep.completion_queue);
  uint64_t start = wall_clock64();
  while (s.retired < target) {
    uint32_t flags = __builtin_bswap32(cq->qid_type_flags);
    if (flags & 2) fail(s, (uint64_t(__builtin_bswap32(cq->status)) << 32) | 2);
    uint32_t msn = __builtin_bswap32(cq->msn) & kIonicMsnMask;
    uint32_t delta = (msn - uint32_t(s.retired)) & kIonicMsnMask;
    if (delta && delta <= ep.send_mask) s.retired += delta;
    if (wall_clock64() - start > ep.timeout_ticks) fail(s, 3);
  }
  __threadfence_system();
}
__device__ IonicWqe *slot(IonicEndpointDescriptor &ep, uint64_t index) {
  return reinterpret_cast<IonicWqe *>(ep.send_queue + (index & ep.send_mask) * ep.send_stride);
}
__device__ void header(IonicWqe &wqe, IonicEndpointDescriptor &ep,
                       uint64_t index, uint64_t remote, uint32_t key,
                       uint32_t bytes, bool inlined) {
  wqe.index = index;
  wqe.opcode = kIonicWrite;
  wqe.sge_count = bytes && !inlined ? 1 : 0;
  wqe.immediate = 0;
  wqe.remote_high = __builtin_bswap32(uint32_t(remote >> 32));
  wqe.remote_low = __builtin_bswap32(uint32_t(remote));
  wqe.remote_key = __builtin_bswap32(key);
  wqe.length = __builtin_bswap32(bytes);
}
__device__ void ready(IonicWqe &wqe, IonicEndpointDescriptor &ep,
                      uint64_t index, bool inlined, bool fenced = false) {
  uint16_t flags = kIonicSignaled | (inlined ? kIonicInline : 0) |
                  (fenced ? kIonicFence : 0) |
                  ((index & (ep.send_mask + 1)) ? 0 : kIonicColor);
  __hip_atomic_store(&wqe.flags, __builtin_bswap16(flags), __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_AGENT);
}
__device__ void ring(IonicEndpointDescriptor &ep, uint64_t end) {
  // Stronger than the qualified vendor path's device fence: make payload and
  // GPU queue stores visible before the MMIO announcement leaves the GPU.
  __threadfence_system();
  __atomic_store_n(reinterpret_cast<uint64_t *>(ep.send_doorbell),
      ep.send_dbval | (end & ep.send_mask), __ATOMIC_SEQ_CST);
  state(ep).published = end;
}
}

extern "C" {
__device__ uint64_t aboleth_ionic_wait_signal(uint64_t address, uint64_t epoch) {
  auto *cell = reinterpret_cast<uint64_t *>(address);
  while (__hip_atomic_load(cell, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {}
  return epoch;
}
__device__ uint64_t aboleth_ionic_fence() {
  __threadfence_system(); return 0;
}
__device__ uint64_t aboleth_ionic_reserve(uint64_t address, uint32_t count) {
  auto &ep = endpoint(address); auto &s = state(ep);
  lock(ep);
  if (ep.version != kIonicEndpointVersion || ep.completion_mask ||
      !count || count > ep.send_mask) fail(s, 4);
  uint64_t base = s.producer;
  if (base + count - s.retired > ep.send_mask) retire(ep, s.published);
  s.producer = base + count;
  return base;
}
__device__ uint64_t aboleth_ionic_put(uint64_t address, uint64_t index,
    uint64_t source, uint32_t lkey, uint64_t remote, uint32_t rkey,
    uint32_t bytes, uint32_t predicate) {
  auto &ep = endpoint(address); auto &wqe = *slot(ep, index);
  // A suppressed command still owns its reserved slot. Ionic has no qualified
  // NOP opcode: an inline write into private peer scratch consumes that slot
  // without reading or writing any application address, including null ones.
  if (!predicate || !bytes) {
    header(wqe, ep, index, ep.scratch_address, ep.scratch_key, 8, true);
    *reinterpret_cast<uint64_t *>(wqe.payload) = 0;
    ready(wqe, ep, index, true); return index + 1;
  }
  header(wqe, ep, index, remote, rkey, bytes, false);
  auto *data = reinterpret_cast<uint64_t *>(wqe.payload);
  data[0] = __builtin_bswap64(source);
  auto *words = reinterpret_cast<uint32_t *>(wqe.payload);
  words[2] = __builtin_bswap32(bytes); words[3] = __builtin_bswap32(lkey);
  ready(wqe, ep, index, false);
  return index + 1;
}
__device__ uint64_t aboleth_ionic_inline(uint64_t address, uint64_t index,
    uint64_t remote, uint32_t rkey, uint64_t lo, uint64_t hi, uint32_t bytes,
    uint32_t predicate) {
  auto &ep = endpoint(address); auto &s = state(ep);
  if (bytes > 16) fail(s, 5);
  if (!predicate || !bytes) {
    remote = ep.scratch_address; rkey = ep.scratch_key;
    lo = hi = 0; bytes = 8;
  }
  auto &wqe = *slot(ep, index);
  header(wqe, ep, index, remote, rkey, bytes, true);
  auto *data = reinterpret_cast<uint64_t *>(wqe.payload);
  data[0] = lo; data[1] = hi;
  ready(wqe, ep, index, true);
  return index + 1;
}
// The fence bit belongs to the release-published flags, never a later store.
__device__ uint64_t aboleth_ionic_inline_fenced(uint64_t address, uint64_t index,
    uint64_t remote, uint32_t rkey, uint64_t lo, uint64_t hi, uint32_t bytes,
    uint32_t predicate) {
  auto &ep = endpoint(address); auto &s = state(ep);
  if (bytes > 16) fail(s, 5);
  if (!predicate || !bytes) {
    remote = ep.scratch_address; rkey = ep.scratch_key;
    lo = hi = 0; bytes = 8;
  }
  auto &wqe = *slot(ep, index);
  header(wqe, ep, index, remote, rkey, bytes, true);
  auto *data = reinterpret_cast<uint64_t *>(wqe.payload);
  data[0] = lo; data[1] = hi;
  ready(wqe, ep, index, true, true);
  return index + 1;
}
__device__ uint64_t aboleth_ionic_publish_prefix(uint64_t address, uint64_t end) {
  auto &ep = endpoint(address);
  // Called with reserve's lock still held. No other producer may interleave
  // an epoch while this payload prefix retires.
  if (end > state(ep).published) ring(ep, end);
  retire(ep, end);
  return end;
}
__device__ uint64_t aboleth_ionic_commit(uint64_t address, uint64_t end) {
  auto &ep = endpoint(address);
  ring(ep, end); unlock(state(ep)); return end;
}
__device__ uint64_t aboleth_ionic_poll(uint64_t address, uint64_t target) {
  auto &ep = endpoint(address);
  lock(ep); retire(ep, target); unlock(state(ep)); return target;
}
__device__ uint64_t aboleth_ionic_drain(uint64_t address) {
  auto &ep = endpoint(address);
  lock(ep); auto target = state(ep).published;
  retire(ep, target); unlock(state(ep)); return target;
}

// Only the compiler-proved CTA issuer uses these entries. The endpoint's
// existing capacity, publication and completion obligations are unchanged.
// Other CTAs, detached thread callers and unproved selections keep the lock.
__device__ uint64_t aboleth_ionic_reserve_exclusive(uint64_t address, uint32_t count) {
  auto &ep = endpoint(address); auto &s = state(ep);
  if (ep.version != kIonicEndpointVersion || ep.completion_mask ||
      !count || count > ep.send_mask) fail(s, 4);
  uint64_t base = s.producer;
  if (base + count - s.retired > ep.send_mask) retire(ep, s.published);
  s.producer = base + count;
  return base;
}
__device__ uint64_t aboleth_ionic_commit_exclusive(uint64_t address, uint64_t end) {
  auto &ep = endpoint(address); ring(ep, end); return end;
}
__device__ uint64_t aboleth_ionic_poll_exclusive(uint64_t address, uint64_t target) {
  auto &ep = endpoint(address); retire(ep, target); return target;
}

// Triton externs execute on every active lane. A kernel submission has one
// issuer; detached Tier-1 functions instead call the per-thread entries above.
// The compiler currently requires one wave for Ionic kernels. Broadcasting
// lane zero's reservation makes the scalar result uniform without using LDS.
#define IONIC_CTA(name, arguments, invocation) \
__device__ uint64_t name##_cta arguments { \
  __threadfence_system(); __syncwarp(); \
  uint64_t value = 0; \
  if (threadIdx.x == 0) value = name invocation; \
  return __shfl(value, 0); \
}
IONIC_CTA(aboleth_ionic_reserve, (uint64_t ep, uint32_t count), (ep, count))
IONIC_CTA(aboleth_ionic_put,
    (uint64_t ep, uint64_t index, uint64_t source, uint32_t lkey,
     uint64_t remote, uint32_t rkey, uint32_t bytes, uint32_t predicate),
    (ep, index, source, lkey, remote, rkey, bytes, predicate))
IONIC_CTA(aboleth_ionic_inline,
    (uint64_t ep, uint64_t index, uint64_t remote, uint32_t rkey,
     uint64_t lo, uint64_t hi, uint32_t bytes, uint32_t predicate),
    (ep, index, remote, rkey, lo, hi, bytes, predicate))
IONIC_CTA(aboleth_ionic_inline_fenced,
    (uint64_t ep, uint64_t index, uint64_t remote, uint32_t rkey,
     uint64_t lo, uint64_t hi, uint32_t bytes, uint32_t predicate),
    (ep, index, remote, rkey, lo, hi, bytes, predicate))
IONIC_CTA(aboleth_ionic_publish_prefix, (uint64_t ep, uint64_t end), (ep, end))
IONIC_CTA(aboleth_ionic_commit, (uint64_t ep, uint64_t end), (ep, end))
IONIC_CTA(aboleth_ionic_poll, (uint64_t ep, uint64_t end), (ep, end))
IONIC_CTA(aboleth_ionic_drain, (uint64_t ep), (ep))
IONIC_CTA(aboleth_ionic_reserve_exclusive, (uint64_t ep, uint32_t count), (ep, count))
IONIC_CTA(aboleth_ionic_commit_exclusive, (uint64_t ep, uint64_t end), (ep, end))
IONIC_CTA(aboleth_ionic_poll_exclusive, (uint64_t ep, uint64_t end), (ep, end))
#undef IONIC_CTA

// These are not standalone producer primitives. The compiler selects them
// only after reserve_exclusive_cta fenced every lane's completed producer
// stores, inside one proved CTA-owned SubmitOp. WQE stores belong to lane zero;
// ring's system fence still publishes them, and retire's fence still protects
// the completion boundary. Detached/shared issuers keep their original ABI.
#define IONIC_POSTED_CTA(name, arguments, invocation) \
__device__ uint64_t name##_cta arguments { \
  __syncwarp(); uint64_t value = 0; \
  if (threadIdx.x == 0) value = invocation; \
  return __shfl(value, 0); \
}
IONIC_POSTED_CTA(aboleth_ionic_put_posted,
    (uint64_t ep, uint64_t index, uint64_t source, uint32_t lkey,
     uint64_t remote, uint32_t rkey, uint32_t bytes, uint32_t predicate),
    aboleth_ionic_put(ep, index, source, lkey, remote, rkey, bytes, predicate))
IONIC_POSTED_CTA(aboleth_ionic_inline_posted,
    (uint64_t ep, uint64_t index, uint64_t remote, uint32_t rkey,
     uint64_t lo, uint64_t hi, uint32_t bytes, uint32_t predicate),
    aboleth_ionic_inline(ep, index, remote, rkey, lo, hi, bytes, predicate))
IONIC_POSTED_CTA(aboleth_ionic_inline_fenced_posted,
    (uint64_t ep, uint64_t index, uint64_t remote, uint32_t rkey,
     uint64_t lo, uint64_t hi, uint32_t bytes, uint32_t predicate),
    aboleth_ionic_inline_fenced(ep, index, remote, rkey, lo, hi, bytes, predicate))
IONIC_POSTED_CTA(aboleth_ionic_publish_prefix_posted,
    (uint64_t ep, uint64_t end), aboleth_ionic_publish_prefix(ep, end))
IONIC_POSTED_CTA(aboleth_ionic_commit_posted,
    (uint64_t ep, uint64_t end), aboleth_ionic_commit_exclusive(ep, end))
IONIC_POSTED_CTA(aboleth_ionic_poll_posted,
    (uint64_t ep, uint64_t end), aboleth_ionic_poll_exclusive(ep, end))
#undef IONIC_POSTED_CTA

}


// Scalar, predicated volatile loads for Ionic receiver modules. Keeping these
// in HIP bitcode avoids the pinned Triton AMD conversion dropping volatility.
// False masks do not dereference the address, including a null/out-of-bounds
// inactive lane. Float and pointer clients pass their bit representations.
extern "C" {
#define IONIC_VOLATILE_LOAD(bits, type) \
__device__ type aboleth_ionic_load_u##bits(uint64_t address, uint32_t active, type other) { \
  if (!active) return other; \
  return *reinterpret_cast<const volatile type *>(address); \
}
IONIC_VOLATILE_LOAD(8, uint8_t)
IONIC_VOLATILE_LOAD(16, uint16_t)
IONIC_VOLATILE_LOAD(32, uint32_t)
IONIC_VOLATILE_LOAD(64, uint64_t)
#undef IONIC_VOLATILE_LOAD
}
