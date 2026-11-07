#include <cstdint>
#include <hip/hip_runtime.h>

// Scalar ABI: addresses cross the Triton extern boundary as i64, so generic
// versus global pointer address spaces cannot change the function signature.
// These are real atomic loads/stores, not read-modify-write substitutes: a
// polling GPU must never write a stale value over a concurrent NIC update.
extern "C" {
__device__ uint64_t aboleth_hip_acquire64(uint64_t address) {
  return __hip_atomic_load(reinterpret_cast<uint64_t *>(address),
                           __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ uint64_t aboleth_hip_release64(uint64_t address, uint64_t value,
                                          uint32_t first_lane) {
  if (!first_lane || threadIdx.x == 0)
    __hip_atomic_store(reinterpret_cast<uint64_t *>(address), value,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  return value;
}
__device__ uint32_t aboleth_hip_release32(uint64_t address, uint32_t value,
                                          uint32_t first_lane) {
  if (!first_lane || threadIdx.x == 0)
    __hip_atomic_store(reinterpret_cast<uint32_t *>(address), value,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  return value;
}
__device__ uint64_t aboleth_hip_reserve64(uint64_t address) {
  return __hip_atomic_fetch_add(reinterpret_cast<uint64_t *>(address), uint64_t(1),
                                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ uint64_t aboleth_hip_copy(uint64_t destination, uint64_t source,
                                    uint32_t bytes, uint32_t cooperative) {
  auto *dst = reinterpret_cast<uint8_t *>(destination);
  auto *src = reinterpret_cast<const uint8_t *>(source);
  unsigned lane = cooperative ? threadIdx.x % warpSize : 0;
  unsigned lanes = cooperative ? warpSize : 1;
  if (((destination | source) & 7) == 0) {
    auto *d = reinterpret_cast<uint64_t *>(dst);
    auto *s = reinterpret_cast<const uint64_t *>(src);
    for (unsigned word = lane; word < bytes / 8; word += lanes)
      d[word] = s[word];
    for (unsigned byte = (bytes / 8) * 8 + lane; byte < bytes; byte += lanes)
      dst[byte] = src[byte];
  } else {
    for (unsigned byte = lane; byte < bytes; byte += lanes)
      dst[byte] = src[byte];
  }
  if (cooperative) {
    __threadfence_system();
    __syncwarp();
  }
  return 0;
}
__device__ uint64_t aboleth_hip_thread_id() { return threadIdx.x; }
__device__ uint64_t aboleth_hip_clock() { return wall_clock64(); }
__device__ uint64_t aboleth_hip_fence() {
  __threadfence_system();
  return 0;
}
}
