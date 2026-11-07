#pragma once

// The Tier-1 benchmark kernels, written once and instantiated per arm
// through a comm policy. Every arm runs these exact loops; only the policy's
// five operations differ, so a difference between arms measures the
// transport, not the benchmark.
//
// Policy contract:
//
//   struct XComm {
//     struct Local {...}; struct Remote {...};  using Flag = ...;
//     struct Ctx {...};   static constexpr bool kHasPutInline;
//     static __device__ Local  offset(Local,  uint64_t byte_offset);
//     static __device__ Remote offset(Remote, uint64_t byte_offset);
//     static __device__ void put(Ctx&, int peer, Remote dst, Local src,
//                                uint32_t bytes);
//     static __device__ void put_inline8(Ctx&, int peer, Remote dst,
//                                        uint64_t value);
//     static __device__ void signal(Ctx&, int peer, Remote flag,
//                                   uint64_t epoch);   // strong-ordered
//     static __device__ void quiet(Ctx&, int peer);    // blocking
//     static __device__ void wait_signal(Ctx&, Flag, uint64_t epoch); // >=
//   };

#include <cstdint>
#include <type_traits>

#include "bench_common.h"

// Optional policy capability: a composite put_many (N descriptor-driven puts
// in one batched submission — one reservation, one doorbell). An adapter
// declares `kHasPutMany = true` plus `PutDescT`, `make_desc(PutDescT&,
// Remote, Local, bytes)`, and `put_many(Ctx&, peer, const PutDescT*, n)`;
// adapters without the members are detected as not having it.
namespace bench_detail {
template <class Comm, class = void>
struct has_put_many : std::false_type {};
template <class Comm>
struct has_put_many<Comm, std::void_t<decltype(Comm::kHasPutMany)>>
    : std::integral_constant<bool, Comm::kHasPutMany> {};

// Optional policy capability: a warp-collective put (`put_warp(Ctx&, peer,
// Remote, Local, bytes)`, entered by a complete warp). On peer-memory
// transports all 32 lanes copy cooperatively; on rdma one elected lane
// posts the WQE, so the kernels can issue the same way everywhere.
template <class Comm, class = void>
struct has_put_warp : std::false_type {};
template <class Comm>
struct has_put_warp<Comm, std::void_t<decltype(Comm::kHasPutWarp)>>
    : std::integral_constant<bool, Comm::kHasPutWarp> {};

// Optional policy capability: fused put+signal (`put_signal(Ctx&, peer,
// Remote, Local, bytes, Remote flag, epoch)` and `put_many_signal(...)`) —
// the payload and its publication in one batched submission.
template <class Comm, class = void>
struct has_put_signal : std::false_type {};
template <class Comm>
struct has_put_signal<Comm, std::void_t<decltype(Comm::kHasPutSignal)>>
    : std::integral_constant<bool, Comm::kHasPutSignal> {};

// Optional policy capability: a window that carries its own retirement
// (`put_many_flush(Ctx&, peer, descs, n)` — puts + signalled NOP + poll in
// one batch, equivalent to put_many + quiet).
template <class Comm, class = void>
struct has_put_many_flush : std::false_type {};
template <class Comm>
struct has_put_many_flush<
    Comm, std::void_t<decltype(Comm::kHasPutManyFlush)>>
    : std::integral_constant<bool, Comm::kHasPutManyFlush> {};

// Optional policy capability: a warp-collective put that carries its
// signal (`put_signal_warp(Ctx&, peer, Remote, Local, bytes, Remote flag,
// epoch)`, entered by a complete warp) — the library's own composite for
// large NVLink copies that still publish in one call.
template <class Comm, class = void>
struct has_put_signal_warp : std::false_type {};
template <class Comm>
struct has_put_signal_warp<
    Comm, std::void_t<decltype(Comm::kHasPutSignalWarp)>>
    : std::integral_constant<bool, Comm::kHasPutSignalWarp> {};

// Optional policy property: quiet completes every endpoint of the rank,
// not just the named peer's (NVSHMEM's nvshmem_quiet). The all-to-all then
// quiets once per rank per round instead of once per peer CTA, which would
// be world-1 rank-wide quiets a round.
template <class Comm, class = void>
struct quiet_is_global : std::false_type {};
template <class Comm>
struct quiet_is_global<Comm, std::void_t<decltype(Comm::kQuietIsGlobal)>>
    : std::integral_constant<bool, Comm::kQuietIsGlobal> {};

// Optional policy property: the library reclaims its send queue by itself
// (a put waits for room and polls completions internally), so a caller may
// post without ever completing until it is done (NVSHMEM, NCCL GIN). A
// Tier-1 put waits for room only a quiet frees.
template <class Comm, class = void>
struct self_reclaim : std::false_type {};
template <class Comm>
struct self_reclaim<Comm, std::void_t<decltype(Comm::kSelfReclaim)>>
    : std::integral_constant<bool, Comm::kSelfReclaim> {};

}  // namespace bench_detail

// Latency ping-pong. The leader posts the payload put(s) plus a signal, the
// follower echoes a signal back; one sample is the full round trip on the
// leader, the tree's existing single-point protocol. Template constants let
// a JIT-per-shape arm bake the swept values (0 / -1 selects the runtime
// argument instead). Launch with one CTA; use >= 2 warps only with l2_buffer
// set, where the extra threads exist to stride the flush buffer.
template <class Comm, std::uint32_t kBytesC = 0, std::uint32_t kPutsC = 0,
          int kOpC = -1, bool kWarpC = false, bool kFuseC = false>
__global__ void bench_pingpong(
    typename Comm::Ctx ctx, int peer, int leader,
    typename Comm::Remote dst, typename Comm::Local src,
    typename Comm::Remote peer_flag, typename Comm::Flag my_flag,
    std::uint32_t bytes_arg, std::uint32_t puts_arg, int op_arg,
    std::uint32_t warmup, std::uint32_t iters, std::uint64_t* samples_ns,
    std::uint8_t* l2_buffer, std::uint32_t l2_bytes,
    void* put_descs = nullptr) {
    const std::uint32_t bytes = kBytesC ? kBytesC : bytes_arg;
    const std::uint32_t puts = kPutsC ? kPutsC : puts_arg;
    const int op = kOpC >= 0 ? kOpC : op_arg;
    // The composite path: the host passes descriptor scratch (global memory)
    // only when it wants put_many. The offsets are round-invariant, so the
    // leader stages the descriptors once and every round is N WQEs plus ONE
    // doorbell instead of a doorbell per put.
    bool use_put_many = false;
    if constexpr (bench_detail::has_put_many<Comm>::value) {
        use_put_many =
            put_descs != nullptr && op == bench::kOpPut && puts > 1;
        if (use_put_many && leader
            && threadIdx.x == 0 && blockIdx.x == 0) {
            auto* descs = static_cast<typename Comm::PutDescT*>(put_descs);
            for (std::uint32_t index = 0; index < puts; ++index)
                Comm::make_desc(
                    descs[index],
                    Comm::offset(dst, std::uint64_t(index) * bytes),
                    Comm::offset(src, std::uint64_t(index) * bytes),
                    bytes);
        }
    }
    static_assert(!kWarpC || bench_detail::has_put_warp<Comm>::value,
                  "warp-issued puts need a put_warp policy member");
    static_assert(!kFuseC || bench_detail::has_put_signal<Comm>::value,
                  "fused signal needs a put_signal policy member");
    static_assert(!(kFuseC && kWarpC)
                      || bench_detail::has_put_signal_warp<Comm>::value,
                  "warp-issued fused signal needs put_signal_warp");
    for (std::uint32_t round = 0; round < warmup + iters; ++round) {
        if constexpr (kWarpC) {
            // The first warp drives complete: every lane joins each
            // put_warp, while the protocol ops (signal, wait, quiet) and
            // the timer stay on lane 0. Host-side validation restricts
            // this shape to op == put without put_many.
            if (threadIdx.x < 32 && blockIdx.x == 0) {
                const std::uint64_t epoch = round + 1;
                if (leader) {
                    std::uint64_t start = 0;
                    if (threadIdx.x == 0)
                        start = bench::globaltimer();
                    __syncwarp();
                    const std::uint32_t plain = kFuseC ? puts - 1 : puts;
                    for (std::uint32_t index = 0; index < plain; ++index)
                        Comm::put_warp(
                            ctx, peer,
                            Comm::offset(dst, std::uint64_t(index) * bytes),
                            Comm::offset(src, std::uint64_t(index) * bytes),
                            bytes);
                    if constexpr (kFuseC) {
                        const std::uint64_t last =
                            std::uint64_t(puts - 1) * bytes;
                        Comm::put_signal_warp(
                            ctx, peer, Comm::offset(dst, last),
                            Comm::offset(src, last), bytes, peer_flag,
                            epoch);
                    }
                    if (threadIdx.x == 0) {
                        if constexpr (!kFuseC)
                            Comm::signal(ctx, peer, peer_flag, epoch);
                        Comm::wait_signal(ctx, my_flag, epoch);
                        if (round >= warmup)
                            samples_ns[round - warmup] =
                                bench::globaltimer() - start;
                        Comm::quiet(ctx, peer);
                    }
                    __syncwarp();
                } else {
                    if (threadIdx.x == 0) {
                        Comm::wait_signal(ctx, my_flag, epoch);
                        Comm::signal(ctx, peer, peer_flag, epoch);
                        Comm::quiet(ctx, peer);
                    }
                    __syncwarp();
                }
            }
        } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            const std::uint64_t epoch = round + 1;
            if (leader) {
                const std::uint64_t start = bench::globaltimer();
                if (op == bench::kOpPutInline) {
                    Comm::put_inline8(ctx, peer, peer_flag, epoch);
                } else if (use_put_many) {
                    if constexpr (kFuseC) {
                        // Puts and epoch share the final batch: no posting
                        // sequence for the signal at all.
                        Comm::put_many_signal(
                            ctx, peer,
                            static_cast<const typename Comm::PutDescT*>(
                                put_descs),
                            puts, peer_flag, epoch);
                    } else {
                        if constexpr (
                            bench_detail::has_put_many<Comm>::value)
                            Comm::put_many(
                                ctx, peer,
                                static_cast<const typename Comm::PutDescT*>(
                                    put_descs),
                                puts);
                        Comm::signal(ctx, peer, peer_flag, epoch);
                    }
                } else if (kFuseC && op == bench::kOpPut) {
                    if constexpr (kFuseC) {
                        for (std::uint32_t index = 0; index + 1 < puts;
                             ++index)
                            Comm::put(
                                ctx, peer,
                                Comm::offset(
                                    dst, std::uint64_t(index) * bytes),
                                Comm::offset(
                                    src, std::uint64_t(index) * bytes),
                                bytes);
                        const std::uint64_t last =
                            std::uint64_t(puts - 1) * bytes;
                        Comm::put_signal(
                            ctx, peer, Comm::offset(dst, last),
                            Comm::offset(src, last), bytes,
                            peer_flag, epoch);
                    }
                } else {
                    for (std::uint32_t index = 0; index < puts; ++index) {
                        Comm::put(
                            ctx, peer,
                            Comm::offset(dst, std::uint64_t(index) * bytes),
                            Comm::offset(src, std::uint64_t(index) * bytes),
                            bytes);
                        if (op == bench::kOpPutSignal && index + 1 < puts)
                            Comm::signal(ctx, peer, peer_flag, epoch);
                    }
                    Comm::signal(ctx, peer, peer_flag, epoch);
                }
                Comm::wait_signal(ctx, my_flag, epoch);
                if (round >= warmup)
                    samples_ns[round - warmup] =
                        bench::globaltimer() - start;
                // Off the critical path: the echo already came back.
                Comm::quiet(ctx, peer);
            } else {
                Comm::wait_signal(ctx, my_flag, epoch);
                Comm::signal(ctx, peer, peer_flag, epoch);
                Comm::quiet(ctx, peer);
            }
        }
        if (l2_buffer) {
            __syncthreads();
            bench::l2_flush(l2_buffer, l2_bytes);
            __syncthreads();
        }
    }
}

// One-way windowed bandwidth, sender side only: each CTA posts `window` puts
// at slots cycling through the arena, then reclaims the queue with one
// quiet. Every CTA records the %globaltimer at its start and after its last
// quiet in `spans[2 * cta]`; bandwidth and message rate are the bytes and
// puts over max(end) - min(start), the same device clock every arm uses.
// CTA 0 also records each iteration's (window + quiet) time in
// `samples_ns`. The receiver needs no kernel: quiet means the RC ack
// arrived, so the data is resident at the peer and a host barrier is enough
// before validation.
//
// `issuers` > 1 (lane-issued puts only; the CTA is launched with exactly
// that many threads, a multiple of 32): the CTA's window is split evenly
// over its threads, each posting its own window / issuers consecutive
// slots, every thread syncing its warp after each put (a converged warp,
// so a library may coalesce the warp's postings into one doorbell itself).
// The window then retires as before: one quiet on thread 0 once every
// thread has posted. Slots, bytes per window and the retirement count are
// those of the single-issuer window, so bandwidth and message rate keep
// their definitions.
//
// A policy whose quiet is rank-wide (kQuietIsGlobal: nvshmem_quiet waits on
// every QP of the PE) retires each window once per rank instead, as the
// all-to-all does: every CTA posts its window, the grid meets, CTA 0
// quiets, and a second barrier releases the next window. A quiet per CTA
// would make each CTA wait on every other CTA's QPs (2610.01380 Fig. 4).
// `barrier_state` is a zeroed counter (one per launch).
//
// `stream` (self-reclaiming policies only): no completion per window; each
// CTA posts every window back to back and retires once after the last, as
// the dissecting study measures message rate (2610.01380, ibgda_bench
// msgrate: one quiet per thread at the end). The per-window samples are
// then issue times; the rate is still puts over the device-clock span,
// which includes the final retirement.
template <class Comm, std::uint32_t kBytesC = 0, bool kWarpC = false,
          bool kFlushC = false>
__global__ void bench_windowed_bw(
    typename Comm::Ctx ctx, int peer,
    typename Comm::Remote dst, typename Comm::Local src,
    std::uint32_t bytes_arg, std::uint32_t window, std::uint32_t iters,
    unsigned long long arena_bytes, std::uint64_t* samples_ns,
    void* put_descs = nullptr, std::uint64_t* spans = nullptr,
    std::uint32_t issuers = 1, unsigned* barrier_state = nullptr,
    bool stream = false) {
    static_assert(!kWarpC || bench_detail::has_put_warp<Comm>::value,
                  "warp-issued puts need a put_warp policy member");
    static_assert(!kFlushC
                      || bench_detail::has_put_many_flush<Comm>::value,
                  "fused windows need a put_many_flush policy member");
    static_assert(!(kFlushC && kWarpC),
                  "fused windows compose with lane-issued puts only");
    // The arena slot after `index`: (cursor + slot) % slots one slot at a
    // time, so the window pays one 64-bit division rather than one per put
    // (a division is a ~40-instruction subroutine call, which dominated a
    // single issuer's per-put cost).
    auto next_slot = [](std::uint64_t index, std::uint64_t slots) {
        return index + 1 == slots ? std::uint64_t(0) : index + 1;
    };
    // One thread per CTA retires the window (see above).
    auto retire = [&]() {
        if constexpr (bench_detail::quiet_is_global<Comm>::value) {
            bench::grid_epoch_barrier(barrier_state, gridDim.x);
            if (blockIdx.x == 0)
                Comm::quiet(ctx, peer);
            bench::grid_epoch_barrier(barrier_state, gridDim.x);
        } else {
            Comm::quiet(ctx, peer);
        }
    };
    if constexpr (!kWarpC && !kFlushC) {
        if (issuers > 1) {
            const std::uint32_t bytes = kBytesC ? kBytesC : bytes_arg;
            const std::uint64_t slots = arena_bytes / bytes;
            const std::uint32_t share = window / issuers;
            std::uint64_t cursor = std::uint64_t(blockIdx.x) * window;
            if (spans && threadIdx.x == 0)
                spans[2 * blockIdx.x] = bench::globaltimer();
            for (std::uint32_t iteration = 0; iteration < iters;
                 ++iteration) {
                const std::uint64_t start = bench::globaltimer();
                std::uint64_t index =
                    (cursor + std::uint64_t(threadIdx.x) * share) % slots;
                for (std::uint32_t slot = 0; slot < share; ++slot) {
                    const std::uint64_t offset = index * bytes;
                    Comm::put(ctx, peer, Comm::offset(dst, offset),
                              Comm::offset(src, offset), bytes);
                    index = next_slot(index, slots);
                    __syncwarp();
                }
                if (!stream) {
                    __syncthreads();
                    if (threadIdx.x == 0)
                        retire();
                    __syncthreads();
                }
                if (threadIdx.x == 0 && blockIdx.x == 0 && samples_ns)
                    samples_ns[iteration] = bench::globaltimer() - start;
                cursor += std::uint64_t(gridDim.x) * window;
            }
            if (stream) {
                __syncthreads();
                if (threadIdx.x == 0)
                    retire();
            }
            if (spans && threadIdx.x == 0)
                spans[2 * blockIdx.x + 1] = bench::globaltimer();
            return;
        }
    }
    if constexpr (kWarpC) {
        // The whole (32-thread) CTA drives each put collectively; quiet
        // and the timer stay on lane 0.
        if (threadIdx.x >= 32)
            return;
    } else if (threadIdx.x != 0) {
        return;
    }
    const std::uint32_t bytes = kBytesC ? kBytesC : bytes_arg;
    // Whole slots only, so no put crosses the end of the arena.
    const std::uint64_t slots = arena_bytes / bytes;
    std::uint64_t cursor = std::uint64_t(blockIdx.x) * window;
    if (spans && threadIdx.x == 0)
        spans[2 * blockIdx.x] = bench::globaltimer();
    for (std::uint32_t iteration = 0; iteration < iters; ++iteration) {
        const std::uint64_t start = bench::globaltimer();
        std::uint64_t index = cursor % slots;
        if constexpr (kFlushC) {
            // The whole window is one batched submission carrying its own
            // retirement: stage this CTA's descriptors (their cost is part
            // of the window, as it would be for any real caller), then one
            // reservation, one doorbell, one CQE.
            auto* descs =
                static_cast<typename Comm::PutDescT*>(put_descs)
                + std::uint64_t(blockIdx.x) * window;
            for (std::uint32_t slot = 0; slot < window; ++slot) {
                const std::uint64_t offset = index * bytes;
                index = next_slot(index, slots);
                Comm::make_desc(descs[slot], Comm::offset(dst, offset),
                                Comm::offset(src, offset), bytes);
            }
            Comm::put_many_flush(ctx, peer, descs, window);
        } else {
            for (std::uint32_t slot = 0; slot < window; ++slot) {
                const std::uint64_t offset = index * bytes;
                index = next_slot(index, slots);
                if constexpr (kWarpC) {
                    Comm::put_warp(ctx, peer, Comm::offset(dst, offset),
                                   Comm::offset(src, offset), bytes);
                } else {
                    Comm::put(ctx, peer, Comm::offset(dst, offset),
                              Comm::offset(src, offset), bytes);
                }
            }
            if (!stream) {
                if constexpr (kWarpC) {
                    if (threadIdx.x == 0)
                        retire();
                    __syncwarp();
                } else {
                    retire();
                }
            }
        }
        if (blockIdx.x == 0 && samples_ns && threadIdx.x == 0)
            samples_ns[iteration] = bench::globaltimer() - start;
        cursor += std::uint64_t(gridDim.x) * window;
    }
    if (stream) {
        if constexpr (kWarpC) {
            if (threadIdx.x == 0)
                retire();
            __syncwarp();
        } else {
            retire();
        }
    }
    if (spans && threadIdx.x == 0)
        spans[2 * blockIdx.x + 1] = bench::globaltimer();
}

// Symmetric all-to-all: world-1 CTAs, CTA b drives peer b + (b >= rank).
// Per round each CTA puts its per-peer slice, signals the slot indexed by
// this rank on the peer, waits for its inbound signal, quiets, and joins the
// grid barrier; CTA 0 records the round time once every peer's exchange has
// completed. Each CTA owns one endpoint, so no doorbell locks are needed.
// A policy whose quiet is rank-wide quiets once per round instead: every
// CTA joins the barrier, CTA 0 quiets, and a second barrier releases the
// round.
// The per-peer tables are indexed by CTA (peer order with self skipped).
// kQuietEveryC > 1 is a labelled diagnostic (Tier 1 only): quiet on every
// kQuietEveryC-th round and the last, to price the per-round retirement.
template <class Comm, std::uint32_t kBytesC = 0, bool kFuseC = false,
          bool kWarpC = false, std::uint32_t kQuietEveryC = 1>
__global__ void bench_alltoall(
    typename Comm::Ctx ctx, int rank,
    const typename Comm::Remote* dsts, const typename Comm::Local* srcs,
    const typename Comm::Remote* peer_flags,
    const typename Comm::Flag* my_flags,
    std::uint32_t bytes_arg, std::uint32_t warmup, std::uint32_t iters,
    std::uint64_t* samples_ns, unsigned* barrier_state) {
    static_assert(!kWarpC || bench_detail::has_put_warp<Comm>::value,
                  "warp-issued puts need a put_warp policy member");
    static_assert(!(kFuseC && kWarpC)
                      || bench_detail::has_put_signal_warp<Comm>::value,
                  "warp-issued fused signal needs put_signal_warp");
    if constexpr (kWarpC) {
        // Each CTA's first warp drives its put collectively; the protocol
        // ops (signal, wait, quiet, barrier) and the timer stay on lane 0.
        if (threadIdx.x >= 32)
            return;
    } else if (threadIdx.x != 0) {
        return;
    }
    const std::uint32_t bytes = kBytesC ? kBytesC : bytes_arg;
    const int lane = static_cast<int>(blockIdx.x);
    const int peer = lane + (lane >= rank);
    for (std::uint32_t round = 0; round < warmup + iters; ++round) {
        const std::uint64_t epoch = round + 1;
        std::uint64_t start = 0;
        if (lane == 0 && threadIdx.x == 0)
            start = bench::globaltimer();
        static_assert(!kFuseC || bench_detail::has_put_signal<Comm>::value,
                      "fused signal needs a put_signal policy member");
        if constexpr (kWarpC && kFuseC) {
            Comm::put_signal_warp(ctx, peer, dsts[lane], srcs[lane], bytes,
                                  peer_flags[lane], epoch);
        } else if constexpr (kWarpC) {
            // put_warp returns with every lane's stores fenced, so lane 0's
            // signal below publishes the whole slice.
            Comm::put_warp(ctx, peer, dsts[lane], srcs[lane], bytes);
        } else if constexpr (kFuseC) {
            Comm::put_signal(ctx, peer, dsts[lane], srcs[lane], bytes,
                             peer_flags[lane], epoch);
        } else {
            Comm::put(ctx, peer, dsts[lane], srcs[lane], bytes);
            Comm::signal(ctx, peer, peer_flags[lane], epoch);
        }
        if (threadIdx.x == 0) {
            if constexpr (kWarpC && !kFuseC)
                Comm::signal(ctx, peer, peer_flags[lane], epoch);
            Comm::wait_signal(ctx, my_flags[lane], epoch);
            if constexpr (bench_detail::quiet_is_global<Comm>::value) {
                bench::grid_epoch_barrier(barrier_state, gridDim.x);
                if (lane == 0)
                    Comm::quiet(ctx, peer);
            } else if (kQuietEveryC == 1 || epoch % kQuietEveryC == 0
                       || round + 1 == warmup + iters) {
                Comm::quiet(ctx, peer);
            }
            bench::grid_epoch_barrier(barrier_state, gridDim.x);
            if (lane == 0 && round >= warmup)
                samples_ns[round - warmup] = bench::globaltimer() - start;
        }
        // Keep the warp converged across rounds: lanes must not enter the
        // next put_warp before lane 0 clears this round's barrier.
        if constexpr (kWarpC)
            __syncwarp();
    }
}
