// Launch surface for the Python-hosted Aboleth arms. Not compiled by CMake:
// experiments/bench/common.py renders a prelude that #defines BENCH_COMM,
// includes the chosen adapter, optionally bakes shape constants, and then
// includes this file; nvcc turns the result into a .so that ladder.py /
// alltoall.py drive through ctypes. All pointer arguments arrive as integers
// because they cross a ctypes boundary.

#include <cstdint>

#include <cuda_runtime.h>

#include "bench_kernels.cuh"

#ifndef BENCH_BAKED_BYTES
#define BENCH_BAKED_BYTES 0u
#endif
#ifndef BENCH_BAKED_PUTS
#define BENCH_BAKED_PUTS 0u
#endif
#ifndef BENCH_BAKED_OP
#define BENCH_BAKED_OP (-1)
#endif
// Non-zero bakes warp-collective puts: kernels keep the first warp alive and
// issue through the policy's put_warp instead of lane-0 put.
// > 1: the all-to-all quiets every Nth round (a labelled diagnostic).
#ifndef BENCH_QUIET_EVERY
#define BENCH_QUIET_EVERY 1u
#endif
#ifndef BENCH_BAKED_WARP_PUTS
#define BENCH_BAKED_WARP_PUTS 0
#endif
// Non-zero bakes the fused publication: the payload put(s) and the epoch
// share one batched submission (policy put_signal / put_many_signal).
#ifndef BENCH_BAKED_FUSE_SIGNAL
#define BENCH_BAKED_FUSE_SIGNAL 0
#endif
// Non-zero bakes fused bandwidth windows: each window is one batched
// submission carrying its own retirement (policy put_many_flush).
#ifndef BENCH_BAKED_BW_FLUSH
#define BENCH_BAKED_BW_FLUSH 0
#endif

extern "C" int bench_launch_pingpong(
    unsigned long long endpoints, unsigned long long doorbell_locks,
    int peer, int leader,
    unsigned long long dst_address, unsigned dst_key,
    unsigned long long src_address, unsigned src_key,
    unsigned long long peer_flag_address, unsigned peer_flag_key,
    unsigned long long my_flag_address,
    unsigned bytes, unsigned puts, int op,
    unsigned warmup, unsigned iters, unsigned long long samples_address,
    unsigned long long l2_address, unsigned l2_bytes, int threads,
    unsigned long long put_descs) {
    using Comm = BENCH_COMM;
    Comm::Ctx ctx = Comm::make_ctx(
        reinterpret_cast<const std::uint64_t*>(endpoints),
        reinterpret_cast<std::uint32_t*>(doorbell_locks));
    bench_pingpong<Comm, BENCH_BAKED_BYTES, BENCH_BAKED_PUTS, BENCH_BAKED_OP,
                   BENCH_BAKED_WARP_PUTS != 0, BENCH_BAKED_FUSE_SIGNAL != 0>
        <<<1, threads>>>(
            ctx, peer, leader,
            typename Comm::Remote{dst_address, dst_key},
            typename Comm::Local{src_address, src_key},
            typename Comm::Remote{peer_flag_address, peer_flag_key},
            reinterpret_cast<typename Comm::Flag>(my_flag_address),
            bytes, puts, op, warmup, iters,
            reinterpret_cast<std::uint64_t*>(samples_address),
            reinterpret_cast<std::uint8_t*>(l2_address), l2_bytes,
            reinterpret_cast<void*>(put_descs));
    return static_cast<int>(cudaGetLastError());
}

extern "C" int bench_launch_windowed_bw(
    unsigned long long endpoints, unsigned long long doorbell_locks,
    int peer,
    unsigned long long dst_address, unsigned dst_key,
    unsigned long long src_address, unsigned src_key,
    unsigned bytes, unsigned window, unsigned iters,
    unsigned long long arena_bytes, unsigned long long samples_address,
    int ctas, unsigned long long put_descs, unsigned long long spans,
    unsigned issuers) {
    using Comm = BENCH_COMM;
    Comm::Ctx ctx = Comm::make_ctx(
        reinterpret_cast<const std::uint64_t*>(endpoints),
        reinterpret_cast<std::uint32_t*>(doorbell_locks));
    // Several issuers: the CTA is exactly that many threads.
    bench_windowed_bw<Comm, BENCH_BAKED_BYTES,
                      BENCH_BAKED_WARP_PUTS != 0,
                      BENCH_BAKED_BW_FLUSH != 0>
        <<<ctas, issuers > 1 ? issuers : 32>>>(
        ctx, peer,
        typename Comm::Remote{dst_address, dst_key},
        typename Comm::Local{src_address, src_key},
        bytes, window, iters, arena_bytes,
        reinterpret_cast<std::uint64_t*>(samples_address),
        reinterpret_cast<void*>(put_descs),
        reinterpret_cast<std::uint64_t*>(spans), issuers);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int bench_launch_alltoall(
    unsigned long long endpoints, unsigned long long doorbell_locks,
    int rank, int world,
    unsigned long long dsts, unsigned long long srcs,
    unsigned long long peer_flags, unsigned long long my_flags,
    unsigned bytes, unsigned warmup, unsigned iters,
    unsigned long long samples_address, unsigned long long barrier_state) {
    using Comm = BENCH_COMM;
    Comm::Ctx ctx = Comm::make_ctx(
        reinterpret_cast<const std::uint64_t*>(endpoints),
        reinterpret_cast<std::uint32_t*>(doorbell_locks));
    bench_alltoall<Comm, BENCH_BAKED_BYTES,
                   BENCH_BAKED_FUSE_SIGNAL != 0,
                   BENCH_BAKED_WARP_PUTS != 0, BENCH_QUIET_EVERY><<<world - 1, 32>>>(
        ctx, rank,
        reinterpret_cast<const typename Comm::Remote*>(dsts),
        reinterpret_cast<const typename Comm::Local*>(srcs),
        reinterpret_cast<const typename Comm::Remote*>(peer_flags),
        reinterpret_cast<const typename Comm::Flag*>(my_flags),
        bytes, warmup, iters,
        reinterpret_cast<std::uint64_t*>(samples_address),
        reinterpret_cast<unsigned*>(barrier_state));
    return static_cast<int>(cudaGetLastError());
}

// Registers, local (spill) bytes and static shared bytes of the kernel each
// launcher above instantiates: 0 ping-pong, 1 windowed bandwidth, 2
// all-to-all. Returns a cudaError_t.
extern "C" int bench_kernel_attributes(int which, int* registers,
                                       int* local_bytes, int* shared_bytes) {
    using Comm = BENCH_COMM;
    cudaFuncAttributes attributes{};
    cudaError_t status = cudaErrorInvalidValue;
    if (which == 0)
        status = cudaFuncGetAttributes(
            &attributes,
            bench_pingpong<Comm, BENCH_BAKED_BYTES, BENCH_BAKED_PUTS,
                           BENCH_BAKED_OP, BENCH_BAKED_WARP_PUTS != 0,
                           BENCH_BAKED_FUSE_SIGNAL != 0>);
    else if (which == 1)
        status = cudaFuncGetAttributes(
            &attributes,
            bench_windowed_bw<Comm, BENCH_BAKED_BYTES,
                              BENCH_BAKED_WARP_PUTS != 0,
                              BENCH_BAKED_BW_FLUSH != 0>);
    else if (which == 2)
        status = cudaFuncGetAttributes(
            &attributes,
            bench_alltoall<Comm, BENCH_BAKED_BYTES,
                           BENCH_BAKED_FUSE_SIGNAL != 0,
                           BENCH_BAKED_WARP_PUTS != 0, BENCH_QUIET_EVERY>);
    *registers = attributes.numRegs;
    *local_bytes = static_cast<int>(attributes.localSizeBytes);
    *shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
    return static_cast<int>(status);
}
