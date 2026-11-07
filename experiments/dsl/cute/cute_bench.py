"""bench_kernels.cuh's three benchmarks as CuTe DSL kernels.

One transcription, every CuTe arm: the kernels call a communication policy
(`use(policy)`) with the Tier-1 op signatures, and only the policy differs
between arms:

    put(endpoints, peer, dst, dst_key, src, src_key, nbytes)
    put_signal(endpoints, peer, dst, dst_key, src, src_key, nbytes,
               flag, flag_key, epoch)
    signal(endpoints, peer, flag, flag_key, epoch)
    put_inline(endpoints, peer, dst, dst_key, value)
    quiet(endpoints, peer)
    wait_signal(flag, epoch)

Addresses are 64-bit integers and keys 32-bit; a policy that needs neither
keys nor an endpoint table (NVSHMEM) ignores them. A policy may also define
`fence()`: its put-with-signal then orders only its own payload, so a fused
round fences its earlier puts first (NVSHMEM's semantics, as the CUDA adapter). The loops mirror the CUDA
kernels line for line: one issuing thread, %globaltimer samples, the
windowed kernel's per-CTA spans and slot cycling, the all-to-all's sense-free
grid barrier. Sizes, put counts and the op are compile-time constants, as in
the Tier-2 arm; counts that the CUDA kernel takes at run time (window, iters)
stay run time.

The CUDA kernels' other issue shapes are here too, for policies that have
them: a warp-issued all-to-all (`put_warp`, and `put_signal_warp` when the
signal is fused; kWarpC), the windowed kernel's several issuers (a CTA of
ISSUERS threads, each posting its share of the window, the warp converged
after every put) and streaming (one retirement after the last window).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Uint64

OP_PUT, OP_PUT_SIGNAL, OP_PUT_INLINE = 0, 1, 2

# The policy the next compilation traces against (module state, read at trace
# time) and whether its quiet completes every endpoint of the rank.
COMM = None
QUIET_IS_GLOBAL = False


def use(policy, quiet_is_global: bool = False) -> None:
    global COMM, QUIET_IS_GLOBAL
    COMM, QUIET_IS_GLOBAL = policy, quiet_is_global


@cute.jit
def _grid_barrier(barrier: cute.Tensor, ctas):
    # bench::grid_epoch_barrier: arrive, then wait for the epoch's last CTA.
    arrived = cute.arch.atomic_add(barrier.iterator, Int32(1)) + 1
    target = ((arrived - 1) // ctas + 1) * ctas
    seen = cute.arch.load(barrier.iterator, Int32, sem="relaxed", scope="gpu")
    while seen < target:
        seen = cute.arch.load(barrier.iterator, Int32, sem="relaxed",
                              scope="gpu")


@cute.kernel
def _pingpong(endpoints: cute.Tensor, peer: Int32, dst: Uint64,
              dst_key: Uint32, src: Uint64, src_key: Uint32,
              peer_flag: Uint64, flag_key: Uint32, my_flag: Uint64,
              warmup: Int32, iters: Int32, samples: cute.Tensor,
              BYTES: cutlass.Constexpr, PUTS: cutlass.Constexpr,
              OP: cutlass.Constexpr, LEADER: cutlass.Constexpr,
              FUSE: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        for round_index in cutlass.range(warmup + iters):
            epoch = Uint64(round_index + 1)
            if cutlass.const_expr(LEADER):
                start = cute.arch.globaltimer()
                if cutlass.const_expr(OP == OP_PUT_INLINE):
                    COMM.put_inline(endpoints, peer, peer_flag, flag_key,
                                    epoch)
                elif cutlass.const_expr(FUSE and OP == OP_PUT):
                    for index in cutlass.range_constexpr(PUTS - 1):
                        COMM.put(endpoints, peer, dst + index * BYTES,
                                 dst_key, src + index * BYTES, src_key,
                                 BYTES)
                    last = (PUTS - 1) * BYTES
                    # A policy whose put-with-signal orders only its own
                    # payload (NVSHMEM) fences the earlier puts first.
                    if cutlass.const_expr(PUTS > 1 and hasattr(COMM,
                                                               "fence")):
                        COMM.fence()
                    COMM.put_signal(endpoints, peer, dst + last, dst_key,
                                    src + last, src_key, BYTES, peer_flag,
                                    flag_key, epoch)
                else:
                    for index in cutlass.range_constexpr(PUTS):
                        COMM.put(endpoints, peer, dst + index * BYTES,
                                 dst_key, src + index * BYTES, src_key,
                                 BYTES)
                        if cutlass.const_expr(OP == OP_PUT_SIGNAL
                                              and index + 1 < PUTS):
                            COMM.signal(endpoints, peer, peer_flag, flag_key,
                                        epoch)
                    COMM.signal(endpoints, peer, peer_flag, flag_key, epoch)
                COMM.wait_signal(my_flag, epoch)
                if round_index >= warmup:
                    samples[round_index - warmup] = (
                        cute.arch.globaltimer() - start)
                # Off the critical path: the echo already came back.
                COMM.quiet(endpoints, peer)
            else:
                COMM.wait_signal(my_flag, epoch)
                COMM.signal(endpoints, peer, peer_flag, flag_key, epoch)
                COMM.quiet(endpoints, peer)


@cute.jit
def pingpong(endpoints: cute.Tensor, peer: Int32, dst: Uint64,
             dst_key: Uint32, src: Uint64, src_key: Uint32,
             peer_flag: Uint64, flag_key: Uint32, my_flag: Uint64,
             warmup: Int32, iters: Int32, samples: cute.Tensor,
             BYTES: cutlass.Constexpr, PUTS: cutlass.Constexpr,
             OP: cutlass.Constexpr, LEADER: cutlass.Constexpr,
             FUSE: cutlass.Constexpr):
    _pingpong(endpoints, peer, dst, dst_key, src, src_key, peer_flag,
              flag_key, my_flag, warmup, iters, samples, BYTES, PUTS, OP,
              LEADER, FUSE).launch(grid=[1, 1, 1], block=[32, 1, 1])


@cute.jit
def _retire(endpoints: cute.Tensor, peer, barrier: cute.Tensor, ctas,
            block):
    if cutlass.const_expr(QUIET_IS_GLOBAL):
        _grid_barrier(barrier, ctas)
        if block == 0:
            COMM.quiet(endpoints, peer)
        _grid_barrier(barrier, ctas)
    else:
        COMM.quiet(endpoints, peer)


@cute.kernel
def _windowed(endpoints: cute.Tensor, peer: Int32, dst: Uint64,
              dst_key: Uint32, src: Uint64, src_key: Uint32, window: Int32,
              iters: Int32, slots: Int64, samples: cute.Tensor,
              spans: cute.Tensor, record: Int32, barrier: cute.Tensor,
              stream: Int32, BYTES: cutlass.Constexpr,
              ISSUERS: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    ctas, _, _ = cute.arch.grid_dim()
    if cutlass.const_expr(ISSUERS > 1):
        # Thread t posts window / ISSUERS consecutive slots, the warp
        # converged after each put; the window retires once every thread
        # has posted (bench_windowed_bw's several issuers).
        share = window // ISSUERS
        cursor = Int64(block) * Int64(window)
        if tidx == 0:
            if record != 0:
                spans[2 * block] = cute.arch.globaltimer()
        for iteration in cutlass.range(iters):
            start = cute.arch.globaltimer()
            index = (cursor + Int64(tidx) * Int64(share)) % slots
            for _slot in cutlass.range(share):
                offset = index * BYTES
                COMM.put(endpoints, peer, dst + Uint64(offset), dst_key,
                         src + Uint64(offset), src_key, BYTES)
                index = index + 1
                if index == slots:
                    index = Int64(0)
                cute.arch.sync_warp()
            if stream == 0:
                cute.arch.sync_threads()
                if tidx == 0:
                    _retire(endpoints, peer, barrier, ctas, block)
                cute.arch.sync_threads()
            if tidx == 0:
                if record != 0:
                    if block == 0:
                        samples[iteration] = cute.arch.globaltimer() - start
            cursor = cursor + Int64(ctas) * Int64(window)
        if stream != 0:
            cute.arch.sync_threads()
            if tidx == 0:
                _retire(endpoints, peer, barrier, ctas, block)
        if tidx == 0:
            if record != 0:
                spans[2 * block + 1] = cute.arch.globaltimer()
    else:
        if tidx == 0:
            cursor = Int64(block) * Int64(window)
            if record != 0:
                spans[2 * block] = cute.arch.globaltimer()
            for iteration in cutlass.range(iters):
                start = cute.arch.globaltimer()
                index = cursor % slots
                for _slot in cutlass.range(window):
                    offset = index * BYTES
                    index = index + 1
                    if index == slots:
                        index = Int64(0)
                    COMM.put(endpoints, peer, dst + Uint64(offset), dst_key,
                             src + Uint64(offset), src_key, BYTES)
                if stream == 0:
                    _retire(endpoints, peer, barrier, ctas, block)
                if record != 0:
                    if block == 0:
                        samples[iteration] = cute.arch.globaltimer() - start
                cursor = cursor + Int64(ctas) * Int64(window)
            if stream != 0:
                # Streaming: every window back to back, one retirement.
                _retire(endpoints, peer, barrier, ctas, block)
            if record != 0:
                spans[2 * block + 1] = cute.arch.globaltimer()


@cute.jit
def windowed(endpoints: cute.Tensor, peer: Int32, dst: Uint64,
             dst_key: Uint32, src: Uint64, src_key: Uint32, window: Int32,
             iters: Int32, slots: Int64, samples: cute.Tensor,
             spans: cute.Tensor, record: Int32, barrier: cute.Tensor,
             ctas: Int32, stream: Int32, BYTES: cutlass.Constexpr,
             ISSUERS: cutlass.Constexpr):
    _windowed(endpoints, peer, dst, dst_key, src, src_key, window, iters,
              slots, samples, spans, record, barrier, stream, BYTES,
              ISSUERS).launch(grid=[ctas, 1, 1],
                              block=[max(ISSUERS, 32), 1, 1])


@cute.kernel
def _alltoall(endpoints: cute.Tensor, rank: Int32, dst: cute.Tensor,
              dst_key: cute.Tensor, src: cute.Tensor, src_key: Uint32,
              flag: cute.Tensor, flag_key: cute.Tensor, my_flag: cute.Tensor,
              warmup: Int32, iters: Int32, samples: cute.Tensor,
              barrier: cute.Tensor, BYTES: cutlass.Constexpr,
              FUSE: cutlass.Constexpr, WARP: cutlass.Constexpr):
    # CTA b drives peer b + (b >= rank); the tables are indexed by CTA.
    tidx, _, _ = cute.arch.thread_idx()
    lane, _, _ = cute.arch.block_idx()
    ctas, _, _ = cute.arch.grid_dim()
    if cutlass.const_expr(WARP):
        # kWarpC: the CTA's warp enters each collective put; the signal
        # (unless fused into it), the wait, the retirement and the timer
        # stay on thread 0, behind the warp's converged copy.
        peer = Int32(lane)
        if lane >= rank:
            peer = Int32(lane + 1)
        destination = Uint64(dst[lane])
        destination_key = Uint32(dst_key[lane])
        source = Uint64(src[lane])
        peer_flag = Uint64(flag[lane])
        peer_flag_key = Uint32(flag_key[lane])
        mine = Uint64(my_flag[lane])
        for round_index in cutlass.range(warmup + iters):
            epoch = Uint64(round_index + 1)
            start = cute.arch.globaltimer()
            if cutlass.const_expr(FUSE):
                COMM.put_signal_warp(endpoints, peer, destination,
                                     destination_key, source, src_key, BYTES,
                                     peer_flag, peer_flag_key, epoch)
            else:
                COMM.put_warp(endpoints, peer, destination, destination_key,
                              source, src_key, BYTES)
            cute.arch.sync_warp()
            if tidx == 0:
                if cutlass.const_expr(not FUSE):
                    COMM.signal(endpoints, peer, peer_flag, peer_flag_key,
                                epoch)
                COMM.wait_signal(mine, epoch)
                _retire(endpoints, peer, barrier, ctas, lane)
                if cutlass.const_expr(not QUIET_IS_GLOBAL):
                    _grid_barrier(barrier, ctas)
                if lane == 0:
                    if round_index >= warmup:
                        samples[round_index - warmup] = (
                            cute.arch.globaltimer() - start)
            cute.arch.sync_warp()
    elif tidx == 0:
        peer = Int32(lane)
        if lane >= rank:
            peer = Int32(lane + 1)
        destination = Uint64(dst[lane])
        destination_key = Uint32(dst_key[lane])
        source = Uint64(src[lane])
        peer_flag = Uint64(flag[lane])
        peer_flag_key = Uint32(flag_key[lane])
        mine = Uint64(my_flag[lane])
        for round_index in cutlass.range(warmup + iters):
            epoch = Uint64(round_index + 1)
            start = cute.arch.globaltimer()
            if cutlass.const_expr(FUSE):
                COMM.put_signal(endpoints, peer, destination, destination_key,
                                source, src_key, BYTES, peer_flag,
                                peer_flag_key, epoch)
            else:
                COMM.put(endpoints, peer, destination, destination_key,
                         source, src_key, BYTES)
                COMM.signal(endpoints, peer, peer_flag, peer_flag_key, epoch)
            COMM.wait_signal(mine, epoch)
            _retire(endpoints, peer, barrier, ctas, lane)
            if cutlass.const_expr(not QUIET_IS_GLOBAL):
                _grid_barrier(barrier, ctas)
            if lane == 0:
                if round_index >= warmup:
                    samples[round_index - warmup] = (
                        cute.arch.globaltimer() - start)


@cute.jit
def alltoall(endpoints: cute.Tensor, rank: Int32, dst: cute.Tensor,
             dst_key: cute.Tensor, src: cute.Tensor, src_key: Uint32,
             flag: cute.Tensor, flag_key: cute.Tensor, my_flag: cute.Tensor,
             warmup: Int32, iters: Int32, samples: cute.Tensor,
             barrier: cute.Tensor, lanes: Int32, BYTES: cutlass.Constexpr,
             FUSE: cutlass.Constexpr, WARP: cutlass.Constexpr):
    _alltoall(endpoints, rank, dst, dst_key, src, src_key, flag, flag_key,
              my_flag, warmup, iters, samples, barrier, BYTES, FUSE,
              WARP).launch(grid=[lanes, 1, 1], block=[32, 1, 1])


COLLECTIVE_THREADS = 256


@cute.kernel
def _alltoall_collective(dst: Uint64, src: Uint64, warmup: Int32,
                         iters: Int32, samples: cute.Tensor,
                         BYTES: cutlass.Constexpr, WORLD: cutlass.Constexpr):
    # bench_tier1_nvshmem.cu's collective arm: every round is the library's
    # own block-scoped all-to-all over contiguous slices; thread 0 times it.
    tidx, _, _ = cute.arch.thread_idx()
    for round_index in cutlass.range(warmup + iters):
        start = cute.arch.globaltimer()
        COMM.alltoall_block(dst, src, BYTES * WORLD)
        if tidx == 0:
            if round_index >= warmup:
                samples[round_index - warmup] = (
                    cute.arch.globaltimer() - start)
        cute.arch.sync_threads()


@cute.jit
def alltoall_collective(dst: Uint64, src: Uint64, warmup: Int32,
                        iters: Int32, samples: cute.Tensor,
                        BYTES: cutlass.Constexpr, WORLD: cutlass.Constexpr):
    # 256 threads: NVSHMEM 3.8's collective needs 254 registers, which the
    # CUDA arm's 512-thread launch bound cannot hold (bench_tier1_nvshmem.cu).
    _alltoall_collective(dst, src, warmup, iters, samples, BYTES,
                         WORLD).launch(grid=[1, 1, 1],
                                       block=[COLLECTIVE_THREADS, 1, 1])
