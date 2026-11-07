"""bench_kernels.cuh's three benchmarks as Triton kernels over library calls.

One transcription for every Triton arm that links a communication library
(the Tier-1 bitcode, NVSHMEM's bitcode); triton_launch.load() instantiates
this module once per arm with its policy as COMM, which provides the
cute_bench.py signatures:

    put(endpoints, peer, dst, dst_key, src, src_key, nbytes)
    put_signal(endpoints, peer, dst, dst_key, src, src_key, nbytes,
               flag, flag_key, epoch)
    signal(endpoints, peer, flag, flag_key, epoch)
    put_inline(endpoints, peer, dst, dst_key, value)
    quiet(endpoints, peer)
    wait_signal(flag, epoch)
    fence()                          (optional; see cute_bench.py)
    put_warp(...)                    (optional: as put, entered by a whole
    put_signal_warp(...)              warp; the large all-to-all's issue)

A Triton extern call runs on every thread, so the protocol is issued under
`if _tid() == 0:`, Triton-distributed's idiom and the CUDA kernels' one
issuing thread; the grid barrier stays outside it, since Triton synchronizes
the CTA around a scalar atomic. Sizes, put counts and the op are constexpr,
as in the Tier-2 arm (tier2.py); window and iteration counts are run time.

The CUDA kernels' other issue shapes (bench_kernels.cuh) are here too: a
warp-issued all-to-all (kWarpC: the CTA's one warp enters each put_warp,
the protocol stays on thread 0), and the windowed kernel's several issuers
(a CTA of ISSUERS threads, each posting its share of the window, converged
after every put) and streaming (one retirement after the last window, for
libraries that reclaim their own queue).
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language import core

# Injected by triton_launch.load() before this module executes. FENCE is the
# policy's fence, or none: Triton resolves every name a kernel mentions, even
# in a branch a constexpr removes, so the policy cannot simply lack it.
COMM = globals().get("COMM")
QUIET_IS_GLOBAL = globals().get("QUIET_IS_GLOBAL", tl.constexpr(False))
HAS_FENCE = globals().get("HAS_FENCE", tl.constexpr(False))
HAS_PUT_SIGNAL_WARP = globals().get("HAS_PUT_SIGNAL_WARP",
                                    tl.constexpr(False))

OP_PUT = tl.constexpr(0)
OP_PUT_SIGNAL = tl.constexpr(1)
OP_PUT_INLINE = tl.constexpr(2)


@core.extern
def _tid(_semantic=None):
    return core.extern_elementwise(
        "", "", [], {(): ("llvm.nvvm.read.ptx.sreg.tid.x", core.int32)},
        is_pure=True, _semantic=_semantic)


@triton.jit
def _no_fence():
    pass


FENCE = globals().get("FENCE") or _no_fence


@triton.jit
def _no_put_warp(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    tl.static_assert(False, "the policy has no warp-collective put")


@triton.jit
def _no_put_signal_warp(endpoints, peer, dst, dst_key, src, src_key, nbytes,
                        flag, flag_key, epoch):
    tl.static_assert(False, "the policy has no warp-collective "
                     "put-with-signal")


# As FENCE: the warp-collective ops the policy may lack (tracing a call to a
# missing one fails, rather than issuing per-thread puts or nothing).
PUT_WARP = globals().get("PUT_WARP") or _no_put_warp
PUT_SIGNAL_WARP = globals().get("PUT_SIGNAL_WARP") or _no_put_signal_warp


@triton.jit
def _sync_warp():
    # __syncwarp(): converge the warp after each put, as the CUDA kernel's
    # several issuers do, so a library may coalesce the warp's postings.
    tl.inline_asm_elementwise("bar.warp.sync -1;\n mov.u32 $0, $1;", "=r,r",
                              [_tid()], dtype=tl.int32, is_pure=False,
                              pack=1)


@triton.jit
def _grid_barrier(barrier, ctas):
    # bench::grid_epoch_barrier, as tier2.py's all-to-all spells it.
    tl.debug_barrier()
    arrived = tl.atomic_add(barrier, 1, sem="acq_rel") + 1
    target = ((arrived - 1) // ctas + 1) * ctas
    while tl.load(barrier, volatile=True) < target:
        pass


@triton.jit
def _retire(endpoints, peer, barrier, ctas, block):
    if QUIET_IS_GLOBAL:
        _grid_barrier(barrier, ctas)
        if _tid() == 0:
            if block == 0:
                COMM.quiet(endpoints, peer)
        _grid_barrier(barrier, ctas)
    else:
        if _tid() == 0:
            COMM.quiet(endpoints, peer)


@triton.jit(do_not_specialize=["peer", "warmup", "iters"])
def pingpong(endpoints, peer, dst, dst_key, src, src_key, peer_flag, flag_key,
             my_flag, warmup, iters, samples, BYTES: tl.constexpr,
             PUTS: tl.constexpr, OP: tl.constexpr, LEADER: tl.constexpr,
             FUSE: tl.constexpr):
    if _tid() == 0:
        for round_index in range(0, warmup + iters):
            epoch = (round_index + 1).to(tl.int64)
            if LEADER:
                start = tl.extra.cuda.globaltimer()
                if OP == OP_PUT_INLINE:
                    COMM.put_inline(endpoints, peer, peer_flag, flag_key,
                                    epoch)
                elif FUSE and OP == OP_PUT:
                    for index in tl.static_range(PUTS - 1):
                        COMM.put(endpoints, peer, dst + index * BYTES,
                                 dst_key, src + index * BYTES, src_key, BYTES)
                    if PUTS > 1 and HAS_FENCE:
                        FENCE()
                    COMM.put_signal(endpoints, peer, dst + (PUTS - 1) * BYTES,
                                    dst_key, src + (PUTS - 1) * BYTES,
                                    src_key, BYTES, peer_flag, flag_key,
                                    epoch)
                else:
                    for index in tl.static_range(PUTS):
                        COMM.put(endpoints, peer, dst + index * BYTES,
                                 dst_key, src + index * BYTES, src_key, BYTES)
                        if OP == OP_PUT_SIGNAL and index + 1 < PUTS:
                            COMM.signal(endpoints, peer, peer_flag, flag_key,
                                        epoch)
                    COMM.signal(endpoints, peer, peer_flag, flag_key, epoch)
                COMM.wait_signal(my_flag, epoch)
                if round_index >= warmup:
                    tl.store(samples + round_index - warmup,
                             tl.extra.cuda.globaltimer() - start)
                # Off the critical path: the echo already came back.
                COMM.quiet(endpoints, peer)
            else:
                COMM.wait_signal(my_flag, epoch)
                COMM.signal(endpoints, peer, peer_flag, flag_key, epoch)
                COMM.quiet(endpoints, peer)


@triton.jit(do_not_specialize=["peer", "window", "iters", "slots",
                               "record", "stream"])
def windowed(endpoints, peer, dst, dst_key, src, src_key, window, iters,
             slots, samples, spans, record, barrier, stream,
             BYTES: tl.constexpr, ISSUERS: tl.constexpr):
    block = tl.program_id(0)
    ctas = tl.num_programs(0)
    cursor = block.to(tl.int64) * window
    if _tid() == 0:
        if record != 0:
            tl.store(spans + 2 * block, tl.extra.cuda.globaltimer())
    if ISSUERS > 1:
        # bench_windowed_bw's several issuers: thread t posts window /
        # ISSUERS consecutive slots, the warp converged after each put, and
        # the window retires once every thread has posted.
        share = window // ISSUERS
        thread = _tid().to(tl.int64)
        for iteration in range(0, iters):
            start = tl.extra.cuda.globaltimer()
            index = (cursor + thread * share) % slots
            for _slot in range(0, share):
                offset = index * BYTES
                COMM.put(endpoints, peer, dst + offset, dst_key, src + offset,
                         src_key, BYTES)
                index = tl.where(index + 1 == slots, 0, index + 1)
                _sync_warp()
            if stream == 0:
                tl.debug_barrier()
                _retire(endpoints, peer, barrier, ctas, block)
                tl.debug_barrier()
            if _tid() == 0:
                if record != 0:
                    if block == 0:
                        tl.store(samples + iteration,
                                 tl.extra.cuda.globaltimer() - start)
            cursor += ctas.to(tl.int64) * window
    else:
        for iteration in range(0, iters):
            start = tl.extra.cuda.globaltimer()
            if _tid() == 0:
                index = cursor % slots
                for _slot in range(0, window):
                    offset = index * BYTES
                    index = tl.where(index + 1 == slots, 0, index + 1)
                    COMM.put(endpoints, peer, dst + offset, dst_key,
                             src + offset, src_key, BYTES)
            if stream == 0:
                _retire(endpoints, peer, barrier, ctas, block)
            if _tid() == 0:
                if record != 0:
                    if block == 0:
                        tl.store(samples + iteration,
                                 tl.extra.cuda.globaltimer() - start)
            cursor += ctas.to(tl.int64) * window
    if stream != 0:
        # Streaming: every window back to back, one retirement at the end.
        tl.debug_barrier()
        _retire(endpoints, peer, barrier, ctas, block)
    if _tid() == 0:
        if record != 0:
            tl.store(spans + 2 * block + 1, tl.extra.cuda.globaltimer())


@triton.jit(do_not_specialize=["rank", "warmup", "iters"])
def alltoall(endpoints, rank, dst, dst_key, src, src_key, flag, flag_key,
             my_flag, warmup, iters, samples, barrier, BYTES: tl.constexpr,
             FUSE: tl.constexpr, WARP: tl.constexpr):
    # Program b drives peer b + (b >= rank); the tables are indexed by
    # program.
    lane = tl.program_id(0)
    ctas = tl.num_programs(0)
    peer = tl.where(lane >= rank, lane + 1, lane)
    destination = tl.load(dst + lane)
    destination_key = tl.load(dst_key + lane)
    source = tl.load(src + lane)
    peer_flag = tl.load(flag + lane)
    peer_flag_key = tl.load(flag_key + lane)
    mine = tl.load(my_flag + lane)
    for round_index in range(0, warmup + iters):
        epoch = (round_index + 1).to(tl.int64)
        start = tl.extra.cuda.globaltimer()
        if WARP:
            # kWarpC: the CTA's warp enters the collective put; the signal
            # (unless fused into it), the wait and the timer stay on thread
            # 0, behind the warp's converged copy.
            if FUSE:
                PUT_SIGNAL_WARP(endpoints, peer, destination, destination_key,
                                source, src_key, BYTES, peer_flag,
                                peer_flag_key, epoch)
            else:
                PUT_WARP(endpoints, peer, destination, destination_key,
                         source, src_key, BYTES)
            tl.debug_barrier()
            if _tid() == 0:
                if not FUSE:
                    COMM.signal(endpoints, peer, peer_flag, peer_flag_key,
                                epoch)
                COMM.wait_signal(mine, epoch)
        else:
            if _tid() == 0:
                if FUSE:
                    COMM.put_signal(endpoints, peer, destination,
                                    destination_key, source, src_key, BYTES,
                                    peer_flag, peer_flag_key, epoch)
                else:
                    COMM.put(endpoints, peer, destination, destination_key,
                             source, src_key, BYTES)
                    COMM.signal(endpoints, peer, peer_flag, peer_flag_key,
                                epoch)
                COMM.wait_signal(mine, epoch)
        _retire(endpoints, peer, barrier, ctas, lane)
        if not QUIET_IS_GLOBAL:
            _grid_barrier(barrier, ctas)
        if _tid() == 0:
            if lane == 0:
                if round_index >= warmup:
                    tl.store(samples + round_index - warmup,
                             tl.extra.cuda.globaltimer() - start)
