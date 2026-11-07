"""W4: FP8 MoE dispatch, paper Listing 2's shape, as Triton kernels.

One program per peer scans the routes, packs and FP8-quantizes (groups of
128, amax/448, as the MoE path) every token routed to that peer into
consecutive records of its own staging slot, then sends the records, their
count and an epoch. `pack_rows` is shared, so every arm runs the same
computation; only the communication after it differs:

- `dispatch`: a library's calls (COMM, injected by triton_launch.load as for
  triton_bench.py: the Tier-1 bitcode or NVSHMEM's). The records go out with
  the library's collective put, its best form for a large copy (NVSHMEM's
  putmem_nbi_block by the whole CTA, as Triton-distributed sends; Tier 1's
  put_warp by warp 0); the count and the epoch from one thread;
- `dispatch_tier2`: the same chain as Aboleth operations in the kernel IR,
  the compiler choosing issue, batching, publication and completion.

A record is RB bytes: H FP8 values, H/GROUP fp32 scales, the int32 token
index, padding. Each round packs and sends again; a round ends when every
peer's records arrived (grid barrier), and lane 0 records its device time.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language import core

import aboleth.language as aboleth_ops

# Injected by triton_launch.load() before this module executes.
COMM = globals().get("COMM")
QUIET_IS_GLOBAL = globals().get("QUIET_IS_GLOBAL", tl.constexpr(False))
HAS_FENCE = globals().get("HAS_FENCE", tl.constexpr(False))


@core.extern
def _tid(_semantic=None):
    return core.extern_elementwise(
        "", "", [], {(): ("llvm.nvvm.read.ptx.sreg.tid.x", core.int32)},
        is_pure=True, _semantic=_semantic)


@triton.jit
def _grid_barrier(barrier, ctas):
    tl.debug_barrier()
    arrived = tl.atomic_add(barrier, 1, sem="acq_rel") + 1
    target = ((arrived - 1) // ctas + 1) * ctas
    while tl.load(barrier, volatile=True) < target:
        pass


@triton.jit
def pack_rows(x, dest, n, record_base, peer, H: tl.constexpr,
              TOPK: tl.constexpr, GROUP: tl.constexpr, RB: tl.constexpr):
    """Quantize every token routed to `peer` into consecutive records at the
    address `record_base`; return how many."""
    G: tl.constexpr = H // GROUP
    hidden = tl.arange(0, H)
    choice = tl.arange(0, TOPK)
    count = tl.zeros((), tl.int32)
    for token in range(0, n):
        routed = tl.load(dest + token * TOPK + choice)
        if tl.max((routed == peer).to(tl.int32), axis=0) > 0:
            values = tl.load(x + token.to(tl.int64) * H + hidden)
            grouped = tl.reshape(values.to(tl.float32), (G, GROUP))
            amax = tl.max(tl.maximum(tl.abs(grouped), 1e-4), axis=1)
            scale = tl.div_rn(amax, 448.0)
            quantized = tl.clamp(tl.div_rn(grouped, scale[:, None]),
                                 -448.0, 448.0).to(tl.float8e4nv)
            record = record_base + count.to(tl.int64) * RB
            tl.store(record.to(tl.pointer_type(tl.float8e4nv)) + hidden,
                     tl.reshape(quantized, (H,)))
            tl.store((record + H).to(tl.pointer_type(tl.float32))
                     + tl.arange(0, G), scale)
            tl.store((record + H + 4 * G).to(tl.pointer_type(tl.int32)),
                     token)
            count += 1
    return count


@triton.jit
def payload_block(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    COMM.put_block(endpoints, peer, dst, dst_key, src, src_key, nbytes)


@triton.jit
def payload_warp(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    if _tid() < 32:
        COMM.put_warp(endpoints, peer, dst, dst_key, src, src_key, nbytes)


# The policy's collective payload put (set by dispatch.py after loading:
# payload_block where the policy has put_block, else payload_warp).
PAYLOAD = None


@triton.jit
def _retire(endpoints, peer, barrier, ctas, lane):
    if QUIET_IS_GLOBAL:
        _grid_barrier(barrier, ctas)
        if _tid() == 0:
            if lane == 0:
                COMM.quiet(endpoints, peer)
        _grid_barrier(barrier, ctas)
    else:
        if _tid() == 0:
            COMM.quiet(endpoints, peer)
        _grid_barrier(barrier, ctas)


@triton.jit(do_not_specialize=["n", "warmup", "iters"])
def dispatch(endpoints, rank, x, dest, n, staging, staging_key, recv,
             recv_key, cell, cell_key, flag, flag_key, my_flag, warmup,
             iters, samples, barrier, H: tl.constexpr, TOPK: tl.constexpr,
             GROUP: tl.constexpr, RB: tl.constexpr, MAXN: tl.constexpr):
    lane = tl.program_id(0)
    ctas = tl.num_programs(0)
    peer = tl.where(lane >= rank, lane + 1, lane)
    base = staging + lane.to(tl.int64) * (MAXN * RB)
    destination = tl.load(recv + lane)
    destination_key = tl.load(recv_key + lane)
    count_cell = tl.load(cell + lane)
    count_key = tl.load(cell_key + lane)
    peer_flag = tl.load(flag + lane)
    peer_flag_key = tl.load(flag_key + lane)
    mine = tl.load(my_flag + lane)
    for round_index in range(0, warmup + iters):
        epoch = (round_index + 1).to(tl.int64)
        start = tl.extra.cuda.globaltimer()
        count = pack_rows(x, dest, n, base, peer, H, TOPK, GROUP, RB)
        tl.debug_barrier()
        PAYLOAD(endpoints, peer, destination, destination_key, base,
                staging_key, count * RB)
        tl.debug_barrier()
        if _tid() == 0:
            COMM.put_inline(endpoints, peer, count_cell, count_key,
                            count.to(tl.int64))
            COMM.signal(endpoints, peer, peer_flag, peer_flag_key, epoch)
            COMM.wait_signal(mine, epoch)
        _retire(endpoints, peer, barrier, ctas, lane)
        if _tid() == 0:
            if lane == 0:
                if round_index >= warmup:
                    tl.store(samples + round_index - warmup,
                             tl.extra.cuda.globaltimer() - start)


@triton.jit(do_not_specialize=["n", "warmup", "iters"])
def dispatch_tier2(endpoints, rank, x, dest, n, staging, staging_key,
                   data_table, count_table, flag_table, my_flags, warmup,
                   iters, samples, barrier, H: tl.constexpr,
                   TOPK: tl.constexpr, GROUP: tl.constexpr,
                   RB: tl.constexpr, MAXN: tl.constexpr,
                   RECEIVE: tl.constexpr, RANK: tl.constexpr,
                   PROGRAMS: tl.constexpr, RECORD: tl.constexpr,
                   PEERS: tl.constexpr):
    # One program per peer of this transport partition (tier2.py's
    # all-to-all): `endpoints` is the partition's own table.
    program = tl.program_id(0)
    peer = tl.full((), PEERS[0], tl.int64)
    for index in tl.static_range(1, len(PEERS)):
        peer = tl.where(program == index, PEERS[index], peer)
    lane = tl.where(peer > RANK, peer - 1, peer)
    ep = aboleth_ops.endpoint(endpoints, program)
    data = aboleth_ops.remote(data_table, peer)
    counts = aboleth_ops.remote(count_table, peer, tl.int64)
    flags = aboleth_ops.remote(flag_table, peer, tl.int64)
    base = staging + lane * (MAXN * RB)
    for round_index in range(0, warmup + iters):
        epoch = (round_index + 1).to(tl.int64)
        start = tl.extra.cuda.globaltimer()
        count = pack_rows(x, dest, n, base, peer, H, TOPK, GROUP, RB)
        tl.debug_barrier()
        t = aboleth_ops.put(ep, data.at(RECEIVE + RANK * MAXN * RB),
                            base.to(tl.pointer_type(tl.int8)), count * RB,
                            source_key=staging_key)
        t = aboleth_ops.put_inline(ep, counts.at(RANK), count.to(tl.int64),
                                   after=t)
        s = aboleth_ops.signal(ep, flags.at(RANK), epoch, after=t)
        aboleth_ops.wait_signal(my_flags + peer, epoch)
        aboleth_ops.wait(s)
        _grid_barrier(barrier, PROGRAMS)
        if RECORD:
            if program == 0:
                if round_index >= warmup:
                    tl.store(samples + round_index - warmup,
                             tl.extra.cuda.globaltimer() - start)


@triton.jit
def send_tier2(endpoints, rank, staging, staging_key, counts, data_table,
               count_table, flag_table, my_flags, epoch_cell, barrier,
               RB: tl.constexpr, MAXN: tl.constexpr, RECEIVE: tl.constexpr,
               RANK: tl.constexpr, PROGRAMS: tl.constexpr,
               PEERS: tl.constexpr, BARRIER: tl.constexpr = False):
    """The send half of dispatch_tier2 as a standalone kernel: records
    another kernel packed (cuTile's pack, which cannot communicate) into
    staging slot `peer`, with counts[peer] of them, go to every peer of the
    partition. One launch is one round; `epoch_cell` holds the round's epoch
    (device memory, so a CUDA graph of the round replays as is).

    A program ends once its peer's records arrived and its own sends
    completed, so the round ends when every partition's launch has: the
    caller joins the partitions' streams. No grid barrier across the
    partitions: they are separate launches, and CUDA does not promise that
    kernels on different streams run at once, so a barrier between them can
    deadlock (seen at world 16). BARRIER restores it, as
    a diagnostic only."""
    program = tl.program_id(0)
    peer = tl.full((), PEERS[0], tl.int64)
    for index in tl.static_range(1, len(PEERS)):
        peer = tl.where(program == index, PEERS[index], peer)
    ep = aboleth_ops.endpoint(endpoints, program)
    data = aboleth_ops.remote(data_table, peer)
    counts_remote = aboleth_ops.remote(count_table, peer, tl.int64)
    flags = aboleth_ops.remote(flag_table, peer, tl.int64)
    count = tl.load(counts + peer).to(tl.int32)
    epoch = tl.load(epoch_cell)
    base = staging + peer * (MAXN * RB)
    t = aboleth_ops.put(ep, data.at(RECEIVE + RANK * MAXN * RB),
                        base.to(tl.pointer_type(tl.int8)), count * RB,
                        source_key=staging_key)
    t = aboleth_ops.put_inline(ep, counts_remote.at(RANK),
                               count.to(tl.int64), after=t)
    s = aboleth_ops.signal(ep, flags.at(RANK), epoch, after=t)
    aboleth_ops.wait_signal(my_flags + peer, epoch)
    aboleth_ops.wait(s)
    if BARRIER:
        _grid_barrier(barrier, PROGRAMS)
