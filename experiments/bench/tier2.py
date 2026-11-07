"""The Tier-2 arm: bench_kernels.cuh's three benchmarks as Triton kernels.

Same protocol as the CUDA kernels, written as a Tier-2 client would: the
communication is `aboleth.*` operations in the kernel and the compiler
chooses batching, publication, completion and issue. ladder.py and
alltoall.py drive these through the launchers below; the drivers own world
setup, correctness checks and reporting, exactly as for the Tier-1 arms.

Each launcher returns `run(...)` with the CUDA launcher's protocol arguments
and records the compiled kernel's registers, spills and cubin size.

A mixed world (NVLink and RDMA peers) compiles one kernel per
backend-homogeneous peer partition, as World.compilation requires; the
all-to-all launches the partitions concurrently on their own streams and
joins them in one grid barrier, so a round still ends when every peer's
exchange has.
"""

from __future__ import annotations

import aboleth  # noqa: F401  (configures the compiler plugin before Triton)
import torch
import triton
import triton.language as tl

import aboleth.language as aboleth_ops
from aboleth.planning import plan_transport_partitions

OP_PUT, OP_PUT_SIGNAL, OP_PUT_INLINE = 0, 1, 2


@triton.jit(do_not_specialize=["warmup", "iters"])
def _pingpong(endpoints, peer, destination, destination_key, source,
              source_key, peer_flag, flag_key, my_flag, warmup, iters,
              samples, BYTES: tl.constexpr, PUTS: tl.constexpr,
              OP: tl.constexpr, LEADER: tl.constexpr):
    # One program; the index is the program id plus a uniform offset, so the
    # compiler proves a single issuer as it would for a per-peer grid.
    ep = aboleth_ops.endpoint(endpoints, tl.program_id(0) + peer)
    dst = destination.to(tl.pointer_type(tl.int8))
    src = source.to(tl.pointer_type(tl.int8))
    flag = peer_flag.to(tl.pointer_type(tl.int64))
    mine = my_flag.to(tl.pointer_type(tl.int64))
    for round_index in tl.range(0, warmup + iters):
        epoch = (round_index + 1).to(tl.int64)
        if LEADER:
            start = tl.extra.cuda.globaltimer()
            if OP == 2:
                s = aboleth_ops.put_inline(ep, flag, epoch,
                                           destination_key=flag_key)
            elif OP == 1:
                # A signal after every put: one completion group retires
                # them all, as the library arms' quiet does.
                group = aboleth_ops.begin_group(ep)
                for index in tl.static_range(PUTS):
                    t = aboleth_ops.put(
                        ep, dst + index * BYTES, src + index * BYTES, BYTES,
                        source_key=source_key,
                        destination_key=destination_key)
                    aboleth_ops.signal(ep, flag, epoch, after=t,
                                       destination_key=flag_key, group=group)
            else:
                t = None
                for index in tl.static_range(PUTS):
                    t = aboleth_ops.put(
                        ep, dst + index * BYTES, src + index * BYTES, BYTES,
                        source_key=source_key,
                        destination_key=destination_key, after=t)
                s = aboleth_ops.signal(ep, flag, epoch, after=t,
                                       destination_key=flag_key)
            aboleth_ops.wait_signal(mine, epoch)
            if round_index >= warmup:
                tl.store(samples + round_index - warmup,
                         tl.extra.cuda.globaltimer() - start)
            # Off the critical path: the echo already came back.
            if OP == 1:
                aboleth_ops.wait_group(group)
            else:
                aboleth_ops.wait(s)
        else:
            aboleth_ops.wait_signal(mine, epoch)
            s = aboleth_ops.signal(ep, flag, epoch, destination_key=flag_key)
            aboleth_ops.wait(s)


@triton.jit(do_not_specialize=["iters", "record"])
def _windowed(endpoints, slot_base, destination, destination_key, source,
              source_key, iters, slots, samples, spans, record,
              BYTES: tl.constexpr, WINDOW: tl.constexpr,
              STREAM: tl.constexpr = False):
    # bench_windowed_bw: per-CTA device spans, whole slots of the arena.
    # STREAM: the windows complete once, after the last, as the libraries'
    # streaming arms do; the wait's own predicate defers the rest, and the
    # compiler retires the queue whenever it fills (self-reclaim).
    program = tl.program_id(0)
    ep = aboleth_ops.endpoint(endpoints, program + slot_base)
    dst = destination.to(tl.pointer_type(tl.int8))
    src = source.to(tl.pointer_type(tl.int8))
    cursor = program.to(tl.int64) * WINDOW
    # A runtime flag: the timed launch must reuse the warm-up's kernel.
    if record != 0:
        tl.store(spans + 2 * program, tl.extra.cuda.globaltimer())
    for iteration in tl.range(0, iters):
        start = tl.extra.cuda.globaltimer()
        t = None
        # (cursor + slot) % slots one slot at a time, as the CUDA kernel
        # does: one 64-bit division per window, not one per put.
        index = cursor % slots
        for slot in tl.static_range(WINDOW):
            offset = index * BYTES
            t = aboleth_ops.put(ep, dst + offset, src + offset, BYTES,
                                source_key=source_key,
                                destination_key=destination_key, after=t)
            index = tl.where(index + 1 == slots, 0, index + 1)
        if STREAM:
            aboleth_ops.wait(t, predicate=iteration == iters - 1)
        else:
            aboleth_ops.wait(t)
        if record != 0:
            if program == 0:
                tl.store(samples + iteration,
                         tl.extra.cuda.globaltimer() - start)
        cursor += tl.num_programs(0).to(tl.int64) * WINDOW
    if record != 0:
        tl.store(spans + 2 * program + 1, tl.extra.cuda.globaltimer())


@triton.jit(do_not_specialize=["warmup", "iters"])
def _alltoall(endpoints, data_table, source, source_key, flag_table,
              my_flags, warmup, iters, samples, barrier,
              BYTES: tl.constexpr, STRIDE: tl.constexpr,
              RECEIVE: tl.constexpr, RANK: tl.constexpr,
              PROGRAMS: tl.constexpr, RECORD: tl.constexpr,
              DRAIN: tl.constexpr, PEERS: tl.constexpr):
    # One program per peer of this partition; `endpoints` is the
    # partition's own table, so program p owns endpoint p (one issuer).
    program = tl.program_id(0)
    peer = tl.full((), PEERS[0], tl.int64)
    for index in tl.static_range(1, len(PEERS)):
        peer = tl.where(program == index, PEERS[index], peer)
    ep = aboleth_ops.endpoint(endpoints, program)
    data = aboleth_ops.remote(data_table, peer)
    flag = aboleth_ops.remote(flag_table, peer, tl.int64)
    src = source.to(tl.pointer_type(tl.int8))
    for round_index in tl.range(0, warmup + iters):
        epoch = (round_index + 1).to(tl.int64)
        start = tl.extra.cuda.globaltimer()
        t = aboleth_ops.put(ep, data.at(RECEIVE + RANK * STRIDE),
                            src + peer * STRIDE, BYTES,
                            source_key=source_key)
        s = aboleth_ops.signal(ep, flag.at(RANK), epoch, after=t)
        aboleth_ops.wait_signal(my_flags + peer, epoch)
        aboleth_ops.wait(s)
        if DRAIN:
            # Diagnostic: also retire as the Tier-1 quiet does, through a
            # drain (a signalled no-op behind its own doorbell, and its
            # poll). It must follow wait(s): the batch's doorbell rings in
            # the wait, and a drain before it queues behind the unsubmitted
            # batch and never completes.
            aboleth_ops.drain(ep)
        # Sense-free barrier over every partition's programs, as
        # bench::grid_epoch_barrier.
        arrived = tl.atomic_add(barrier, 1, sem="acq_rel") + 1
        target = ((arrived - 1) // PROGRAMS + 1) * PROGRAMS
        while tl.load(barrier, volatile=True) < target:
            pass
        if RECORD:
            if program == 0:
                if round_index >= warmup:
                    tl.store(samples + round_index - warmup,
                             tl.extra.cuda.globaltimer() - start)


class _Launch:
    """Record the compiled kernel's resources on the first call."""

    def __init__(self, records: list, name: str, size: int, ctas: int = 0):
        self.records, self.name, self.size = records, name, size
        self.ctas = ctas
        self.compiled = None

    def __call__(self, kernel, grid, *args, **constants):
        compiled = kernel[grid](*args, num_warps=1, **constants)
        if self.compiled is None:
            self.compiled = compiled
            self.records.append({
                "kernel": self.name, "size": self.size,
                "registers": compiled.n_regs, "spills": compiled.n_spills,
                "shared_bytes": compiled.metadata.shared,
                "threads": 32, "ctas": self.ctas or grid[0],
                "cubin_bytes": len(compiled.asm.get("cubin", b""))})


def pingpong(setup, size: int, puts: int, opcode: int):
    launch = _Launch(setup.kernels, "pingpong", size)

    def run(warmup: int, iters: int) -> None:
        with setup.world.compilation():
            launch(_pingpong, (1,), setup.table.endpoint_handles, setup.peer,
                   setup.remote_data.address, setup.remote_data.rkey,
                   setup.local_data.address, setup.local_data.lkey,
                   setup.remote_flags.address + setup.rank * 8,
                   setup.remote_flags.rkey,
                   setup.local_flags.address + setup.peer * 8,
                   warmup, iters, setup.samples, BYTES=size, PUTS=puts,
                   OP=opcode, LEADER=setup.leader)
        torch.cuda.synchronize()
    return run


def windowed(setup, size: int):
    launch = _Launch(setup.kernels, "bandwidth", size, setup.args.ctas)
    fan_out = setup.world.endpoints_per_peer

    def run(iters: int, record: bool, ctas: int) -> None:
        with setup.world.compilation():
            launch(_windowed, (ctas,), setup.table.endpoint_handles,
                   setup.peer * fan_out,
                   setup.remote_data.address, setup.remote_data.rkey,
                   setup.local_data.address, setup.local_data.lkey,
                   iters, setup.data_bytes // size, setup.samples,
                   setup.spans, int(record), BYTES=size,
                   WINDOW=setup.args.window,
                   STREAM=setup.args.stream == "on")
    return run


def alltoall(world, table, source, source_key, flags, size, stride, receive,
             samples, barrier, records, drain=False):
    rank = world.rank
    launches = plan_transport_partitions(world.transport_geography)
    handles = table.endpoint_handles.view(torch.int64)
    partitions = []
    for index, spec in enumerate(launches):
        selection = torch.tensor(spec.peers, device=handles.device)
        partitions.append((
            spec, handles[selection].contiguous(),
            _Launch(records, f"alltoall[{spec.backend}]", size),
            torch.cuda.current_stream() if index == 0
            else torch.cuda.Stream()))
    programs = world.world_size - 1

    def launch_all(warmup: int, iters: int) -> None:
        for index, (spec, endpoints, launch, stream) in enumerate(
                partitions):
            with torch.cuda.stream(stream), world.compilation(peers=spec.peers):
                launch(_alltoall, (spec.grid_extent,), endpoints,
                       table.regions("data"), source, source_key,
                       table.regions("flags"), flags, warmup, iters,
                       samples, barrier, BYTES=size, STRIDE=stride,
                       RECEIVE=receive, RANK=rank, PROGRAMS=programs,
                       RECORD=index == 0, DRAIN=drain, PEERS=spec.peers)
            # A zero-round launch only compiles and loads: with lazy module
            # loading, loading one partition's kernel waits for the others to
            # finish, and they wait for it in the barrier.
            if iters == 0:
                torch.cuda.synchronize()

    def run(warmup: int, iters: int) -> None:
        if partitions[0][2].compiled is None:
            launch_all(0, 0)
        barrier.zero_()
        torch.cuda.synchronize()
        # Every partition must be resident at once: they meet in the
        # barrier, so each runs on its own stream.
        launch_all(warmup, iters)
        torch.cuda.synchronize()
    return run
