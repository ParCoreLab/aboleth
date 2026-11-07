"""The cuTile NVLink arm: the all-to-all as a cuTile kernel over peer arrays.

cuTile 1.x cannot call SIMT code, link bitcode or call a library inside a
tile kernel, so neither Tier 1 nor NVSHMEM reaches it; what a cuTile kernel
can do is load and store through peer-mapped arrays (cuTile's own
AllGatherMatmul sample reads peers' symmetric-memory tensors in-kernel). This
arm is that: bench_alltoall's protocol written with tile loads and stores,
release and acquire at system scope, and an atomic grid barrier; every peer
an NVLink peer, so one node. The world is Aboleth's p2p World (its region
table maps every peer's arena into this process), only to obtain the mapped
addresses: no Aboleth code runs in the kernel.

cuTile has no device clock, so a sample is the launch's CUDA-event time over
its rounds (every sample of a launch is that mean; the record says so).
"""

from __future__ import annotations

# Known issue: this arm can deadlock with 2 ranks across nodes; the cuTile
# results use the cutile-tier2 and cutile-nccl dispatch arms instead.

import torch

import cuda.tile as ct

ConstInt = ct.Constant[int]


class _View:
    """A peer-mapped device address as a CUDA array."""

    def __init__(self, address: int, elements: int, device: int,
                 typestr: str = "<i4"):
        self.__cuda_array_interface__ = {
            "shape": (elements,), "typestr": typestr, "version": 3,
            "data": (address, False), "strides": None, "stream": None}
        self.device = device


@ct.kernel
def _alltoall(sources, destinations, flags, my_flags, barrier, rounds,
              start_epoch, rank, LANES: ConstInt, WORDS: ConstInt,
              PER_SLOT: ConstInt, BARRIER: ConstInt):
    # Block b drives one peer; the host hands each block its own views (send
    # slot, the peer's arrays, the inbound flag), so blocks index by b only.
    # Destination arrays are indexed in WORDS-word tiles, a slot PER_SLOT.
    lane = ct.bid(0)
    for index in range(rounds):
        epoch = start_epoch + index + 1
        tile = ct.load(sources[lane], index=(0,), shape=(WORDS,),
                       allow_tma=False)
        # Plain stores: TMA does not reach peer-mapped memory.
        ct.store(destinations[lane], index=(rank * PER_SLOT,), tile=tile,
                 allow_tma=False)
        ct.store(flags[lane], index=(rank,),
                 tile=ct.full((1,), epoch, dtype=ct.int64), allow_tma=False,
                 memory_order=ct.MemoryOrder.RELEASE,
                 memory_scope=ct.MemoryScope.SYS)
        seen = ct.load(my_flags[lane], index=(0,), shape=(1,), allow_tma=False,
                       memory_order=ct.MemoryOrder.ACQUIRE,
                       memory_scope=ct.MemoryScope.SYS)
        while ct.sum(seen) < epoch:
            seen = ct.load(my_flags[lane], index=(0,), shape=(1,), allow_tma=False,
                           memory_order=ct.MemoryOrder.ACQUIRE,
                           memory_scope=ct.MemoryScope.SYS)
        # Sense-free grid barrier, as bench::grid_epoch_barrier (one block:
        # nothing to wait for).
        if BARRIER:
            arrived = ct.atomic_add(barrier, (0,), 1,
                                    memory_order=ct.MemoryOrder.ACQ_REL,
                                    memory_scope=ct.MemoryScope.DEVICE) + 1
            target = ((arrived - 1) // LANES + 1) * LANES
            count = ct.load(barrier, index=(0,), shape=(1,), allow_tma=False,
                            memory_order=ct.MemoryOrder.ACQUIRE,
                            memory_scope=ct.MemoryScope.DEVICE)
            while ct.sum(count) < target:
                count = ct.load(barrier, index=(0,), shape=(1,),
                                allow_tma=False,
                                memory_order=ct.MemoryOrder.ACQUIRE,
                                memory_scope=ct.MemoryScope.DEVICE)


def alltoall(world, table, source, source_key, flags, size, stride, receive,
             samples, barrier, records, drain=False, args=None):
    """alltoall.py's launcher protocol. The slices are int32 tiles of
    size/4 words at stride/4-word pitch, so size is a power of two >= 4."""
    if size % 4 or size & (size - 1):
        raise ValueError("cutile-p2p: sizes are powers of two >= 4 bytes")
    rank, world_size = world.rank, world.world_size
    device = samples.device
    peers = [peer for peer in range(world_size) if peer != rank]
    if stride % size:
        raise ValueError("cutile-p2p: the slot stride must be a multiple of "
                         "the size")
    words, per_slot = size // 4, stride // size
    local = table.local("data")
    sources = [_View(local.address + peer * stride, words, device.index)
               for peer in peers]
    inbound = [_View(flags.data_ptr() + peer * 8, 1, device.index, "<i8")
               for peer in peers]
    destinations = [_View(table.remote(peer, "data").address + receive,
                          world_size * per_slot * words, device.index)
                    for peer in peers]
    flag_views = [_View(table.remote(peer, "flags").address, world_size,
                        device.index, "<i8") for peer in peers]
    stream = torch.cuda.current_stream()
    records.append({"kernel": "alltoall", "size": size, "threads": None,
                    "ctas": len(peers), "timer": "cuda events per launch"})
    state = {"epoch": 0}

    def launch(rounds: int) -> None:
        ct.launch(stream, (len(peers),), _alltoall,
                  (sources, destinations, flag_views,
                   inbound, barrier, rounds, state["epoch"], rank,
                   len(peers), words, per_slot, int(len(peers) > 1)))
        state["epoch"] += rounds

    def run(warmup: int, iters: int) -> None:
        barrier.zero_()
        # The drivers zero the flags before each run, so epochs restart.
        state["epoch"] = 0
        if warmup:
            launch(warmup)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        launch(iters)
        end.record(stream)
        torch.cuda.synchronize()
        mean_ns = int(begin.elapsed_time(end) * 1e6 / max(iters, 1))
        samples[:iters] = mean_ns
    return run
