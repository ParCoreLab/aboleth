"""W4 with a cuTile pack: the forms a cuTile kernel's data can leave in.

A tile kernel cannot communicate (no library or SIMT calls; peer arrays
reach NVLink peers only), so a cuTile user packs in one kernel and sends in
another, on the same stream:

- cutile-tier2: the records go out through Aboleth's Tier-2 send kernel
  (`send_tier2`, per transport partition, as tier2.py's all-to-all);
- cutile-nccl: through NCCL's all_to_all_single at fixed capacity (counts
  first, then MAXN records per peer), the host collective a PyTorch user has.

Per round: a routing plan on the GPU without a host sync (scatter of each
token's position among the peer's tokens), the cuTile pack (one block per
(peer, slot), the same quantization as dispatch_kernels.pack_rows), and the
send. After the first (eager) round, a round is one CUDA graph (plan, pack,
send), replayed per round and timed with CUDA events: a deployment captures
it so, and the kernel boundaries are the cost this arm measures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

import cuda.tile as ct

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "triton_dsl"))

ConstInt = ct.Constant[int]
GROUP = 128


@ct.kernel
def pack(x, plan, quantized, scales, token_ids, H: ConstInt,
         GROUPS: ConstInt, QGROUP: ConstInt):
    peer = ct.bid(0)
    slot = ct.bid(1)
    token = ct.load(plan, index=(peer, slot), shape=(1, 1))
    row = ct.maximum(token, 0)
    columns = ct.reshape(ct.arange(H, dtype=ct.int32), (1, H))
    values = ct.astype(ct.gather(x, (row, columns)), ct.float32)
    grouped = ct.reshape(values, (GROUPS, QGROUP))
    amax = ct.max(ct.maximum(ct.abs(grouped), 1e-4), axis=1, keepdims=True)
    scale = ct.truediv(amax, 448.0)
    clamped = ct.minimum(ct.maximum(ct.truediv(grouped, scale), -448.0),
                         448.0)
    ct.store(quantized, index=(peer, slot, 0),
             tile=ct.reshape(ct.astype(clamped, ct.float8_e4m3fn), (1, 1, H)))
    ct.store(scales, index=(peer, slot, 0),
             tile=ct.reshape(scale, (1, 1, GROUPS)))
    ct.store(token_ids, index=(peer, slot), tile=token)


class _Plan:
    """Each peer's tokens in order, padded with -1, and their counts."""

    def __init__(self, world_size: int, rank: int, maxn: int, device):
        self.world_size, self.rank, self.maxn = world_size, rank, maxn
        self.ranks = torch.arange(world_size, device=device)
        self.table = torch.full((world_size, maxn + 1), -1,
                                dtype=torch.int32, device=device)
        self.counts = torch.zeros(world_size, dtype=torch.int64,
                                  device=device)

    def build(self, dest: torch.Tensor) -> torch.Tensor:
        n = dest.shape[0]
        routed = (dest[:, :, None] == self.ranks[None, None, :]).any(1)
        routed[:, self.rank] = False
        position = torch.cumsum(routed.to(torch.int64), 0) - 1
        # Unrouted tokens land in the spare last column.
        target = torch.where(routed, position, self.maxn)
        tokens = torch.arange(n, dtype=torch.int32,
                              device=dest.device)[:, None].expand(
                                  n, self.world_size)
        self.table.fill_(-1)
        self.table.scatter_(1, target.T.contiguous(), tokens.T.contiguous())
        self.counts.copy_(routed.sum(0))
        return self.table[:, :self.maxn]


def launcher(args, world, table, arena_info, records):
    import triton_launch
    from aboleth.planning import plan_transport_partitions
    rank, world_size = world.rank, world.world_size
    device, rb, maxn = arena_info["device"], arena_info["rb"], \
        arena_info["maxn"]
    hidden, groups = args.hidden, args.hidden // GROUP
    arena = arena_info["arena"]
    staging = arena[0: world_size * maxn * rb].view(world_size, maxn, rb)
    quantized = staging[:, :, :hidden].view(torch.float8_e4m3fn)
    scales = staging[:, :, hidden: hidden + 4 * groups].view(torch.float32)
    token_ids = staging[:, :, hidden + 4 * groups:
                        hidden + 4 * groups + 4].view(torch.int32)[:, :, 0]
    plan = _Plan(world_size, rank, maxn, device)
    epoch = torch.zeros(1, dtype=torch.int64, device=device)
    records.append({"kernel": "pack (cuTile)", "ctas": world_size * maxn,
                    "timer": "cuda events per round"})

    def pack_round(x, dest):
        table_view = plan.build(dest)
        epoch.add_(1)
        ct.launch(torch.cuda.current_stream(), (world_size, maxn), pack,
                  (x, table_view, quantized, scales, token_ids, hidden,
                   groups, GROUP))

    if args.arm == "cutile-nccl":
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl", init_method="tcp://{}:{}".format(
                    os.environ.get("MASTER_ADDR", "127.0.0.1"),
                    int(os.environ.get("MASTER_PORT", "29500")) + 97),
                rank=rank, world_size=world_size, device_id=device)
        receive = arena[arena_info["recv"]:
                        arena_info["recv"] + world_size * maxn * rb]
        counts_in = arena_info["counts_view"]

        def send_round():
            dist.all_to_all_single(counts_in, plan.counts)
            dist.all_to_all_single(receive, arena[0: world_size * maxn * rb])
    else:
        kernels = triton_launch.load(None, quiet_is_global=False,
                                     name="cutile-tier2",
                                     source="dispatch_kernels.py")
        local = table.local("data")
        handles = table.endpoint_handles.view(torch.int64)
        partitions = []
        for index, spec in enumerate(
                plan_transport_partitions(world.transport_geography)):
            selection = torch.tensor(spec.peers, device=handles.device)
            partitions.append((spec, handles[selection].contiguous(),
                               None if index == 0 else torch.cuda.Stream()))
        barrier = torch.zeros(1, dtype=torch.int32, device=device)
        state = {"recorded": False}

        def send_arguments(spec, endpoints):
            return ((endpoints, rank, local.address, local.lkey, plan.counts,
                     table.regions("data"), table.regions("counts"),
                     table.regions("flags"), arena_info["flags_view"], epoch,
                     barrier),
                    dict(RB=rb, MAXN=maxn, RECEIVE=arena_info["recv"],
                         RANK=rank, PROGRAMS=world_size - 1,
                         PEERS=spec.peers, num_warps=1))
        # send_tier2 has no grid barrier across the partitions (it could
        # deadlock); a round ends when the joined streams have.

        def load_partitions():
            # Every partition's kernel compiled and loaded before the first
            # launch, so no round pays a compile or a module load.
            for spec, endpoints, _ in partitions:
                arguments, constants = send_arguments(spec, endpoints)
                with world.compilation(peers=spec.peers):
                    compiled = kernels.send_tier2.warmup(
                        *arguments, grid=(spec.grid_extent,), **constants)
                compiled._init_handles()
            torch.cuda.synchronize()
            state["loaded"] = True

        def send_round():
            if not state.get("loaded"):
                load_partitions()
            stream = torch.cuda.current_stream()
            barrier.zero_()
            packed = torch.cuda.Event()
            packed.record(stream)
            done = []
            for spec, endpoints, side in partitions:
                side = side or stream
                side.wait_event(packed)
                arguments, constants = send_arguments(spec, endpoints)
                with torch.cuda.stream(side), \
                        world.compilation(peers=spec.peers):
                    compiled = kernels.send_tier2[(spec.grid_extent,)](
                        *arguments, **constants)
                if not state["recorded"]:
                    records.append({"kernel": f"send[{spec.backend}]",
                                    "registers": compiled.n_regs,
                                    "spills": compiled.n_spills,
                                    "ctas": spec.grid_extent})
                event = torch.cuda.Event()
                event.record(side)
                done.append(event)
            for event in done:
                stream.wait_event(event)
            state["recorded"] = True

    graphs = {}

    def run(x, dest, tokens, warmup, iters, samples, barrier_unused):
        stream = torch.cuda.current_stream()
        epoch.zero_()
        key = (x.data_ptr(), dest.data_ptr(), tokens)
        if key not in graphs and not graphs.get("eager_done", {}).get(key):
            # The first round of a shape runs eagerly: it compiles, loads and
            # (NCCL) warms the communicator; the drivers check its records.
            for _ in range(warmup + iters):
                pack_round(x, dest)
                send_round()
            torch.cuda.synchronize()
            graphs.setdefault("eager_done", {})[key] = True
            return
        if key not in graphs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pack_round(x, dest)
                send_round()
            graphs[key] = graph
            epoch.zero_()
        graph = graphs[key]
        begin = [torch.cuda.Event(enable_timing=True)
                 for _ in range(warmup + iters)]
        end = [torch.cuda.Event(enable_timing=True)
               for _ in range(warmup + iters)]
        for index in range(warmup + iters):
            begin[index].record(stream)
            graph.replay()
            end[index].record(stream)
        torch.cuda.synchronize()
        for index in range(iters):
            samples[index] = int(begin[warmup + index].elapsed_time(
                end[warmup + index]) * 1e6)
    return run
