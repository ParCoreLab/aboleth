"""dsl-moe arms for cuTile: a tile kernel packs and reduces, another moves.

A cuTile kernel cannot communicate (no library or SIMT calls; peer arrays
reach NVLink peers only), so a cuTile user's MoE exchange is a cuTile kernel
around a transfer on the same stream:

- dispatch: a routing plan (torch, no host sync: each destination rank's
  tokens in order and each token's position there), a cuTile kernel that
  packs and FP8-quantizes (group 128) each destination's records (row,
  scales, token index, the expert ids local to that rank, the weights),
  then the transfer: `cutile-tier2` Aboleth's Tier-2 send kernel
  (dispatch_kernels.send_tier2: records, count and epoch per peer, one
  launch per transport partition), `cutile-nccl` NCCL's all_to_all_single
  at fixed capacity (counts, then every peer's records), as a PyTorch user
  exchanges;
- combine: each expert rank's rows (one per received record, the local
  experts' weighted sum) return the same way, and a cuTile kernel sums each
  token's returned rows at its source.

Capacity per peer is the cell's token count (every baseline gets the same).

ABOLETH_CUTILE_SEND_BARRIER=1 (diagnostic) restores the grid barrier across
the send's partition launches, which deadlocks when the launches do not run
at once (dispatch_kernels.send_tier2).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import aboleth  # noqa: F401  (configures the compiler plugin before Triton)
import torch
import torch.distributed as dist

import cuda.tile as ct

import moe_common

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "triton_dsl"))
ConstInt = ct.Constant[int]
GROUP = moe_common.GROUP
SEND_BARRIER = os.environ.get("ABOLETH_CUTILE_SEND_BARRIER", "0") == "1"


@ct.kernel
def _pack(x, ids, weights, plan, quantized, scales, token_ids, ids_out,
          weights_out, H: ConstInt, GROUPS: ConstInt, QGROUP: ConstInt,
          K: ConstInt, LOCAL: ConstInt):
    # One block per (destination rank, slot): the slot's token, quantized,
    # with the expert ids that rank owns (others -1) and the weights.
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
    choices = ct.reshape(ct.arange(K, dtype=ct.int32), (1, K))
    experts = ct.gather(ids, (row, choices))
    first = peer * LOCAL
    local = ct.astype((experts >= first) & (experts < first + LOCAL),
                      ct.int32)
    ct.store(ids_out, index=(peer, slot, 0),
             tile=ct.reshape(experts * local + (local - 1), (1, 1, K)))
    ct.store(weights_out, index=(peer, slot, 0),
             tile=ct.reshape(ct.gather(weights, (row, choices)), (1, 1, K)))


@ct.kernel
def _reduce(returned, positions, out, H: ConstInt, WORLD: ConstInt,
            MAXN: ConstInt):
    # Token t's rows from every rank it went to, summed in FP32, one rank's
    # row at a time (a (WORLD, H) gather made tileiras run for minutes).
    token = ct.bid(0)
    columns = ct.reshape(ct.arange(H, dtype=ct.int32), (1, H))
    total = ct.full((1, H), 0.0, dtype=ct.float32)
    for rank in range(WORLD):
        position = ct.load(positions, index=(token, rank), shape=(1, 1))
        row = rank * MAXN + ct.maximum(position, 0)
        values = ct.astype(ct.gather(returned, (row, columns)), ct.float32)
        total = total + values * ct.astype(position >= 0, ct.float32)
    ct.store(out, index=(token, 0), tile=ct.astype(total, ct.bfloat16))


class _Plan:
    """Each destination rank's tokens in order (padded with -1), counts,
    and each token's slot at every rank (-1 where it does not go)."""

    def __init__(self, world: int, local: int, maxn: int, device):
        self.world, self.local, self.maxn = world, local, maxn
        self.ranks = torch.arange(world, device=device)
        self.table = torch.full((world, maxn + 1), -1, dtype=torch.int32,
                                device=device)
        self.counts = torch.zeros(world, dtype=torch.int64, device=device)

    def build(self, ids: torch.Tensor):
        n = ids.shape[0]
        routed = ((ids // self.local)[:, :, None]
                  == self.ranks[None, None, :]).any(1)
        position = torch.cumsum(routed.to(torch.int64), 0) - 1
        target = torch.where(routed, position, self.maxn)
        tokens = torch.arange(n, dtype=torch.int32, device=ids.device)[
            :, None].expand(n, self.world)
        self.table.fill_(-1)
        self.table.scatter_(1, target.T.contiguous(), tokens.T.contiguous())
        self.counts.copy_(routed.sum(0))
        positions = torch.where(routed, position, -1).to(torch.int32)
        return self.table[:, :self.maxn], positions.contiguous()


def _record_bytes(hidden: int, topk: int) -> int:
    return (hidden + 4 * (hidden // GROUP) + 4 + 8 * topk + 15) // 16 * 16


class _Cell:
    def __init__(self, ctx, tokens: int):
        s = ctx.spec
        self.ctx, self.tokens = ctx, tokens
        world, hidden, k = s.world, s.hidden, s.topk
        groups = hidden // GROUP
        maxn = max(tokens, 1)
        rb, rb_back = _record_bytes(hidden, k), 2 * hidden
        self.maxn, self.rb, self.rb_back = maxn, rb, rb_back
        device = ctx.device
        self.plan = _Plan(world, s.local_experts, maxn, device)
        # Forward: [staging | receive | counts | flags]; return the same.
        forward = 2 * world * maxn * rb
        back = 2 * world * maxn * rb_back
        counts_bytes = flags_bytes = 8 * world
        size = forward + back + 2 * (counts_bytes + flags_bytes)
        self.arena = torch.zeros(size, dtype=torch.uint8, device=device)
        a = self.arena
        self.staging = a[0: world * maxn * rb].view(world, maxn, rb)
        self.receive = a[world * maxn * rb: forward].view(world, maxn, rb)
        base = forward
        self.back_staging = a[base: base + world * maxn * rb_back].view(
            world, maxn, rb_back)
        self.back_receive = a[base + world * maxn * rb_back:
                              base + back].view(world, maxn, rb_back)
        tail = forward + back
        self.counts_in = a[tail: tail + 8 * world].view(torch.int64)
        self.flags = a[tail + 8 * world: tail + 16 * world].view(torch.int64)
        self.back_counts = a[tail + 16 * world:
                             tail + 24 * world].view(torch.int64)
        self.back_flags = a[tail + 24 * world:
                            tail + 32 * world].view(torch.int64)
        layout = (hidden, 4 * groups, 4, 4 * k, 4 * k)

        def fields(records):
            offsets = [sum(layout[:i]) for i in range(len(layout) + 1)]
            q = records[:, :, offsets[0]:offsets[1]].view(torch.float8_e4m3fn)
            sc = records[:, :, offsets[1]:offsets[2]].view(torch.float32)
            tok = records[:, :, offsets[2]:offsets[3]].view(torch.int32)[
                :, :, 0]
            ids = records[:, :, offsets[3]:offsets[4]].view(torch.int32)
            w = records[:, :, offsets[4]:offsets[5]].view(torch.float32)
            return q, sc, tok, ids, w
        self.out_fields = fields(self.staging)
        self.in_fields = fields(self.receive)
        self.back_rows = self.back_staging.view(torch.bfloat16)  # [w,m,H]
        self.returned = self.back_receive.view(torch.bfloat16).reshape(
            world * maxn, hidden)
        self.output = torch.empty((maxn, hidden), dtype=torch.bfloat16,
                                  device=device)
        self.epoch = torch.zeros(1, dtype=torch.int64, device=device)
        self.back_epoch = torch.zeros(1, dtype=torch.int64, device=device)
        self.barrier = torch.zeros(1, dtype=torch.int32, device=device)
        if ctx.nccl:
            self.send, self.send_back = self._nccl_send, self._nccl_send_back
        else:
            self._tier2_setup()
            self.send, self.send_back = self._tier2_send, self._tier2_send_back

    # -- Tier-2 transfer (Aboleth's send kernel, per transport partition) --
    def _tier2_setup(self):
        import triton_launch
        from aboleth.planning import plan_transport_partitions
        ctx, world = self.ctx, self.ctx.spec.world
        registration = ctx.world.register(self.arena)
        maxn, rb, rb_back = self.maxn, self.rb, self.rb_back
        forward, back = 2 * world * maxn * rb, 2 * world * maxn * rb_back
        tail = forward + back

        def table(tag, data_offset, data_length, counts_offset,
                  flags_offset):
            return ctx.world.exchange_regions({
                "data": ctx.world.region("data", registration,
                                         offset=data_offset,
                                         length=data_length),
                "counts": ctx.world.region("counts", registration,
                                           offset=counts_offset,
                                           length=8 * world),
                "flags": ctx.world.region("flags", registration,
                                          offset=flags_offset,
                                          length=8 * world)},
                tag=f"cutile-{tag}-{self.tokens}")
        self.forward_table = table("fwd", 0, forward, tail, tail + 8 * world)
        self.back_table = table("back", forward, back, tail + 16 * world,
                                tail + 24 * world)
        self.kernels = triton_launch.load(None, quiet_is_global=False,
                                          name="cutile-moe",
                                          source="dispatch_kernels.py")
        handles = self.forward_table.endpoint_handles.view(torch.int64)
        self.partitions = []
        for index, spec in enumerate(
                plan_transport_partitions(ctx.world.transport_geography)):
            selection = torch.tensor(spec.peers, device=handles.device)
            self.partitions.append((spec, handles[selection].contiguous(),
                                    None if index == 0
                                    else torch.cuda.Stream()))
        self.loaded = False

    def _send_arguments(self, round_, spec, endpoints):
        table, base, rb, counts, flags, epoch, receive_offset = round_
        rank, world = self.ctx.spec.rank, self.ctx.spec.world
        return ((endpoints, rank, base, table.local("data").lkey, counts,
                 table.regions("data"), table.regions("counts"),
                 table.regions("flags"), flags, epoch, self.barrier),
                dict(RB=rb, MAXN=self.maxn, RECEIVE=receive_offset,
                     RANK=rank, PROGRAMS=world - 1, PEERS=spec.peers,
                     BARRIER=SEND_BARRIER, num_warps=1))

    def _tier2_round(self, round_):
        stream = torch.cuda.current_stream()
        if not self.loaded:
            # Every partition's kernels compiled and loaded before the first
            # round, so no round pays a compile or a module load (and the
            # BARRIER diagnostic cannot deadlock on lazy loading, the
            # world-8 hang).
            for each in self._rounds():
                for spec, endpoints, _ in self.partitions:
                    a, k = self._send_arguments(each, spec, endpoints)
                    with self.ctx.world.compilation(peers=spec.peers):
                        compiled = self.kernels.send_tier2.warmup(
                            *a, grid=(spec.grid_extent,), **k)
                    compiled._init_handles()
            torch.cuda.synchronize()
            self.loaded = True
        self.barrier.zero_()
        ready = torch.cuda.Event()
        ready.record(stream)
        done = []
        for spec, endpoints, side in self.partitions:
            side = side or stream
            side.wait_event(ready)
            a, k = self._send_arguments(round_, spec, endpoints)
            with torch.cuda.stream(side), \
                    self.ctx.world.compilation(peers=spec.peers):
                self.kernels.send_tier2[(spec.grid_extent,)](*a, **k)
            event = torch.cuda.Event()
            event.record(side)
            done.append(event)
        for event in done:
            stream.wait_event(event)

    def _rounds(self):
        world, maxn = self.ctx.spec.world, self.maxn
        forward = self.forward_table.local("data").address
        back = self.back_table.local("data").address
        return ((self.forward_table, forward, self.rb, self.plan.counts,
                 self.flags, self.epoch, world * maxn * self.rb),
                (self.back_table, back, self.rb_back, self.counts_in,
                 self.back_flags, self.back_epoch,
                 world * maxn * self.rb_back))

    def _tier2_send(self):
        rank = self.ctx.spec.rank
        self.receive[rank].copy_(self.staging[rank])
        self.counts_in[rank:rank + 1].copy_(self.plan.counts[rank:rank + 1])
        self.epoch.add_(1)
        self._tier2_round(self._rounds()[0])

    def _tier2_send_back(self):
        rank = self.ctx.spec.rank
        self.back_receive[rank].copy_(self.back_staging[rank])
        self.back_epoch.add_(1)
        self._tier2_round(self._rounds()[1])

    # -- NCCL transfer (torch.distributed all_to_all_single) ---------------
    def _nccl_send(self):
        dist.all_to_all_single(self.counts_in, self.plan.counts)
        dist.all_to_all_single(self.receive.view(-1), self.staging.view(-1))

    def _nccl_send_back(self):
        dist.all_to_all_single(self.back_receive.view(-1),
                               self.back_staging.view(-1))

    # -- the arm protocol ---------------------------------------------------
    def dispatch(self, x, ids, weights):
        s = self.ctx.spec
        table, positions = self.plan.build(ids)
        q, sc, tok, out_ids, out_w = self.out_fields
        ct.launch(torch.cuda.current_stream(), (s.world, self.maxn), _pack,
                  (x, ids.to(torch.int32), weights, table, q, sc, tok,
                   out_ids, out_w, s.hidden, s.hidden // GROUP, GROUP,
                   s.topk, s.local_experts))
        self.send()
        return positions

    def counts(self, positions):
        s = self.ctx.spec
        _, _, _, ids, _ = self.in_fields
        live = (torch.arange(self.maxn, device=ids.device)[None, :]
                < self.counts_in[:, None])
        local = ids[live] - s.rank * s.local_experts
        local = local[(local >= 0) & (local < s.local_experts)]
        return torch.bincount(local.long(), minlength=s.local_experts)

    def expert(self, positions):
        s = self.ctx.spec
        q, sc, _, ids, weights = self.in_fields
        values = moe_common.dequantize(q, sc)
        live = (torch.arange(self.maxn, device=ids.device)[None, :]
                < self.counts_in[:, None])[:, :, None]
        valid = (ids >= 0) & live
        coefficient = torch.where(
            valid, weights * moe_common.expert_factor(ids.long()), 0).sum(-1)
        self.back_rows.copy_((values * coefficient[..., None]).to(
            torch.bfloat16))
        return self.back_rows

    def combine(self, positions, y):
        s = self.ctx.spec
        self.send_back()
        ct.launch(torch.cuda.current_stream(), (self.tokens,), _reduce,
                  (self.returned, positions, self.output, s.hidden, s.world,
                   self.maxn))
        return self.output[:self.tokens]

    def resources(self):
        return [{"kernel": "pack, reduce (cuTile)", "size": self.tokens}]

    def describe(self):
        return {"capacity": self.maxn, "record_bytes": self.rb}

    def close(self):
        torch.cuda.synchronize()
        if not self.ctx.nccl:
            self.forward_table.close()
            self.back_table.close()


class _Context:
    def __init__(self, args, spec):
        if not spec.fp8:
            raise SystemExit(f"{args.arm}: FP8 dispatch only")
        import importlib.metadata as metadata
        self.spec, self.nccl = spec, args.arm == "cutile-nccl-moe"
        self.device = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(self.device)
        free_before = torch.cuda.mem_get_info()[0]
        self.bootstrap = moe_common.bootstrap_class().from_env(timeout=float(
            os.environ.get("ABOLETH_BOOTSTRAP_TIMEOUT_S", "300")))
        if self.nccl:
            dist.init_process_group(
                "nccl", store=dist.PrefixStore("cutile",
                                               self.bootstrap._store),
                rank=spec.rank, world_size=spec.world,
                device_id=torch.device("cuda", self.device))
            dist.all_reduce(torch.zeros(1, device=self.device))
            self.world = None
        else:
            from aboleth.world import World
            self.world = World(spec.rank, spec.world, device=self.device,
                               bootstrap=self.bootstrap)
            self.world.connect_all("auto")
        free_after = torch.cuda.mem_get_info()[0]
        self.memory = {"free_before": free_before, "free_after": free_after,
                       "library_bytes": free_before - free_after}
        self.meta = {"library": "nccl" if self.nccl else "aboleth",
                     "frontend": "cutile",
                     "transfer": ("torch.distributed all_to_all_single"
                                  if self.nccl else "Tier-2 send kernel"),
                     "cuda_tile": metadata.version("cuda-tile")}
        if not self.nccl and SEND_BARRIER:
            self.meta["diagnostic"] = "send grid barrier"

    def cell(self, tokens: int) -> _Cell:
        return _Cell(self, tokens)

    def close(self):
        torch.cuda.synchronize()
        if self.world is not None:
            self.world.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def open(args, spec):  # noqa: A001 - the arm protocol's name
    return _Context(args, spec)
