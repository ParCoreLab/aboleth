"""dsl-moe arm: Aboleth's compiled MoE dispatch and combine (Tier 2).

The kernels of the paper's MoE results: `NativeMoESlot` over a plan from
`compile_moe_plan`, set up as aboleth_moe_fused_pack.py sets it up (LANES
endpoint lanes per slot, the platform's qualified profiles).
`aboleth-moe` packs in a separate kernel ahead of the transfer chains (the
default); `aboleth-moe-fused` packs inside each peer's send program (paper
Listing 2's form). One slot per token count, its capacity that count (the
compiler's shape specialization; every baseline gets its buffers sized to
the same count). Its combine contract: the expert side applies the routing
weights to its local experts (the slot hands the rows' expert ids and
weights), combine sums the partials at the source.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import aboleth  # noqa: F401  (configures the compiler plugin before Triton)
import torch

from aboleth.bootstrap import TcpStoreBootstrap
from aboleth.endpoint_profiles import QUALIFIED_ENDPOINT_PROFILES
from aboleth.moe.compiled_plan import MoESpec, compile_moe_plan
from aboleth.moe.native import NativeMoESlot
from aboleth.world import World

import moe_common

LANES = int(os.environ.get("ABOLETH_MOE_LANES", "4"))


def _fused_pack_driver():
    path = (Path(__file__).resolve().parents[3] / "experiments" / "python"
            / "aboleth_moe_fused_pack.py")
    spec = importlib.util.spec_from_file_location("_fused_pack_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Cell:
    def __init__(self, ctx, tokens: int, index: int):
        s = ctx.spec
        self.ctx, self.tokens = ctx, tokens
        quant = (dict(value_type="f8E4M3FN", scale_group=moe_common.GROUP)
                 if s.fp8 else {})
        plan = compile_moe_plan(
            MoESpec(max(tokens, 1), s.hidden, s.topk, s.experts, s.world,
                    **quant),
            geography=ctx.world.transport_geography,
            capacities=ctx.capacities, available_lanes=LANES,
            profiles=QUALIFIED_ENDPOINT_PROFILES, **ctx.facts)
        self.slot = NativeMoESlot(
            ctx.world, plan, owners=ctx.owners, placeholder=ctx.placeholder,
            tag=f"moe-{tokens}", endpoint_base=index * LANES,
            endpoint_capacity=LANES, fuse_pack=ctx.fused)
        NativeMoESlot.bind_input_domain([self.slot])
        self.slot.warmup_transport()
        torch.cuda.synchronize()

    def dispatch(self, x, ids, weights):
        return self.slot.dispatch(x, ids, weights)

    def counts(self, handle):
        s = self.ctx.spec
        first = s.rank * s.local_experts
        return handle[1][first:first + s.local_experts]

    def expert(self, handle):
        s = self.ctx.spec
        x, _, ids, weights = handle
        values = moe_common.dequantize(
            x, self.slot.expert_scales if s.fp8 else None)
        first = s.rank * s.local_experts
        local = (ids >= first) & (ids < first + s.local_experts)
        coefficient = torch.where(
            local, weights * moe_common.expert_factor(ids.long()), 0).sum(
                dim=1)
        y = self.slot.expert_output
        y.copy_((values * coefficient[:, None]).to(torch.bfloat16))
        return y

    def combine(self, handle, y):
        return self.slot.combine(y)

    def resources(self):
        manifest = self.slot.manifest()
        return [{"kernel": name, "size": self.tokens, "registers": registers,
                 "spills": manifest["natural_spills"].get(name),
                 "warps": manifest["natural_warps"].get(name)}
                for name, registers in manifest["natural_registers"].items()]

    def describe(self):
        manifest = self.slot.manifest()
        return {"capacity": manifest["spec"]["capacity"],
                "allocated_bytes": manifest["allocated_root_bytes"],
                "lanes": LANES}

    def close(self):
        torch.cuda.synchronize()
        self.slot.close()


class _Context:
    def __init__(self, args, spec):
        self.spec, self.fused = spec, args.arm == "aboleth-moe-fused"
        self.device = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(self.device)
        free_before = torch.cuda.mem_get_info()[0]
        self.bootstrap = TcpStoreBootstrap.from_env(timeout=float(
            os.environ.get("ABOLETH_BOOTSTRAP_TIMEOUT_S", "300")))
        cells = len(args.tokens.split(","))
        self.world = World(spec.rank, spec.world, device=self.device,
                           bootstrap=self.bootstrap)
        self.world.connect_all("auto", remote_transport="rdma",
                               endpoints_per_peer=cells * LANES)
        self.memory = {"free_before": free_before,
                       "free_after": torch.cuda.mem_get_info()[0]}
        self.memory["library_bytes"] = (self.memory["free_before"]
                                        - self.memory["free_after"])
        driver = _fused_pack_driver()
        driver.LANES = LANES
        self.facts = driver.platform_facts(self.world, self.bootstrap,
                                           self.device)
        self.capacities = {
            kind: min(self.world.endpoint(peer).capacity
                      for peer, backend in self.world.transport_geography
                      if backend == kind)
            for kind in {b for _, b in self.world.transport_geography}}
        self.owners = (torch.arange(spec.experts, device=self.device,
                                    dtype=torch.int32)
                       // spec.local_experts)
        self.placeholder = ((spec.rank + 1) % spec.world) * spec.local_experts
        self._index = 0
        self.meta = {"library": "aboleth", "frontend": "triton",
                     "pack": "fused" if self.fused else "separate",
                     "lanes": LANES,
                     "passes": os.environ.get("ABOLETH_PASSES", ""),
                     "platform_key": self.facts.get("platform")}

    def cell(self, tokens: int) -> _Cell:
        cell = _Cell(self, tokens, self._index)
        self._index += 1
        return cell

    def close(self):
        torch.cuda.synchronize()
        self.world.close()


def open(args, spec):  # noqa: A001 - the arm protocol's name
    return _Context(args, spec)
