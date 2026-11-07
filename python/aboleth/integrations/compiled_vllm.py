"""vLLM facts and prepare/finalize over compiler-owned native MoE artifacts."""

from dataclasses import dataclass
from threading import Lock
import torch
from ..moe.native import NativeMoEFamily


@dataclass(frozen=True)
class PreparedExperts:
    expert_input: torch.Tensor
    local_counts: torch.Tensor
    global_ids: torch.Tensor
    expert_weights: torch.Tensor
    scales: torch.Tensor | None = None


class AbolethMoERuntimeManager:
    def __init__(self, *, rank, world_size, device, bootstrap, internode):
        self.facts = dict(
            rank=rank,
            ranks=world_size,
            device=device,
            bootstrap=bootstrap,
            internode=internode,
        )
        self.handles = {}
        self.lock = Lock()

    def get_handle(self, kwargs, *, num_slots, placement_strategy):
        if kwargs["dtype"] != torch.bfloat16:
            raise NotImplementedError(
                "compiled MoE requires BF16 model activations"
            )
        quant = kwargs.get("quant_dtype")
        group = 0
        value_type = "bf16"
        rounded = kwargs.get("quant_round_scale", False)
        floor = 1e-4
        if quant is not None:
            block = kwargs.get("quant_block_shape")
            if "quant_round_scale" not in kwargs:
                raise ValueError(
                    "FP8 requires the optimized consumer scale-rounding fact"
                )
            if (
                quant != torch.float8_e4m3fn
                or block is None
                or len(block) != 2
            ):
                raise NotImplementedError(
                    "FP8 dispatch requires E4M3 per-token K-block scales"
                )
            group = block[1]
            value_type = "f8E4M3FN"
            floor = 1e-10
        elif rounded:
            raise ValueError("unquantized dispatch has no rounding contract")
        e, r = kwargs["num_experts"], self.facts["ranks"]
        if e % r or kwargs["num_local_experts"] != e // r:
            raise ValueError("expert counts must be balanced")
        if placement_strategy == "linear":
            owners = tuple(i // (e // r) for i in range(e))
        elif placement_strategy == "round_robin":
            owners = tuple(i % r for i in range(e))
        else:
            raise ValueError("expert placement must be linear or round_robin")
        key = (
            kwargs["num_max_tokens_per_rank"],
            kwargs["hidden"],
            kwargs["num_topk"],
            e,
            num_slots,
            owners,
            value_type,
            group,
            rounded,
            floor,
        )
        with self.lock:
            if key not in self.handles:
                self.handles[key] = NativeMoEFamily(
                    **self.facts,
                    capacity=key[0],
                    hidden=key[1],
                    top_k=key[2],
                    experts=e,
                    slots=num_slots,
                    owners=owners,
                    tag=f"compiled-moe-{len(self.handles)}",
                    value_type=value_type,
                    scale_group=group,
                    round_scale=rounded,
                    quant_amax_floor=floor,
                )
            return self.handles[key]

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


class ModularMoERuntime:
    def __init__(
        self,
        handle,
        *,
        rank_expert_offset=0,
        layer_name="",
        slot_provider=None,
    ):
        self.family = handle
        self.slot_provider = slot_provider or (lambda: 0)
        self.has_decode_sibling = len(handle.family.variants) > 1
        self.selected = {}
        self.routed = set()

    def num_dispatchers(self):
        return self.family.ranks

    def output_is_reduced(self):
        return True

    def max_num_tokens_per_rank(self):
        return self.family.capacity

    def topk_indices_dtype(self):
        return torch.int32

    def expert_output_alias(self, expert_input):
        handle = self.selected[self.slot_provider()]
        if handle.expert_output.shape != expert_input.shape:
            raise RuntimeError("expert binding shape mismatch")
        return handle.expert_output

    def prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        *,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        required_tokens,
        quantized_input,
        defer_input_quant,
    ):
        if num_experts != self.family.experts or expert_map is None:
            raise ValueError("missing or mismatched expert contract")
        if apply_router_weight_on_input:
            if topk_ids.shape[1] != 1:
                raise ValueError("router weight on input requires top-k one")
            a1 = a1 * topk_weights.to(a1.dtype)
            topk_weights = torch.ones_like(topk_weights)
        slot = self.slot_provider()
        if slot in self.routed:
            raise RuntimeError(
                "same in-flight slot prepared twice before finalize"
            )
        handle = self.family.select(
            max(int(a1.shape[0]), int(required_tokens)), slot
        )
        if bool(handle.spec.scale_cols) != bool(
            quantized_input and not defer_input_quant
        ):
            raise ValueError(
                "expert quantization request disagrees with the compiled dispatch contract"
            )
        self.selected[slot] = handle
        x, counts, ids, weights = handle.dispatch(a1, topk_ids, topk_weights)
        self.routed.add(slot)
        return PreparedExperts(
            x,
            self.family.local_counts(counts),
            ids,
            weights,
            handle.expert_scales,
        )

    def finalize(self, output, fused_expert_output):
        slot = self.slot_provider()
        if slot not in self.routed:
            raise RuntimeError("finalize without a matching prepared slot")
        result = self.selected[slot].combine(fused_expert_output)
        output.copy_(result)
        self.routed.remove(slot)
