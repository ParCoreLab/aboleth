"""A bounded MoE program expressed through generic compiler planning APIs.

The library states rank-deduplicated records and the external expert contract.
C++ derives live communication, storage, destination bindings and endpoint
geometry. This module contains no transport implementation or cost heuristic.
"""

from dataclasses import dataclass
from math import prod, isfinite

from ..pattern import (
    Pattern,
    CompiledPatternPlan,
    OpaqueCapabilities,
    PhaseResources,
)
from ..planning import (
    EndpointLaunchPlan,
    EndpointPartitionFacts,
    EndpointProfile,
    RegionEffect,
    plan_endpoint_work,
    plan_storage_mlir,
    plan_transport_partitions,
)

SCALAR_BYTES = {"bf16": 2, "f8E4M3FN": 1, "i32": 4, "i64": 8, "f32": 4}
FIELDS = ("payload", "metadata", "count")


@dataclass(frozen=True)
class MoESpec:
    capacity: int
    hidden: int
    top_k: int
    experts: int
    ranks: int
    value_type: str = "bf16"
    scale_group: int = 0
    round_scale: bool = False
    quant_amax_floor: float = 1e-4

    def __post_init__(self):
        if any(
            type(v) is not int or v <= 0
            for v in (
                self.capacity,
                self.hidden,
                self.top_k,
                self.experts,
                self.ranks,
            )
        ):
            raise ValueError("MoE dimensions must be positive integers")
        if (
            self.ranks < 2
            or self.experts % self.ranks
            or self.top_k > self.experts
        ):
            raise ValueError(
                "MoE requires balanced expert counts and a valid top-k"
            )
        if self.experts >= 2**31:
            raise ValueError(
                "expert IDs must fit the optimized int32 consumer interface"
            )
        if self.capacity * self.hidden * 2 >= 2**31:
            raise ValueError(
                "a bounded per-peer payload must fit a native transfer length"
            )
        if (
            self.value_type not in ("bf16", "f8E4M3FN")
            or type(self.round_scale) is not bool
        ):
            raise ValueError(
                "dispatch requires BF16 or explicit FP8 E4M3 scale facts"
            )
        if (
            type(self.quant_amax_floor) not in (float, int)
            or not isfinite(self.quant_amax_floor)
            or not 0 < self.quant_amax_floor <= 3.402823466e38
        ):
            raise ValueError("quantization floor must be positive finite FP32")
        if self.value_type == "bf16":
            if (
                self.scale_group != 0
                or self.round_scale
                or self.quant_amax_floor != 1e-4
            ):
                raise ValueError("BF16 dispatch has no quantization scale")
        elif (
            type(self.scale_group) is not int
            or self.scale_group <= 0
            or self.scale_group & (self.scale_group - 1)
            or self.hidden % self.scale_group
        ):
            raise ValueError(
                "FP8 scale group must be a power of two dividing hidden"
            )

    @property
    def scale_cols(self):
        return self.hidden // self.scale_group if self.scale_group else 0

    @property
    def metadata_words(self):
        return 2 + 2 * self.top_k + self.scale_cols


@dataclass(frozen=True)
class CompiledMoEPlan:
    spec: MoESpec
    pattern: CompiledPatternPlan
    dispatch_fields: tuple[str, ...]
    return_fields: tuple[str, ...]
    dispatch_launches: tuple[EndpointLaunchPlan, ...]
    return_launches: tuple[EndpointLaunchPlan, ...]
    expert_destination: str
    residual_expert_copy: bool
    dispatch_phase: str
    return_phase: str


def build_pattern(spec: MoESpec, *, bindable_expert_output: bool):
    p = Pattern("native-moe")
    b, h, k, r, e = (
        spec.capacity,
        spec.hidden,
        spec.top_k,
        spec.ranks,
        spec.experts,
    )
    w = spec.metadata_words

    def region(name, shape, dtype="bf16", **facts):
        strides = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
        return p.region(
            name, element_type=dtype, shape=shape, strides=strides, **facts
        )

    tokens = region(
        "tokens",
        (b, h),
        immutable=True,
        valid_count="num_tokens",
        dead_padding=True,
    )
    ids = region(
        "input_ids",
        (b, k),
        "i32",
        immutable=True,
        valid_count="num_tokens",
        dead_padding=True,
    )
    weights = region(
        "input_weights",
        (b, k),
        "f32",
        immutable=True,
        valid_count="num_tokens",
        dead_padding=True,
    )
    owners = region("owners", (e,), "i32", immutable=True)
    packed = region(
        "send_payload",
        (r * b, h),
        spec.value_type,
        visibility="peer",
        registered=True,
    )
    packed_meta = region(
        "send_metadata", (r * b, w), "i32", visibility="peer", registered=True
    )
    packed_count = region(
        "send_counts", (r,), "i64", visibility="peer", registered=True
    )
    slots = region("source_slots", (b, r), "i32")
    overflow = region("overflow", (1,), "i32")
    receive = region(
        "received_payload",
        (r * b * k, h),
        spec.value_type,
        visibility="peer",
        registered=True,
    )
    # One rank-deduplicated record feeds its top-k expert choices. The actual
    # expert interface reads this bounded row view plus the separate choices;
    # the storage pass derives the reached extent instead of a Python property.
    received_rows = p.view(
        "received_rows", receive, shape=(r * b, h), strides=(h, 1)
    )
    received_meta = region(
        "received_metadata",
        (r * b, w),
        "i32",
        visibility="peer",
        registered=True,
    )
    received_count = region(
        "received_counts", (r,), "i64", visibility="peer", registered=True
    )
    expert_ids = region("expert_ids", (r * b, k), "i32")
    expert_weights = region("expert_weights", (r * b, k), "f32")
    expert_counts = region("expert_counts", (e,), "i32")
    scales = (
        (region("expert_scales", (r * b, spec.scale_cols), "f32"),)
        if spec.scale_cols
        else ()
    )
    expert_output = region("expert_output", (r * b, h))
    return_source = region(
        "return_source",
        (r * b, h),
        visibility="peer",
        registered=True,
        donated=True,
        reuse_interval=1,
        pool_depth=1,
    )
    returned = region(
        "returned_payload", (r * b, h), visibility="peer", registered=True
    )
    returned_meta = region(
        "returned_metadata",
        (r * b, w),
        "i32",
        visibility="peer",
        registered=True,
    )
    returned_count = region(
        "returned_counts", (r,), "i64", visibility="peer", registered=True
    )
    final = region("final_output", (b, h), escapes=True)
    packed_values = p.opaque_compute(
        "pack",
        reads=(tokens, ids, weights, owners),
        writes=(packed, packed_meta, packed_count, slots, overflow),
        capabilities=OpaqueCapabilities(
            prefix_read_counts=("num_tokens",) * 3 + ("",)
        ),
    )
    p.use("dispatch-valid-count", packed_values[2])

    def exchange(name, sources, destinations):
        # A generic record interface returns all fields. Consumer demand
        # decides which remote writes survive compiler dataflow simplification.
        return tuple(
            p.transfer(f"{name}.{field}", source, destination)
            for field, source, destination in zip(
                FIELDS, sources, destinations, strict=True
            )
        )

    dispatched = exchange(
        "dispatch",
        packed_values[:3],
        (received_rows, received_meta, received_count),
    )
    expert_meta = p.opaque_compute(
        "unpack",
        reads=(dispatched[1], dispatched[2], packed_values[4]),
        writes=(
            expert_ids,
            expert_weights,
            expert_counts,
            packed_values[4],
            *scales,
        ),
    )
    (output,) = p.opaque_compute(
        "expert",
        reads=(dispatched[0], *expert_meta[:3], *expert_meta[4:]),
        writes=(expert_output,),
        capabilities=OpaqueCapabilities(
            accepts_destination_rewrite=bindable_expert_output
        ),
    )
    return_value = p.copy("expert-to-return", output, return_source)
    p.use("return-valid-count", dispatched[2])
    replies = exchange(
        "return",
        (return_value, dispatched[1], dispatched[2]),
        (returned, returned_meta, returned_count),
    )
    p.opaque_compute(
        "gather",
        reads=(replies[0], return_value, packed_values[3], expert_meta[3]),
        writes=(final,),
    )
    return p.compile()


def compile_moe_plan(
    spec: MoESpec,
    *,
    geography: tuple[tuple[int, str], ...],
    capacities: dict[str, int],
    platform: str,
    queue_memory: str,
    available_lanes: int,
    participants: tuple[tuple[str, str, int], ...],
    profiles: tuple[EndpointProfile, ...],
    bindable_expert_output: bool = True,
    peer_link: str = "",
    participant_links: tuple[str, ...] = (),
) -> CompiledMoEPlan:
    compiled = build_pattern(
        spec, bindable_expert_output=bindable_expert_output
    )
    storage = plan_storage_mlir(compiled.analysis.mlir)
    # Stable symbolic allocation identities describe this program before CUDA
    # binding. The runtime allocates distinct roots and binds rewritten outputs
    # to the exact root selected by C++; no unrelated allocations are aliased.
    effects = {
        name: RegionEffect(
            index + 1,
            1,
            0,
            plan.byte_count,
            plan.byte_count,
            1,
            period=plan.byte_count,
        )
        for index, (name, plan) in enumerate(storage.items())
    }
    backends = tuple(sorted(capacities))
    resources = PhaseResources(
        regions=effects,
        peer_transports=backends,
        peer_capacities=tuple(capacities[b] for b in backends),
        num_peers=spec.ranks,
        transport_geography=geography,
    )
    pattern = compiled.compile_plan(resources)
    partitions = plan_transport_partitions(geography)
    facts = tuple(
        EndpointPartitionFacts(
            p,
            "none" if p.backend == "p2p" else queue_memory,
            platform,
            available_lanes,
            participants,
            peer_link if p.backend == "p2p" else "",
            participant_links,
        )
        for p in partitions
    )

    def phase(name):
        payload_type = spec.value_type if name == "dispatch" else "bf16"
        row_bytes = {
            "payload": spec.hidden * SCALAR_BYTES[payload_type],
            "metadata": spec.metadata_words * 4,
            "count": 8,
        }
        fields = tuple(
            field
            for field in FIELDS
            if compiled.analysis.operation(f"{name}.{field}") is not None
        )
        bound = sum(
            (
                spec.capacity * row_bytes[field]
                if field != "count"
                else row_bytes[field]
            )
            for field in fields
        )
        launches = plan_endpoint_work(
            name=name,
            payload_bytes=bound,
            max_items=spec.capacity,
            partitions=facts,
            profiles=profiles,
            rank_count=spec.ranks,
            grouped_receiver_supported=True,
        )
        return fields, launches

    dispatch, dispatch_launches = phase("dispatch")
    returns, return_launches = phase("return")

    def phase_name(operation):
        return next(
            stage.name
            for stage in pattern.stages
            if any(
                getattr(item, "id", None) == operation
                for item in getattr(stage, "operations", ())
            )
        )

    return CompiledMoEPlan(
        spec,
        pattern,
        dispatch,
        returns,
        dispatch_launches,
        return_launches,
        compiled.opaque_stage("expert").writes[0],
        compiled.analysis.operation("expert-to-return") is not None,
        phase_name("dispatch.payload"),
        phase_name("return.payload"),
    )
