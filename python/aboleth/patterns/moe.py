from __future__ import annotations

from dataclasses import dataclass

from ..pattern import (
    OpaqueCapabilities,
    Pattern,
)
from ._exchange import (
    batch_exchange,
    exchange_lane,
    opaque_outputs,
    tensor_region,
)
from ._moe import (
    declare_dispatch_lanes,
    declare_route,
)


@dataclass(frozen=True)
class MoEPatternSpec:
    """Logical expert-parallel protocol around opaque expert compute."""

    max_tokens: int
    hidden: int
    top_k: int
    num_ranks: int = 1
    slots: int = 4
    value_type: str = "bf16"
    scale_width: int = 0
    output_type: str = "bf16"
    experts_asynchronous: bool = True
    expert_max_inflight: int = 1
    expert_event_ordered: bool = True
    preserve_return_staging: bool = False

    def __post_init__(self):
        positive = (
            "max_tokens",
            "hidden",
            "top_k",
            "num_ranks",
            "slots",
            "expert_max_inflight",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.scale_width < 0:
            raise ValueError("scale_width cannot be negative")
        if (
            not self.experts_asynchronous and
            self.expert_max_inflight != 1
        ):
            raise ValueError(
                "synchronous experts require max inflight equal to one"
            )

    @property
    def has_scales(self) -> bool:
        return self.scale_width > 0


def build_moe_pattern(spec: MoEPatternSpec) -> Pattern:
    """Build route → dispatch → experts → return → combine."""

    p = Pattern("moe")
    routes = spec.max_tokens * spec.top_k
    expert_rows = routes * spec.num_ranks

    route_values = declare_route(p, spec, routes)

    dispatch = p.workset(
        "dispatch.work",
        fields={
            "source_row": route_values["route.source_rows"],
            "destination_peer":
                route_values["route.destination_peers"],
            "destination_row":
                route_values["route.destination_rows"],
        },
        bounds={
            "source_row": (0, routes),
            "destination_peer": (0, spec.num_ranks),
            "destination_row": (0, expert_rows),
        },
        max_items=routes,
        unique_destinations=True,
    )

    expert_inputs = declare_dispatch_lanes(
        p, spec, dispatch, route_values, expert_rows
    )
    expert_inputs["counts"] = batch_exchange(
        p,
        name="dispatch.count",
        source=route_values["route.peer_counts"],
        element_type="i32",
        shape=(spec.num_ranks,),
        output="expert.counts",
        pool_depth=spec.slots,
    ).output

    expert_specs = {
        "expert.output": (
            spec.output_type, (expert_rows, spec.hidden)
        ),
    }
    expert_values = opaque_outputs(
        p,
        "experts",
        reads=tuple(expert_inputs.values()),
        outputs=expert_specs,
        asynchronous=spec.experts_asynchronous,
        max_inflight=spec.expert_max_inflight,
        event_ordered=spec.expert_event_ordered,
        capabilities=OpaqueCapabilities(
            accepts_destination_rewrite=True,
            accepts_indexed_scatter=True,
            completion_contract=(
                "external_event"
                if spec.experts_asynchronous
                else "current_stream"
            ),
        ),
    )

    owner_results = batch_exchange(
        p,
        name="return.values",
        source=expert_values["expert.output"],
        element_type=spec.output_type,
        shape=(expert_rows, spec.hidden),
        output="returned.values",
        pool_depth=spec.slots,
        preserve_staging=spec.preserve_return_staging,
    ).output

    output_storage = tensor_region(
        p, "output.values",
        element_type="f32",
        shape=(spec.max_tokens, spec.hidden),
        escapes=True,
    )
    (output,) = p.opaque_compute(
        "combine",
        reads=(
            owner_results,
            route_values["route.source_rows"],
            route_values["route.weights"],
        ),
        writes=(output_storage,),
        outputs=("output",),
    )
    p.use("output.consumer", output)
    return p
