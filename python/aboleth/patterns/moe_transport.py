from __future__ import annotations

from ..physicalization import PhaseBinding, physicalize_pattern
from .moe import MoEPatternSpec, build_moe_pattern


def moe_phase_bindings(
    spec: MoEPatternSpec,
    *,
    items_per_message: int = 0,
) -> tuple[PhaseBinding, ...]:
    return (
        PhaseBinding(
            producer="route",
            consumer="experts",
            name="moe.dispatch",
            kernel="pack_dispatch_alltoall_kernel",
            insertion_point="payload",
            peer_segments=spec.num_ranks,
            items_per_transfer=items_per_message,
            publication="dispatch.ready",
        ),
        PhaseBinding(
            producer="experts",
            consumer="combine",
            name="moe.return",
            kernel="return_alltoall_kernel",
            insertion_point="payload",
            peer_segments=spec.num_ranks,
            publication="return.ready",
        ),
    )


def physicalize_moe_pattern(
    spec: MoEPatternSpec,
    *,
    items_per_message: int = 0,
):
    if spec.has_scales:
        raise NotImplementedError(
            "the MoE runtime attachment does not bind scale buffers"
        )
    return physicalize_pattern(
        build_moe_pattern(spec).compile(),
        moe_phase_bindings(
            spec,
            items_per_message=items_per_message,
        ),
    )
