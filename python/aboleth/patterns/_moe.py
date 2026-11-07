from __future__ import annotations

from ._exchange import (
    exchange_lane,
    opaque_outputs,
    tensor_region,
)


def declare_route(pattern, spec, route_capacity):
    tokens = tensor_region(
        pattern,
        "input.tokens",
        element_type=spec.value_type,
        shape=(spec.max_tokens, spec.hidden),
        immutable=True,
    )
    expert_ids = tensor_region(
        pattern,
        "input.expert_ids",
        element_type="i64",
        shape=(spec.max_tokens, spec.top_k),
        immutable=True,
    )
    topk_weights = tensor_region(
        pattern,
        "input.topk_weights",
        element_type="f32",
        shape=(spec.max_tokens, spec.top_k),
        immutable=True,
    )
    reads = [tokens, expert_ids, topk_weights]
    outputs = {
        "route.values": (
            spec.value_type,
            (route_capacity, spec.hidden),
        ),
        "route.source_rows": ("i64", (route_capacity,)),
        "route.destination_rows": ("i64", (route_capacity,)),
        "route.destination_peers": ("i32", (route_capacity,)),
        "route.expert_ids": ("i32", (route_capacity,)),
        "route.weights": ("f32", (route_capacity,)),
        "route.peer_counts": ("i32", (spec.num_ranks,)),
    }
    if spec.has_scales:
        reads.append(tensor_region(
            pattern,
            "input.scales",
            element_type="f32",
            shape=(spec.max_tokens, spec.scale_width),
            immutable=True,
        ))
        outputs["route.scales"] = (
            "f32",
            (route_capacity, spec.scale_width),
        )
    return opaque_outputs(
        pattern,
        "route",
        reads=tuple(reads),
        outputs=outputs,
    )


def declare_dispatch_lanes(
    pattern,
    spec,
    workset,
    route,
    expert_rows,
):
    lanes = {
        "values": (
            route["route.values"],
            spec.value_type,
            (expert_rows, spec.hidden),
            "expert.values",
        ),
        "expert_ids": (
            route["route.expert_ids"],
            "i32",
            (expert_rows,),
            "expert.ids",
        ),
    }
    if spec.has_scales:
        lanes["scales"] = (
            route["route.scales"],
            "f32",
            (expert_rows, spec.scale_width),
            "expert.scales",
        )
    return {
        lane: exchange_lane(
            pattern,
            name=f"dispatch.{lane}",
            workset=workset,
            source=source,
            element_type=element_type,
            shape=shape,
            output=output,
            pool_depth=spec.slots,
        ).output
        for lane, (
            source, element_type, shape, output
        ) in lanes.items()
    }
