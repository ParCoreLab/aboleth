from __future__ import annotations

from dataclasses import dataclass

from ..pattern import (
    OpaqueCapabilities,
    Pattern,
    Workset,
)


def contiguous_strides(
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def tensor_region(
    pattern: Pattern,
    name: str,
    *,
    element_type: str,
    shape: tuple[int, ...],
    visibility: str = "local",
    immutable: bool = False,
    escapes: bool = False,
    registered: bool = False,
    donated: bool = False,
    reuse_interval: int = 1,
    pool_depth: int = 1,
) -> str:
    return pattern.region(
        name,
        element_type=element_type,
        shape=shape,
        strides=contiguous_strides(shape),
        visibility=visibility,
        immutable=immutable,
        escapes=escapes,
        registered=registered,
        donated=donated,
        reuse_interval=reuse_interval,
        pool_depth=pool_depth,
    )


@dataclass(frozen=True)
class ExchangeLane:
    operation: str
    output: str
    transfer: str
    materialize: str
    staging: str
    storage: str


def batch_exchange(
    pattern: Pattern,
    *,
    name: str,
    source: str,
    element_type: str,
    shape: tuple[int, ...],
    output: str,
    pool_depth: int,
    preserve_staging: bool = False,
) -> ExchangeLane:
    staging = tensor_region(
        pattern,
        f"{name}.staging",
        element_type=element_type,
        shape=shape,
        visibility="peer",
    )
    storage = tensor_region(
        pattern,
        f"{name}.storage",
        element_type=element_type,
        shape=shape,
        visibility="peer",
        escapes=True,
        registered=True,
        donated=True,
        pool_depth=pool_depth,
    )
    transferred = pattern.transfer(
        name,
        source,
        staging,
    )
    if preserve_staging:
        pattern.use(
            f"{name}.staging_observer",
            transferred,
        )
    materialize = f"{name}.materialize"
    result = pattern.copy(
        materialize,
        transferred,
        storage,
        output=output,
    )
    return ExchangeLane(
        operation=name,
        output=result,
        transfer=name,
        materialize=materialize,
        staging=staging,
        storage=storage,
    )


def exchange_lane(
    pattern: Pattern,
    *,
    name: str,
    workset: Workset,
    source: str,
    element_type: str,
    shape: tuple[int, ...],
    output: str,
    pool_depth: int,
    preserve_staging: bool = False,
) -> ExchangeLane:
    staging_name = f"{name}.staging"
    storage_name = f"{name}.storage"
    staging = tensor_region(
        pattern,
        staging_name,
        element_type=element_type,
        shape=shape,
        visibility="peer",
    )
    storage = tensor_region(
        pattern,
        storage_name,
        element_type=element_type,
        shape=shape,
        visibility="peer",
        escapes=True,
        registered=True,
        donated=True,
        pool_depth=pool_depth,
    )
    transfer_id = name
    transferred = pattern.transfer_each(
        transfer_id,
        workset,
        source,
        staging,
    )
    if preserve_staging:
        pattern.use(
            f"{name}.staging_observer",
            transferred,
        )
    materialize_id = f"{name}.materialize"
    result = pattern.copy_each(
        materialize_id,
        workset,
        transferred,
        storage,
        source_index="destination_row",
        destination_index="destination_row",
        output=output,
    )
    return ExchangeLane(
        operation=name,
        output=result,
        transfer=transfer_id,
        materialize=materialize_id,
        staging=staging,
        storage=storage,
    )


def opaque_outputs(
    pattern: Pattern,
    name: str,
    *,
    reads: tuple[str, ...],
    outputs: dict[
        str, tuple[str, tuple[int, ...]]
    ],
    asynchronous: bool = False,
    max_inflight: int = 1,
    event_ordered: bool = False,
    capabilities: OpaqueCapabilities | None = None,
) -> dict[str, str]:
    names = tuple(outputs)
    storage = tuple(
        tensor_region(
            pattern,
            f"{output}.storage",
            element_type=element_type,
            shape=shape,
        )
        for output, (element_type, shape) in outputs.items()
    )
    return dict(zip(
        names,
        pattern.opaque_compute(
            name,
            reads=reads,
            writes=storage,
            outputs=names,
            asynchronous=asynchronous,
            max_inflight=max_inflight,
            event_ordered=event_ordered,
            capabilities=capabilities,
        ),
    ))
