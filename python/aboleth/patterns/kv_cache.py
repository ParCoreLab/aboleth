from __future__ import annotations

from dataclasses import dataclass

from ..pattern import (
    OpaqueCapabilities,
    Pattern,
)


@dataclass(frozen=True)
class KVCacheMigrationSpec:
    max_pages: int
    page_bytes: int
    num_ranks: int = 1
    slots: int | None = None
    max_consumers: int = 1
    unique_destinations: bool = True
    consumer_asynchronous: bool = True
    consumer_max_inflight: int = 1

    def __post_init__(self):
        if self.max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if self.page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        if self.page_bytes % 8:
            raise ValueError(
                "page_bytes must be aligned to eight bytes"
            )
        if self.num_ranks <= 0:
            raise ValueError("num_ranks must be positive")
        if self.slots is not None and self.slots <= 0:
            raise ValueError("slots must be positive")
        if self.max_consumers <= 0:
            raise ValueError("max_consumers must be positive")
        if self.consumer_max_inflight <= 0:
            raise ValueError(
                "consumer_max_inflight must be positive"
            )
        if (
            self.slots is not None
            and self.slots < self.required_slots
        ):
            raise ValueError(
                "slot count cannot satisfy consumer lifetime"
            )

    @property
    def required_slots(self) -> int:
        return (
            self.consumer_max_inflight + 1
            if self.consumer_asynchronous
            else 1
        )

    @property
    def slot_count(self) -> int:
        return self.slots or self.required_slots


def build_kv_cache_migration_pattern(
    spec: KVCacheMigrationSpec,
) -> Pattern:
    """Build page migration around an opaque cache consumer."""

    flow = Pattern("kv-cache-migration")
    page_shape = (spec.max_pages, spec.page_bytes)
    page_strides = (spec.page_bytes, 1)

    source_pages = flow.region(
        "cache.source.pages",
        element_type="i8",
        shape=page_shape,
        strides=page_strides,
        visibility="local",
        immutable=True,
        registered=True,
    )
    source_rows = flow.region(
        "migration.source_rows",
        element_type="i64",
        shape=(spec.max_pages,),
        strides=(1,),
        visibility="local",
        immutable=True,
    )
    destination_peers = flow.region(
        "migration.destination_peers",
        element_type="i32",
        shape=(spec.max_pages,),
        strides=(1,),
        visibility="local",
        immutable=True,
    )
    destination_rows = flow.region(
        "migration.destination_rows",
        element_type="i64",
        shape=(spec.max_pages,),
        strides=(1,),
        visibility="local",
        immutable=True,
    )
    consumer_ids = flow.region(
        "migration.consumer_ids",
        element_type="i32",
        shape=(spec.max_pages,),
        strides=(1,),
        visibility="local",
        immutable=True,
    )
    count = flow.region(
        "migration.count",
        element_type="i32",
        shape=(1,),
        strides=(1,),
        visibility="local",
        immutable=True,
    )
    work = flow.workset(
        "migration.work",
        fields={
            "source_row": source_rows,
            "destination_peer": destination_peers,
            "destination_row": destination_rows,
            "consumer": consumer_ids,
        },
        bounds={
            "source_row": (0, spec.max_pages),
            "destination_peer": (0, spec.num_ranks),
            "destination_row": (0, spec.max_pages),
            "consumer": (0, spec.max_consumers),
        },
        count=count,
        max_items=spec.max_pages,
        unique_destinations=spec.unique_destinations,
    )

    staging = flow.region(
        "cache.destination.staging",
        element_type="i8",
        shape=page_shape,
        strides=page_strides,
        visibility="peer",
    )
    final_storage = flow.region(
        "cache.destination.storage",
        element_type="i8",
        shape=page_shape,
        strides=page_strides,
        visibility="peer",
        escapes=True,
        registered=True,
        donated=True,
        reuse_interval=1,
        pool_depth=spec.slot_count,
    )
    received = flow.transfer_each(
        "migration.transfer",
        work,
        source_pages,
        staging,
    )
    migrated = flow.copy_each(
        "migration.materialize",
        work,
        received,
        final_storage,
        output="cache.destination.pages",
    )
    flow.opaque_compute(
        "cache.consumer",
        reads=(migrated,),
        asynchronous=spec.consumer_asynchronous,
        max_inflight=spec.consumer_max_inflight,
        event_ordered=spec.consumer_asynchronous,
        capabilities=OpaqueCapabilities(
            completion_contract=(
                "external_event"
                if spec.consumer_asynchronous
                else "current_stream"
            ),
        ),
    )
    return flow
