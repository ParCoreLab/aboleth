from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class KVTargetProfile:
    name: str
    generated_copy_workers: int
    generated_block_elements: int
    copy_engine_threshold_bytes: int
    copy_engine_streams: int
    copy_engine_max_runs: int
    consumers_per_readiness: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("KV target profile needs a name")
        for field_name in (
            "generated_copy_workers",
            "generated_block_elements",
            "copy_engine_streams",
            "copy_engine_max_runs",
            "consumers_per_readiness",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(
                    f"{field_name} must be positive"
                )
        if self.copy_engine_threshold_bytes < 0:
            raise ValueError(
                "copy_engine_threshold_bytes cannot be negative"
            )


@dataclass(frozen=True)
class KVRuntimeOverrides:
    generated_copy_workers: int | None = None
    generated_block_elements: int | None = None
    copy_engine_threshold_bytes: int | None = None
    copy_engine_streams: int | None = None
    asynchronous_completion: bool | None = None
    consumer_layer_pipeline: bool | None = None
    consumers_per_readiness: int | None = None
    profile_transfers: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "generated_copy_workers",
            "generated_block_elements",
            "copy_engine_streams",
            "consumers_per_readiness",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(
                    f"{field_name} must be positive"
                )
        if (
            self.copy_engine_threshold_bytes is not None
            and self.copy_engine_threshold_bytes < 0
        ):
            raise ValueError(
                "copy_engine_threshold_bytes cannot be negative"
            )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
    ) -> KVRuntimeOverrides:
        def optional_int(name: str) -> int | None:
            value = values.get(name)
            return int(value) if value is not None else None

        def optional_bool(name: str) -> bool | None:
            value = values.get(name)
            return bool(value) if value is not None else None

        return cls(
            generated_copy_workers=optional_int(
                "aboleth_copy_workers"
            ),
            generated_block_elements=optional_int(
                "aboleth_block_elements"
            ),
            copy_engine_threshold_bytes=optional_int(
                "aboleth_copy_engine_threshold_bytes"
            ),
            copy_engine_streams=optional_int(
                "aboleth_copy_engine_streams"
            ),
            asynchronous_completion=optional_bool(
                "aboleth_async_transfers"
            ),
            consumer_layer_pipeline=optional_bool(
                "aboleth_consumer_layer_pipeline"
            ),
            consumers_per_readiness=optional_int(
                "aboleth_consumers_per_readiness"
            ),
            profile_transfers=bool(
                values.get(
                    "aboleth_profile_transfers",
                    False,
                )
            ),
        )


GENERIC_KV_TARGET_PROFILE = KVTargetProfile(
    name="generic-generated-p2p",
    generated_copy_workers=16,
    generated_block_elements=256,
    copy_engine_threshold_bytes=2**63 - 1,
    copy_engine_streams=1,
    copy_engine_max_runs=1,
    consumers_per_readiness=1,
)


H200_NVLINK_KV_TARGET_PROFILE = KVTargetProfile(
    name="h200-nvlink",
    generated_copy_workers=16,
    generated_block_elements=256,
    copy_engine_threshold_bytes=64 << 20,
    copy_engine_streams=1,
    copy_engine_max_runs=16,
    consumers_per_readiness=4,
)


def select_kv_target_profile(
    device_name: str,
) -> KVTargetProfile:
    normalized = device_name.upper()
    if "H200" in normalized:
        return H200_NVLINK_KV_TARGET_PROFILE
    return GENERIC_KV_TARGET_PROFILE
