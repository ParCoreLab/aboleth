from __future__ import annotations

from dataclasses import dataclass
from math import prod

from ..execution import (
    CapacityContract,
    CompiledExecutionPlan,
    ExecutionContract,
    ExecutionFallback,
    PaddingSemantics,
    RankControl,
    RuntimeGuard,
)
from ..kv.profile import (
    GENERIC_KV_TARGET_PROFILE,
    KVTargetProfile,
)
from ..pattern import (
    P2PRealizationProfile,
    PhaseResources,
)
from ..phase import CompiledPhase, compile_phase
from ..physicalization import (
    PhaseBinding,
    physicalize_pattern,
)
from ..planning import RegionEffect
from .kv_cache import (
    KVCacheMigrationSpec,
    build_kv_cache_migration_pattern,
)


KV_CACHE_P2P_REALIZATION = P2PRealizationProfile(
    direct_destination_kernel=(
        "kv_cache_direct_page_write_kernel"
    ),
    direct_publication_kernels=(
        (
            "coalesced_kernel",
            "kv_cache_consumer_publish_kernel",
        ),
        (
            "separate_kernel",
            "kv_cache_consumer_publish_kernel",
        ),
        ("packed_doorbell", None),
    ),
    parallel_copy_kernel="kv_cache_page_batch_copy_kernel",
    parallel_publication_kernels=(
        ("fused_kernel", None),
        (
            "coalesced_kernel",
            "kv_cache_consumer_publish_kernel",
        ),
        ("consumer_prologue", None),
    ),
)


@dataclass(frozen=True)
class KVCacheLayoutContract:
    contiguous_page_runs: bool = True
    consumer_partitioned: bool = True
    consumer_ordered: bool = False
    graph_stable_addresses: bool = True


@dataclass(frozen=True)
class KVCachePhysicalization:
    name: str
    target_profile: str
    pages_per_transfer: int
    direct_destination: bool
    readiness: str
    slot_count: int
    target_batch_bytes: int
    generated_copy_workers: int
    generated_block_elements: int
    copy_engine_threshold_bytes: int
    copy_engine_streams: int
    copy_engine_max_runs: int
    asynchronous_completion: bool
    consumer_layer_pipeline: bool
    consumers_per_readiness: int

    def __post_init__(self) -> None:
        if not self.target_profile:
            raise ValueError("KV physicalization needs a target profile")
        if self.pages_per_transfer <= 0:
            raise ValueError(
                "pages_per_transfer must be positive"
            )
        if self.readiness not in ("rank", "consumer"):
            raise ValueError("unknown readiness granularity")
        if self.slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if self.generated_copy_workers <= 0:
            raise ValueError(
                "generated_copy_workers must be positive"
            )
        if self.generated_block_elements <= 0:
            raise ValueError(
                "generated_block_elements must be positive"
            )
        if self.copy_engine_threshold_bytes < 0:
            raise ValueError(
                "copy_engine_threshold_bytes cannot be negative"
            )
        if self.copy_engine_streams <= 0:
            raise ValueError(
                "copy_engine_streams must be positive"
            )
        if self.copy_engine_max_runs <= 0:
            raise ValueError(
                "copy_engine_max_runs must be positive"
            )
        if self.consumers_per_readiness <= 0:
            raise ValueError(
                "consumers_per_readiness must be positive"
            )
        if (
            self.consumer_layer_pipeline
            and self.copy_engine_streams != 1
        ):
            raise ValueError(
                "consumer-layer readiness requires one ordered copy stream"
            )


@dataclass(frozen=True)
class CompiledKVCacheMigrationPlan:
    spec: KVCacheMigrationSpec
    phase: CompiledPhase
    fallback_phase: CompiledPhase | None
    execution: CompiledExecutionPlan
    physicalization: KVCachePhysicalization
    layout_contract: KVCacheLayoutContract

    def select_transfer_engine(
        self,
        *,
        transfer_bytes: int,
        run_count: int,
    ) -> str:
        if (
            transfer_bytes
            >= self.physicalization.copy_engine_threshold_bytes
            and run_count
            <= self.physicalization.copy_engine_max_runs
        ):
            return "copy_engine"
        return "generated_p2p"

    def validate_batch_layout(
        self,
        *,
        contiguous_page_runs: bool,
        consumer_partitioned: bool,
    ) -> int:
        if (
            self.physicalization.pages_per_transfer > 1
            and (
                not contiguous_page_runs
                or not consumer_partitioned
            )
        ):
            return 1
        return self.physicalization.pages_per_transfer

    def select_phase(
        self,
        *,
        contiguous_page_runs: bool,
        consumer_partitioned: bool,
    ) -> tuple[CompiledPhase, int]:
        pages = self.validate_batch_layout(
            contiguous_page_runs=contiguous_page_runs,
            consumer_partitioned=consumer_partitioned,
        )
        if pages == self.physicalization.pages_per_transfer:
            return self.phase, pages
        if self.fallback_phase is None:
            return self.phase, 1
        return self.fallback_phase, 1

    def describe(self) -> dict:
        return {
            "name": "kv-cache-migration",
            "max_pages": self.spec.max_pages,
            "page_bytes": self.spec.page_bytes,
            "max_consumers": self.spec.max_consumers,
            "physicalization": {
                "name": self.physicalization.name,
                "target_profile": (
                    self.physicalization.target_profile
                ),
                "pages_per_transfer": (
                    self.physicalization.pages_per_transfer
                ),
                "direct_destination": (
                    self.physicalization.direct_destination
                ),
                "readiness": (
                    self.physicalization.readiness
                ),
                "slot_count": (
                    self.physicalization.slot_count
                ),
                "target_batch_bytes": (
                    self.physicalization.target_batch_bytes
                ),
                "generated_copy_workers": (
                    self.physicalization.generated_copy_workers
                ),
                "generated_block_elements": (
                    self.physicalization.generated_block_elements
                ),
                "copy_engine_threshold_bytes": (
                    self.physicalization.copy_engine_threshold_bytes
                ),
                "copy_engine_streams": (
                    self.physicalization.copy_engine_streams
                ),
                "copy_engine_max_runs": (
                    self.physicalization.copy_engine_max_runs
                ),
                "asynchronous_completion": (
                    self.physicalization.asynchronous_completion
                ),
                "consumer_layer_pipeline": (
                    self.physicalization.consumer_layer_pipeline
                ),
                "consumers_per_readiness": (
                    self.physicalization.consumers_per_readiness
                ),
                "fallback": (
                    self.fallback_phase.name
                    if self.fallback_phase is not None
                    else None
                ),
            },
            "layout_contract": {
                "contiguous_page_runs": (
                    self.layout_contract.contiguous_page_runs
                ),
                "consumer_partitioned": (
                    self.layout_contract.consumer_partitioned
                ),
                "consumer_ordered": (
                    self.layout_contract.consumer_ordered
                ),
                "graph_stable_addresses": (
                    self.layout_contract.graph_stable_addresses
                ),
            },
            "execution": self.execution.describe(),
        }


def kv_cache_phase_binding(
    spec: KVCacheMigrationSpec,
    *,
    pages_per_transfer: int = 1,
    name: str = "kv_cache.migrate",
    kernel: str = "migrate_pages_kernel",
) -> PhaseBinding:
    return PhaseBinding(
        producer=None,
        consumer="cache.consumer",
        name=name,
        kernel=kernel,
        insertion_point="page_ready",
        peer_segments=spec.num_ranks,
        items_per_transfer=pages_per_transfer,
        publication="migration.ready",
    )


def physicalize_kv_cache_pattern(
    spec: KVCacheMigrationSpec,
    *,
    pages_per_transfer: int = 1,
):
    return physicalize_pattern(
        build_kv_cache_migration_pattern(spec).compile(),
        (
            kv_cache_phase_binding(
                spec,
                pages_per_transfer=pages_per_transfer,
            ),
        ),
    )


def _element_bytes(element_type: str) -> int:
    sizes = {
        "i1": 1,
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "i64": 8,
        "f16": 2,
        "bf16": 2,
        "f32": 4,
        "f64": 8,
    }
    if element_type.startswith("f8"):
        return 1
    return sizes[element_type]


def _phase_resources(
    spec: KVCacheMigrationSpec,
    transports: tuple[str, ...],
) -> PhaseResources:
    compiled = build_kv_cache_migration_pattern(spec).compile()
    effects = {}
    for allocation, region in enumerate(
        compiled.analysis.regions
    ):
        length = max(
            1,
            prod(region.shape)
            * _element_bytes(region.element_type),
        )
        effects[region.id] = RegionEffect(
            allocation=allocation,
            generation=1,
            offset=0,
            length=length,
            stride=length,
            count=1,
            period=length,
        )
    return PhaseResources(
        regions=effects,
        peer_transports=transports,
        peer_capacities=(2**31 - 1,) * len(transports),
        num_peers=spec.num_ranks,
        transport_geography=tuple(enumerate(transports)),
        iteration_count=1,
        recycled_buffers=("cache.destination.storage",),
        advanced_values=("migration.ready",),
        consumer_grid_size=spec.max_consumers,
        p2p_parallel_threshold_bytes=0,
        coalesce_return_with_consumer=False,
        p2p_realization=KV_CACHE_P2P_REALIZATION,
        p2p_direct_destination=False,
        p2p_fused_publication=True,
    )


def _select_pages_per_transfer(
    spec: KVCacheMigrationSpec,
    contract: KVCacheLayoutContract,
    target_batch_bytes: int,
) -> int:
    if not contract.contiguous_page_runs:
        return 1
    maximum = min(
        16,
        spec.max_pages,
        max(1, spec.max_pages // spec.max_consumers),
        max(1, target_batch_bytes // spec.page_bytes),
    )
    return 1 << (maximum.bit_length() - 1)


def compile_kv_cache_migration(
    spec: KVCacheMigrationSpec,
    *,
    transports: tuple[str, ...] | None = None,
    layout_contract: KVCacheLayoutContract | None = None,
    execution_contract: ExecutionContract | None = None,
    target_profile: KVTargetProfile | None = None,
    target_batch_bytes: int = 1024 * 1024,
    generated_copy_workers: int | None = None,
    generated_block_elements: int | None = None,
    copy_engine_threshold_bytes: int | None = None,
    copy_engine_streams: int | None = None,
    copy_engine_max_runs: int | None = None,
    asynchronous_completion: bool = True,
    consumer_layer_pipeline: bool | None = None,
    consumers_per_readiness: int | None = None,
) -> CompiledKVCacheMigrationPlan:
    if target_batch_bytes <= 0:
        raise ValueError("target_batch_bytes must be positive")
    transports = transports or ("p2p",) * spec.num_ranks
    if len(transports) != spec.num_ranks:
        raise ValueError(
            "KV topology requires one transport per rank"
        )
    layout_contract = (
        layout_contract or KVCacheLayoutContract()
    )
    target_profile = (
        target_profile or GENERIC_KV_TARGET_PROFILE
    )
    generated_copy_workers = (
        generated_copy_workers
        if generated_copy_workers is not None
        else target_profile.generated_copy_workers
    )
    generated_block_elements = (
        generated_block_elements
        if generated_block_elements is not None
        else target_profile.generated_block_elements
    )
    copy_engine_threshold_bytes = (
        copy_engine_threshold_bytes
        if copy_engine_threshold_bytes is not None
        else target_profile.copy_engine_threshold_bytes
    )
    copy_engine_streams = (
        copy_engine_streams
        if copy_engine_streams is not None
        else target_profile.copy_engine_streams
    )
    copy_engine_max_runs = (
        copy_engine_max_runs
        if copy_engine_max_runs is not None
        else target_profile.copy_engine_max_runs
    )
    if consumer_layer_pipeline is None:
        consumer_layer_pipeline = (
            spec.consumer_asynchronous
            and spec.max_consumers > 1
            and layout_contract.consumer_ordered
        )
    if consumers_per_readiness is None:
        consumers_per_readiness = min(
            target_profile.consumers_per_readiness,
            spec.max_consumers,
        )
    pages_per_transfer = _select_pages_per_transfer(
        spec,
        layout_contract,
        target_batch_bytes,
    )
    execution_contract = execution_contract or ExecutionContract(
        capacity=CapacityContract(
            max_items=spec.max_pages,
            dynamic_valid_count=True,
            padding=PaddingSemantics.DEAD,
            graph_stable_addresses=(
                layout_contract.graph_stable_addresses
            ),
        ),
        rank_control=RankControl.DYNAMIC,
        guards=(
            RuntimeGuard(
                "page_count_within_capacity",
                ExecutionFallback.ERROR,
            ),
            RuntimeGuard(
                "contiguous_batch_layout",
                ExecutionFallback.EAGER,
            ),
        ),
        fallback=ExecutionFallback.EAGER,
    )
    binding = kv_cache_phase_binding(
        spec,
        pages_per_transfer=pages_per_transfer,
    )
    phase = compile_phase(
        build_kv_cache_migration_pattern(spec),
        binding,
        _phase_resources(spec, transports),
        cyclic=spec.slot_count > 1,
    )
    fallback_phase = (
        compile_phase(
            build_kv_cache_migration_pattern(spec),
            kv_cache_phase_binding(
                spec,
                pages_per_transfer=1,
                name="kv_cache.migrate.fallback",
                kernel="migrate_pages_fallback_kernel",
            ),
            _phase_resources(spec, transports),
            cyclic=spec.slot_count > 1,
        )
        if pages_per_transfer > 1
        else None
    )
    direct_destination = (
        "migration.transfer"
        in phase.plan.analysis.retargeted_destinations
    )
    if spec.unique_destinations and not direct_destination:
        raise RuntimeError(
            "unique KV pages failed direct-destination proof"
        )
    physicalization = KVCachePhysicalization(
        name=(
            "parallel-batched-direct-pages"
            if direct_destination
            and pages_per_transfer > 1
            else (
                "parallel-direct-pages"
                if direct_destination
                else "staged-pages"
            )
        ),
        target_profile=target_profile.name,
        pages_per_transfer=pages_per_transfer,
        direct_destination=direct_destination,
        readiness=(
            "consumer"
            if spec.consumer_asynchronous
            or spec.max_consumers > 1
            else "rank"
        ),
        slot_count=spec.slot_count,
        target_batch_bytes=target_batch_bytes,
        generated_copy_workers=generated_copy_workers,
        generated_block_elements=generated_block_elements,
        copy_engine_threshold_bytes=(
            copy_engine_threshold_bytes
        ),
        copy_engine_streams=copy_engine_streams,
        copy_engine_max_runs=copy_engine_max_runs,
        asynchronous_completion=asynchronous_completion,
        consumer_layer_pipeline=consumer_layer_pipeline,
        consumers_per_readiness=consumers_per_readiness,
    )
    execution = CompiledExecutionPlan(
        phases=(
            (phase, fallback_phase)
            if fallback_phase is not None
            else (phase,)
        ),
        contract=execution_contract,
    )
    return CompiledKVCacheMigrationPlan(
        spec=spec,
        phase=phase,
        fallback_phase=fallback_phase,
        execution=execution,
        physicalization=physicalization,
        layout_contract=layout_contract,
    )
