from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Optional

from .planning import (
    CommunicationDataflow,
    CommunicationDataflowOperation,
    OutputLifetimePlan,
    RegionEffect,
    SimplifiedCommunicationDataflow,
    StorageAllocationPlan,
    TransportKernel,
    TransportPhase,
    TransportPlan,
    TransportPartitionPlan,
    compile_transport_program_mlir,
    plan_output_lifetimes,
    plan_storage_mlir,
    plan_transport_program,
    plan_transport_facts,
    plan_transport_partitions,
)


@dataclass(frozen=True)
class CommunicationStage:
    name: str
    operations: tuple[CommunicationDataflowOperation, ...]


@dataclass(frozen=True)
class OpaqueStage:
    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    outputs: tuple[str, ...]
    asynchronous: bool
    max_inflight: int
    event_ordered: bool


PatternStage = CommunicationStage | OpaqueStage


@dataclass(frozen=True)
class OpaqueCapabilities:
    accepts_destination_rewrite: bool = False
    accepts_indexed_scatter: bool = False
    accepted_address_spaces: tuple[str, ...] = ("global",)
    completion_contract: str = "current_stream"
    # Empty string denotes a full read; other entries name the exact runtime
    # mask used by the opaque implementation for that read operand.
    prefix_read_counts: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.accepted_address_spaces:
            raise ValueError(
                "opaque stage requires an accepted address space"
            )
        if not self.completion_contract:
            raise ValueError(
                "opaque completion contract cannot be empty"
            )


# The artifact forms materialization must support (ARCHITECTURE.md, "Artifact
# forms"). Mirrors physicalization.Artifact's values, which is the typed
# authoring surface; pattern.py carries the string so the enum can live beside
# the attachment that selects it.
ARTIFACT_KINDS = (
    "inline_fragment",
    "bindable_epilogue",
    "standalone_kernel",
)


@dataclass(frozen=True)
class KernelBinding:
    operation: str
    kernel: str
    insertion_point: str
    artifact: str = "inline_fragment"

    def __post_init__(self):
        if not self.operation or not self.kernel:
            raise ValueError(
                "kernel bindings require operation and kernel names"
            )
        if not self.insertion_point:
            raise ValueError(
                "kernel binding insertion point cannot be empty"
            )
        if self.artifact not in ARTIFACT_KINDS:
            raise ValueError(
                f"unknown artifact kind {self.artifact!r}; expected one of "
                + ", ".join(ARTIFACT_KINDS)
            )

@dataclass(frozen=True)
class Workset:
    name: str
    fields: dict[str, str]
    bounds: dict[str, tuple[int, int]]
    max_items: int
    count: Optional[str]
    unique_destinations: bool
    ordered: bool

    def field(self, name: str) -> str:
        try:
            return self.fields[name]
        except KeyError as error:
            raise KeyError(
                f"workset {self.name!r} has no field {name!r}"
            ) from error


@dataclass(frozen=True)
class KernelTransportArgument:
    name: str
    mlir_type: str

    def __post_init__(self):
        if not self.name:
            raise ValueError(
                "kernel transport arguments require a name"
            )
        if self.mlir_type not in ("i1", "i32", "i64"):
            raise ValueError(
                "kernel transport argument type must be i1, i32, or i64"
            )


@dataclass(frozen=True)
class KernelTransportTransfer:
    operations: tuple[str, ...]
    destination: str
    source: str
    byte_count: str
    source_key: str
    destination_key: str
    predicate: str
    max_bytes: int

    def __post_init__(self):
        if self.max_bytes <= 0:
            raise ValueError(
                "kernel transport transfers require positive max bytes"
            )


@dataclass(frozen=True)
class KernelTransportSignal:
    destination: str
    source: str
    source_key: str
    destination_key: str
    iteration: str
    iterations: str


@dataclass(frozen=True)
class KernelTransportReduction:
    """The reduce contract one physicalized reduction carries.

    The transfer geometry says how much moves; this says what combines it.
    `cardinality` is the declared upper bound on contributions and
    `cardinality_source` names the index region that supplies the runtime count
    when the reduction reads an indexed view, or None when the bound is static.
    """

    operation: str
    operator: str
    cardinality: int
    cardinality_source: Optional[str] = None

    def __post_init__(self):
        if not self.operation or not self.operator:
            raise ValueError(
                "reduce contracts require an operation and an operator"
            )
        if self.cardinality <= 0:
            raise ValueError(
                "reduce contract cardinality must be positive"
            )


@dataclass(frozen=True)
class KernelTransportTransform:
    """The kernel obligation one physicalized transform carries.

    A transform is not a wire command: the client kernel that packs the
    phase must apply `transform` while producing the destination regions.
    Transforms sharing an id stem fuse into one obligation with several
    destinations (values plus scale sidecars); `source_type` and
    `destination_types` record the dtype change the wire then assumes.
    """

    operation: str
    transform: str
    source: str
    source_type: str
    destinations: tuple[str, ...]
    destination_types: tuple[str, ...]

    def __post_init__(self):
        if not self.operation or not self.transform:
            raise ValueError(
                "transform contracts require an operation and a "
                "transform name"
            )
        if not self.destinations or (
            len(self.destinations) != len(self.destination_types)
        ):
            raise ValueError(
                "transform contracts require matched destination "
                "regions and element types"
            )


@dataclass(frozen=True)
class KernelTransportProgram:
    name: str
    operations: tuple[str, ...]
    kernel: str
    insertion_point: str
    arguments: tuple[KernelTransportArgument, ...]
    transfers: tuple[KernelTransportTransfer, ...]
    signal: Optional[KernelTransportSignal]
    program_id: Optional[int] = None
    artifact: str = "inline_fragment"
    reductions: tuple[KernelTransportReduction, ...] = ()
    transforms: tuple[KernelTransportTransform, ...] = ()
    external_publication: bool = False
    system_fence: bool = False
    drain_endpoint: bool = False
    publish_head: bool = False
    planned_reservation: bool = False
    reservation_scope: str = "gpu"
    issuer_scope: str = "cta"

    def __post_init__(self):
        if self.program_id is not None and self.program_id < 0:
            raise ValueError(
                "kernel transport program id cannot be negative"
            )
        if self.artifact not in ARTIFACT_KINDS:
            raise ValueError(
                f"unknown artifact kind {self.artifact!r}; expected one of "
                + ", ".join(ARTIFACT_KINDS)
            )
        if not self.name or not self.kernel:
            raise ValueError(
                "kernel transport programs require names and kernels"
            )
        if (
            not self.operations
            and self.signal is None
            and not self.publish_head
            and not self.system_fence
            and not self.drain_endpoint
        ):
            raise ValueError(
                "kernel transport programs require logical operations"
            )
        if len(self.operations) != len(set(self.operations)):
            raise ValueError(
                "kernel transport program operations must be unique"
            )
        if not self.insertion_point:
            raise ValueError(
                "kernel transport programs require an insertion point"
            )
        if (
            not self.transfers
            and self.signal is None
            and not self.publish_head
            and not self.system_fence
            and not self.drain_endpoint
        ):
            raise ValueError(
                "kernel transport programs require at least one transfer"
            )
        modes = (
            int(self.signal is not None)
            + int(self.external_publication)
            + int(self.system_fence)
            + int(self.drain_endpoint)
            + int(self.publish_head)
        )
        if modes != 1:
            raise ValueError(
                "a transport program must select exactly one terminal "
                "contract: signal, external publication, system fence, "
                "endpoint drain, or head publication"
            )
        if (
            self.publish_head
            or self.system_fence
            or self.drain_endpoint
        ) and self.transfers:
            raise ValueError(
                "a boundary-only program cannot carry payload transfers"
            )
        if self.reservation_scope not in ("gpu", "cta"):
            raise ValueError(
                "transport program reservation_scope must be gpu or cta"
            )
        if self.issuer_scope not in ("cta", "thread"):
            raise ValueError(
                "transport program issuer_scope must be cta or thread"
            )
        realized = tuple(
            operation
            for transfer in self.transfers
            for operation in transfer.operations
        )
        if (
            len(realized) != len(set(realized)) or
            set(realized) != set(self.operations)
        ):
            raise ValueError(
                "kernel transport program must realize each logical "
                "operation exactly once"
            )
        names = tuple(
            argument.name for argument in self.arguments
        )
        if len(names) != len(set(names)):
            raise ValueError(
                "kernel transport argument names must be unique"
            )
        available = set(names)
        required = set()
        if not self.system_fence:
            required.add("endpoint")
        if self.system_fence:
            required.add("predicate")
        if self.publish_head:
            required.add("only_if_advanced")
        for transfer in self.transfers:
            required.update((
                transfer.destination,
                transfer.source,
                transfer.byte_count,
                transfer.source_key,
                transfer.destination_key,
                transfer.predicate,
            ))
        if self.signal is not None:
            required.update((
                self.signal.destination,
                self.signal.source,
                self.signal.source_key,
                self.signal.destination_key,
                self.signal.iteration,
                self.signal.iterations,
            ))
        missing = required - available
        if missing:
            raise ValueError(
                "kernel transport program has unbound arguments: "
                + ", ".join(sorted(missing))
            )
        reduced = tuple(
            reduction.operation
            for reduction in self.reductions
        )
        if len(reduced) != len(set(reduced)):
            raise ValueError(
                "kernel transport program reduce contracts must be unique"
            )
        stray = set(reduced) - set(self.operations)
        if stray:
            raise ValueError(
                "reduce contracts name operations the program does not "
                "realize: " + ", ".join(sorted(stray))
            )
        transformed = tuple(
            transform.operation
            for transform in self.transforms
        )
        if len(transformed) != len(set(transformed)):
            raise ValueError(
                "kernel transport program transform contracts must "
                "be unique"
            )
        collisions = set(transformed) & set(self.operations)
        if collisions:
            raise ValueError(
                "transform contracts are kernel obligations and "
                "cannot name wire transfers: "
                + ", ".join(sorted(collisions))
            )
        if self.transforms and (
            self.publish_head
            or self.system_fence
            or self.drain_endpoint
        ):
            raise ValueError(
                "a boundary-only program cannot carry transform "
                "obligations"
            )

    @property
    def command_count(self) -> int:
        return len(self.transfers) + (
            1 if self.signal is not None else 0
        )

    @property
    def payload_bytes(self) -> int:
        return sum(
            transfer.max_bytes
            for transfer in self.transfers
        ) + (8 if self.signal is not None else 0)

    def _fragment_close(self) -> str:
        # Transform obligations ride as a discardable attribute so a
        # program without them prints byte-identically to before the
        # attribute existed.
        if not self.transforms:
            return "    }) : () -> ()"
        rendered = ", ".join(
            "{"
            f'operation = "{transform.operation}", '
            f'transform = "{transform.transform}", '
            f'source = "{transform.source}", '
            f'sourceType = "{transform.source_type}", '
            "destinations = ["
            + ", ".join(
                f'"{name}"' for name in transform.destinations
            )
            + "], destinationTypes = ["
            + ", ".join(
                f'"{name}"'
                for name in transform.destination_types
            )
            + "]}"
            for transform in self.transforms
        )
        return "    }) {transforms = [" + rendered + "]} : () -> ()"

    def to_mlir(self) -> str:
        if self.program_id is None:
            raise ValueError(
                "kernel transport program must be bound before MLIR emission"
            )
        arguments = {
            argument.name: f"%arg{index}"
            for index, argument in enumerate(self.arguments)
        }
        names = ", ".join(
            f'"{argument.name}"'
            for argument in self.arguments
        )
        block = ", ".join(
            f"{arguments[argument.name]}: {argument.mlir_type}"
            for argument in self.arguments
        )
        lines = [
            '    "aboleth.transport_fragment"() <{',
            f"      programId = {self.program_id} : i32,",
            f'      name = "{self.name}",',
            f"      argumentNames = [{names}]",
            "    }> ({",
            f"    ^bb0({block}):",
        ]
        if self.system_fence:
            lines.extend((
                (
                    "      aboleth.system_fence "
                    f'{arguments["predicate"]} : i1'
                ),
                "      ub.unreachable",
                "    }) : () -> ()",
            ))
            return "\n".join(lines)
        lines.append(
            (
                "      %endpoint = aboleth.bind_endpoint "
                f'{arguments["endpoint"]} : i64 -> !aboleth.endpoint'
            )
        )
        if self.drain_endpoint:
            lines.extend((
                (
                    "      aboleth.drain %endpoint "
                    f'{{issuerScope = "{self.issuer_scope}"}} '
                    ": !aboleth.endpoint"
                ),
                "      ub.unreachable",
                "    }) : () -> ()",
            ))
            return "\n".join(lines)
        if self.publish_head:
            lines.extend((
                (
                    "      aboleth.publish_head %endpoint, "
                    f'{arguments["only_if_advanced"]} '
                    ": !aboleth.endpoint, i1"
                ),
                "      ub.unreachable",
                "    }) : () -> ()",
            ))
            return "\n".join(lines)
        previous = None
        for index, transfer in enumerate(self.transfers):
            destination = f"%destination{index}"
            source = f"%source{index}"
            token = f"%transfer{index}"
            lines.extend((
                (
                    f"      {destination} = aboleth.bind_remote_region "
                    f'{arguments[transfer.destination]}, '
                    f'{arguments[transfer.destination_key]} '
                    ": (i64, i32) -> !aboleth.remote_region"
                ),
                (
                    f"      {source} = aboleth.bind_local_region "
                    f'{arguments[transfer.source]}, '
                    f'{arguments[transfer.source_key]} '
                    ": (i64, i32) -> !aboleth.local_region"
                ),
            ))
            operands = (
                f"%endpoint, {destination}, {source}, "
                f'{arguments[transfer.byte_count]}, '
                f'{arguments[transfer.predicate]}'
            )
            types = (
                "!aboleth.endpoint, !aboleth.remote_region, "
                "!aboleth.local_region, i32, i1"
            )
            if previous is not None:
                operands += f", {previous}"
                types += ", !aboleth.token"
            operation = (
                "aboleth.post_put"
                if self.external_publication
                else "aboleth.put"
            )
            lines.append(
                f"      {token} = {operation} {operands} "
                f": ({types}) -> !aboleth.token"
            )
            previous = token
        if self.external_publication:
            tokens = ", ".join(
                f"%transfer{index}"
                for index in range(len(self.transfers))
            )
            types = ", ".join(
                "!aboleth.token"
                for _ in self.transfers
            )
            attributes = [
                f"batchSize = {len(self.transfers)} : i32",
                "deferCompletion",
                "externalPublication",
            ]
            if self.planned_reservation:
                attributes.append("plannedReservation")
            attributes.append(
                f'reservationScope = "{self.reservation_scope}"'
            )
            attributes.append(
                f'issuerScope = "{self.issuer_scope}"'
            )
            lines.extend((
                (
                    "      %submission = aboleth.submit "
                    f"%endpoint, {tokens} "
                    f"{{{', '.join(attributes)}}} "
                    f": (!aboleth.endpoint, {types}) "
                    "-> !aboleth.token"
                ),
                "      ub.unreachable",
                self._fragment_close(),
            ))
            return "\n".join(lines)
        assert self.signal is not None
        signal_dependency = (
            ""
            if previous is None
            else f", {previous}"
        )
        signal_dependency_type = (
            ""
            if previous is None
            else ", !aboleth.token"
        )
        lines.extend((
            (
                "      %signal_destination = "
                "aboleth.bind_remote_region "
                f'{arguments[self.signal.destination]}, '
                f'{arguments[self.signal.destination_key]} '
                ": (i64, i32) -> !aboleth.remote_region"
            ),
            (
                "      %signal_source = aboleth.bind_local_region "
                f'{arguments[self.signal.source]}, '
                f'{arguments[self.signal.source_key]} '
                ": (i64, i32) -> !aboleth.local_region"
            ),
            (
                "      %published = aboleth.signal %endpoint, "
                "%signal_destination, %signal_source"
                f"{signal_dependency} : (!aboleth.endpoint, "
                "!aboleth.remote_region, !aboleth.local_region"
                f"{signal_dependency_type}) -> !aboleth.token"
            ),
            (
                "      aboleth.planned_wait %published, "
                f'{arguments[self.signal.iteration]}, '
                f'{arguments[self.signal.iterations]} '
                ": !aboleth.token, i32, i32"
            ),
            "      ub.unreachable",
            self._fragment_close(),
        ))
        return "\n".join(lines)

@dataclass(frozen=True)
class P2PRealizationProfile:
    """Client-owned kernels for compiler-selected P2P realizations."""

    direct_destination_kernel: str
    direct_publication_kernels: tuple[
        tuple[str, str | None], ...
    ]
    parallel_copy_kernel: str
    parallel_publication_kernels: tuple[
        tuple[str, str | None], ...
    ]

    def __post_init__(self):
        if (
            not self.direct_destination_kernel
            or not self.parallel_copy_kernel
        ):
            raise ValueError(
                "P2P realization kernels cannot be empty"
            )
        for mappings in (
            self.direct_publication_kernels,
            self.parallel_publication_kernels,
        ):
            strategies = tuple(
                strategy for strategy, _kernel in mappings
            )
            if (
                len(set(strategies)) != len(strategies)
                or any(not strategy for strategy in strategies)
            ):
                raise ValueError(
                    "P2P publication strategies must be unique and nonempty"
                )

    def direct_publication_kernel(
        self,
        strategy: str,
    ) -> str | None:
        return dict(self.direct_publication_kernels)[strategy]

    def parallel_publication_kernel(
        self,
        strategy: str,
    ) -> str | None:
        return dict(self.parallel_publication_kernels)[strategy]


@dataclass(frozen=True)
class PhaseResources:
    """Runtime facts needed to compile one or more communication phases.

    Integrations provide stable region effects and the endpoint capabilities
    available to the phase. Pointers and per-item predicates remain kernel
    operands; launches and streams remain client-owned.
    """

    regions: dict[str, RegionEffect]
    peer_transports: tuple[str, ...]
    peer_capacities: tuple[int, ...]
    num_peers: int
    # Optional canonical rank -> transport geography. `peer_transports` above
    # remains the capability/capacity classes consumed by transport planning;
    # this mapping keys code specialization and describes mixed-world routing.
    transport_geography: tuple[tuple[int, str], ...] = ()
    iteration_count: int = 1
    completion_intervals: tuple[int, ...] = ()
    p2p_parallel_threshold_bytes: int = 4 * 1024
    coalesce_publication_wait: bool = True
    coalesce_return_with_consumer: bool = True
    recycled_buffers: tuple[str, ...] = ()
    advanced_values: tuple[str, ...] = ()
    consumer_grid_size: int = 0
    p2p_packed_doorbell: bool = False
    p2p_realization: P2PRealizationProfile | None = None
    p2p_direct_destination: bool = True
    p2p_fused_publication: bool = False

    def __post_init__(self):
        if not self.peer_transports:
            raise ValueError(
                "pattern requires at least one peer transport"
            )
        if (
            len(self.peer_transports) !=
            len(self.peer_capacities)
        ):
            raise ValueError(
                "pattern requires one capacity per transport"
            )
        if (
            self.completion_intervals and
            len(self.completion_intervals) !=
                len(self.peer_transports)
        ):
            raise ValueError(
                "pattern requires one completion interval per transport"
            )
        if any(
            interval <= 0
            for interval in self.completion_intervals
        ):
            raise ValueError(
                "completion intervals must be positive"
            )
        if self.iteration_count <= 0:
            raise ValueError(
                "iteration_count must be positive"
            )
        if self.num_peers <= 0:
            raise ValueError("num_peers must be positive")
        if self.transport_geography:
            peers = tuple(peer for peer, _ in self.transport_geography)
            invalid_peer = any(
                isinstance(peer, bool)
                or not isinstance(peer, int)
                or peer < 0
                or peer >= self.num_peers
                for peer in peers
            )
            invalid_transport = any(
                not isinstance(transport, str) or not transport
                for _, transport in self.transport_geography
            )
            if (
                tuple(sorted(self.transport_geography))
                != self.transport_geography
                or len(set(peers)) != len(peers)
                or invalid_peer
                or invalid_transport
            ):
                raise ValueError(
                    "transport_geography must be sorted unique "
                    "(peer, transport) entries inside num_peers"
                )
        if self.p2p_parallel_threshold_bytes < 0:
            raise ValueError(
                "P2P parallel threshold cannot be negative"
            )
        if self.consumer_grid_size < 0:
            raise ValueError(
                "consumer grid size cannot be negative"
            )
        if not isinstance(self.p2p_direct_destination, bool):
            raise TypeError(
                "p2p_direct_destination must be a bool"
            )
        if not isinstance(self.p2p_fused_publication, bool):
            raise TypeError(
                "p2p_fused_publication must be a bool"
            )
        for kind, names in (
            ("recycled buffer", self.recycled_buffers),
            ("advanced value", self.advanced_values),
        ):
            if (
                len(set(names)) != len(names)
                or any(not name for name in names)
            ):
                raise ValueError(
                    f"{kind} names must be unique and nonempty"
                )
@dataclass(frozen=True)
class PhaseSchedule:
    """Compiler-selected realization of one physicalized phase."""

    name: str
    strategy: str
    kernel: str
    publication_kernel: str | None
    publication_strategy: str
    recycled_buffers: tuple[str, ...]
    advanced_values: tuple[str, ...]
    grid_axes: tuple[str, ...]
    payload_bytes: int
    block_elements: int
    num_warps: int
    buffer_depth: int
    rationale: str
    # Chunk width of the P2P cooperative copy, in 8-byte elements. None keeps
    # the backend default; the planner may narrow or widen it per phase as
    # payload geometry demands.
    p2p_coop_elements: int | None = None

    def __post_init__(self):
        if self.strategy not in (
            "inline_transport",
            "direct_destination",
            "parallel_copy",
        ):
            raise ValueError(
                f"unknown phase strategy {self.strategy!r}"
            )
        if self.publication_strategy not in (
            "inline",
            "separate_kernel",
            "coalesced_kernel",
            "consumer_prologue",
            "packed_doorbell",
            "fused_kernel",
        ):
            raise ValueError(
                "unknown publication strategy "
                f"{self.publication_strategy!r}"
            )
        if (
            len(set(self.recycled_buffers))
            != len(self.recycled_buffers)
            or any(not name for name in self.recycled_buffers)
        ):
            raise ValueError(
                "recycled phase buffer names must be unique and nonempty"
            )
        if (
            len(set(self.advanced_values))
            != len(self.advanced_values)
            or any(not name for name in self.advanced_values)
        ):
            raise ValueError(
                "advanced phase value names must be unique and nonempty"
            )
        if self.payload_bytes < 0:
            raise ValueError("phase payload cannot be negative")
        if self.block_elements < 0:
            raise ValueError(
                "phase block element count cannot be negative"
            )
        if self.num_warps <= 0:
            raise ValueError("phase warp count must be positive")
        if self.buffer_depth <= 0:
            raise ValueError("phase buffer depth must be positive")
        if not self.kernel or not self.grid_axes:
            raise ValueError(
                "phase schedule requires a kernel and launch grid"
            )
        if self.p2p_coop_elements is not None and (
            self.p2p_coop_elements < 32
            or self.p2p_coop_elements > 4096
            or self.p2p_coop_elements
            & (self.p2p_coop_elements - 1)
        ):
            raise ValueError(
                "p2p_coop_elements must be a power of two in [32, 4096]"
            )


@dataclass(frozen=True)
class RegionRequirement:
    name: str
    element_type: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    visibility: str
    registered: bool
    donated: bool
    pool_depth: int


def _select_phase_schedules(
    analysis: SimplifiedCommunicationDataflow,
    programs: tuple[KernelTransportProgram, ...],
    resources: PhaseResources,
    *,
    cyclic: bool,
) -> tuple[PhaseSchedule, ...]:
    operations = {
        operation.id: operation
        for operation in analysis.operations
    }
    geography_transports = (
        {
            transport
            for _, transport in resources.transport_geography
        }
        if resources.transport_geography
        else set(resources.peer_transports)
    )
    p2p = (
        resources.num_peers > 1
        and resources.iteration_count == 1
        and geography_transports == {"p2p"}
    )
    schedules = []
    for program in programs:
        phase_operations = tuple(
            operations[operation]
            for operation in program.operations
            if operation in operations
        )
        per_item = any(
            operation.kind in ("transfer_each", "copy_each")
            for operation in phase_operations
        )
        direct_legal = any(
            operation.id in analysis.retargeted_destinations
            for operation in phase_operations
            if operation.kind in ("transfer_each", "copy_each")
        )
        large = (
            program.payload_bytes
            >= resources.p2p_parallel_threshold_bytes
        )
        profile = resources.p2p_realization
        if (
            p2p
            and large
            and per_item
            and direct_legal
            and profile is not None
            and resources.p2p_direct_destination
        ):
            packed_doorbell = resources.p2p_packed_doorbell
            strategy = "direct_destination"
            kernel = profile.direct_destination_kernel
            publication_strategy = (
                "packed_doorbell"
                if packed_doorbell
                else (
                    "coalesced_kernel"
                    if resources.coalesce_publication_wait
                    else "separate_kernel"
                )
            )
            publication_kernel = (
                profile.direct_publication_kernel(
                    publication_strategy
                )
            )
            recycled_buffers = resources.recycled_buffers
            advanced_values = resources.advanced_values
            grid_axes = ("routes",)
            block_elements = 0
            num_warps = 4
            buffer_depth = 1
            rationale = (
                "P2P is load/store addressable and the indexed transfer "
                "was legally retargeted to destination storage; "
                + (
                    "count and epoch share one release doorbell; "
                    if packed_doorbell
                    else ""
                )
                + (
                    "publication and wait share one kernel"
                    if resources.coalesce_publication_wait
                    else "publication and wait use separate kernels"
                )
            )
        elif (
            p2p
            and large
            and per_item
            and profile is not None
            and not resources.p2p_direct_destination
        ):
            strategy = "parallel_copy"
            kernel = profile.parallel_copy_kernel
            publication_strategy = (
                "fused_kernel"
                if resources.p2p_fused_publication
                else "coalesced_kernel"
            )
            publication_kernel = (
                profile.parallel_publication_kernel(
                    publication_strategy
                )
            )
            recycled_buffers = ()
            advanced_values = ()
            grid_axes = ("peers", "payload_blocks")
            block_elements = 256
            num_warps = 8
            buffer_depth = 1
            rationale = (
                "low expected destination fanout favors the "
                "parallel staged-copy realization"
            )
        elif (
            p2p
            and large
            and not per_item
            and profile is not None
        ):
            consumer_prologue = (
                resources.coalesce_return_with_consumer
                and resources.num_peers == 2
                and resources.consumer_grid_size
                >= resources.num_peers - 1
            )
            strategy = "parallel_copy"
            kernel = profile.parallel_copy_kernel
            publication_strategy = (
                "fused_kernel"
                if resources.p2p_fused_publication
                else (
                    "consumer_prologue"
                    if consumer_prologue
                    else "coalesced_kernel"
                )
            )
            publication_kernel = (
                profile.parallel_publication_kernel(
                    publication_strategy
                )
            )
            recycled_buffers = ()
            advanced_values = ()
            grid_axes = ("peers", "payload_blocks")
            block_elements = 1024
            num_warps = 4
            buffer_depth = 2 if cyclic else 1
            rationale = (
                "payload exceeds the P2P single-CTA threshold and has "
                "contiguous phase geometry; "
                + (
                    "publication and wait are sunk into the consumer prologue"
                    if consumer_prologue
                    else "publication and wait share one kernel"
                )
            )
        else:
            strategy = "inline_transport"
            kernel = program.kernel
            publication_kernel = None
            publication_strategy = "inline"
            recycled_buffers = ()
            advanced_values = ()
            grid_axes = ("peers",)
            block_elements = 0
            num_warps = 8 if per_item else 16
            buffer_depth = 1
            rationale = (
                "inline transport is required by topology, legality, "
                "or payload size"
            )
        schedules.append(PhaseSchedule(
            name=program.name,
            strategy=strategy,
            kernel=kernel,
            publication_kernel=publication_kernel,
            publication_strategy=publication_strategy,
            recycled_buffers=recycled_buffers,
            advanced_values=advanced_values,
            grid_axes=grid_axes,
            payload_bytes=program.payload_bytes,
            block_elements=block_elements,
            num_warps=num_warps,
            buffer_depth=buffer_depth,
            rationale=rationale,
        ))
    return tuple(schedules)


def _realize_phase_schedules(
    analysis: SimplifiedCommunicationDataflow,
    programs: tuple[KernelTransportProgram, ...],
    schedules: tuple[PhaseSchedule, ...],
) -> tuple[KernelTransportProgram, ...]:
    by_name = {
        schedule.name: schedule
        for schedule in schedules
    }
    operation_kinds = {
        operation.id: operation.kind
        for operation in analysis.operations
    }
    realized = []
    for program in programs:
        schedule = by_name[program.name]
        if schedule.strategy == "inline_transport":
            realized.append(program)
            continue
        if program.signal is None:
            raise ValueError(
                f"phase {program.name!r} requires publication "
                "but declares no signal"
            )
        transfers = (
            tuple(
                transfer
                for transfer in program.transfers
                if all(
                    operation_kinds.get(operation)
                    not in ("transfer_each", "copy_each")
                    for operation in transfer.operations
                )
            )
            if (
                schedule.strategy == "direct_destination"
                and (
                    schedule.publication_strategy
                    != "packed_doorbell"
                )
            )
            else ()
        )
        required = {
            "endpoint",
            program.signal.destination,
            program.signal.source,
            program.signal.source_key,
            program.signal.destination_key,
            program.signal.iteration,
            program.signal.iterations,
            *(
                field
                for transfer in transfers
                for field in (
                    transfer.destination,
                    transfer.source,
                    transfer.byte_count,
                    transfer.source_key,
                    transfer.destination_key,
                    transfer.predicate,
                )
            ),
        }
        realized.append(replace(
            program,
            operations=tuple(
                operation
                for transfer in transfers
                for operation in transfer.operations
            ),
            arguments=tuple(
                argument
                for argument in program.arguments
                if argument.name in required
            ),
            transfers=transfers,
            reductions=(),
        ))
    return tuple(realized)


@dataclass(frozen=True)
class CompiledPatternPlan:
    name: str
    analysis: SimplifiedCommunicationDataflow
    stages: tuple[PatternStage, ...]
    transport_operations: tuple[
        TransportPhase | TransportKernel, ...
    ]
    transport_plan: dict[str, TransportPlan]
    output_lifetimes: dict[str, OutputLifetimePlan]
    storage_plan: dict[str, StorageAllocationPlan]
    transport_mlir: str
    kernel_bindings: tuple[KernelBinding, ...]
    kernel_transport_programs: tuple[
        KernelTransportProgram, ...
    ]
    opaque_capabilities: tuple[
        tuple[str, OpaqueCapabilities], ...
    ]
    regions: tuple[RegionRequirement, ...]
    phase_schedules: tuple[PhaseSchedule, ...]
    transport_geography: tuple[tuple[int, str], ...]
    transport_partitions: tuple[TransportPartitionPlan, ...]
    retirement_programs: dict[str, KernelTransportProgram]
    cyclic: bool

    def binding(self, operation: str) -> KernelBinding:
        return next(
            binding
            for binding in self.kernel_bindings
            if binding.operation == operation
        )

    def completion_interval_for(
        self, kernel: str
    ) -> Optional[int]:
        intervals = {
            self.transport_plan[
                operation.name
            ].completion_interval
            for operation in self.transport_operations
            if (
                isinstance(operation, TransportPhase) and
                operation.bound_kernel == kernel and
                operation.name in self.transport_plan
            )
        }
        if not intervals:
            return None
        # One enclosing kernel can carry several transport fragments with
        # different reuse deadlines. Polling at the shortest interval satisfies
        # every fragment; requiring equality rejects otherwise legal fusion.
        return min(intervals)

    def transport_program_for_kernel(
        self, kernel: str
    ) -> KernelTransportProgram:
        programs = tuple(
            program
            for program in self.kernel_transport_programs
            if program.kernel == kernel
        )
        if len(programs) != 1:
            raise ValueError(
                f"kernel {kernel!r} requires exactly one transport program"
            )
        return programs[0]

    def phase_schedule(self, name: str) -> PhaseSchedule:
        schedules = tuple(
            schedule
            for schedule in self.phase_schedules
            if schedule.name == name
        )
        if len(schedules) != 1:
            raise ValueError(
                f"phase {name!r} requires exactly one schedule"
            )
        return schedules[0]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cyclic": self.cyclic,
            "transport_geography": [
                {"peer": peer, "transport": transport}
                for peer, transport in self.transport_geography
            ],
            "transport_partitions": [asdict(partition) for partition in self.transport_partitions],
            "retirement_programs": {
                kernel: {"program_id": program.program_id,
                         "boundary": "slot_retirement_and_teardown",
                         "drain_endpoint": program.drain_endpoint}
                for kernel, program in self.retirement_programs.items()
            },
            "removed_operations": sorted(
                self.analysis.removed_operation_ids
            ),
            "retargeted_destinations": dict(
                self.analysis.retargeted_destinations
            ),
            "kernel_bindings": [
                {
                    "operation": binding.operation,
                    "kernel": binding.kernel,
                    "artifact": binding.artifact,
                    "insertion_point":
                        binding.insertion_point,
                }
                for binding in self.kernel_bindings
            ],
            "kernel_transport_programs": [
                {
                    "program_id": program.program_id,
                    "name": program.name,
                    "operations": list(program.operations),
                    "kernel": program.kernel,
                    "artifact": program.artifact,
                    "insertion_point": program.insertion_point,
                    "command_count": program.command_count,
                    "payload_bytes": program.payload_bytes,
                    "transfers": [
                        {
                            "operations": list(
                                transfer.operations
                            ),
                            "max_bytes":
                                transfer.max_bytes,
                        }
                        for transfer in program.transfers
                    ],
                    "reductions": [
                        {
                            "operation": reduction.operation,
                            "operator": reduction.operator,
                            "cardinality":
                                reduction.cardinality,
                            "cardinality_source":
                                reduction.cardinality_source,
                        }
                        for reduction in program.reductions
                    ],
                }
                for program in self.kernel_transport_programs
            ],
            "phase_schedules": [
                {
                    "name": schedule.name,
                    "strategy": schedule.strategy,
                    "kernel": schedule.kernel,
                    "publication_kernel":
                        schedule.publication_kernel,
                    "publication_strategy":
                        schedule.publication_strategy,
                    "recycled_buffers": list(
                        schedule.recycled_buffers
                    ),
                    "advanced_values": list(
                        schedule.advanced_values
                    ),
                    "grid_axes": list(schedule.grid_axes),
                    "payload_bytes": schedule.payload_bytes,
                    "block_elements": schedule.block_elements,
                    "num_warps": schedule.num_warps,
                    "buffer_depth": schedule.buffer_depth,
                    "rationale": schedule.rationale,
                    "p2p_coop_elements":
                        schedule.p2p_coop_elements,
                }
                for schedule in self.phase_schedules
            ],
            "opaque_capabilities": {
                name: {
                    "accepts_destination_rewrite":
                        value.accepts_destination_rewrite,
                    "accepts_indexed_scatter":
                        value.accepts_indexed_scatter,
                    "accepted_address_spaces": list(
                        value.accepted_address_spaces
                    ),
                    "completion_contract":
                        value.completion_contract,
                }
                for name, value in self.opaque_capabilities
            },
            "regions": [
                {
                    "name": region.name,
                    "element_type": region.element_type,
                    "shape": list(region.shape),
                    "strides": list(region.strides),
                    "visibility": region.visibility,
                    "registered": region.registered,
                    "donated": region.donated,
                    "pool_depth": region.pool_depth,
                }
                for region in self.regions
            ],
            "transport_plan": {
                name: {
                    "completion_interval":
                        value.completion_interval,
                    "reuse_distance": value.reuse_distance,
                    "completion_before":
                        value.completion_before,
                    "completion_iteration_delta":
                        value.completion_iteration_delta,
                    "publication_before":
                        value.publication_before,
                    "publication_iteration_delta":
                        value.publication_iteration_delta,
                }
                for name, value in self.transport_plan.items()
            },
            "output_lifetimes": {
                name: {
                    "last_consumer": value.last_consumer,
                    "last_consumer_iteration_delta":
                        value.last_consumer_iteration_delta,
                    "reuse_before": value.reuse_before,
                    "reuse_iteration_delta":
                        value.reuse_iteration_delta,
                    "pool_depth": value.pool_depth,
                }
                for name, value in self.output_lifetimes.items()
            },
            "storage_plan": {
                name: asdict(value) for name, value in self.storage_plan.items()
            },
        }


@dataclass(frozen=True)
class CompiledPattern:
    analysis: SimplifiedCommunicationDataflow
    stages: tuple[PatternStage, ...]
    kernel_bindings: tuple[KernelBinding, ...] = ()
    kernel_transport_programs: tuple[
        KernelTransportProgram, ...
    ] = ()
    opaque_capabilities: tuple[
        tuple[str, OpaqueCapabilities], ...
    ] = ()

    def with_transport_programs(
        self,
        programs: tuple[KernelTransportProgram, ...],
    ) -> "CompiledPattern":
        operation_ids = {
            operation.id
            for operation in self.analysis.operations
        }
        bindings = list(self.kernel_bindings)
        bound_programs = list(self.kernel_transport_programs)
        bound_operations = {
            binding.operation for binding in bindings
        }
        names = {
            program.name for program in bound_programs
        }
        for program in programs:
            if program.program_id is not None:
                raise ValueError(
                    "transport program ids are assigned when attached"
                )
            if program.name in names:
                raise ValueError(
                    f"transport program {program.name!r} is already attached"
                )
            unknown = set(program.operations) - operation_ids
            if unknown:
                raise KeyError(
                    "transport program references unknown operations: "
                    + ", ".join(sorted(unknown))
                )
            duplicate = set(program.operations) & bound_operations
            if duplicate:
                raise ValueError(
                    "transport operations already have kernel bindings: "
                    + ", ".join(sorted(duplicate))
                )
            for operation in program.operations:
                bindings.append(KernelBinding(
                    operation=operation,
                    kernel=program.kernel,
                    insertion_point=program.insertion_point,
                    artifact=program.artifact,
                ))
            bound_programs.append(replace(
                program,
                program_id=len(bound_programs),
            ))
            bound_operations.update(program.operations)
            names.add(program.name)
        return replace(
            self,
            kernel_bindings=tuple(bindings),
            kernel_transport_programs=tuple(bound_programs),
        )

    def opaque_stage(self, name: str) -> OpaqueStage:
        return next(
            stage
            for stage in self.stages
            if (
                isinstance(stage, OpaqueStage) and
                stage.name == name
            )
        )

    def binding(self, operation: str) -> KernelBinding:
        return next(
            binding
            for binding in self.kernel_bindings
            if binding.operation == operation
        )

    def capabilities(
        self, stage: str
    ) -> OpaqueCapabilities:
        return dict(self.opaque_capabilities)[stage]

    def transport_operations(
        self,
        bindings: PhaseResources,
    ) -> tuple[TransportPhase | TransportKernel, ...]:
        regions = {
            region.id: region
            for region in self.analysis.regions
        }
        outputs: dict[
            str, tuple[CommunicationDataflowOperation, int]
        ] = {}
        worksets = {}
        kernel_bindings = {
            binding.operation: binding
            for binding in self.kernel_bindings
        }
        transport_programs = {
            operation: program
            for program in self.kernel_transport_programs
            for operation in program.operations
        }
        for operation in self.analysis.operations:
            if operation.output_id is not None:
                outputs[operation.output_id] = (
                    operation, 0
                )
            for index, output_id in enumerate(
                operation.output_ids
            ):
                outputs[output_id] = (
                    operation, index
                )
            if (
                operation.kind == "workset" and
                operation.output_id is not None
            ):
                worksets[operation.output_id] = operation

        for workset in worksets.values():
            if "destination_peer" not in workset.field_names:
                continue
            index = workset.field_names.index(
                "destination_peer"
            )
            if (
                workset.field_lower_bounds[index] < 0 or
                workset.field_upper_bounds[index] >
                    bindings.num_peers
            ):
                raise ValueError(
                    f"workset {workset.output_id!r} peer bounds "
                    "exceed the bound runtime mesh"
                )

        cache: dict[str, RegionEffect] = {}

        def effect(value: str) -> RegionEffect:
            if value in cache:
                return cache[value]
            if value in bindings.regions:
                result = bindings.regions[value]
            elif value in regions and regions[value].source:
                result = effect(regions[value].source)
            elif value in outputs:
                operation, index = outputs[value]
                if operation.kind == "opaque_stage":
                    result = effect(
                        operation.destinations[index]
                    )
                elif operation.destination is not None:
                    result = effect(operation.destination)
                else:
                    raise KeyError(
                        f"no region effect for {value!r}"
                    )
            else:
                raise KeyError(
                    f"no region effect binding for {value!r}"
                )
            cache[value] = result
            return result

        def unique_effects(
            values: tuple[str, ...],
        ) -> tuple[RegionEffect, ...]:
            result = []
            for value in values:
                candidate = effect(value)
                if candidate not in result:
                    result.append(candidate)
            return tuple(result)

        operations: list[
            TransportPhase | TransportKernel
        ] = []
        for stage in self.stages:
            if isinstance(stage, OpaqueStage):
                capabilities = self.capabilities(stage.name)
                if (
                    stage.asynchronous and
                    capabilities.completion_contract ==
                        "current_stream"
                ):
                    raise ValueError(
                        f"asynchronous opaque stage {stage.name!r} "
                        "requires an external completion contract"
                    )
                operations.append(
                    TransportKernel(
                        name=stage.name,
                        reads=unique_effects(stage.reads),
                        writes=unique_effects(stage.writes),
                        asynchronous=stage.asynchronous,
                        max_inflight=stage.max_inflight,
                        event_ordered=stage.event_ordered,
                        bound_kernel=(
                            kernel_bindings[stage.name].kernel
                            if stage.name in kernel_bindings
                            else None
                        ),
                        artifact_kind=(
                            kernel_bindings[stage.name].artifact
                            if stage.name in kernel_bindings
                            else None
                        ),
                        insertion_point=(
                            kernel_bindings[
                                stage.name
                            ].insertion_point
                            if stage.name in kernel_bindings
                            else None
                        ),
                    )
                )
                continue

            communication = []
            phase_index = 0

            def flush_communication() -> None:
                nonlocal phase_index
                if not communication:
                    return
                reads = tuple(
                    operation.source
                    for operation in communication
                    if operation.source is not None
                )
                writes = tuple(
                    operation.destination
                    for operation in communication
                    if operation.destination is not None
                )
                command_count = 0
                payload_bytes = 0
                programs = {
                    transport_programs[operation.id]
                    for operation in communication
                    if operation.id in transport_programs
                }
                if len(programs) > 1:
                    raise ValueError(
                        "communication phase spans multiple transport programs"
                    )
                if programs:
                    program = next(iter(programs))
                    uncovered = [
                        operation.id
                        for operation in communication
                        if operation.id not in program.operations
                    ]
                    if uncovered:
                        raise ValueError(
                            "transport program does not cover phase operations: "
                            + ", ".join(uncovered)
                        )
                    command_count = program.command_count
                    payload_bytes = program.payload_bytes
                else:
                    for operation in communication:
                        if operation.kind == "transfer_each":
                            workset = worksets[
                                operation.workset
                            ]
                            command_count += (
                                workset.max_items or 1
                            )
                        elif operation.kind == "reduction":
                            command_count += (
                                operation.cardinality or 1
                            )
                        else:
                            command_count += 1
                        payload_bytes += effect(
                            operation.source
                        ).length
                name = (
                    stage.name
                    if phase_index == 0
                    else f"{stage.name}.{phase_index}"
                )
                bound_kernels = {
                    kernel_bindings[operation.id].kernel
                    for operation in communication
                    if operation.id in kernel_bindings
                }
                if len(bound_kernels) > 1:
                    raise ValueError(
                        f"communication phase {name!r} spans "
                        "multiple enclosing kernels"
                    )
                phase_bindings = [
                    kernel_bindings[operation.id]
                    for operation in communication
                    if operation.id in kernel_bindings
                ]
                phase_binding = (
                    phase_bindings[0]
                    if phase_bindings
                    else None
                )
                operations.append(
                    TransportPhase(
                        name=name,
                        program_id=(next(iter(programs)).program_id if programs else None),
                        commands_per_iteration=command_count,
                        iteration_count=(
                            bindings.iteration_count
                        ),
                        payload_bytes=payload_bytes,
                        peer_transports=(
                            bindings.peer_transports
                        ),
                        peer_capacities=(
                            bindings.peer_capacities
                        ),
                        completion_intervals=(
                            bindings.completion_intervals
                        ),
                        reads=unique_effects(reads),
                        writes=unique_effects(writes),
                        bound_kernel=next(
                            iter(bound_kernels),
                            None,
                        ),
                        artifact_kind=(
                            phase_binding.artifact
                            if phase_binding
                            else None
                        ),
                        insertion_point=(
                            phase_binding.insertion_point
                            if phase_binding
                            else None
                        ),
                    )
                )
                communication.clear()
                phase_index += 1

            for operation in stage.operations:
                # A physicalized copy is one of its attachment's commands,
                # so it stays inside the phase whose program already counts
                # it; an unbound copy stays a local materialization kernel.
                if operation.kind in (
                    "transfer",
                    "transfer_each",
                    "reduction",
                ) or (
                    operation.kind in ("copy", "copy_each") and
                    operation.id in kernel_bindings
                ):
                    if communication:
                        previous = kernel_bindings.get(
                            communication[-1].id
                        )
                        current = kernel_bindings.get(
                            operation.id
                        )
                        previous_key = (
                            previous.kernel,
                            previous.insertion_point,
                        ) if previous else None
                        current_key = (
                            current.kernel,
                            current.insertion_point,
                        ) if current else None
                        if previous_key != current_key:
                            flush_communication()
                    communication.append(operation)
                    continue
                flush_communication()
                if operation.kind in (
                    "copy",
                    "copy_each",
                    "transform",
                ):
                    operations.append(
                        TransportKernel(
                            name=operation.id,
                            reads=unique_effects(
                                (operation.source,)
                            ),
                            writes=unique_effects(
                                (operation.destination,)
                            ),
                            bound_kernel=(
                                kernel_bindings[
                                    operation.id
                                ].kernel
                                if operation.id in kernel_bindings
                                else None
                            ),
                            artifact_kind=(
                                kernel_bindings[
                                    operation.id
                                ].artifact
                                if operation.id in kernel_bindings
                                else None
                            ),
                            insertion_point=(
                                kernel_bindings[
                                    operation.id
                                ].insertion_point
                                if operation.id in kernel_bindings
                                else None
                            ),
                        )
                    )
            flush_communication()
        return tuple(operations)

    def plan_transport(
        self,
        bindings: PhaseResources,
        *,
        cyclic: bool = False,
    ) -> dict[str, TransportPlan]:
        return self.compile_plan(
            bindings,
            cyclic=cyclic,
        ).transport_plan

    def compile_plan(
        self,
        bindings: PhaseResources,
        *,
        cyclic: bool = False,
    ) -> CompiledPatternPlan:
        transport_operations = self.transport_operations(
            bindings
        )
        transport_plan = plan_transport_program(
            transport_operations,
            cyclic=cyclic,
        )
        output_lifetimes = plan_output_lifetimes(
            transport_operations,
            cyclic=cyclic,
        )
        phase_schedules = _select_phase_schedules(
            self.analysis,
            self.kernel_transport_programs,
            bindings,
            cyclic=cyclic,
        )
        scheduled_programs = _realize_phase_schedules(
            self.analysis,
            self.kernel_transport_programs,
            phase_schedules,
        )
        # The decision comes from C++ lifetime/capacity planning. This bridge
        # supplies the existing generic semantic drain fragment, never a
        # framework-specific completion interval or handwritten NIC kernel.
        phase_facts = {
            phase["name"]: phase
            for program in plan_transport_facts(transport_operations, cyclic)["programs"]
            for phase in program["phases"]
        }
        retirement_programs = {}
        next_program_id = max((program.program_id for program in scheduled_programs), default=-1) + 1
        for operation in transport_operations:
            if (isinstance(operation, TransportPhase) and operation.bound_kernel
                    and phase_facts[operation.name].get("requiresFinalDrain", False)
                    and operation.bound_kernel not in retirement_programs):
                kernel = operation.bound_kernel
                retirement_programs[kernel] = KernelTransportProgram(
                    name=f"{kernel}.retire", operations=(), kernel=f"{kernel}.retire",
                    insertion_point="slot_retirement_and_teardown",
                    arguments=(KernelTransportArgument("endpoint", "i64"),),
                    transfers=(), signal=None, drain_endpoint=True,
                    program_id=next_program_id+len(retirement_programs))
        transport_mlir = compile_transport_program_mlir(
            transport_operations,
            cyclic=cyclic,
            transport_geography=bindings.transport_geography,
            fragments="\n".join(
                program.to_mlir()
                for program in (*scheduled_programs, *retirement_programs.values())
            ),
        )
        storage_plan = plan_storage_mlir(self.analysis.mlir)
        storage_identity = hashlib.sha256(json.dumps(
            {name: asdict(value) for name, value in storage_plan.items()},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        transport_mlir += f"\n// aboleth storage plan: {storage_identity}\n"
        regions = tuple(
            RegionRequirement(
                name=region.id,
                element_type=region.element_type,
                shape=region.shape,
                strides=region.strides,
                visibility=region.visibility,
                registered=region.registered,
                donated=region.donated,
                pool_depth=region.pool_depth,
            )
            for region in self.analysis.regions
        )
        return CompiledPatternPlan(
            name=self.analysis.name,
            analysis=self.analysis,
            stages=self.stages,
            transport_operations=transport_operations,
            transport_plan=transport_plan,
            output_lifetimes=output_lifetimes,
            storage_plan=storage_plan,
            transport_mlir=transport_mlir,
            kernel_bindings=self.kernel_bindings,
            kernel_transport_programs=(
                scheduled_programs
            ),
            opaque_capabilities=(
                self.opaque_capabilities
            ),
            regions=regions,
            phase_schedules=phase_schedules,
            transport_geography=bindings.transport_geography,
            transport_partitions=plan_transport_partitions(bindings.transport_geography),
            retirement_programs=retirement_programs,
            cyclic=cyclic,
        )


class Pattern(CommunicationDataflow):
    """Authoring surface for communication around opaque compute."""

    def __init__(self, name: str):
        super().__init__(name)
        self._opaque_capabilities: dict[
            str, OpaqueCapabilities
        ] = {}

    def workset(
        self,
        workset_id: str,
        *,
        fields: dict[str, str],
        bounds: dict[str, tuple[int, int]],
        max_items: int,
        count: Optional[str] = None,
        unique_destinations: bool = False,
        ordered: bool = False,
    ) -> Workset:
        name = super().workset(
            workset_id,
            fields=fields,
            bounds=bounds,
            max_items=max_items,
            count=count,
            unique_destinations=unique_destinations,
            ordered=ordered,
        )
        return Workset(
            name=name,
            fields=dict(fields),
            bounds=dict(bounds),
            max_items=max_items,
            count=count,
            unique_destinations=unique_destinations,
            ordered=ordered,
        )

    def transfer_each(
        self,
        operation_id: str,
        workset: Workset | str,
        source: str,
        destination: str,
        *,
        source_index: str = "source_row",
        destination_peer: str = "destination_peer",
        destination_index: str = "destination_row",
        bytes_field: Optional[str] = None,
        predicate_field: Optional[str] = None,
        output: Optional[str] = None,
    ) -> str:
        return super().transfer_each(
            operation_id,
            (
                workset.name
                if isinstance(workset, Workset)
                else workset
            ),
            source,
            destination,
            source_index=source_index,
            destination_peer=destination_peer,
            destination_index=destination_index,
            bytes_field=bytes_field,
            predicate_field=predicate_field,
            output=output,
        )

    def copy_each(
        self,
        operation_id: str,
        workset: Workset | str,
        source: str,
        destination: str,
        *,
        source_index: str = "source_row",
        destination_index: str = "destination_row",
        output: Optional[str] = None,
    ) -> str:
        return super().copy_each(
            operation_id,
            (
                workset.name
                if isinstance(workset, Workset)
                else workset
            ),
            source,
            destination,
            source_index=source_index,
            destination_index=destination_index,
            output=output,
        )

    def opaque_compute(
        self,
        name: str,
        *,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        outputs: Optional[tuple[str, ...]] = None,
        asynchronous: bool = False,
        max_inflight: int = 1,
        event_ordered: bool = False,
        capabilities: Optional[
            OpaqueCapabilities
        ] = None,
    ) -> tuple[str, ...]:
        if name in self._opaque_capabilities:
            raise ValueError(
                f"opaque stage {name!r} is already declared"
            )
        self._opaque_capabilities[name] = (
            capabilities or OpaqueCapabilities()
        )
        return self.opaque_stage(
            name,
            reads=reads,
            writes=writes,
            outputs=outputs,
            asynchronous=asynchronous,
            max_inflight=max_inflight,
            event_ordered=event_ordered,
            bindable_outputs=self._opaque_capabilities[name].accepts_destination_rewrite,
            prefix_read_counts=self._opaque_capabilities[name].prefix_read_counts,
        )

    def analyze(self) -> SimplifiedCommunicationDataflow:
        return self.simplify()

    def compile(self) -> CompiledPattern:
        analysis = self.analyze()
        stages: list[PatternStage] = []
        communication: list[
            CommunicationDataflowOperation
        ] = []

        def flush_communication() -> None:
            if not communication:
                return
            stages.append(
                CommunicationStage(
                    name=(
                        "communication."
                        f"{sum(isinstance(stage, CommunicationStage) for stage in stages)}"
                    ),
                    operations=tuple(communication),
                )
            )
            communication.clear()

        for operation in analysis.operations:
            if operation.kind in (
                "region",
                "view",
                "indexed_view",
                "workset",
                "use",
                "async_use",
            ):
                continue
            if operation.kind == "opaque_stage":
                flush_communication()
                stages.append(
                    OpaqueStage(
                        name=operation.id,
                        reads=operation.inputs,
                        writes=operation.destinations,
                        outputs=operation.output_ids,
                        asynchronous=bool(
                            operation.asynchronous
                        ),
                        max_inflight=(
                            operation.max_inflight or 1
                        ),
                        event_ordered=bool(
                            operation.event_ordered
                        ),
                    )
                )
                continue
            communication.append(operation)

        flush_communication()
        return CompiledPattern(
            analysis=analysis,
            stages=tuple(stages),
            opaque_capabilities=tuple(
                self._opaque_capabilities.items()
            ),
        )
