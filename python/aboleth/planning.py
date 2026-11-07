from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from triton._C.libtriton import ir, passes


@dataclass(frozen=True)
class RegionEffect:
    allocation: int
    generation: int
    offset: int
    length: int
    stride: int
    count: int
    period: int = 0
    unknown: bool = False

    @classmethod
    def from_runtime(
        cls,
        tensor,
        *,
        registration=None,
        iteration_count: int = 1,
    ) -> "RegionEffect":
        if iteration_count <= 0:
            raise ValueError(
                "iteration_count must be positive"
            )
        total_bytes = (
            tensor.numel() * tensor.element_size()
        )
        if total_bytes % iteration_count:
            raise ValueError(
                "runtime region does not divide into iterations"
            )
        item_bytes = total_bytes // iteration_count
        allocation = (
            registration.address
            if registration is not None
            else tensor.data_ptr()
        )
        generation = (
            registration.generation
            if registration is not None
            else 1
        )
        offset = tensor.data_ptr() - allocation
        if offset < 0 or (
            registration is not None and
            offset + total_bytes > registration.length
        ):
            raise ValueError(
                "runtime tensor is outside its registration"
            )
        return cls(
            allocation=allocation,
            generation=generation,
            offset=offset,
            length=item_bytes,
            stride=item_bytes,
            count=iteration_count,
            period=total_bytes,
        )

    def __post_init__(self):
        if not self.unknown and (
            self.allocation < 0 or self.generation < 0
        ):
            raise ValueError(
                "known region allocation and generation cannot be negative"
            )
        if (
            self.offset < 0
            or self.length <= 0
            or self.stride < 0
            or self.period < 0
            or self.count <= 0
        ):
            raise ValueError(
                "region offset/stride must be non-negative and "
                "length/count must be positive"
            )


@dataclass(frozen=True)
class TransportPhase:
    name: str
    commands_per_iteration: int
    iteration_count: int
    payload_bytes: int
    peer_transports: tuple[str, ...]
    peer_capacities: tuple[int, ...]
    reads: tuple[RegionEffect, ...]
    writes: tuple[RegionEffect, ...] = ()
    tiny_payload_bytes: int = 4096
    # One optional override per peer transport. Zero, or an empty tuple, leaves
    # the choice to the transport capability registry.
    completion_intervals: tuple[int, ...] = ()
    bound_kernel: Optional[str] = None
    artifact_kind: Optional[str] = None
    insertion_point: Optional[str] = None
    # The client's own name for the queue this phase's commands go to. Two
    # phases are on one endpoint when their labels are equal — never because
    # their peer transports and capacities happen to match, which two distinct
    # queue pairs of the same depth also do. Optional: a session that never
    # has to compare two phases' endpoints never needs it (see
    # docs/PROGRAMMING_MODEL.md).
    endpoint: Optional[str] = None
    # Stable attachment identity, not a completion-policy choice.
    program_id: Optional[int] = None


@dataclass(frozen=True)
class TransportKernel:
    name: str
    reads: tuple[RegionEffect, ...]
    writes: tuple[RegionEffect, ...]
    asynchronous: bool = False
    max_inflight: int = 1
    event_ordered: bool = False
    bound_kernel: Optional[str] = None
    artifact_kind: Optional[str] = None
    insertion_point: Optional[str] = None
    # The client compiles this kernel outside Aboleth, so no transform may
    # place work inside it: a deadline that lands here stays eager.
    opaque: bool = False


@dataclass(frozen=True)
class TransportPlan:
    completion_interval: int
    reuse_distance: int
    completion_before: str
    completion_iteration_delta: int
    publication_before: str
    publication_iteration_delta: int


@dataclass(frozen=True)
class OutputLifetimePlan:
    last_consumer: str
    last_consumer_iteration_delta: int
    reuse_before: str
    reuse_iteration_delta: int
    pool_depth: int


@dataclass(frozen=True)
class StorageAllocationPlan:
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    strides: tuple[int, ...]
    byte_count: Optional[int]
    pool_depth: int
    last_reader: Optional[str]
    event_ordered: bool
    max_inflight: int
    reader_lifetime_known: bool
    escaped: bool
    prefix_binding: Optional[dict] = None


@dataclass(frozen=True)
class TransportPartitionPlan:
    """One compiler-derived homogeneous launch; the client owns execution."""

    backend: str
    peers: tuple[int, ...]
    grid_extent: int


def plan_transport_partitions(
    geography: tuple[tuple[int, str], ...],
) -> tuple[TransportPartitionPlan, ...]:
    if not geography:
        return ()
    _, facts = _run_transport_planner((), transport_geography=geography)
    return tuple(TransportPartitionPlan(
        backend=launch["backend"], peers=tuple(launch["peers"]),
        grid_extent=launch["grid_extent"],
    ) for launch in facts["programs"][0]["launch_partitions"])


@dataclass(frozen=True)
class EndpointPartitionFacts:
    """Registered resources for one compiler-derived transport partition."""

    partition: TransportPartitionPlan
    queue_memory: str
    platform: str = ""
    available_lanes: int = 1
    # Canonical rank order: (exact platform key, queue memory, allocated lanes).
    participants: tuple[tuple[str, str, int], ...] = ()
    peer_link: str = ""
    participant_links: tuple[str, ...] = ()


@dataclass(frozen=True)
class EndpointProfile:
    backend: str
    platform: str
    queue_memory: str
    bytes_per_endpoint: int
    max_lanes: int
    identity: str
    peer_link: str = ""
    sender_warps: int = 1


@dataclass(frozen=True)
class EndpointLaunchPlan:
    backend: str
    peers: tuple[int, ...]
    lanes: int
    slices: tuple[tuple[int, int], ...]
    slice_quotient: int
    slice_remainder: int
    count_lane: int
    signal_slots: int
    sender_grid: tuple[int, int]
    receiver_grid: tuple[int, int]
    profile_identity: str
    ready_stride: int
    receiver_lanes_per_program: int = 1
    sender_warps: int = 1
    receiver_warps: int = 1


def plan_endpoint_work(*, name: str, payload_bytes: int, max_items: int,
                       partitions: tuple[EndpointPartitionFacts, ...],
                       profiles: tuple[EndpointProfile, ...] = (),
                       rank_count: int | None = None,
                       grouped_receiver_supported: bool = False) -> tuple[EndpointLaunchPlan, ...]:
    """Compile lanes, record slices and readiness extents from declared facts.

    Records must be unordered across slices; all fields of a record share its
    slice. Lane zero sends the count. Every lane publishes a strong signal,
    even for an empty valid slice, and the receiver waits on every lane.
    This function serializes facts; all policy and arithmetic run in C++.
    The result is immutable and includes profile identity for artifact keys.
    """
    from pathlib import Path
    from .compiler import enabled_optimizations

    def integer(value):
        if type(value) is not int or not -(2**63) <= value < 2**63:
            raise ValueError("endpoint planning facts must be signed i64 integers")
        return str(value)

    def string(value):
        if not isinstance(value, str):
            raise ValueError("endpoint planning names must be strings")
        return json.dumps(value)

    parts = []
    for facts in partitions:
        partition = facts.partition
        peers = ", ".join(integer(peer) for peer in partition.peers)
        if partition.grid_extent != len(partition.peers):
            raise ValueError("partition grid does not match its registered peers")
        image = ", ".join("{platform = " + string(platform) + ", queueMemory = " + string(memory)
            + ", availableLanes = " + integer(lanes) + " : i64}"
            for platform, memory, lanes in facts.participants)
        participants = ", participants = [" + image + "]" if facts.participants else ""
        link = ", peerLink = " + string(facts.peer_link) if facts.peer_link else ""
        links = (", participantLinks = [" + ", ".join(string(v) for v in facts.participant_links)
            + "]") if facts.participant_links else ""
        parts.append("{backend = " + string(partition.backend)
            + ", peers = array<i32: " + peers + ">, queueMemory = "
            + string(facts.queue_memory) + ", platform = " + string(facts.platform)
            + ", availableLanes = " + integer(facts.available_lanes) + " : i64" + participants + link + links + "}")
    measured = []
    for profile in profiles:
        integer(profile.sender_warps)
        measured.append("{backend = " + string(profile.backend)
            + ", platform = " + string(profile.platform)
            + ", queueMemory = " + string(profile.queue_memory)
            + ", bytesPerEndpoint = " + integer(profile.bytes_per_endpoint) + " : i64"
            + ", maxLanes = " + integer(profile.max_lanes) + " : i64"
            + ", identity = " + string(profile.identity)
            + (", peerLink = " + string(profile.peer_link) if profile.peer_link else "")
            + (", senderWarps = " + integer(profile.sender_warps) + " : i64"
               if profile.sender_warps != 1 else "") + "}")
    ranks = ", rankCount = " + integer(rank_count) + " : i64" if rank_count is not None else ""
    if type(grouped_receiver_supported) is not bool:
        raise ValueError("grouped receiver support must be a boolean ABI fact")
    receiver = ", groupedReceiverSupported = true" if grouped_receiver_supported else ""
    source = ("module { aboleth.endpoint_work {name = " + string(name)
        + ", payloadBytes = " + integer(payload_bytes) + " : i64, maxItems = "
        + integer(max_items) + " : i64, partitions = [" + ", ".join(parts)
        + "], profiles = [" + ", ".join(measured) + "]" + ranks + receiver + "} }")
    enabled = "endpoint-fanout" in enabled_optimizations()
    with tempfile.TemporaryDirectory(prefix="aboleth-endpoints-") as folder:
        directory = Path(folder)
        input_file, output_file = directory / "input.mlir", directory / "plan.json"
        input_file.write_text(source)
        context = ir.context()
        ir.load_dialects(context)
        module = ir.parse_mlir_module(str(input_file), context)
        manager = ir.pass_manager(context)
        passes.plugin.add_aboleth_plan_endpoints(
            manager, [str(output_file), "" if enabled else "unoptimized",
                      "" if "coalesce-receives" in enabled_optimizations() else "ungrouped"])
        manager.run(module, "aboleth-plan-endpoints")
        launches = json.loads(output_file.read_text())["programs"][0]["launches"]
    return tuple(EndpointLaunchPlan(
        backend=launch["backend"], peers=tuple(launch["peers"]),
        lanes=launch["lanes"], slices=tuple(tuple(s) for s in launch["slices"]),
        slice_quotient=launch["slice_quotient"], slice_remainder=launch["slice_remainder"],
        count_lane=launch["count_lane"], signal_slots=launch["signal_slots"],
        sender_grid=tuple(launch["sender_grid"]), receiver_grid=tuple(launch["receiver_grid"]),
        profile_identity=launch["profile_identity"], ready_stride=launch["ready_stride"],
        receiver_lanes_per_program=launch["receiver_lanes_per_program"],
        sender_warps=launch.get("sender_warps", 1), receiver_warps=launch.get("receiver_warps", 1),
    ) for launch in launches)


def plan_storage_mlir(source: str) -> dict[str, StorageAllocationPlan]:
    """Read compiler-derived allocation facts without changing logical shapes.

    Allocation consumers use this plan; Python neither selects a compact
    consumer nor calculates a workload-specific payload bound.
    """
    from .compiler import enabled_optimizations

    enabled = "plan-storage" in enabled_optimizations()
    with tempfile.TemporaryDirectory(prefix="aboleth-storage-") as folder:
        from pathlib import Path

        directory = Path(folder)
        input_file, output_file = directory / "input.mlir", directory / "plan.json"
        input_file.write_text(source)
        context = ir.context()
        ir.load_dialects(context)
        module = ir.parse_mlir_module(str(input_file), context)
        manager = ir.pass_manager(context)
        passes.plugin.add_aboleth_plan_storage(
            manager, [str(output_file), "" if enabled else "unoptimized"])
        manager.run(module, "aboleth-plan-storage")
        programs = json.loads(output_file.read_text())["programs"]
    if len(programs) != 1:
        raise ValueError("a storage plan requires one communication dataflow")
    result = {}
    for allocation in programs[0]["allocations"]:
        name = allocation.pop("name")
        if name in result:
            raise ValueError(f"duplicate storage allocation {name!r}")
        allocation["byte_count"] = allocation.pop("bytes")
        allocation["last_reader"] = allocation["last_reader"] or None
        for field in ["logical_shape", "physical_shape", "strides"]:
            allocation[field] = tuple(allocation[field])
        result[name] = StorageAllocationPlan(**allocation)
    return result


@dataclass(frozen=True)
class CommunicationRegion:
    id: str
    element_type: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    index_map: str
    visibility: str
    immutable: bool
    escapes: bool
    registered: bool
    donated: bool
    reuse_interval: int
    pool_depth: int
    source: Optional[str] = None
    indices: Optional[str] = None
    injective: bool = False
    index_lower_bounds: tuple[int, ...] = ()
    index_upper_bounds: tuple[int, ...] = ()
    valid_count: Optional[str] = None
    dead_padding: bool = False

    @property
    def is_view(self) -> bool:
        return self.source is not None

    @property
    def is_indexed(self) -> bool:
        return self.indices is not None


@dataclass(frozen=True)
class CommunicationDataflowOperation:
    id: str
    kind: str
    output_id: Optional[str] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    inputs: tuple[str, ...] = ()
    destinations: tuple[str, ...] = ()
    output_ids: tuple[str, ...] = ()
    workset: Optional[str] = None
    field_names: tuple[str, ...] = ()
    field_lower_bounds: tuple[int, ...] = ()
    field_upper_bounds: tuple[int, ...] = ()
    count: Optional[str] = None
    max_items: Optional[int] = None
    unique_destinations: Optional[bool] = None
    ordered: Optional[bool] = None
    source_index_field: Optional[str] = None
    destination_peer_field: Optional[str] = None
    destination_index_field: Optional[str] = None
    bytes_field: Optional[str] = None
    predicate_field: Optional[str] = None
    cardinality: Optional[int] = None
    operator: Optional[str] = None
    transform: Optional[str] = None
    index_map: Optional[str] = None
    max_inflight: Optional[int] = None
    event_ordered: Optional[bool] = None
    asynchronous: Optional[bool] = None
    bindable_outputs: bool = False
    prefix_read_counts: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimplifiedCommunicationDataflow:
    name: str
    regions: tuple[CommunicationRegion, ...]
    operations: tuple[CommunicationDataflowOperation, ...]
    removed_operation_ids: frozenset[str]
    removed_region_ids: frozenset[str]
    use_provenance: dict[
        str, tuple[tuple[str, str], ...]
    ]
    retargeted_destinations: dict[str, tuple[str, str]]
    input_mlir: str
    mlir: str

    def operation(
        self, operation_id: str
    ) -> Optional[CommunicationDataflowOperation]:
        return next(
            (
                operation
                for operation in self.operations
                if operation.id == operation_id
            ),
            None,
        )

    def operations_of_kind(
        self, kind: str
    ) -> tuple[CommunicationDataflowOperation, ...]:
        return tuple(
            operation
            for operation in self.operations
            if operation.kind == kind
        )

    def producer(
        self, output_id: str
    ) -> Optional[CommunicationDataflowOperation]:
        return next(
            (
                operation
                for operation in self.operations
                if (
                    operation.output_id == output_id or
                    output_id in operation.output_ids
                )
            ),
            None,
        )

    def use_inputs(self, use_id: str) -> tuple[str, ...]:
        operation = self.operation(use_id)
        if (
            operation is None or
            operation.kind not in ("use", "async_use")
        ):
            raise KeyError(use_id)
        return operation.inputs


def _identity_index_map(rank: int) -> str:
    dimensions = ", ".join(
        f"d{index}" for index in range(rank)
    )
    return (
        f"affine_map<({dimensions}) -> "
        f"({dimensions})>"
    )


# A dataflow ID reaches generated MLIR as a quoted attribute and generated C
# and Triton as an identifier stem, so it has to survive both. Dots are the
# namespacing every pattern uses ("source.live.window"); hyphens appear in
# dataflow names ("kv-cache-migration") and reach an identifier only through
# the slot and symbol manglers, which map them to '_' and refuse the
# collisions that creates. A leading digit does not survive either, and a
# quote, brace, or newline has no business in a name at all.
_DATAFLOW_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")


def _validate_dataflow_id(value: str, kind: str) -> None:
    if _DATAFLOW_ID.fullmatch(value) is None:
        raise ValueError(
            f"invalid {kind} ID {value!r}; a dataflow ID begins with a letter "
            "or underscore and continues with letters, digits, underscores, "
            "dots, or hyphens"
        )


def _validate_element_type(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_<>.,?x]*", value):
        raise ValueError(f"invalid MLIR element type {value!r}")


def _validate_index_map(value: str) -> None:
    if (
        not value.startswith("affine_map<") or
        not value.endswith(">") or
        "\n" in value or
        "\r" in value
    ):
        raise ValueError(f"invalid affine index map {value!r}")


class CommunicationDataflow:
    """Builder for frontend-neutral logical communication dataflow."""

    def __init__(self, name: str):
        _validate_dataflow_id(name, "dataflow")
        self.name = name
        self._regions: dict[str, CommunicationRegion] = {}
        self._operations: list[CommunicationDataflowOperation] = []
        self._sequence: list[tuple[str, str]] = []
        self._identifiers: set[str] = set()
        self._values: set[str] = set()

    @property
    def regions(self) -> tuple[CommunicationRegion, ...]:
        return tuple(self._regions.values())

    @property
    def operations(
        self,
    ) -> tuple[CommunicationDataflowOperation, ...]:
        return tuple(self._operations)

    def _add_region(
        self,
        region: CommunicationRegion,
    ) -> str:
        _validate_dataflow_id(region.id, "region")
        _validate_element_type(region.element_type)
        _validate_index_map(region.index_map)
        if region.valid_count is not None:
            if not isinstance(region.valid_count,str) or not region.valid_count or not region.dead_padding:
                raise ValueError("a valid input count requires a name and dead padding")
        elif region.dead_padding:
            raise ValueError("dead input padding requires a valid count")
        if region.id in self._identifiers:
            raise ValueError(
                f"duplicate dataflow identifier {region.id!r}"
            )
        if not region.shape or len(region.shape) != len(region.strides):
            raise ValueError(
                "region shape and strides must have equal positive rank"
            )
        if any(value <= 0 for value in (*region.shape, *region.strides)):
            raise ValueError("region shape and strides must be positive")
        if region.source is not None and region.source not in self._values:
            raise ValueError(f"unknown view source {region.source!r}")
        if region.indices is not None and region.indices not in self._values:
            raise ValueError(f"unknown indexed-view map {region.indices!r}")
        if region.reuse_interval < 0 or region.pool_depth < 0:
            raise ValueError(
                "reuse interval and pool depth cannot be negative"
            )
        if region.donated and not region.registered:
            raise ValueError("donated storage must be registered")
        self._regions[region.id] = region
        self._sequence.append(("region", region.id))
        self._identifiers.add(region.id)
        self._values.add(region.id)
        return region.id

    def region(
        self,
        region_id: str,
        *,
        element_type: str,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        index_map: Optional[str] = None,
        visibility: str = "local",
        immutable: bool = False,
        escapes: bool = False,
        registered: bool = False,
        donated: bool = False,
        reuse_interval: int = 0,
        pool_depth: int = 0,
        valid_count: Optional[str] = None,
        dead_padding: bool = False,
    ) -> str:
        return self._add_region(
            CommunicationRegion(
                id=region_id,
                element_type=element_type,
                shape=tuple(shape),
                strides=tuple(strides),
                index_map=(
                    index_map or
                    _identity_index_map(len(shape))
                ),
                visibility=visibility,
                immutable=immutable,
                escapes=escapes,
                registered=registered,
                donated=donated,
                reuse_interval=reuse_interval,
                pool_depth=pool_depth,
                valid_count=valid_count,
                dead_padding=dead_padding,
            )
        )

    def view(
        self,
        view_id: str,
        source: str,
        *,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        index_map: Optional[str] = None,
        element_type: Optional[str] = None,
        visibility: Optional[str] = None,
        immutable: Optional[bool] = None,
        escapes: Optional[bool] = None,
        registered: Optional[bool] = None,
        donated: Optional[bool] = None,
        reuse_interval: Optional[int] = None,
        pool_depth: Optional[int] = None,
    ) -> str:
        source_region = self._region_metadata(source)
        if source_region is None:
            raise ValueError(
                "views require a logical storage source"
            )
        return self._add_region(
            CommunicationRegion(
                id=view_id,
                element_type=(
                    element_type or
                    source_region.element_type
                ),
                shape=tuple(shape),
                strides=tuple(strides),
                index_map=(
                    index_map or
                    _identity_index_map(len(shape))
                ),
                visibility=(
                    visibility
                    if visibility is not None
                    else source_region.visibility
                ),
                immutable=(
                    immutable
                    if immutable is not None
                    else source_region.immutable
                ),
                escapes=(
                    escapes
                    if escapes is not None
                    else source_region.escapes
                ),
                source=source,
                registered=(
                    registered
                    if registered is not None
                    else source_region.registered
                ),
                donated=(
                    donated
                    if donated is not None
                    else source_region.donated
                ),
                reuse_interval=(
                    reuse_interval
                    if reuse_interval is not None
                    else source_region.reuse_interval
                ),
                pool_depth=(
                    pool_depth
                    if pool_depth is not None
                    else source_region.pool_depth
                ),
            )
        )

    def indexed_view(
        self,
        view_id: str,
        source: str,
        indices: str,
        *,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        index_lower_bounds: tuple[int, ...],
        index_upper_bounds: tuple[int, ...],
        injective: bool,
        index_map: Optional[str] = None,
    ) -> str:
        source_region = self._region_metadata(source)
        if source_region is None:
            raise ValueError(
                "indexed views require a logical storage source"
            )
        if len(index_lower_bounds) != len(index_upper_bounds):
            raise ValueError("indexed-view bounds must have equal rank")
        return self._add_region(
            CommunicationRegion(
                id=view_id,
                element_type=source_region.element_type,
                shape=tuple(shape),
                strides=tuple(strides),
                index_map=(
                    index_map or
                    _identity_index_map(len(shape))
                ),
                visibility=source_region.visibility,
                immutable=source_region.immutable,
                escapes=source_region.escapes,
                registered=source_region.registered,
                donated=source_region.donated,
                reuse_interval=source_region.reuse_interval,
                pool_depth=source_region.pool_depth,
                source=source,
                indices=indices,
                injective=injective,
                index_lower_bounds=tuple(index_lower_bounds),
                index_upper_bounds=tuple(index_upper_bounds),
            )
        )

    def _region_metadata(self, value: str) -> Optional[CommunicationRegion]:
        """Authoring metadata for a destination-passing SSA storage value."""
        if value in self._regions:
            return self._regions[value]
        for operation in reversed(self._operations):
            if value in operation.output_ids:
                index = operation.output_ids.index(value)
                if index < len(operation.destinations):
                    return self._region_metadata(operation.destinations[index])
            if value == operation.output_id and operation.destination is not None:
                return self._region_metadata(operation.destination)
        return None

    def _add_operation(
        self,
        operation: CommunicationDataflowOperation,
    ) -> Optional[str]:
        _validate_dataflow_id(operation.id, "operation")
        if operation.id in self._identifiers:
            raise ValueError(
                f"duplicate dataflow identifier {operation.id!r}"
            )
        for value in (
            operation.source,
            operation.destination,
            operation.workset,
            operation.count,
            *operation.inputs,
            *operation.destinations,
        ):
            if value is not None and value not in self._values:
                raise ValueError(f"unknown dataflow value {value!r}")
        self._identifiers.add(operation.id)
        if operation.output_id is not None:
            _validate_dataflow_id(operation.output_id, "output")
            if operation.output_id in self._identifiers:
                raise ValueError(
                    "duplicate dataflow identifier "
                    f"{operation.output_id!r}"
                )
            self._identifiers.add(operation.output_id)
            self._values.add(operation.output_id)
        for output_id in operation.output_ids:
            _validate_dataflow_id(output_id, "output")
            if output_id in self._identifiers:
                raise ValueError(
                    "duplicate dataflow identifier "
                    f"{output_id!r}"
                )
            self._identifiers.add(output_id)
            self._values.add(output_id)
        self._operations.append(operation)
        self._sequence.append(("operation", operation.id))
        return operation.output_id

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
    ) -> str:
        if not fields:
            raise ValueError(
                "workset requires at least one field"
            )
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        field_names = tuple(fields)
        if set(bounds) != set(fields):
            raise ValueError(
                "workset bounds must cover every field exactly"
            )
        lower_bounds = tuple(
            int(bounds[name][0])
            for name in field_names
        )
        upper_bounds = tuple(
            int(bounds[name][1])
            for name in field_names
        )
        if any(
            lower >= upper
            for lower, upper in zip(
                lower_bounds,
                upper_bounds,
                strict=True,
            )
        ):
            raise ValueError(
                "workset field bounds must be non-empty"
            )
        self._add_operation(
            CommunicationDataflowOperation(
                id=f"{workset_id}.workset",
                kind="workset",
                output_id=workset_id,
                inputs=tuple(fields.values()),
                field_names=field_names,
                field_lower_bounds=lower_bounds,
                field_upper_bounds=upper_bounds,
                count=count,
                max_items=max_items,
                unique_destinations=unique_destinations,
                ordered=ordered,
            )
        )
        return workset_id

    def transfer(
        self,
        operation_id: str,
        source: str,
        destination: str,
        *,
        output: Optional[str] = None,
    ) -> str:
        return self._add_operation(
            CommunicationDataflowOperation(
                id=operation_id,
                kind="transfer",
                output_id=output or f"{operation_id}.result",
                source=source,
                destination=destination,
            )
        )

    def transfer_each(
        self,
        operation_id: str,
        workset: str,
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
        return self._add_operation(
            CommunicationDataflowOperation(
                id=operation_id,
                kind="transfer_each",
                output_id=output or f"{operation_id}.result",
                source=source,
                destination=destination,
                workset=workset,
                source_index_field=source_index,
                destination_peer_field=destination_peer,
                destination_index_field=destination_index,
                bytes_field=bytes_field,
                predicate_field=predicate_field,
            )
        )

    def copy_each(
        self,
        operation_id: str,
        workset: str,
        source: str,
        destination: str,
        *,
        source_index: str = "source_row",
        destination_index: str = "destination_row",
        output: Optional[str] = None,
    ) -> str:
        return self._add_operation(
            CommunicationDataflowOperation(
                id=operation_id,
                kind="copy_each",
                output_id=output or f"{operation_id}.result",
                source=source,
                destination=destination,
                workset=workset,
                source_index_field=source_index,
                destination_index_field=destination_index,
            )
        )

    def reduction(
        self,
        operation_id: str,
        source: str,
        destination: str,
        *,
        cardinality: int,
        operator: str,
        output: Optional[str] = None,
    ) -> str:
        if cardinality <= 0:
            raise ValueError("reduction cardinality must be positive")
        if not operator:
            raise ValueError("reduction operator cannot be empty")
        return self._add_operation(
            CommunicationDataflowOperation(
                id=operation_id,
                kind="reduction",
                output_id=output or f"{operation_id}.result",
                source=source,
                destination=destination,
                cardinality=cardinality,
                operator=operator,
            )
        )

    def copy(
        self,
        operation_id: str,
        source: str,
        destination: str,
        *,
        output: Optional[str] = None,
    ) -> str:
        return self._add_operation(
            CommunicationDataflowOperation(
                id=operation_id,
                kind="copy",
                output_id=output or f"{operation_id}.result",
                source=source,
                destination=destination,
            )
        )

    def transform(
        self,
        operation_id: str,
        source: str,
        destination: str,
        *,
        transform: str,
        index_map: str,
        output: Optional[str] = None,
    ) -> str:
        if not transform:
            raise ValueError("transform name cannot be empty")
        _validate_index_map(index_map)
        return self._add_operation(
            CommunicationDataflowOperation(
                id=operation_id,
                kind="transform",
                output_id=output or f"{operation_id}.result",
                source=source,
                destination=destination,
                transform=transform,
                index_map=index_map,
            )
        )

    def use(self, operation_id: str, *inputs: str) -> None:
        if not inputs:
            raise ValueError("use requires at least one input")
        self._add_operation(
            CommunicationDataflowOperation(
                id=operation_id,
                kind="use",
                inputs=tuple(inputs),
            )
        )

    def async_use(
        self,
        operation_id: str,
        *inputs: str,
        max_inflight: int,
        event_ordered: bool,
    ) -> None:
        if not inputs:
            raise ValueError("async_use requires at least one input")
        if max_inflight <= 0:
            raise ValueError(
                "async_use max_inflight must be positive"
            )
        self._add_operation(
            CommunicationDataflowOperation(
                id=operation_id,
                kind="async_use",
                inputs=tuple(inputs),
                max_inflight=max_inflight,
                event_ordered=event_ordered,
            )
        )

    def opaque_stage(
        self,
        operation_id: str,
        *,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        outputs: Optional[tuple[str, ...]] = None,
        asynchronous: bool = False,
        max_inflight: int = 1,
        event_ordered: bool = False,
        bindable_outputs: bool = False,
        prefix_read_counts: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if not reads and not writes:
            raise ValueError(
                "opaque stage requires at least one read or write region"
            )
        if prefix_read_counts and (len(prefix_read_counts)!=len(reads)
                or any(not isinstance(count,str)for count in prefix_read_counts)):
            raise ValueError("prefix read counts require one runtime-count string per read")
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if not asynchronous and max_inflight != 1:
            raise ValueError(
                "synchronous opaque stages require max_inflight=1"
            )
        output_ids = (
            tuple(outputs)
            if outputs is not None
            else tuple(
                f"{operation_id}.output.{index}"
                for index in range(len(writes))
            )
        )
        if len(output_ids) != len(writes):
            raise ValueError(
                "opaque stage requires one output name per write region"
            )
        self._add_operation(
            CommunicationDataflowOperation(
                id=operation_id,
                kind="opaque_stage",
                inputs=tuple(reads),
                destinations=tuple(writes),
                output_ids=output_ids,
                asynchronous=asynchronous,
                max_inflight=max_inflight,
                event_ordered=event_ordered,
                bindable_outputs=bindable_outputs,
                prefix_read_counts=tuple(prefix_read_counts),
            )
        )
        return output_ids

    def simplify(self) -> SimplifiedCommunicationDataflow:
        return simplify_communication_dataflow(self)


def _format_i64_array(values: tuple[int, ...]) -> str:
    return "array<i64: " + ", ".join(map(str, values)) + ">"


def _communication_dataflow_mlir(
    dataflow: CommunicationDataflow,
) -> str:
    value_names: dict[str, str] = {}
    lines = [
        "module {",
        "  aboleth.communication_dataflow "
        f"{_quote(dataflow.name)} {{",
    ]

    def value_name(value_id: str) -> str:
        return value_names[value_id]

    next_value = 0

    def allocate_value(value_id: str) -> str:
        nonlocal next_value
        result = f"%v{next_value}"
        next_value += 1
        value_names[value_id] = result
        return result

    def emit_region(region: CommunicationRegion) -> None:
        result = allocate_value(region.id)
        attributes = (
            f"name = {_quote(region.id)}, "
            f"elementType = {region.element_type}, "
            f"shape = {_format_i64_array(region.shape)}, "
            f"strides = {_format_i64_array(region.strides)}, "
            f"indexMap = {region.index_map}, "
            f"visibility = {_quote(region.visibility)}, "
            f"immutable = {str(region.immutable).lower()}, "
            f"escapes = {str(region.escapes).lower()}, "
            f"registered = {str(region.registered).lower()}, "
            f"donated = {str(region.donated).lower()}, "
            f"reuseInterval = {region.reuse_interval} : i64, "
            f"poolDepth = {region.pool_depth} : i64"
        )
        if region.valid_count is not None:
            attributes += f", validCount = {_quote(region.valid_count)}, deadPadding = true"
        if region.is_indexed:
            lines.append(
                f"    {result} = aboleth.indexed_view "
                f"{value_name(region.source)} using "
                f"{value_name(region.indices)} "
                f"{{{attributes}, "
                f"injective = {str(region.injective).lower()}, "
                "indexLowerBounds = "
                f"{_format_i64_array(region.index_lower_bounds)}, "
                "indexUpperBounds = "
                f"{_format_i64_array(region.index_upper_bounds)}}} : "
                "!aboleth.logical_region, "
                "!aboleth.logical_region -> "
                "!aboleth.logical_region"
            )
        elif region.is_view:
            lines.append(
                f"    {result} = aboleth.view "
                f"{value_name(region.source)} "
                f"{{{attributes}}} : "
                "!aboleth.logical_region -> "
                "!aboleth.logical_region"
            )
        else:
            lines.append(
                f"    {result} = aboleth.region "
                f"{{{attributes}}} : "
                "!aboleth.logical_region"
            )

    def emit_operation(
        operation: CommunicationDataflowOperation,
    ) -> None:
        if operation.kind == "workset":
            operands = (
                *operation.inputs,
                *((operation.count,) if operation.count else ()),
            )
            operand_values = ", ".join(
                value_name(value) for value in operands
            )
            operand_types = ", ".join(
                "!aboleth.logical_region"
                for _ in operands
            )
            field_names = ", ".join(
                _quote(value)
                for value in operation.field_names
            )
            result = allocate_value(operation.output_id)
            lines.append(
                f'    {result} = "aboleth.workset"'
                f"({operand_values}) <{{"
                f"id = {_quote(operation.id)}, "
                f"name = {_quote(operation.output_id)}, "
                f"fieldNames = [{field_names}], "
                "fieldLowerBounds = "
                f"{_format_i64_array(operation.field_lower_bounds)}, "
                "fieldUpperBounds = "
                f"{_format_i64_array(operation.field_upper_bounds)}, "
                f"maxItems = {operation.max_items} : i64, "
                "uniqueDestinations = "
                f"{str(operation.unique_destinations).lower()}, "
                f"ordered = {str(operation.ordered).lower()}, "
                "operandSegmentSizes = "
                f"array<i32: {len(operation.inputs)}, "
                f"{1 if operation.count else 0}>"
                "}> : "
                f"({operand_types}) -> !aboleth.workset"
            )
            return

        if operation.kind in ("use", "async_use"):
            operands = ", ".join(
                value_name(value) for value in operation.inputs
            )
            types = ", ".join(
                "!aboleth.logical_region"
                for _ in operation.inputs
            )
            attributes = [
                f"name = {_quote(operation.id)}",
                f"id = {_quote(operation.id)}",
            ]
            if operation.kind == "async_use":
                attributes.extend([
                    f"maxInflight = {operation.max_inflight} : i64",
                    "eventOrdered = " +
                    str(operation.event_ordered).lower(),
                ])
            lines.append(
                f"    aboleth.{operation.kind} {operands} "
                f"{{{', '.join(attributes)}}} : {types}"
            )
            return

        if operation.kind == "transfer_each":
            result = allocate_value(operation.output_id)
            optional = ""
            if operation.bytes_field is not None:
                optional += (
                    ", bytesField = "
                    f"{_quote(operation.bytes_field)}"
                )
            if operation.predicate_field is not None:
                optional += (
                    ", predicateField = "
                    f"{_quote(operation.predicate_field)}"
                )
            lines.append(
                f'    {result} = "aboleth.transfer_each"('
                f"{value_name(operation.workset)}, "
                f"{value_name(operation.source)}, "
                f"{value_name(operation.destination)}) <{{"
                f"id = {_quote(operation.id)}, "
                f"outputId = {_quote(operation.output_id)}, "
                "sourceIndexField = "
                f"{_quote(operation.source_index_field)}, "
                "destinationPeerField = "
                f"{_quote(operation.destination_peer_field)}, "
                "destinationIndexField = "
                f"{_quote(operation.destination_index_field)}"
                f"{optional}"
                "}> : (!aboleth.workset, "
                "!aboleth.logical_region, "
                "!aboleth.logical_region) -> "
                "!aboleth.logical_region"
            )
            return

        if operation.kind == "copy_each":
            result = allocate_value(operation.output_id)
            lines.append(
                f'    {result} = "aboleth.copy_each"('
                f"{value_name(operation.workset)}, "
                f"{value_name(operation.source)}, "
                f"{value_name(operation.destination)}) <{{"
                f"id = {_quote(operation.id)}, "
                f"outputId = {_quote(operation.output_id)}, "
                "sourceIndexField = "
                f"{_quote(operation.source_index_field)}, "
                "destinationIndexField = "
                f"{_quote(operation.destination_index_field)}"
                "}> : (!aboleth.workset, "
                "!aboleth.logical_region, "
                "!aboleth.logical_region) -> "
                "!aboleth.logical_region"
            )
            return

        if operation.kind == "opaque_stage":
            operands = (
                *operation.inputs,
                *operation.destinations,
            )
            operand_values = ", ".join(
                value_name(value) for value in operands
            )
            operand_types = ", ".join(
                "!aboleth.logical_region"
                for _ in operands
            )
            result_names = []
            for output_id in operation.output_ids:
                result_names.append(
                    allocate_value(output_id)
                )
            result_prefix = (
                ", ".join(result_names) + " = "
                if result_names
                else ""
            )
            result_types = ", ".join(
                "!aboleth.logical_region"
                for _ in result_names
            )
            output_ids = ", ".join(
                _quote(value)
                for value in operation.output_ids
            )
            binding = "bindableOutputs = true, " if operation.bindable_outputs else ""
            if operation.prefix_read_counts:
                binding += "prefixReadCounts = [" + ", ".join(_quote(count)
                    for count in operation.prefix_read_counts) + "], "
            lines.append(
                f"    {result_prefix}"
                f'"aboleth.opaque_stage"({operand_values}) <{{'
                f"name = {_quote(operation.id)}, "
                f"outputIds = [{output_ids}], "
                f"{binding}"
                "asynchronous = "
                f"{str(operation.asynchronous).lower()}, "
                f"maxInflight = {operation.max_inflight} : i64, "
                "eventOrdered = "
                f"{str(operation.event_ordered).lower()}, "
                "operandSegmentSizes = "
                f"array<i32: {len(operation.inputs)}, "
                f"{len(operation.destinations)}>"
                "}> : "
                f"({operand_types}) -> ({result_types})"
            )
            return

        result = allocate_value(operation.output_id)
        attributes = [
            f"id = {_quote(operation.id)}",
            f"outputId = {_quote(operation.output_id)}",
        ]
        if operation.kind == "reduction":
            attributes.extend([
                f"cardinality = {operation.cardinality} : i64",
                f"operatorName = {_quote(operation.operator)}",
            ])
        elif operation.kind == "transform":
            attributes.extend([
                f"transform = {_quote(operation.transform)}",
                f"indexMap = {operation.index_map}",
            ])
        attribute_text = ", ".join(attributes)
        lines.append(
            f"    {result} = aboleth.{operation.kind} "
            f"{value_name(operation.source)} into "
            f"{value_name(operation.destination)} "
            f"{{{attribute_text}}} : "
            "(!aboleth.logical_region, "
            "!aboleth.logical_region) -> "
            "!aboleth.logical_region"
        )

    operations_by_id = {
        operation.id: operation
        for operation in dataflow.operations
    }
    for kind, identifier in dataflow._sequence:
        if kind == "region":
            emit_region(dataflow._regions[identifier])
        else:
            emit_operation(operations_by_id[identifier])

    lines.extend([
        "    ub.unreachable",
        "  }",
        "}",
    ])
    return "\n".join(lines) + "\n"


def _extract_simplified_dataflow(
    module,
    dataflow: CommunicationDataflow,
    input_mlir: str,
) -> SimplifiedCommunicationDataflow:
    value_ids: dict[int, str] = {}
    operations: list[CommunicationDataflowOperation] = []
    remaining_region_ids: set[str] = set()
    original_by_id = {
        operation.id: operation
        for operation in dataflow.operations
    }

    def operand_ids(operation) -> tuple[str, ...]:
        return tuple(
            value_ids[operation.get_operand(index).id()]
            for index in range(operation.get_num_operands())
        )

    def visit(operation) -> None:
        full_name = operation.get_name()
        if not full_name.startswith("aboleth."):
            return
        kind = full_name.removeprefix("aboleth.")
        if kind == "communication_dataflow":
            return

        operands = operand_ids(operation)
        if kind in ("region", "view", "indexed_view"):
            region_id = operation.get_str_attr("name")
            output_id = region_id
            value_ids[operation.get_result(0).id()] = output_id
            remaining_region_ids.add(region_id)
            operations.append(
                CommunicationDataflowOperation(
                    id=region_id,
                    kind=kind,
                    output_id=output_id,
                    source=(
                        operands[0]
                        if kind in ("view", "indexed_view")
                        else None
                    ),
                )
            )
            return

        operation_id = (
            operation.get_str_attr("id") or
            operation.get_str_attr("name")
        )
        if kind in ("use", "async_use"):
            original = original_by_id.get(operation_id)
            operations.append(
                CommunicationDataflowOperation(
                    id=operation_id,
                    kind=kind,
                    inputs=operands,
                    max_inflight=(
                        original.max_inflight
                        if kind == "async_use"
                        else None
                    ),
                    event_ordered=(
                        original.event_ordered
                        if kind == "async_use"
                        else None
                    ),
                )
            )
            return

        if kind == "workset":
            original = original_by_id[operation_id]
            output_id = original.output_id
            value_ids[
                operation.get_result(0).id()
            ] = output_id
            operations.append(original)
            return

        if kind == "transfer_each":
            original = original_by_id[operation_id]
            output_id = (
                operation.get_str_attr("outputId") or
                original.output_id
            )
            value_ids[
                operation.get_result(0).id()
            ] = output_id
            operations.append(
                CommunicationDataflowOperation(
                    id=operation_id,
                    kind=kind,
                    output_id=output_id,
                    workset=operands[0],
                    source=operands[1],
                    destination=operands[2],
                    source_index_field=(
                        original.source_index_field
                    ),
                    destination_peer_field=(
                        original.destination_peer_field
                    ),
                    destination_index_field=(
                        original.destination_index_field
                    ),
                    bytes_field=original.bytes_field,
                    predicate_field=(
                        original.predicate_field
                    ),
                )
            )
            return

        if kind == "copy_each":
            original = original_by_id[operation_id]
            output_id = (
                operation.get_str_attr("outputId") or
                original.output_id
            )
            value_ids[
                operation.get_result(0).id()
            ] = output_id
            operations.append(
                CommunicationDataflowOperation(
                    id=operation_id,
                    kind=kind,
                    output_id=output_id,
                    workset=operands[0],
                    source=operands[1],
                    destination=operands[2],
                    source_index_field=(
                        original.source_index_field
                    ),
                    destination_index_field=(
                        original.destination_index_field
                    ),
                )
            )
            return

        if kind == "opaque_stage":
            original = original_by_id[operation_id]
            read_count = len(original.inputs)
            write_count = len(original.destinations)
            for index, output_id in enumerate(
                original.output_ids
            ):
                value_ids[
                    operation.get_result(index).id()
                ] = output_id
            operations.append(
                CommunicationDataflowOperation(
                    id=operation_id,
                    kind=kind,
                    inputs=operands[:read_count],
                    destinations=operands[
                        read_count:
                        read_count + write_count
                    ],
                    output_ids=original.output_ids,
                    asynchronous=original.asynchronous,
                    max_inflight=original.max_inflight,
                    event_ordered=original.event_ordered,
                    bindable_outputs=original.bindable_outputs,
                    prefix_read_counts=original.prefix_read_counts,
                )
            )
            return

        output_id = (
            operation.get_str_attr("outputId") or
            operation_id
        )
        value_ids[operation.get_result(0).id()] = output_id
        operations.append(
            CommunicationDataflowOperation(
                id=operation_id,
                kind=kind,
                output_id=output_id,
                source=operands[0],
                destination=operands[1],
                cardinality=(
                    operation.get_int_attr("cardinality")
                    if kind == "reduction"
                    else None
                ),
                operator=(
                    operation.get_str_attr("operatorName")
                    if kind == "reduction"
                    else None
                ),
                transform=(
                    operation.get_str_attr("transform")
                    if kind == "transform"
                    else None
                ),
                index_map=(
                    original_by_id[operation_id].index_map
                    if (
                        kind == "transform" and
                        operation_id in original_by_id
                    )
                    else None
                ),
            )
        )

    module.walk(visit)
    actual_by_id = {
        operation.id: operation
        for operation in operations
        if operation.kind not in ("region", "view", "indexed_view")
    }
    original_uses = {
        operation.id: operation
        for operation in dataflow.operations
        if operation.kind in ("use", "async_use")
    }
    actual_uses = {
        operation.id: operation
        for operation in operations
        if operation.kind in ("use", "async_use")
    }
    use_provenance: dict[
        str, tuple[tuple[str, str], ...]
    ] = {}
    for use_id, original in original_uses.items():
        actual = actual_uses.get(use_id)
        if actual is None:
            continue
        use_provenance[use_id] = tuple(zip(
            original.inputs, actual.inputs, strict=True
        ))

    retargeted = {}
    for operation_id, original in original_by_id.items():
        actual = actual_by_id.get(operation_id)
        if (
            actual is not None and
            original.destination is not None and
            actual.destination != original.destination
        ):
            retargeted[operation_id] = (
                original.destination,
                actual.destination,
            )

    for operation_id, original in original_by_id.items():
        actual = actual_by_id.get(operation_id)
        if original.kind != "opaque_stage" or actual is None:
            continue
        for index, (before, after) in enumerate(zip(original.destinations, actual.destinations, strict=True)):
            if before != after:
                retargeted[f"{operation_id}.output.{index}"] = (before, after)

    remaining_operation_ids = {
        operation.id
        for operation in operations
        if operation.kind not in ("region", "view", "indexed_view")
    }
    return SimplifiedCommunicationDataflow(
        name=dataflow.name,
        regions=tuple(
            region
            for region in dataflow.regions
            if region.id in remaining_region_ids
        ),
        operations=tuple(operations),
        removed_operation_ids=frozenset(
            original_by_id.keys() -
            remaining_operation_ids
        ),
        removed_region_ids=frozenset(
            dataflow._regions.keys() -
            remaining_region_ids
        ),
        use_provenance=use_provenance,
        retargeted_destinations=retargeted,
        input_mlir=input_mlir,
        mlir=module.str_nodebug(),
    )


def simplify_communication_dataflow(
    dataflow: CommunicationDataflow,
) -> SimplifiedCommunicationDataflow:
    """Run the compiler and return its actual rewritten SSA operation graph."""
    source = _communication_dataflow_mlir(dataflow)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".mlir",
    ) as file:
        file.write(source)
        file.flush()
        context = ir.context()
        ir.load_dialects(context)
        module = ir.parse_mlir_module(file.name, context)
        manager = ir.pass_manager(context)
        from .compiler import enabled_optimizations
        passes.plugin.add_aboleth_simplify_communication_dataflow(
            manager, [
                "optimized" if "plan-storage" in enabled_optimizations() else "unoptimized",
                "prune-dead" if "dead-communication" in enabled_optimizations() else "keep-dead",
            ]
        )
        manager.run(module, "aboleth-communication-dataflow")
        return _extract_simplified_dataflow(
            module, dataflow, source
        )


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _regions(regions: tuple[RegionEffect, ...]) -> str:
    encoded = []
    for region in regions:
        unknown = ", unknown = true" if region.unknown else ""
        encoded.append(
            "{"
            f"allocation = {region.allocation} : i64, "
            f"generation = {region.generation} : i64, "
            f"offset = {region.offset} : i64, "
            f"length = {region.length} : i64, "
            f"stride = {region.stride} : i64, "
            f"period = {region.period} : i64, "
            f"count = {region.count} : i32"
            f"{unknown}"
            "}"
        )
    return "[" + ", ".join(encoded) + "]"


def _artifact_attributes(
    *,
    bound_kernel: Optional[str],
    artifact_kind: Optional[str],
    insertion_point: Optional[str],
) -> str:
    result = ""
    if bound_kernel:
        result += (
            f"      boundKernel = {_quote(bound_kernel)},\n"
        )
    if artifact_kind:
        result += (
            f"      artifactKind = {_quote(artifact_kind)},\n"
        )
    if insertion_point:
        result += (
            "      insertionPoint = "
            f"{_quote(insertion_point)},\n"
        )
    return result


def _phase_operation(phase: TransportPhase) -> str:
    transports = ", ".join(
        _quote(value)
        for value in phase.peer_transports
    )
    capacities = ", ".join(
        str(value)
        for value in phase.peer_capacities
    )
    intervals = ""
    binding = _artifact_attributes(
        bound_kernel=phase.bound_kernel,
        artifact_kind=phase.artifact_kind,
        insertion_point=phase.insertion_point,
    )
    if phase.program_id is not None:
        binding += f"      programId = {phase.program_id} : i32,\n"
    if phase.completion_intervals:
        values = ", ".join(
            str(value)
            for value in phase.completion_intervals
        )
        intervals = (
            f"      peerCompletionIntervals = "
            f"array<i32: {values}>,\n"
        )
    return f"""
    "aboleth.transport_phase"() <{{
      name = {_quote(phase.name)},
{binding}      commandsPerIteration = {phase.commands_per_iteration} : i32,
      iterationCount = {phase.iteration_count} : i32,
      payloadBytes = {phase.payload_bytes} : i64,
      peerTransports = [{transports}],
      peerCapacities = array<i32: {capacities}>,
      policyTinyPayloadBytes = {phase.tiny_payload_bytes} : i64,
{intervals}      reads = {_regions(phase.reads)},
      writes = {_regions(phase.writes)}
    }}> : () -> ()
"""


def _kernel_operation(kernel: TransportKernel) -> str:
    binding = _artifact_attributes(
        bound_kernel=kernel.bound_kernel,
        artifact_kind=kernel.artifact_kind,
        insertion_point=kernel.insertion_point,
    )
    asynchronous = ""
    if kernel.asynchronous:
        asynchronous = (
            "      asynchronous = true,\n"
            f"      maxInflight = {kernel.max_inflight} : i32,\n"
            f"      eventOrdered = "
            f"{str(kernel.event_ordered).lower()},\n"
        )
    opaque = "      opaque = true,\n" if kernel.opaque else ""
    return f"""
    "aboleth.transport_kernel"() <{{
      name = {_quote(kernel.name)},
{binding}{asynchronous}{opaque}      reads = {_regions(kernel.reads)},
      writes = {_regions(kernel.writes)}
    }}> : () -> ()
"""


def _validate_phase(phase: TransportPhase):
    if not phase.name:
        raise ValueError(
            "transport phase name cannot be empty"
        )
    if (
        len(phase.peer_transports) !=
        len(phase.peer_capacities)
    ):
        raise ValueError(
            "transport phase requires one "
            "capacity per peer transport"
        )
    if not phase.peer_transports:
        raise ValueError(
            "transport phase requires at least one peer"
        )
    if (
        phase.completion_intervals and
        len(phase.completion_intervals) !=
        len(phase.peer_transports)
    ):
        raise ValueError(
            "transport phase requires one completion "
            "interval per peer transport"
        )
    if not phase.reads:
        raise ValueError(
            "transport phase requires source read effects"
        )


def _validate_kernel(kernel: TransportKernel):
    if not kernel.name:
        raise ValueError(
            "transport kernel name cannot be empty"
        )
    if kernel.max_inflight <= 0:
        raise ValueError(
            "transport kernel max_inflight must be positive"
        )
    if (
        kernel.asynchronous and
        kernel.max_inflight > 1 and
        not kernel.event_ordered
    ):
        raise ValueError(
            "multi-generation asynchronous kernels require "
            "event ordering"
        )


def _planned_facts(
    facts: dict,
    kind: str,
) -> dict[str, dict]:
    result = {}
    for program in facts["programs"]:
        for record in program[kind]:
            name = record.get("name")
            if name is None:
                raise RuntimeError(
                    "transport planner emitted an "
                    "unnamed transport operation"
                )
            result[name] = record
    return result


@lru_cache(maxsize=128)
def _run_transport_planner(
    operations: tuple[
        TransportPhase | TransportKernel, ...
    ],
    cyclic: bool = False,
    transport_geography: tuple[tuple[int, str], ...] = (),
) -> tuple[str, dict]:
    if not operations and not transport_geography:
        raise ValueError(
            "transport program cannot be empty"
        )
    names = set()
    encoded = []
    for operation in operations:
        if operation.name in names:
            raise ValueError(
                "transport operation names must be unique"
            )
        names.add(operation.name)
        if isinstance(operation, TransportPhase):
            _validate_phase(operation)
            encoded.append(
                _phase_operation(operation)
            )
        else:
            _validate_kernel(operation)
            encoded.append(
                _kernel_operation(operation)
            )
    body = "".join(encoded)
    cyclic_attribute = (
        "true" if cyclic else "false"
    )
    geography = ""
    if transport_geography:
        if any(type(peer) is not int or not 0 <= peer < 2**31
               or not isinstance(backend, str) for peer, backend in transport_geography):
            raise ValueError("geography requires nonnegative i32 peer ranks and transport names")
        ranks = ", ".join(str(peer) for peer, _ in transport_geography)
        backends = ", ".join(_quote(backend) for _, backend in transport_geography)
        geography = (f", peerRanks = array<i32: {ranks}>, "
                     f"peerTransports = [{backends}]")
    source = f"""
module {{
  aboleth.transport_program {{
{body}
    ub.unreachable
  }} {{
    name = "transport-program",
    cyclic = {cyclic_attribute}{geography}
  }}
}}
"""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".mlir",
    ) as file, tempfile.NamedTemporaryFile(
        suffix=".json",
    ) as plan:
        file.write(source)
        file.flush()
        context = ir.context()
        ir.load_dialects(context)
        module = ir.parse_mlir_module(file.name, context)
        manager = ir.pass_manager(context)
        passes.plugin.add_aboleth_plan_transport(
            manager,
            [plan.name],
        )
        manager.run(module, "aboleth-transport-plan")
        with open(plan.name) as facts:
            return (
                module.str_nodebug(),
                json.load(facts),
            )


def compile_transport_program_mlir(
    operations: tuple[
        TransportPhase | TransportKernel, ...
    ],
    *,
    cyclic: bool = False,
    fragments: str = "",
    transport_geography: tuple[tuple[int, str], ...] = (),
) -> str:
    """Return the planned transport_program as importable MLIR."""
    planned, _ = _run_transport_planner(
        operations,
        cyclic,
        transport_geography,
    )
    if not fragments:
        return planned
    marker = "    ub.unreachable"
    if marker not in planned:
        raise RuntimeError(
            "transport planner output has no program terminator"
        )
    return planned.replace(
        marker,
        f"{fragments}\n{marker}",
        1,
    )


@lru_cache(maxsize=128)
def plan_transport_facts(
    operations: tuple[
        TransportPhase | TransportKernel, ...
    ],
    cyclic: bool = False,
) -> dict:
    """Return the planner's JSON facts for one transport program.

    The document is the same one `emit-plan-json` writes: keys are the MLIR
    attribute names, so a fact reads the same in the module and here. The
    result is shared, so callers must treat it as read-only.
    """
    _, facts = _run_transport_planner(
        operations,
        cyclic,
    )
    return facts


@lru_cache(maxsize=128)
def plan_transport_program(
    operations: tuple[
        TransportPhase | TransportKernel, ...
    ],
    cyclic: bool = False,
) -> dict[str, TransportPlan]:
    _, facts = _run_transport_planner(
        operations,
        cyclic,
    )

    result = {}
    for name, phase in _planned_facts(
        facts, "phases"
    ).items():
        try:
            result[name] = TransportPlan(
                completion_interval=phase[
                    "resolvedCompletionInterval"
                ],
                reuse_distance=phase[
                    "reuseDistance"
                ],
                completion_before=phase[
                    "completionBefore"
                ],
                completion_iteration_delta=phase[
                    "completionIterationDelta"
                ],
                publication_before=phase[
                    "publicationBefore"
                ],
                publication_iteration_delta=phase[
                    "publicationIterationDelta"
                ],
            )
        except KeyError as error:
            raise RuntimeError(
                "transport planner omitted "
                "required phase output"
            ) from error
    expected = {
        operation.name
        for operation in operations
        if isinstance(
            operation, TransportPhase
        )
    }
    if result.keys() != expected:
        raise RuntimeError(
            "transport planner omitted a phase"
        )
    return result


@lru_cache(maxsize=128)
def plan_output_lifetimes(
    operations: tuple[
        TransportPhase | TransportKernel, ...
    ],
    cyclic: bool = False,
) -> dict[str, OutputLifetimePlan]:
    _, facts = _run_transport_planner(
        operations,
        cyclic,
    )
    result = {}
    for name, kernel in _planned_facts(
        facts, "kernels"
    ).items():
        if not kernel.get("outputReusable", False):
            continue
        try:
            result[name] = OutputLifetimePlan(
                last_consumer=kernel[
                    "outputLastConsumer"
                ],
                last_consumer_iteration_delta=kernel[
                    "outputLastConsumerIterationDelta"
                ],
                reuse_before=kernel[
                    "outputReuseBefore"
                ],
                reuse_iteration_delta=kernel[
                    "outputReuseIterationDelta"
                ],
                pool_depth=kernel[
                    "outputPoolDepth"
                ],
            )
        except KeyError as error:
            raise RuntimeError(
                "transport planner emitted an incomplete "
                "kernel output lifetime"
            ) from error
    return result


@lru_cache(maxsize=128)
def plan_transport(
    phase: TransportPhase,
) -> TransportPlan:
    return plan_transport_program(
        (phase,)
    )[phase.name]
