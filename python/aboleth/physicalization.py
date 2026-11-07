from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Callable

from .pattern import (
    CommunicationStage,
    CompiledPattern,
    KernelTransportProgram,
    KernelTransportReduction,
    KernelTransportTransform,
    OpaqueStage,
)
from .planning import RegionEffect
from .transport_binding import (
    BoundScalar,
    InvokeManifest,
    PublicationBinding,
    RegionBinding,
    RegionSide,
    TransportProgramBinding,
    invoke_manifest,
)


class Artifact(Enum):
    """Which form the attachment's communication is materialized as.

    The three forms of ARCHITECTURE.md's "Artifact forms"; the values are
    the `artifactKind` strings a transport program prints, and mirror
    pattern.ARTIFACT_KINDS.
    """

    INLINE_FRAGMENT = "inline_fragment"
    BINDABLE_EPILOGUE = "bindable_epilogue"
    STANDALONE_KERNEL = "standalone_kernel"


# Operation kinds physicalization derives a transport command for. A copy is a
# local transfer: it binds the same region pair, with this rank on both sides.
# A transform derives a kernel obligation, not a wire command: the client
# kernel that packs the phase applies it while producing the destination.
# Everything else in a communication stage is still rejected by name.
_PHYSICALIZED_KINDS = (
    "transfer",
    "transfer_each",
    "reduction",
    "copy",
    "copy_each",
    "transform",
)

# Transform names physicalization knows how to obligate a kernel with.
# Dotted suffixes select the output role (".values", ".scales").
_KNOWN_TRANSFORM_STEMS = ("quantize.e4m3.g128",)


@dataclass(frozen=True)
class PhaseBinding:
    """Where one generated communication phase integrates with client code."""

    producer: str | None
    consumer: str | None
    name: str
    kernel: str
    insertion_point: str
    peer_segments: int
    publication: str
    items_per_transfer: int = 0
    artifact: Artifact = Artifact.INLINE_FRAGMENT

    def __post_init__(self):
        for field in (
            "name",
            "kernel",
            "insertion_point",
            "publication",
        ):
            if not getattr(self, field):
                raise ValueError(
                    f"communication attachment {field} cannot be empty"
                )
        if self.producer is None and self.consumer is None:
            raise ValueError(
                "communication attachment requires a producer "
                "or consumer boundary"
            )
        if self.peer_segments <= 0:
            raise ValueError(
                "communication attachment peer_segments must be positive"
            )
        if self.items_per_transfer < 0:
            raise ValueError(
                "items_per_transfer cannot be negative"
            )
        if not isinstance(self.artifact, Artifact):
            raise ValueError(
                f"communication attachment artifact must be an Artifact, "
                f"got {self.artifact!r}; expected one of "
                + ", ".join(
                    f"Artifact.{kind.name}" for kind in Artifact
                )
            )


@dataclass(frozen=True)
class PhysicalizedPattern:
    compiled: CompiledPattern
    bindings: tuple[TransportProgramBinding, ...]

    def binding(self, name: str) -> TransportProgramBinding:
        return next(
            binding
            for binding in self.bindings
            if binding.name == name
        )

    def program(self, name: str) -> KernelTransportProgram:
        return next(
            program
            for program in self.compiled.kernel_transport_programs
            if program.name == name
        )

    def manifest(self, name: str) -> InvokeManifest:
        return invoke_manifest(self.program(name))


@dataclass(frozen=True)
class RuntimeBufferBinding:
    buffer: str
    regions: tuple[str, ...]

    def __post_init__(self):
        if not self.buffer or not self.regions:
            raise ValueError(
                "runtime buffer binding requires a buffer and regions"
            )
        if len(self.regions) != len(set(self.regions)):
            raise ValueError(
                "runtime buffer binding regions must be unique"
            )


@dataclass(frozen=True)
class RuntimeBindings:
    buffers: tuple[RuntimeBufferBinding, ...]

    def __post_init__(self):
        regions = tuple(
            region
            for binding in self.buffers
            for region in binding.regions
        )
        if len(regions) != len(set(regions)):
            raise ValueError(
                "logical regions may bind to only one runtime buffer"
            )

    def effects(
        self,
        compiled: CompiledPattern,
        resolve: Callable[[str], RegionEffect],
    ) -> dict[str, RegionEffect]:
        runtime_by_region = {
            region: binding.buffer
            for binding in self.buffers
            for region in binding.regions
        }
        required = {
            region.id
            for region in compiled.analysis.regions
        }
        missing = required - set(runtime_by_region)
        if missing:
            raise KeyError(
                "no runtime buffer binding for logical regions: "
                + ", ".join(sorted(missing))
            )
        unknown = (
            set(runtime_by_region)
            - required
            - set(compiled.analysis.removed_region_ids)
        )
        if unknown:
            raise KeyError(
                "runtime bindings reference unknown logical regions: "
                + ", ".join(sorted(unknown))
            )
        runtime_by_region = {
            region: buffer
            for region, buffer in runtime_by_region.items()
            if region in required
        }
        used_buffers = set(runtime_by_region.values())
        resolved = {
            binding.buffer: resolve(binding.buffer)
            for binding in self.buffers
            if binding.buffer in used_buffers
        }
        return {
            region: resolved[buffer]
            for region, buffer in runtime_by_region.items()
        }


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
    try:
        return sizes[element_type]
    except KeyError as error:
        raise ValueError(
            f"unknown physical element size for {element_type!r}"
        ) from error


def _slot_name(value: str) -> str:
    return "".join(
        character
        if character.isalnum()
        else "_"
        for character in value
    )


def _validate_slot_stems(
    attachment: PhaseBinding,
    operations: tuple,
) -> None:
    """Refuse two slot sources that mangle to one stem.

    Every slot this binding declares is a mangled stem plus a role suffix, so
    two sources sharing a stem name the same slots: the binding would bind one
    source's pointers to the other's commands instead of failing.
    """
    stems: dict[str, str] = {}
    for described, value in (
        *(
            (f"operation {operation.id!r}", operation.id)
            for operation in operations
        ),
        (f"publication {attachment.publication!r}", attachment.publication),
    ):
        stem = _slot_name(value)
        previous = stems.get(stem)
        if previous is not None:
            raise ValueError(
                f"attachment {attachment.name!r} derives the slot stem "
                f"{stem!r} from both {previous} and {described}; rename one "
                "so each names its own slots"
            )
        stems[stem] = described


def _side(pointer: str, key: str) -> RegionSide:
    return RegionSide(
        pointer=BoundScalar(pointer, "i64"),
        key=BoundScalar(key, "i32"),
    )


def _resolve_region(compiled: CompiledPattern, value: str):
    regions = {
        region.id: region
        for region in compiled.analysis.regions
    }
    producers = {}
    for operation in compiled.analysis.operations:
        if operation.output_id is not None:
            producers[operation.output_id] = (
                operation,
                0,
            )
        for index, output in enumerate(operation.output_ids):
            producers[output] = (operation, index)
    seen = set()
    current = value
    while current not in regions:
        if current in seen:
            raise ValueError(
                f"cyclic region resolution for {value!r}"
            )
        seen.add(current)
        try:
            operation, index = producers[current]
        except KeyError as error:
            raise KeyError(
                f"no physical region for logical value {value!r}"
            ) from error
        if operation.kind == "opaque_stage":
            current = operation.destinations[index]
        elif operation.destination is not None:
            current = operation.destination
        else:
            raise KeyError(
                f"logical value {current!r} has no destination region"
            )
    while regions[current].source is not None:
        current = regions[current].source
    return regions[current]


def _workset(compiled: CompiledPattern, name: str):
    return next(
        operation
        for operation in compiled.analysis.operations
        if (
            operation.kind == "workset"
            and operation.output_id == name
        )
    )


def _segment_geometry(
    compiled: CompiledPattern,
    operation,
    peer_segments: int,
) -> tuple[int, int]:
    if operation.source is None:
        raise ValueError(
            f"operation {operation.id!r} has no source"
        )
    region = _resolve_region(compiled, operation.source)
    if not region.shape:
        return 1, _element_bytes(region.element_type)
    row_bytes = (
        prod(region.shape[1:])
        * _element_bytes(region.element_type)
    )
    if operation.kind in ("transfer_each", "copy_each"):
        workset = _workset(compiled, operation.workset)
        if workset.max_items is None:
            raise ValueError(
                f"workset for {operation.id!r} has no maximum"
            )
        return workset.max_items, row_bytes
    if operation.kind == "copy":
        # A copy is a local transfer: it materializes the whole staged
        # extent on this rank, so there is no peer fan-out to divide by.
        return region.shape[0], row_bytes
    if operation.kind not in ("transfer", "reduction"):
        raise NotImplementedError(
            f"physicalization does not derive geometry for "
            f"{operation.kind!r} operation {operation.id!r}; supported kinds "
            "are " + ", ".join(_PHYSICALIZED_KINDS)
        )
    extent = region.shape[0]
    if extent % peer_segments:
        raise ValueError(
            f"operation {operation.id!r} source extent {extent} "
            f"is not divisible by {peer_segments} peer segments"
        )
    return extent // peer_segments, row_bytes


def _transfer_regions(
    compiled: CompiledPattern,
    operation,
    attachment: PhaseBinding,
) -> tuple[RegionBinding, ...]:
    max_items, item_bytes = _segment_geometry(
        compiled,
        operation,
        attachment.peer_segments,
    )
    items = (
        max_items
        if attachment.items_per_transfer == 0
        else min(attachment.items_per_transfer, max_items)
    )
    count = (max_items + items - 1) // items
    stem = _slot_name(operation.id)
    regions = []
    for index in range(count):
        suffix = "" if count == 1 else f"_{index}"
        chunk_items = min(
            items,
            max_items - index * items,
        )
        regions.append(RegionBinding(
            operations=(
                (operation.id,) if index == 0 else ()
            ),
            remote=_side(
                f"{stem}_destination{suffix}",
                f"{stem}_destination_key{suffix}",
            ),
            local=_side(
                f"{stem}_source{suffix}",
                f"{stem}_source_key{suffix}",
            ),
            byte_count=BoundScalar(
                f"{stem}_bytes{suffix}",
                "i32",
            ),
            predicate=BoundScalar(
                f"{stem}_predicate{suffix}",
                "i1",
            ),
            max_bytes=chunk_items * item_bytes,
        ))
    return tuple(regions)


def _reduce_contract(
    compiled: CompiledPattern,
    operation,
) -> KernelTransportReduction:
    if operation.cardinality is None or not operation.operator:
        raise ValueError(
            f"reduction {operation.id!r} declares no cardinality and operator"
        )
    regions = {
        region.id: region
        for region in compiled.analysis.regions
    }
    source = regions.get(operation.source)
    return KernelTransportReduction(
        operation=operation.id,
        operator=operation.operator,
        cardinality=operation.cardinality,
        # An indexed view carries the count that bounds the live contributions;
        # a plain region leaves the declared cardinality as the only bound.
        cardinality_source=(
            source.indices
            if source is not None and source.is_indexed
            else None
        ),
    )


def _transform_contracts(
    compiled: CompiledPattern,
    operations,
) -> tuple[KernelTransportTransform, ...]:
    if not operations:
        return ()
    grouped: dict[tuple[str, str], list] = {}
    order: list[tuple[str, str]] = []
    for operation in operations:
        name = getattr(operation, "transform", "") or ""
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if stem not in _KNOWN_TRANSFORM_STEMS:
            raise NotImplementedError(
                f"physicalization does not know transform "
                f"{name!r} on {operation.id!r}; known transform "
                "stems are " + ", ".join(_KNOWN_TRANSFORM_STEMS)
            )
        operation_stem = operation.id.rsplit(".", 1)[0]
        key = (operation.source, operation_stem)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((operation, stem))
    contracts = []
    for source, operation_stem in order:
        members = grouped[(source, operation_stem)]
        stems = {stem for _, stem in members}
        if len(stems) != 1:
            raise ValueError(
                f"transforms fused under {operation_stem!r} name "
                "different transform stems: "
                + ", ".join(sorted(stems))
            )
        source_region = _resolve_region(compiled, source)
        contracts.append(KernelTransportTransform(
            operation=(
                operation_stem
                if len(members) > 1
                else members[0][0].id
            ),
            transform=next(iter(stems)),
            source=source,
            source_type=source_region.element_type,
            destinations=tuple(
                member.destination for member, _ in members
            ),
            destination_types=tuple(
                _resolve_region(
                    compiled, member.destination
                ).element_type
                for member, _ in members
            ),
        ))
    return tuple(contracts)


def _opaque_stage_names(
    compiled: CompiledPattern,
) -> tuple[str, ...]:
    return tuple(sorted(
        stage.name
        for stage in compiled.stages
        if isinstance(stage, OpaqueStage)
    ))


def _validate_boundaries(
    compiled: CompiledPattern,
    attachment: PhaseBinding,
) -> None:
    known = _opaque_stage_names(compiled)
    for role, name in (
        ("producer", attachment.producer),
        ("consumer", attachment.consumer),
    ):
        if name is None or name in known:
            continue
        raise ValueError(
            f"communication attachment {attachment.name!r} names unknown "
            f"{role} stage {name!r}; the pattern declares "
            + (", ".join(known) or "no opaque stages")
        )


def _validate_kernels(
    compiled: CompiledPattern,
    attachments: tuple[PhaseBinding, ...],
) -> None:
    """Reject two programs at one kernel's insertion point.

    A Pattern declares no client kernel names, so the attachment's `kernel`
    has no registry to be checked against; what is checkable is the
    collision, since `transport_program_for_kernel` demands exactly one
    program per kernel.
    """
    occupied = {
        (binding.kernel, binding.insertion_point): (
            "an already attached program"
        )
        for binding in compiled.kernel_bindings
    }
    for attachment in attachments:
        key = (attachment.kernel, attachment.insertion_point)
        previous = occupied.get(key)
        if previous is not None:
            raise ValueError(
                f"communication attachment {attachment.name!r} binds kernel "
                f"{attachment.kernel!r} at insertion point "
                f"{attachment.insertion_point!r}, which {previous} occupies"
            )
        occupied[key] = f"attachment {attachment.name!r}"


def _stage_declares(
    compiled: CompiledPattern,
    stage: str,
    region: str,
) -> bool:
    """Whether `stage` names the region behind `region` in its effects.

    A declared effect that resolves to no physical region is a failure of the
    question, not an answer to it: swallowing it would permit the artifact the
    stage was never asked about.
    """
    target = _resolve_region(compiled, region).id
    declared = compiled.opaque_stage(stage)
    resolved = []
    for value in declared.reads + declared.writes:
        try:
            resolved.append(_resolve_region(compiled, value).id)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"cannot check stage {stage!r} for {region!r}: the stage "
                f"declares {value!r}, which resolves to no physical region "
                f"({error})"
            ) from error
    return target in resolved


def _validate_artifact(
    compiled: CompiledPattern,
    attachment: PhaseBinding,
    operations: tuple,
    neighbours: tuple[str, ...],
) -> None:
    """Check the artifact kind against the neighbours' permissions.

    `neighbours` are the opaque stages that physically border the matched
    communication stage, not the ones the attachment happened to name: an
    attachment may leave a boundary out to match on the other one, and a stage
    left out still runs beside the artifact and still has to permit it.
    """
    if attachment.artifact is Artifact.INLINE_FRAGMENT:
        # An inline fragment is emitted into a kernel the client already
        # compiles through Aboleth, so it needs no permission from the opaque
        # stages beside it.
        return
    kind = attachment.artifact.value
    for stage in neighbours:
        capabilities = compiled.capabilities(stage)
        # A detached artifact addresses its payloads as plain global device
        # pointers, so every stage it sits between must accept that space.
        if "global" not in capabilities.accepted_address_spaces:
            raise ValueError(
                f"attachment {attachment.name!r} cannot be a {kind}: stage "
                f"{stage!r} accepts address spaces "
                + ", ".join(capabilities.accepted_address_spaces)
                + ", not global"
            )
        # A standalone kernel is launched on the client's own stream, so a
        # neighbour that publishes completion only through an external event
        # cannot order it.
        if (
            attachment.artifact is Artifact.STANDALONE_KERNEL and
            capabilities.completion_contract != "current_stream"
        ):
            raise ValueError(
                f"attachment {attachment.name!r} cannot be a {kind}: stage "
                f"{stage!r} publishes completion as "
                f"{capabilities.completion_contract!r}, and a standalone "
                "kernel needs 'current_stream'"
            )
    retargeted = compiled.analysis.retargeted_destinations
    for operation in operations:
        if operation.id not in retargeted:
            continue
        _, destination = retargeted[operation.id]
        for stage in neighbours:
            if not _stage_declares(compiled, stage, destination):
                continue
            capabilities = compiled.capabilities(stage)
            # The plan retargeted this operation straight into storage
            # the stage declares, rewriting a destination it owns.
            if not capabilities.accepts_destination_rewrite:
                raise ValueError(
                    f"attachment {attachment.name!r} cannot be a {kind}: "
                    f"{operation.id!r} was retargeted into {destination!r}, "
                    f"which stage {stage!r} declares without "
                    "accepts_destination_rewrite"
                )
            # A per-item retarget lands each item through the workset's index
            # map, which is an indexed scatter into that same storage.
            if (
                operation.kind in ("transfer_each", "copy_each") and
                not capabilities.accepts_indexed_scatter
            ):
                raise ValueError(
                    f"attachment {attachment.name!r} cannot be a {kind}: "
                    f"{operation.id!r} scatters items into {destination!r} "
                    f"through a workset index map, and stage {stage!r} "
                    "declares no accepts_indexed_scatter"
                )


def _physicalize_attachment(
    compiled: CompiledPattern,
    attachment: PhaseBinding,
) -> TransportProgramBinding:
    _validate_boundaries(compiled, attachment)
    matches = []
    boundaries = []
    for index, stage in enumerate(compiled.stages):
        if not isinstance(stage, CommunicationStage):
            continue
        previous = (
            compiled.stages[index - 1]
            if index > 0
            else None
        )
        following = (
            compiled.stages[index + 1]
            if index + 1 < len(compiled.stages)
            else None
        )
        boundaries.append("{} -> {} -> {}".format(
            previous.name
            if isinstance(previous, OpaqueStage)
            else "-",
            stage.name,
            following.name
            if isinstance(following, OpaqueStage)
            else "-",
        ))
        if (
            attachment.producer is not None
            and (
                not isinstance(previous, OpaqueStage)
                or previous.name != attachment.producer
            )
        ):
            continue
        if (
            attachment.consumer is not None
            and (
                not isinstance(following, OpaqueStage)
                or following.name != attachment.consumer
            )
        ):
            continue
        matches.append((stage, tuple(
            neighbour.name
            for neighbour in (previous, following)
            if isinstance(neighbour, OpaqueStage)
        )))
    if len(matches) != 1:
        raise ValueError(
            f"communication attachment {attachment.name!r} matched "
            f"{len(matches)} stages between producer "
            f"{attachment.producer!r} and consumer {attachment.consumer!r}; "
            "the pattern's communication boundaries are "
            + (" | ".join(boundaries) or "none")
        )
    stage, neighbours = matches[0]
    transport_operations = tuple(
        operation
        for operation in stage.operations
        if operation.kind in _PHYSICALIZED_KINDS
        and operation.kind != "transform"
    )
    transform_operations = tuple(
        operation
        for operation in stage.operations
        if operation.kind == "transform"
    )
    unsupported = tuple(
        f"{operation.id} ({operation.kind})"
        for operation in stage.operations
        if operation.kind not in _PHYSICALIZED_KINDS
    )
    if unsupported:
        raise NotImplementedError(
            "communication physicalization supports "
            + ", ".join(_PHYSICALIZED_KINDS)
            + " and does not support: "
            + ", ".join(unsupported)
        )
    if not transport_operations:
        raise ValueError(
            f"attachment {attachment.name!r} has no transport operations"
        )
    _validate_artifact(
        compiled,
        attachment,
        transport_operations,
        neighbours,
    )
    _validate_slot_stems(attachment, transport_operations)
    transfer_groups = tuple(
        _transfer_regions(
            compiled,
            operation,
            attachment,
        )
        for operation in transport_operations
    )
    longest = max(map(len, transfer_groups))
    partitioned = tuple(
        group
        for group in transfer_groups
        if len(group) == longest
    )
    remainder = tuple(
        group
        for group in transfer_groups
        if len(group) != longest
    )
    regions = tuple(
        group[index]
        for index in range(longest)
        for group in partitioned
    ) + tuple(
        region
        for group in remainder
        for region in group
    )
    publication = _slot_name(attachment.publication)
    return TransportProgramBinding(
        name=attachment.name,
        operations=tuple(
            operation.id
            for operation in transport_operations
        ),
        kernel=attachment.kernel,
        insertion_point=attachment.insertion_point,
        endpoint=BoundScalar("endpoint", "i64"),
        regions=regions,
        publication=PublicationBinding(
            remote=_side(
                f"{publication}_destination",
                f"{publication}_destination_key",
            ),
            local=_side(
                f"{publication}_source",
                f"{publication}_source_key",
            ),
            iteration=BoundScalar("iteration", "i32"),
            iterations=BoundScalar("iterations", "i32"),
        ),
        artifact=attachment.artifact.value,
        reductions=tuple(
            _reduce_contract(compiled, operation)
            for operation in transport_operations
            if operation.kind == "reduction"
        ),
        transforms=_transform_contracts(
            compiled,
            transform_operations,
        ),
    )


def physicalize_pattern(
    compiled: CompiledPattern,
    attachments: tuple[PhaseBinding, ...],
) -> PhysicalizedPattern:
    _validate_kernels(compiled, attachments)
    bindings = tuple(
        _physicalize_attachment(compiled, attachment)
        for attachment in attachments
    )
    attached = compiled.with_transport_programs(tuple(
        binding.to_program()
        for binding in bindings
    ))
    return PhysicalizedPattern(
        compiled=attached,
        bindings=bindings,
    )
