from __future__ import annotations

from dataclasses import dataclass

from .pattern import (
    ARTIFACT_KINDS,
    KernelTransportArgument,
    KernelTransportProgram,
    KernelTransportReduction,
    KernelTransportSignal,
    KernelTransportTransfer,
    KernelTransportTransform,
)


@dataclass(frozen=True)
class BoundScalar:
    name: str
    mlir_type: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("bound scalar requires a name")
        if self.mlir_type not in ("i1", "i32", "i64"):
            raise ValueError(
                "bound scalar type must be i1, i32, or i64"
            )


@dataclass(frozen=True)
class RegionSide:
    pointer: BoundScalar
    key: BoundScalar

    def __post_init__(self):
        if self.pointer.mlir_type != "i64":
            raise ValueError("region pointers must use i64 slots")
        if self.key.mlir_type != "i32":
            raise ValueError("region keys must use i32 slots")


@dataclass(frozen=True)
class RegionBinding:
    remote: RegionSide
    local: RegionSide
    byte_count: BoundScalar
    predicate: BoundScalar
    max_bytes: int
    operations: tuple[str, ...] = ()

    def __post_init__(self):
        if self.byte_count.mlir_type != "i32":
            raise ValueError("region byte count must use an i32 slot")
        if self.predicate.mlir_type != "i1":
            raise ValueError("region predicate must use an i1 slot")
        if self.max_bytes <= 0:
            raise ValueError("region max bytes must be positive")

    def slots(self) -> tuple[BoundScalar, ...]:
        return (
            self.remote.pointer,
            self.local.pointer,
            self.byte_count,
            self.local.key,
            self.remote.key,
            self.predicate,
        )


# What each slot the two bindings export is for, in the order they export
# them; `TransportProgramBinding.slots` names these when two collide.
_REGION_SLOT_ROLES = (
    "remote pointer",
    "local pointer",
    "byte count",
    "local key",
    "remote key",
    "predicate",
)


@dataclass(frozen=True)
class PublicationBinding:
    remote: RegionSide
    local: RegionSide
    iteration: BoundScalar
    iterations: BoundScalar

    def __post_init__(self):
        if (
            self.iteration.mlir_type != "i32"
            or self.iterations.mlir_type != "i32"
        ):
            raise ValueError(
                "publication iteration slots must use i32"
            )

    def slots(self) -> tuple[BoundScalar, ...]:
        return (
            self.remote.pointer,
            self.local.pointer,
            self.local.key,
            self.remote.key,
            self.iteration,
            self.iterations,
        )


_PUBLICATION_SLOT_ROLES = (
    "remote pointer",
    "local pointer",
    "local key",
    "remote key",
    "iteration",
    "iterations",
)


@dataclass(frozen=True)
class TransportProgramBinding:
    name: str
    operations: tuple[str, ...]
    kernel: str
    insertion_point: str
    endpoint: BoundScalar
    regions: tuple[RegionBinding, ...]
    publication: PublicationBinding
    artifact: str = "inline_fragment"
    reductions: tuple[KernelTransportReduction, ...] = ()
    transforms: tuple[KernelTransportTransform, ...] = ()

    def __post_init__(self):
        if not self.name or not self.kernel or not self.insertion_point:
            raise ValueError(
                "transport binding requires name, kernel, and insertion point"
            )
        if self.artifact not in ARTIFACT_KINDS:
            raise ValueError(
                f"unknown artifact kind {self.artifact!r}; expected one of "
                + ", ".join(ARTIFACT_KINDS)
            )
        if self.endpoint.mlir_type != "i64":
            raise ValueError("transport endpoint must use an i64 slot")
        if not self.operations or not self.regions:
            raise ValueError(
                "transport binding requires operations and regions"
            )
        realized = tuple(
            operation
            for region in self.regions
            for operation in region.operations
        )
        if (
            len(realized) != len(set(realized))
            or set(realized) != set(self.operations)
        ):
            raise ValueError(
                "transport binding must realize each logical operation once"
            )
    def slots(self) -> tuple[BoundScalar, ...]:
        """The bound scalars this program exports, in invocation order.

        One name is one role. Two roles that arrive under one name are a
        collision, not a shared slot: the program would issue the second role's
        command through the first role's storage, so the name is refused rather
        than folded away.
        """
        candidates = (
            ("endpoint", self.endpoint),
            *(
                (f"region {index} {role}", slot)
                for index, region in enumerate(self.regions)
                for role, slot in zip(
                    _REGION_SLOT_ROLES,
                    region.slots(),
                )
            ),
            *(
                (f"publication {role}", slot)
                for role, slot in zip(
                    _PUBLICATION_SLOT_ROLES,
                    self.publication.slots(),
                )
            ),
        )
        result = []
        roles: dict[str, tuple[str, str]] = {}
        for role, slot in candidates:
            previous = roles.get(slot.name)
            if previous is None:
                roles[slot.name] = (role, slot.mlir_type)
                result.append(slot)
                continue
            if previous == (role, slot.mlir_type):
                continue
            raise ValueError(
                f"transport binding {self.name!r} binds slot "
                f"{slot.name!r} to two roles: the {previous[0]} "
                f"({previous[1]}) and the {role} ({slot.mlir_type})"
            )
        return tuple(result)

    def to_program(self) -> KernelTransportProgram:
        return KernelTransportProgram(
            name=self.name,
            operations=self.operations,
            kernel=self.kernel,
            insertion_point=self.insertion_point,
            arguments=tuple(
                KernelTransportArgument(
                    slot.name,
                    slot.mlir_type,
                )
                for slot in self.slots()
            ),
            transfers=tuple(
                KernelTransportTransfer(
                    operations=region.operations,
                    destination=region.remote.pointer.name,
                    source=region.local.pointer.name,
                    byte_count=region.byte_count.name,
                    source_key=region.local.key.name,
                    destination_key=region.remote.key.name,
                    predicate=region.predicate.name,
                    max_bytes=region.max_bytes,
                )
                for region in self.regions
            ),
            signal=KernelTransportSignal(
                destination=self.publication.remote.pointer.name,
                source=self.publication.local.pointer.name,
                source_key=self.publication.local.key.name,
                destination_key=self.publication.remote.key.name,
                iteration=self.publication.iteration.name,
                iterations=self.publication.iterations.name,
            ),
            artifact=self.artifact,
            reductions=self.reductions,
            transforms=self.transforms,
        )


@dataclass(frozen=True)
class InvokeManifest:
    program_id: int
    arguments: tuple[KernelTransportArgument, ...]
    transfers: tuple[KernelTransportTransfer, ...]
    signal: KernelTransportSignal | None

    @property
    def slot_names(self) -> tuple[str, ...]:
        return tuple(argument.name for argument in self.arguments)


def invoke_manifest(
    program: KernelTransportProgram,
) -> InvokeManifest:
    if program.program_id is None:
        raise ValueError("transport program is not bound")
    return InvokeManifest(
        program_id=program.program_id,
        arguments=program.arguments,
        transfers=program.transfers,
        signal=program.signal,
    )
