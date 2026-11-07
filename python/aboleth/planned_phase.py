"""Compiler inputs for externally attached planned transport fragments."""

from __future__ import annotations

import os

from .artifact import FragmentArtifact, compile_fragment_artifact
from .pattern import (
    KernelTransportArgument,
    KernelTransportProgram,
    KernelTransportTransfer,
)
from .physicalization import Artifact

PLANNED_FRAGMENT_MAX_REGISTERS = 64


def planned_put_program(
    *,
    name: str = "phase.put",
    max_bytes: int,
    cta_exclusive: bool,
    artifact: Artifact = Artifact.BINDABLE_EPILOGUE,
) -> KernelTransportProgram:
    if max_bytes <= 0:
        raise ValueError("planned put max_bytes must be positive")
    return KernelTransportProgram(
        name=name,
        operations=("payload",),
        kernel="client_transport",
        insertion_point="payload_put",
        arguments=tuple(
            KernelTransportArgument(slot, kind)
            for slot, kind in (
                ("endpoint", "i64"),
                ("destination", "i64"),
                ("source", "i64"),
                ("bytes", "i32"),
                ("source_key", "i32"),
                ("destination_key", "i32"),
                ("predicate", "i1"),
            )
        ),
        transfers=(
            KernelTransportTransfer(
                operations=("payload",),
                destination="destination",
                source="source",
                byte_count="bytes",
                source_key="source_key",
                destination_key="destination_key",
                predicate="predicate",
                max_bytes=max_bytes,
            ),
        ),
        signal=None,
        program_id=0,
        artifact=artifact.value,
        external_publication=True,
        planned_reservation=True,
        reservation_scope=("cta" if cta_exclusive else "gpu"),
        issuer_scope="thread",
    )


def planned_publish_program(
    *,
    name: str = "phase.publish",
    artifact: Artifact = Artifact.BINDABLE_EPILOGUE,
) -> KernelTransportProgram:
    return KernelTransportProgram(
        name=name,
        operations=(),
        kernel="client_transport",
        insertion_point="phase_publish",
        arguments=(
            KernelTransportArgument("endpoint", "i64"),
            KernelTransportArgument("only_if_advanced", "i1"),
        ),
        transfers=(),
        signal=None,
        program_id=0,
        artifact=artifact.value,
        publish_head=True,
        issuer_scope="thread",
    )


def planned_fence_program(
    *,
    name: str = "phase.fence",
    artifact: Artifact = Artifact.BINDABLE_EPILOGUE,
) -> KernelTransportProgram:
    return KernelTransportProgram(
        name=name,
        operations=(),
        kernel="client_transport",
        insertion_point="phase_fence",
        arguments=(KernelTransportArgument("predicate", "i1"),),
        transfers=(),
        signal=None,
        program_id=0,
        artifact=artifact.value,
        system_fence=True,
        issuer_scope="thread",
    )


def planned_drain_program(
    *,
    name: str = "phase.drain",
    artifact: Artifact = Artifact.BINDABLE_EPILOGUE,
) -> KernelTransportProgram:
    return KernelTransportProgram(
        name=name,
        operations=(),
        kernel="client_transport",
        insertion_point="phase_reclaim",
        arguments=(KernelTransportArgument("endpoint", "i64"),),
        transfers=(),
        signal=None,
        program_id=0,
        artifact=artifact.value,
        drain_endpoint=True,
        issuer_scope="thread",
    )


def compile_planned_put_artifact(
    *,
    arch: str,
    max_bytes: int,
    cta_exclusive: bool,
    device_bitcode: str | os.PathLike,
    output_dir: str | os.PathLike,
) -> FragmentArtifact:
    """Compile the put-only Queue IR slice into an NVCC-linkable artifact."""
    return compile_fragment_artifact(
        planned_put_program(
            max_bytes=max_bytes,
            cta_exclusive=cta_exclusive,
        ),
        arch=arch,
        completion_interval=1,
        device_bitcode=device_bitcode,
        output_dir=output_dir,
        max_registers=PLANNED_FRAGMENT_MAX_REGISTERS,
        cuda_inline=True,
    )


def compile_planned_publish_artifact(
    *,
    arch: str,
    device_bitcode: str | os.PathLike,
    output_dir: str | os.PathLike,
) -> FragmentArtifact:
    return compile_fragment_artifact(
        planned_publish_program(),
        arch=arch,
        completion_interval=1,
        device_bitcode=device_bitcode,
        output_dir=output_dir,
        max_registers=PLANNED_FRAGMENT_MAX_REGISTERS,
        cuda_inline=True,
    )


def compile_planned_fence_artifact(
    *,
    arch: str,
    device_bitcode: str | os.PathLike,
    output_dir: str | os.PathLike,
) -> FragmentArtifact:
    return compile_fragment_artifact(
        planned_fence_program(),
        arch=arch,
        completion_interval=1,
        device_bitcode=device_bitcode,
        output_dir=output_dir,
        max_registers=PLANNED_FRAGMENT_MAX_REGISTERS,
        cuda_inline=True,
    )


def compile_planned_drain_artifact(
    *,
    arch: str,
    device_bitcode: str | os.PathLike,
    output_dir: str | os.PathLike,
) -> FragmentArtifact:
    return compile_fragment_artifact(
        planned_drain_program(),
        arch=arch,
        completion_interval=1,
        device_bitcode=device_bitcode,
        output_dir=output_dir,
        max_registers=PLANNED_FRAGMENT_MAX_REGISTERS,
        cuda_inline=True,
    )
