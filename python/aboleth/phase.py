"""Compile one semantic communication phase for a client-owned kernel.

This is Aboleth's primary integration surface. The client supplies a Pattern,
one phase binding, and runtime resource facts. Aboleth returns a planned,
typed phase that can be inlined while compiling the client kernel or realized
as a detached artifact. Aboleth never launches the artifact.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

from .artifact import (
    FragmentArtifact,
    StandaloneKernel,
    realize_bindable_epilogue,
    realize_standalone_kernel,
)
from .compiler import DEFAULT_INLINE_BYTES, compilation
from .pattern import (
    CompiledPattern,
    CompiledPatternPlan,
    KernelTransportProgram,
    Pattern,
    PhaseResources,
)
from .physicalization import (
    Artifact,
    PhaseBinding,
    PhysicalizedPattern,
    physicalize_pattern,
)
from .transport_binding import InvokeManifest, invoke_manifest
from .planning import TransportPartitionPlan


@dataclass(frozen=True)
class CompiledPhase:
    """One compiler-authoritative communication phase.

    Static protocol decisions live in ``plan`` and ``program``. Dynamic
    addresses, byte counts, keys, and predicates are named by ``manifest`` and
    remain operands supplied by the client kernel.
    """

    binding: PhaseBinding
    physicalized: PhysicalizedPattern
    plan: CompiledPatternPlan

    @property
    def name(self) -> str:
        return self.binding.name

    @property
    def program(self) -> KernelTransportProgram:
        return self.plan.transport_program_for_kernel(
            self.binding.kernel
        )

    @property
    def manifest(self) -> InvokeManifest:
        return invoke_manifest(self.program)

    @property
    def launches(self) -> tuple[TransportPartitionPlan, ...]:
        """Compile each listed specialization; do not regroup its peers."""
        return self.plan.transport_partitions

    @property
    def retirement_manifest(self) -> InvokeManifest | None:
        """Drain this artifact's endpoints at slot retirement and teardown."""
        program = self.plan.retirement_programs.get(self.binding.kernel)
        return invoke_manifest(program) if program is not None else None

    def compilation_launch(self, launch: TransportPartitionPlan):
        if launch not in self.launches:
            raise ValueError("launch is not a partition of this compiled artifact")
        return self.compilation(launch.backend, peers=launch.peers)

    @property
    def completion_interval(self) -> int:
        interval = self.plan.completion_interval_for(
            self.binding.kernel
        )
        if interval is None:
            raise RuntimeError(
                f"phase {self.name!r} has no completion interval for "
                f"kernel {self.binding.kernel!r}"
            )
        return interval

    def compilation(
        self,
        backend: str,
        *,
        peers: Iterable[int] | None = None,
        inline_bytes: int = DEFAULT_INLINE_BYTES,
    ):
        """Open the compile-time context for an inline client kernel."""
        if self.binding.artifact is not Artifact.INLINE_FRAGMENT:
            raise ValueError(
                f"phase {self.name!r} is a "
                f"{self.binding.artifact.value!r} artifact; only "
                "Artifact.INLINE_FRAGMENT compiles inside a client kernel"
            )
        self._validate_transport_scope(backend, peers)
        schedule = None
        try:
            schedule = self.plan.phase_schedule(self.name)
        except ValueError:
            pass
        return compilation(
            backend,
            pattern_plan=self.plan,
            kernel=self.binding.kernel,
            inline_bytes=inline_bytes,
            p2p_coop_elements=(
                schedule.p2p_coop_elements if schedule else None
            ),
        )

    def realize(
        self,
        *,
        arch: str,
        device_bitcode: str | os.PathLike,
        output_dir: str | os.PathLike,
        transport: str = "rdma",
        peers: Iterable[int] | None = None,
        name: str | None = None,
            max_registers: int | None = None,
        cuda_inline: bool = False,
    ) -> FragmentArtifact | StandaloneKernel:
        """Compile a detached artifact; the client remains its executor."""
        self._validate_transport_scope(transport, peers)
        arguments = {
            "arch": arch,
            "device_bitcode": device_bitcode,
            "output_dir": output_dir,
            "completion_interval": self.completion_interval,
            "transport": transport,
            "name": name,
        }
        if self.binding.artifact is Artifact.BINDABLE_EPILOGUE:
            return realize_bindable_epilogue(
                self.program,
                max_registers=max_registers,
                cuda_inline=cuda_inline,
                **arguments,
            )
        if max_registers is not None or cuda_inline:
            raise ValueError(
                "max_registers and cuda_inline apply only to bindable "
                "epilogues"
            )
        if self.binding.artifact is Artifact.STANDALONE_KERNEL:
            return realize_standalone_kernel(
                self.program,
                **arguments,
            )
        raise ValueError(
            f"phase {self.name!r} is inline; compile its client kernel "
            "inside phase.compilation(...) instead of realizing a "
            "detached artifact"
        )

    def _validate_transport_scope(
        self,
        backend: str,
        peers: Iterable[int] | None,
    ) -> None:
        """Reject lowering a mixed dynamic peer set through one backend."""
        geography = dict(self.plan.transport_geography)
        if not geography:
            return
        if peers is None:
            selected = set(geography.values())
            if selected == {backend}:
                return
            raise ValueError(
                "mixed transport geography requires a backend-homogeneous "
                "peer partition for phase compilation"
            )
        selected_peers = tuple(dict.fromkeys(int(peer) for peer in peers))
        if not selected_peers:
            raise ValueError("phase compilation peer partition cannot be empty")
        missing = set(selected_peers) - set(geography)
        if missing:
            raise ValueError(
                "phase compilation names peers absent from its geography: "
                + ", ".join(str(peer) for peer in sorted(missing))
            )
        wrong = {
            peer: geography[peer]
            for peer in selected_peers
            if geography[peer] != backend
        }
        if wrong:
            detail = ", ".join(
                f"{peer}:{transport}" for peer, transport in wrong.items()
            )
            raise ValueError(
                f"{backend} compilation partition contains {detail}"
            )

    def describe(self) -> dict:
        """Stable, integration-facing explanation of the generated phase."""
        schedule = self.plan.phase_schedule(self.name)
        return {
            "name": self.name,
            "artifact": self.binding.artifact.value,
            "kernel": self.binding.kernel,
            "insertion_point": self.binding.insertion_point,
            "command_count": self.program.command_count,
            "payload_bytes": self.program.payload_bytes,
            "completion_interval": self.completion_interval,
            "transport_geography": [
                {"peer": peer, "transport": transport}
                for peer, transport in self.plan.transport_geography
            ],
            "strategy": schedule.strategy,
            "scheduled_kernel": schedule.kernel,
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
            "block_elements": schedule.block_elements,
            "num_warps": schedule.num_warps,
            "buffer_depth": schedule.buffer_depth,
            "rationale": schedule.rationale,
            "slots": list(self.manifest.slot_names),
        }


def compile_phase(
    pattern: Pattern | CompiledPattern,
    binding: PhaseBinding,
    resources: PhaseResources,
    *,
    cyclic: bool = False,
) -> CompiledPhase:
    """Physicalize and plan one communication phase.

    ``pattern`` describes semantics, ``binding`` describes the client hook, and
    ``resources`` describes stable runtime capabilities. No pointer binding,
    launch, stream, graph capture, or application execution occurs here.
    """
    if isinstance(pattern, Pattern):
        compiled = pattern.compile()
    elif isinstance(pattern, CompiledPattern):
        compiled = pattern
    else:
        raise TypeError(
            "compile_phase pattern must be a Pattern or CompiledPattern"
        )
    if not isinstance(binding, PhaseBinding):
        raise TypeError("compile_phase binding must be a PhaseBinding")
    if not isinstance(resources, PhaseResources):
        raise TypeError("compile_phase resources must be PhaseResources")

    physicalized = physicalize_pattern(compiled, (binding,))
    plan = physicalized.compiled.compile_plan(
        resources,
        cyclic=cyclic,
    )
    # Resolve eagerly so a miss is reported at integration time rather than
    # during a later Triton compilation.
    physicalized.program(binding.name)
    return CompiledPhase(
        binding=binding,
        physicalized=physicalized,
        plan=plan,
    )
