"""Typed execution contracts shared by communication clients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .pattern import CompiledPattern, Pattern, PhaseResources
from .phase import CompiledPhase, compile_phase
from .physicalization import PhaseBinding


class PaddingSemantics(Enum):
    UNKNOWN = "unknown"
    PRESERVED = "preserved"
    DEAD = "dead"


class RankControl(Enum):
    UNKNOWN = "unknown"
    DYNAMIC = "dynamic"
    LOCKSTEP = "lockstep"


class ExecutionFallback(Enum):
    EXACT_AGREEMENT = "exact_agreement"
    EAGER = "eager"
    ERROR = "error"


@dataclass(frozen=True)
class CapacityContract:
    max_items: int
    dynamic_valid_count: bool
    padding: PaddingSemantics
    graph_stable_addresses: bool

    def __post_init__(self) -> None:
        if self.max_items <= 0:
            raise ValueError("capacity max_items must be positive")
        if not isinstance(self.padding, PaddingSemantics):
            raise TypeError("capacity padding must be PaddingSemantics")


@dataclass(frozen=True)
class RuntimeGuard:
    name: str
    fallback: ExecutionFallback

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("runtime guard name cannot be empty")
        if not isinstance(self.fallback, ExecutionFallback):
            raise TypeError(
                "runtime guard fallback must be ExecutionFallback"
            )


@dataclass(frozen=True)
class ExecutionContract:
    capacity: CapacityContract
    rank_control: RankControl
    guards: tuple[RuntimeGuard, ...] = ()
    fallback: ExecutionFallback = (
        ExecutionFallback.EXACT_AGREEMENT
    )

    def __post_init__(self) -> None:
        if not isinstance(self.rank_control, RankControl):
            raise TypeError(
                "execution rank_control must be RankControl"
            )
        if not isinstance(self.fallback, ExecutionFallback):
            raise TypeError(
                "execution fallback must be ExecutionFallback"
            )
        guard_names = tuple(guard.name for guard in self.guards)
        if len(guard_names) != len(set(guard_names)):
            raise ValueError("runtime guard names must be unique")

    @property
    def fixed_capacity_legal(self) -> bool:
        return (
            self.capacity.dynamic_valid_count
            and self.capacity.padding is PaddingSemantics.DEAD
            and self.capacity.graph_stable_addresses
            and self.rank_control is RankControl.LOCKSTEP
        )

    def describe(self) -> dict:
        return {
            "capacity": {
                "max_items": self.capacity.max_items,
                "dynamic_valid_count": (
                    self.capacity.dynamic_valid_count
                ),
                "padding": self.capacity.padding.value,
                "graph_stable_addresses": (
                    self.capacity.graph_stable_addresses
                ),
            },
            "rank_control": self.rank_control.value,
            "fixed_capacity_legal": (
                self.fixed_capacity_legal
            ),
            "fallback": self.fallback.value,
            "guards": [
                {
                    "name": guard.name,
                    "fallback": guard.fallback.value,
                }
                for guard in self.guards
            ],
        }

    def resolve(
        self,
        actual_items: int,
        *,
        graph_stable_addresses: bool = True,
        rank_control: RankControl | None = None,
    ) -> "ExecutionDecision":
        if actual_items < 0:
            raise ValueError("actual_items cannot be negative")
        if actual_items > self.capacity.max_items:
            raise RuntimeError(
                f"runtime items {actual_items} exceed compiled "
                f"capacity {self.capacity.max_items}"
            )
        runtime_rank_control = rank_control or self.rank_control
        fixed = (
            self.fixed_capacity_legal
            and graph_stable_addresses
            and runtime_rank_control is RankControl.LOCKSTEP
        )
        if fixed:
            return ExecutionDecision(
                mode="fixed",
                valid_items=actual_items,
                execution_items=self.capacity.max_items,
                requires_rank_agreement=False,
                fallback_used=False,
            )
        if (
            not graph_stable_addresses
            and self.fallback is ExecutionFallback.EAGER
        ):
            return ExecutionDecision(
                mode="eager",
                valid_items=actual_items,
                execution_items=actual_items,
                requires_rank_agreement=True,
                fallback_used=True,
            )
        if (
            not graph_stable_addresses
            and self.fallback is ExecutionFallback.ERROR
        ):
            raise RuntimeError(
                "runtime graph addresses violate the execution contract"
            )
        return ExecutionDecision(
            mode="exact",
            valid_items=actual_items,
            execution_items=actual_items,
            requires_rank_agreement=True,
            fallback_used=self.fixed_capacity_legal,
        )


@dataclass(frozen=True)
class ExecutionDecision:
    mode: str
    valid_items: int
    execution_items: int
    requires_rank_agreement: bool
    fallback_used: bool

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "exact", "eager"):
            raise ValueError("unknown execution decision mode")
        if (
            self.valid_items < 0
            or self.execution_items < self.valid_items
        ):
            raise ValueError(
                "execution decision has invalid item counts"
            )


@dataclass(frozen=True)
class CompiledExecutionPlan:
    phases: tuple[CompiledPhase, ...]
    contract: ExecutionContract

    def __post_init__(self) -> None:
        names = tuple(phase.name for phase in self.phases)
        if not names:
            raise ValueError(
                "compiled execution plan requires at least one phase"
            )
        if len(names) != len(set(names)):
            raise ValueError(
                "compiled execution phase names must be unique"
            )

    def phase(self, name: str) -> CompiledPhase:
        try:
            return next(
                phase
                for phase in self.phases
                if phase.name == name
            )
        except StopIteration as error:
            raise KeyError(
                f"compiled execution has no phase {name!r}"
            ) from error

    def describe(self) -> dict:
        return {
            "contract": self.contract.describe(),
            "phases": [
                phase.describe()
                for phase in self.phases
            ],
        }


def compile_execution_plan(
    pattern: Pattern | CompiledPattern,
    bindings: tuple[PhaseBinding, ...],
    resources: PhaseResources,
    contract: ExecutionContract,
    *,
    cyclic: bool = False,
) -> CompiledExecutionPlan:
    if not bindings:
        raise ValueError(
            "compile_execution_plan requires phase bindings"
        )
    phases = tuple(
        compile_phase(
            pattern,
            binding,
            resources,
            cyclic=cyclic,
        )
        for binding in bindings
    )
    return CompiledExecutionPlan(
        phases=phases,
        contract=contract,
    )
