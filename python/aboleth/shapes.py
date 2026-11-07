"""Compiler-generated shape variants; clients supply uniform bound/slot facts."""

from bisect import bisect_left
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

from .compiler import enabled_optimizations
from triton._C.libtriton import ir, passes


@dataclass(frozen=True)
class ShapeVariant:
    capacity: int
    control_base: int
    control_count: int

    def control_domain(self, slot: int) -> int:
        if type(slot) is not int or not 0 <= slot < self.control_count:
            raise ValueError("slot is outside its declared in-flight domain")
        return self.control_base + slot


@dataclass(frozen=True)
class ShapeFamilyPlan:
    name: str
    maximum_bound: int
    slots: int
    uniform_required: bool
    variants: tuple[ShapeVariant, ...]
    private_control: str
    mutually_exclusive_within_slot: bool
    payload_aliases_proven: bool
    selector: str
    iteration_bound: int | None = None

    def select(self, uniform_bound: int) -> ShapeVariant:
        """Interpret the compiler's threshold table, without a capacity heuristic.

        Multi-variant plans require the declared bound to be rank-uniform.
        The caller owns obtaining that bound and executing its CUDA launches.
        Storage sharing still requires the compiled last-reader proof.
        """
        if (
            type(uniform_bound) is not int
            or not 0 <= uniform_bound <= self.maximum_bound
        ):
            raise ValueError(
                "runtime bound is outside the declared shape family"
            )
        return self.variants[
            bisect_left(
                tuple(v.capacity for v in self.variants), uniform_bound
            )
        ]


def plan_shape_family(
    *,
    name: str,
    minimum_bound: int,
    maximum_bound: int,
    slots: int,
    uniform_selector: bool,
    iteration_bound: int | None = None,
) -> ShapeFamilyPlan:
    """State iterative and overall bounds; C++ chooses variants and fallback.

    iteration_bound describes the repeated-use domain, not every invocation.
    Larger initialization or mixed-phase inputs use the overall fallback.
    Omitting it preserves specialization over the complete bounded domain.
    """
    values = (minimum_bound, maximum_bound, slots)
    if iteration_bound is not None:
        values += (iteration_bound,)
    for value in values:
        if type(value) is not int or not -(2**63) <= value < 2**63:
            raise ValueError("shape facts must be signed i64 integers")
    if not isinstance(name, str) or type(uniform_selector) is not bool:
        raise ValueError(
            "shape facts require a name and explicit uniform-selector fact"
        )
    source = (
        "module { aboleth.shape_family {name = "
        + json.dumps(name)
        + f", minimumBound = {minimum_bound} : i64, maximumBound = {maximum_bound} : i64, "
        + f"slots = {slots} : i64, uniformSelector = {str(uniform_selector).lower()}"
        + (
            f", iterationBound = {iteration_bound} : i64"
            if iteration_bound is not None
            else ""
        )
        + "} }"
    )
    with tempfile.TemporaryDirectory(prefix="aboleth-shapes-") as folder:
        directory = Path(folder)
        input_file, output_file = (
            directory / "input.mlir",
            directory / "plan.json",
        )
        input_file.write_text(source)
        context = ir.context()
        ir.load_dialects(context)
        module = ir.parse_mlir_module(str(input_file), context)
        manager = ir.pass_manager(context)
        passes.plugin.add_aboleth_plan_shapes(
            manager,
            [
                str(output_file),
                (
                    ""
                    if "shape-specialization" in enabled_optimizations()
                    else "unoptimized"
                ),
            ],
        )
        manager.run(module, "aboleth-plan-shapes")
        result = json.loads(output_file.read_text())["families"][0]
    result["variants"] = tuple(
        ShapeVariant(**variant) for variant in result["variants"]
    )
    return ShapeFamilyPlan(**result)
