"""Phase intervals survive fragment import without becoming a kernel minimum."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import aboleth  # Configure the plugin before loading Triton/LLVM.
from triton._C.libtriton import ir, passes
from aboleth import compile_phase
from aboleth.planning import (
    RegionEffect,
    TransportPhase,
    compile_transport_program_mlir,
)
from test_phase import communication_pattern, phase_binding, resources


def imported_waits(*, unplanned=False, actual_capacity=129):
    pattern = communication_pattern()
    phase = compile_phase(pattern, phase_binding(), resources(pattern))
    fragments = [
        replace(
            phase.program, program_id=i, name=f"fragment{i}", kernel="fused"
        )
        for i in range(2)
    ]
    operations = tuple(
        TransportPhase(
            name=f"phase{i}",
            program_id=i,
            bound_kernel="fused",
            commands_per_iteration=2,
            iteration_count=16,
            payload_bytes=65536,
            peer_transports=("rdma",),
            peer_capacities=(129,),
            reads=(RegionEffect(2 * i, 1, 0, 64, 64, 16, period=64 * reuse),),
            writes=(RegionEffect(2 * i + 1, 1, 0, 64, 64, 16),),
        )
        for i, reuse in enumerate([8, 2])
    )
    plan = compile_transport_program_mlir(
        operations,
        fragments="\n".join(program.to_mlir() for program in fragments),
    )
    arguments = phase.program.arguments
    types = ", ".join(a.mlir_type for a in arguments)
    signature = ", ".join(
        f"%a{i}: {a.mlir_type}" for i, a in enumerate(arguments)
    )
    operands = ", ".join(f"%a{i}" for i in range(len(arguments)))
    calls = "\n".join(
        f"%id{i} = arith.constant {i} : i32\n"
        f'"aboleth.invoke_transport_phase"(%id{i}, {operands}) : (i32, {types}) -> ()'
        for i in range(2)
    )
    source = f"module {{ tt.func public @fused({signature}) {{ {calls}\n tt.return }} }}"
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        (directory / "plan.mlir").write_text(plan)
        (directory / "kernel.mlir").write_text(source)
        context = ir.context()
        ir.load_dialects(context)
        module = ir.parse_mlir_module(str(directory / "kernel.mlir"), context)
        manager = ir.pass_manager(context)
        passes.plugin.add_aboleth_materialize_transport_phase(
            manager, [str(directory / "plan.mlir")]
        )
        manager.run(module, "phase-import")
        imported = module.str_nodebug()
        manager = ir.pass_manager(context)
        passes.plugin.add_aboleth_materialize_completion(
            manager,
            [
                "1",
                "rdma",
                str(actual_capacity),
                "unplanned" if unplanned else "",
            ],
        )
        manager.run(module, "phase-completion")
        return imported, module.str_nodebug()


class PhaseCompletionTest(unittest.TestCase):
    def test_two_phases_keep_distinct_intervals_inside_one_kernel(self):
        imported, lowered = imported_waits()
        self.assertIn("completionInterval = 8 : i64", imported)
        self.assertIn("completionInterval = 2 : i64", imported)
        self.assertEqual(lowered.count("arith.remui"), 2)
        self.assertIn("arith.constant 8 : i32", lowered)
        self.assertIn("arith.constant 2 : i32", lowered)

    def test_completion_switch_disables_phase_intervals_too(self):
        _, lowered = imported_waits(unplanned=True)
        self.assertNotIn("arith.remui", lowered)
        self.assertEqual(lowered.count("aboleth.wait"), 2)

    def test_shared_endpoint_obeys_the_actual_aggregate_queue_bound(self):
        _, lowered = imported_waits(actual_capacity=7)
        # Two two-command chains share six usable WQEBBs: no deferral fits.
        self.assertNotIn("arith.remui", lowered)
        self.assertEqual(lowered.count("aboleth.wait"), 2)

    def test_pattern_attachment_identity_reaches_the_phase(self):
        pattern = communication_pattern()
        phase = compile_phase(pattern, phase_binding(), resources(pattern))
        self.assertEqual(
            phase.plan.transport_operations[0].program_id,
            phase.program.program_id,
        )

    def test_deferred_queue_phase_supplies_its_retirement_fragment(self):
        pattern = communication_pattern()
        phase = compile_phase(pattern, phase_binding(), resources(pattern))
        manifest = phase.retirement_manifest
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.slot_names, ("endpoint",))
        self.assertIsNone(manifest.signal)
        self.assertIn("aboleth.drain", phase.plan.transport_mlir)
        self.assertNotEqual(manifest.program_id, phase.manifest.program_id)
        base = resources(pattern)
        eager = compile_phase(
            pattern,
            phase_binding(),
            replace(
                base,
                iteration_count=1,
                regions={
                    name: replace(effect, count=1)
                    for name, effect in base.regions.items()
                },
            ),
        )
        self.assertIsNone(eager.retirement_manifest)


if __name__ == "__main__":
    unittest.main()
