import json
import pathlib
import tempfile
import unittest

from aboleth.planning import (
    OutputLifetimePlan,
    RegionEffect,
    TransportKernel,
    TransportPhase,
    TransportPlan,
    plan_output_lifetimes,
    plan_transport_program,
)
from aboleth.planning import (
    _run_transport_planner,
    compile_transport_program_mlir,
)
from triton._C.libtriton import ir, passes


def region(
    allocation,
    length,
    stride,
    period,
    count,
):
    return RegionEffect(
        allocation=allocation,
        generation=1,
        offset=0,
        length=length,
        stride=stride,
        period=period,
        count=count,
    )


# The same effect trace as test/compiler/transport-plan.mlir: one phase whose
# source storage a later kernel overwrites, one rotating phase, and one tiny
# phase that needs no early completion.
OPERATIONS = (
    TransportPhase(
        name="independent",
        commands_per_iteration=4,
        iteration_count=100,
        payload_bytes=65536,
        peer_transports=("rdma",),
        peer_capacities=(33,),
        reads=(region(0, 64, 64, 0, 100),),
        writes=(region(1, 64, 64, 0, 100),),
    ),
    TransportKernel(
        name="experts",
        reads=(region(1, 6400, 0, 0, 1),),
        writes=(region(2, 6400, 0, 0, 1),),
    ),
    TransportPhase(
        name="rotating",
        commands_per_iteration=3,
        iteration_count=7,
        payload_bytes=65536,
        peer_transports=("rdma",),
        peer_capacities=(33,),
        reads=(region(3, 64, 64, 192, 7),),
        writes=(region(6, 64, 64, 192, 7),),
    ),
    TransportKernel(
        name="reuse-send-slot-0",
        reads=(),
        writes=(region(0, 6400, 0, 0, 1),),
    ),
    TransportPhase(
        name="tiny",
        commands_per_iteration=1,
        iteration_count=100,
        payload_bytes=1024,
        peer_transports=("rdma",),
        peer_capacities=(129,),
        reads=(region(4, 8, 8, 0, 100),),
        writes=(region(5, 8, 8, 0, 100),),
    ),
)

# Captured from the regex-scraping driver the JSON plan replaced. These are the
# literal results that mechanism produced for OPERATIONS; the JSON extraction
# must reproduce them exactly.
EXPECTED_PLANS = {
    "independent": TransportPlan(
        completion_interval=8,
        reuse_distance=100,
        completion_before="reuse-send-slot-0",
        completion_iteration_delta=0,
        publication_before="experts",
        publication_iteration_delta=0,
    ),
    "rotating": TransportPlan(
        completion_interval=3,
        reuse_distance=3,
        completion_before="program_end",
        completion_iteration_delta=0,
        publication_before="program_end",
        publication_iteration_delta=0,
    ),
    "tiny": TransportPlan(
        completion_interval=100,
        reuse_distance=100,
        completion_before="program_end",
        completion_iteration_delta=0,
        publication_before="program_end",
        publication_iteration_delta=0,
    ),
}

EXPECTED_LIFETIMES = {
    "reuse-send-slot-0": OutputLifetimePlan(
        last_consumer="independent",
        last_consumer_iteration_delta=1,
        reuse_before="reuse-send-slot-0",
        reuse_iteration_delta=1,
        pool_depth=1,
    ),
}

# The JSON document keys are the MLIR attribute names, so this is the printed
# module's planned facts written a second way.
EXPECTED_DOCUMENT = {
    "programs": [
        {
            "name": "transport-program",
            "phases": [
                {
                    "name": "independent",
                    "resolvedCompletionInterval": 8,
                    "reuseDistance": 100,
                    "completionBefore": "reuse-send-slot-0",
                    "completionIterationDelta": 0,
                    "publicationBefore": "experts",
                    "publicationIterationDelta": 0,
                    "outputReusable": True,
                    "outputLastConsumer": "experts",
                    "outputLastConsumerIterationDelta": 0,
                    "outputReuseBefore": "independent",
                    "outputReuseIterationDelta": 1,
                    "outputPoolDepth": 1,
                },
                {
                    "name": "rotating",
                    "resolvedCompletionInterval": 3,
                    "reuseDistance": 3,
                    "completionBefore": "program_end",
                    "completionIterationDelta": 0,
                    "publicationBefore": "program_end",
                    "publicationIterationDelta": 0,
                    "outputReusable": False,
                },
                {
                    "name": "tiny",
                    "resolvedCompletionInterval": 100,
                    "reuseDistance": 100,
                    "completionBefore": "program_end",
                    "completionIterationDelta": 0,
                    "publicationBefore": "program_end",
                    "publicationIterationDelta": 0,
                    "outputReusable": False,
                },
            ],
            "kernels": [
                {
                    "name": "experts",
                    "outputReusable": False,
                },
                {
                    "name": "reuse-send-slot-0",
                    "outputReusable": True,
                    "outputLastConsumer": "independent",
                    "outputLastConsumerIterationDelta": 1,
                    "outputReuseBefore": "reuse-send-slot-0",
                    "outputReuseIterationDelta": 1,
                    "outputPoolDepth": 1,
                },
            ],
        },
    ],
}


def emit_plan_json(source: str, path: str) -> dict:
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
        passes.plugin.add_aboleth_plan_transport(
            manager,
            [path],
        )
        manager.run(module, "aboleth-plan-json")
    return json.loads(pathlib.Path(path).read_text())


class PlanJsonTest(unittest.TestCase):
    def test_transport_plan_matches_captured_result(self):
        self.assertEqual(
            plan_transport_program(OPERATIONS, True),
            EXPECTED_PLANS,
        )
        self.assertEqual(
            plan_transport_program(OPERATIONS),
            EXPECTED_PLANS,
        )

    def test_output_lifetimes_match_captured_result(self):
        self.assertEqual(
            plan_output_lifetimes(OPERATIONS, True),
            EXPECTED_LIFETIMES,
        )
        # Straight-line programs reuse nothing across iterations.
        self.assertEqual(
            plan_output_lifetimes(OPERATIONS),
            {},
        )

    def test_plan_json_carries_the_same_facts(self):
        _, facts = _run_transport_planner(
            OPERATIONS,
            True,
        )
        self.assertEqual(facts, EXPECTED_DOCUMENT)

        phases = {
            phase["name"]: phase
            for phase in facts["programs"][0]["phases"]
        }
        for name, plan in EXPECTED_PLANS.items():
            phase = phases[name]
            self.assertEqual(
                phase["resolvedCompletionInterval"],
                plan.completion_interval,
            )
            self.assertEqual(
                phase["reuseDistance"],
                plan.reuse_distance,
            )
            self.assertEqual(
                phase["completionBefore"],
                plan.completion_before,
            )
            self.assertEqual(
                phase["completionIterationDelta"],
                plan.completion_iteration_delta,
            )
            self.assertEqual(
                phase["publicationBefore"],
                plan.publication_before,
            )
            self.assertEqual(
                phase["publicationIterationDelta"],
                plan.publication_iteration_delta,
            )

        kernels = {
            kernel["name"]: kernel
            for kernel in facts["programs"][0]["kernels"]
        }
        for name, lifetime in EXPECTED_LIFETIMES.items():
            kernel = kernels[name]
            self.assertTrue(kernel["outputReusable"])
            self.assertEqual(
                kernel["outputLastConsumer"],
                lifetime.last_consumer,
            )
            self.assertEqual(
                kernel[
                    "outputLastConsumerIterationDelta"
                ],
                lifetime.last_consumer_iteration_delta,
            )
            self.assertEqual(
                kernel["outputReuseBefore"],
                lifetime.reuse_before,
            )
            self.assertEqual(
                kernel["outputReuseIterationDelta"],
                lifetime.reuse_iteration_delta,
            )
            self.assertEqual(
                kernel["outputPoolDepth"],
                lifetime.pool_depth,
            )

    def test_phase_lifetimes_stay_out_of_the_kernel_plan(self):
        _, facts = _run_transport_planner(
            OPERATIONS,
            True,
        )
        phases = facts["programs"][0]["phases"]
        self.assertTrue(phases[0]["outputReusable"])
        self.assertNotIn(
            phases[0]["name"],
            plan_output_lifetimes(OPERATIONS, True),
        )

    def test_plan_json_is_written_where_requested(self):
        source = compile_transport_program_mlir(
            OPERATIONS,
            cyclic=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(
                pathlib.Path(directory) / "plan.json"
            )
            self.assertEqual(
                emit_plan_json(source, path),
                EXPECTED_DOCUMENT,
            )

    def test_printed_module_is_unchanged_by_the_side_file(self):
        planned, _ = _run_transport_planner(
            OPERATIONS,
            True,
        )
        self.assertNotIn("emit-plan-json", planned)
        self.assertIn(
            'resolvedCompletionInterval = 8 : i32',
            planned,
        )


if __name__ == "__main__":
    unittest.main()
