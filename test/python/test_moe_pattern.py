import unittest
from dataclasses import replace
from pathlib import Path

from aboleth import (
    MoEPatternSpec,
    PhaseResources,
    RegionEffect,
    build_moe_pattern,
)
from aboleth.emit import (
    emit_cuda_fragment,
)
from aboleth.pattern import CommunicationStage, OpaqueStage
from aboleth.patterns.moe_transport import (
    physicalize_moe_pattern,
)


def transport_bindings(compiled, *, iterations=4):
    effects = {
        region.id: RegionEffect(
            allocation=index,
            generation=1,
            offset=0,
            length=4096,
            stride=4096,
            count=iterations,
            period=4096 * iterations,
        )
        for index, region in enumerate(compiled.analysis.regions)
    }
    return PhaseResources(
        regions=effects,
        peer_transports=("p2p",),
        peer_capacities=(2**31 - 1,),
        num_peers=4,
        iteration_count=iterations,
    )


class MoEPatternTest(unittest.TestCase):
    def test_dispatch_binding_supports_fixed_chunks(self):
        spec = MoEPatternSpec(
            max_tokens=64,
            hidden=128,
            top_k=2,
            num_ranks=4,
        )
        full = physicalize_moe_pattern(spec).binding("moe.dispatch").to_program()
        chunked = (
            physicalize_moe_pattern(
                spec,
                items_per_message=32,
            )
            .binding("moe.dispatch")
            .to_program()
        )

        self.assertEqual(full.command_count, 4)
        self.assertEqual(chunked.command_count, 10)
        self.assertEqual(
            chunked.operations,
            full.operations,
        )
        self.assertEqual(
            sum(
                1
                for transfer in chunked.transfers
                if transfer.operations == ("dispatch.values",)
            ),
            1,
        )




    def test_frontend_neutral_pattern_has_no_bindings(self):
        compiled = build_moe_pattern(
            MoEPatternSpec(
                max_tokens=256,
                hidden=1024,
                top_k=2,
                num_ranks=8,
            )
        ).compile()

        self.assertFalse(compiled.kernel_bindings)
        self.assertEqual(
            [(type(stage), stage.name) for stage in compiled.stages],
            [
                (OpaqueStage, "route"),
                (CommunicationStage, "communication.0"),
                (OpaqueStage, "experts"),
                (CommunicationStage, "communication.1"),
                (OpaqueStage, "combine"),
            ],
        )
        self.assertEqual(
            tuple(operation.id for operation in compiled.stages[1].operations),
            (
                "dispatch.values",
                "dispatch.expert_ids",
                "dispatch.count",
            ),
        )
        self.assertEqual(
            tuple(operation.id for operation in compiled.stages[3].operations),
            ("return.values",),
        )

    def test_dispatch_and_return_materialize_directly(self):
        graph = build_moe_pattern(
            MoEPatternSpec(
                max_tokens=256,
                hidden=1024,
                top_k=2,
                num_ranks=8,
                slots=4,
            )
        ).simplify()

        for operation in (
            "dispatch.values.materialize",
            "dispatch.expert_ids.materialize",
            "dispatch.count.materialize",
            "return.values.materialize",
        ):
            self.assertIsNone(graph.operation(operation))
        self.assertEqual(
            graph.operation("dispatch.values").destination,
            "dispatch.values.storage",
        )
        self.assertEqual(
            graph.operation("dispatch.count").destination,
            "dispatch.count.storage",
        )
        self.assertEqual(
            graph.operation("return.values").destination,
            "return.values.storage",
        )
        self.assertEqual(
            graph.operation("combine").inputs,
            (
                "returned.values",
                "route.source_rows",
                "route.weights",
            ),
        )

    def test_topk_sizes_dynamic_dispatch_workset(self):
        graph = build_moe_pattern(
            MoEPatternSpec(
                max_tokens=64,
                hidden=1024,
                top_k=4,
                num_ranks=8,
            )
        ).simplify()

        self.assertEqual(
            graph.operation("dispatch.work.workset").max_items,
            256,
        )
        self.assertEqual(
            graph.operation("return.values").kind,
            "transfer",
        )

    def test_fp8_values_and_scales_share_dispatch(self):
        graph = build_moe_pattern(
            MoEPatternSpec(
                max_tokens=256,
                hidden=1024,
                top_k=2,
                num_ranks=8,
                value_type="f8E4M3FN",
                scale_width=32,
            )
        ).simplify()

        self.assertIsNone(graph.operation("dispatch.scales.materialize"))
        self.assertEqual(
            graph.operation("dispatch.scales").destination,
            "dispatch.scales.storage",
        )
        self.assertIn(
            "expert.scales",
            graph.operation("experts").inputs,
        )

    def test_unsafe_async_depth_keeps_dispatch_staging(self):
        graph = build_moe_pattern(
            MoEPatternSpec(
                max_tokens=256,
                hidden=1024,
                top_k=2,
                num_ranks=8,
                slots=1,
                expert_max_inflight=2,
            )
        ).simplify()

        self.assertIsNotNone(graph.operation("dispatch.values.materialize"))


if __name__ == "__main__":
    unittest.main()
