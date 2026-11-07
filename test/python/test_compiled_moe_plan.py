"""The real MoE consumer graph determines fields, storage and output binding."""

import os
import unittest
from unittest.mock import patch
from aboleth.moe.compiled_plan import MoESpec, compile_moe_plan
from aboleth.endpoint_profiles import B200_NDR400_HOST as PROFILE


def plan(*, enabled=True, bindable=True, spec=None):
    with patch.dict(
        os.environ,
        {"ABOLETH_PASSES": "+plan-storage" if enabled else "-plan-storage"},
    ):
        return compile_moe_plan(
            spec or MoESpec(4, 128, 2, 8, 4),
            geography=((0, "p2p"), (1, "rdma"), (3, "rdma")),
            capacities={"p2p": 2**31 - 1, "rdma": 4096},
            platform=PROFILE.platform,
            queue_memory="host",
            available_lanes=4,
            participants=((PROFILE.platform, "host", 4),) * 4,
            profiles=(PROFILE,),
            bindable_expert_output=bindable,
        )


class CompiledMoEPlanTest(unittest.TestCase):
    def test_dataflow_removes_unused_return_fields(self):
        result = plan()
        self.assertEqual(
            result.dispatch_fields, ("payload", "metadata", "count")
        )
        self.assertEqual(result.return_fields, ("payload",))
        self.assertIn(
            "return.metadata", result.pattern.analysis.removed_operation_ids
        )
        self.assertIn(
            "return.count", result.pattern.analysis.removed_operation_ids
        )
        self.assertNotIn("returned_metadata", result.pattern.storage_plan)
        self.assertNotIn("returned_counts", result.pattern.storage_plan)

    def test_storage_pass_derives_reached_rows_and_expert_destination(self):
        on, off = plan(), plan(enabled=False)
        self.assertEqual(
            on.pattern.storage_plan["received_payload"].physical_shape,
            (16, 128),
        )
        self.assertEqual(
            off.pattern.storage_plan["received_payload"].physical_shape,
            (32, 128),
        )
        self.assertEqual(on.expert_destination, "return_source")
        self.assertFalse(on.residual_expert_copy)
        self.assertNotIn("expert_output", on.pattern.storage_plan)
        self.assertEqual(off.expert_destination, "expert_output")
        self.assertTrue(off.residual_expert_copy)
        for name, row_bytes in [
            ("tokens", 256),
            ("input_ids", 8),
            ("input_weights", 8),
        ]:
            self.assertEqual(
                on.pattern.storage_plan[name].prefix_binding,
                dict(count="num_tokens", capacity=4, row_bytes=row_bytes),
            )
            self.assertIsNone(off.pattern.storage_plan[name].prefix_binding)

    def test_dead_communication_ablation_keeps_generic_return_fields(self):
        with patch.dict(os.environ, {"ABOLETH_PASSES": "-dead-communication"}):
            result = compile_moe_plan(
                MoESpec(4, 128, 2, 8, 4),
                geography=((0, "p2p"), (1, "rdma"), (3, "rdma")),
                capacities={"p2p": 2**31 - 1, "rdma": 4096},
                platform=PROFILE.platform,
                queue_memory="host",
                available_lanes=4,
                participants=((PROFILE.platform, "host", 4),) * 4,
                profiles=(PROFILE,),
            )
        self.assertEqual(
            result.return_fields, ("payload", "metadata", "count")
        )
        self.assertIn("returned_metadata", result.pattern.storage_plan)
        self.assertIn("returned_counts", result.pattern.storage_plan)

    def test_binding_requires_the_client_capability(self):
        result = plan(bindable=False)
        self.assertEqual(result.expert_destination, "expert_output")
        self.assertTrue(result.residual_expert_copy)

    def test_endpoint_geometry_is_compiler_owned(self):
        result = plan()
        self.assertEqual(
            [p.backend for p in result.dispatch_launches], ["p2p", "rdma"]
        )
        self.assertEqual(
            [p.peers for p in result.dispatch_launches], [(0,), (1, 3)]
        )
        self.assertEqual(
            [p.ready_stride for p in result.dispatch_launches], [4, 4]
        )
        self.assertEqual(
            [p.signal_slots for p in result.dispatch_launches], [16, 16]
        )
        self.assertTrue(
            all(
                p.completion_interval == 1
                for p in result.pattern.transport_plan.values()
            )
        )

    def test_fp8_dispatch_and_bf16_return_keep_distinct_physical_widths(self):
        result = plan(
            spec=MoESpec(4, 256, 2, 8, 4, "f8E4M3FN", 128, False, 1e-10)
        )
        storage = result.pattern.storage_plan
        self.assertEqual(storage["received_payload"].physical_shape, (16, 256))
        self.assertEqual(storage["received_payload"].byte_count, 4096)
        self.assertEqual(storage["return_source"].byte_count, 8192)
        self.assertEqual(storage["expert_scales"].byte_count, 128)
        self.assertEqual(storage["received_metadata"].byte_count, 512)
        self.assertEqual(result.return_fields, ("payload",))
        self.assertEqual(result.expert_destination, "return_source")

    def test_fp8_rejects_incompatible_scale_contracts(self):
        invalid = [
            dict(scale_group=0),
            dict(scale_group=96),
            dict(scale_group=512),
            dict(round_scale=1),
            dict(quant_amax_floor=0),
            dict(quant_amax_floor=float("inf")),
            dict(quant_amax_floor=float("nan")),
        ]
        for facts in invalid:
            kwargs = dict(
                value_type="f8E4M3FN",
                scale_group=128,
                round_scale=False,
                quant_amax_floor=1e-10,
            )
            kwargs.update(facts)
            with self.subTest(facts=facts), self.assertRaises(ValueError):
                MoESpec(4, 256, 2, 8, 4, **kwargs)
        with self.assertRaises(ValueError):
            MoESpec(4, 256, 2, 8, 4, scale_group=128)

    def test_invalid_dimensions_rejected(self):
        for args in [
            (0, 128, 2, 8, 4),
            (4, 0, 2, 8, 4),
            (4, 128, 9, 8, 4),
            (4, 128, 2, 7, 4),
            (4, 128, 2, 8, 1),
            (True, 128, 2, 8, 4),
            (2**24, 128, 2, 8, 4),
            (4, 128, 2, 2**31, 4),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                MoESpec(*args)


if __name__ == "__main__":
    unittest.main()
