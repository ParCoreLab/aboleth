"""Destination binding needs explicit capability, donation, layout and lifetime."""

import os
import unittest
from unittest.mock import patch
from aboleth.pattern import Pattern, OpaqueCapabilities
from aboleth.planning import plan_storage_mlir


def example(
    *,
    bindable=True,
    donated=True,
    registered=True,
    escaped=False,
    late=False,
    strided=False,
    extra_reader=False,
    asynchronous=False,
    ordered=False,
    inflight=1,
    pool=1,
    enabled=True,
):
    p = Pattern("bindable-output")
    inputs = p.region(
        "input",
        element_type="bf16",
        shape=(16, 8),
        strides=(8, 1),
        immutable=True,
    )
    staging = p.region(
        "staging",
        element_type="bf16",
        shape=(16, 8),
        strides=(8, 1),
        escapes=escaped,
    )

    def final():
        return p.region(
            "return",
            element_type="bf16",
            shape=(16, 8),
            strides=(16, 1) if strided else (8, 1),
            visibility="peer",
            registered=registered,
            donated=donated,
            pool_depth=pool,
            reuse_interval=1,
        )

    if not late:
        target = final()
    (output,) = p.opaque_compute(
        "expert",
        reads=(inputs,),
        writes=(staging,),
        asynchronous=asynchronous,
        event_ordered=ordered,
        max_inflight=inflight,
        capabilities=OpaqueCapabilities(accepts_destination_rewrite=bindable),
    )
    if late:
        target = final()
    copied = p.copy("stage-for-send", output, target)
    if extra_reader:
        p.use("other-output-consumer", output)
    p.use("send-consumer", copied)
    with patch.dict(
        os.environ,
        {"ABOLETH_PASSES": "+plan-storage" if enabled else "-plan-storage"},
    ):
        return p.compile()


class OpaqueOutputBindingTest(unittest.TestCase):
    def test_compiler_retargets_bindable_output_and_removes_staging(self):
        result = example()
        self.assertEqual(result.opaque_stage("expert").writes, ("return",))
        self.assertIn("stage-for-send", result.analysis.removed_operation_ids)
        self.assertIn("staging", result.analysis.removed_region_ids)
        self.assertEqual(
            result.analysis.retargeted_destinations["expert.output.0"],
            ("staging", "return"),
        )
        allocations = plan_storage_mlir(result.analysis.mlir)
        self.assertNotIn("staging", allocations)
        self.assertEqual(allocations["return"].byte_count, 16 * 8 * 2)

    def test_capability_donation_layout_and_lifetime_are_required(self):
        for changes in [
            dict(bindable=False),
            dict(donated=False),
            dict(registered=False, donated=False),
            dict(escaped=True),
            dict(late=True),
            dict(strided=True),
            dict(extra_reader=True),
            dict(asynchronous=True),
            dict(asynchronous=True, ordered=True, inflight=2, pool=1),
            dict(enabled=False),
        ]:
            with self.subTest(changes=changes):
                result = example(**changes)
                self.assertEqual(
                    result.opaque_stage("expert").writes, ("staging",)
                )
                self.assertIsNotNone(
                    result.analysis.operation("stage-for-send")
                )

    def test_invalid_donation_is_rejected_before_planning(self):
        with self.assertRaisesRegex(
            ValueError, "donated storage must be registered"
        ):
            example(registered=False)

    def test_ordered_asynchronous_output_respects_destination_pool(self):
        result = example(asynchronous=True, ordered=True, inflight=2, pool=2)
        self.assertEqual(result.opaque_stage("expert").writes, ("return",))


if __name__ == "__main__":
    unittest.main()
