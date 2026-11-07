"""Every real reader must respect the same mask before a short input is bound."""

import os
import unittest
from unittest.mock import patch
from aboleth.pattern import Pattern, OpaqueCapabilities
from aboleth.planning import plan_storage_mlir


def storage(
    *,
    mask="n",
    extra_read=False,
    mutable=False,
    registered=False,
    escapes=False,
    view=False,
    async_reader=False,
    duplicate_full_read=False,
    enabled=True,
):
    p = Pattern("prefix-input")
    x = p.region(
        "x",
        element_type="bf16",
        shape=(16, 128),
        strides=(128, 1),
        immutable=not mutable,
        registered=registered,
        escapes=escapes,
        valid_count="n",
        dead_padding=True,
    )
    y = p.region(
        "y",
        element_type="bf16",
        shape=(16, 128),
        strides=(128, 1),
        escapes=True,
    )
    value = p.view("view", x, shape=(16, 128), strides=(128, 1)) if view else x
    reads = (value, value) if duplicate_full_read else (value,)
    masks = (mask, "") if duplicate_full_read else (mask,)
    p.opaque_compute(
        "compute",
        reads=reads,
        writes=(y,),
        asynchronous=async_reader,
        capabilities=OpaqueCapabilities(prefix_read_counts=masks),
    )
    if extra_read:
        p.use("full-observer", x)
    with patch.dict(
        os.environ,
        {"ABOLETH_PASSES": "+plan-storage" if enabled else "-plan-storage"},
    ):
        return plan_storage_mlir(p.compile().analysis.mlir)["x"]


class PrefixStorageTest(unittest.TestCase):
    def test_all_reads_share_the_declared_mask(self):
        self.assertEqual(
            storage().prefix_binding,
            dict(count="n", capacity=16, row_bytes=256),
        )
        self.assertIsNone(storage(enabled=False).prefix_binding)

    def test_unknown_or_unbounded_reader_blocks_binding(self):
        for facts in [
            dict(mask=""),
            dict(mask="other"),
            dict(extra_read=True),
            dict(view=True),
            dict(async_reader=True),
            dict(duplicate_full_read=True),
        ]:
            with self.subTest(facts=facts):
                self.assertIsNone(storage(**facts).prefix_binding)

    def test_only_private_immutable_non_escaping_inputs(self):
        for facts in [
            dict(mutable=True),
            dict(registered=True),
            dict(escapes=True),
        ]:
            with self.subTest(facts=facts):
                self.assertIsNone(storage(**facts).prefix_binding)

    def test_invalid_contract_is_rejected(self):
        p = Pattern("invalid")
        for facts in [
            dict(valid_count="", dead_padding=True),
            dict(valid_count="n"),
            dict(dead_padding=True),
        ]:
            with self.subTest(facts=facts), self.assertRaises(ValueError):
                p.region(
                    "x",
                    element_type="bf16",
                    shape=(16, 128),
                    strides=(128, 1),
                    **facts,
                )
        x = p.region(
            "valid", element_type="bf16", shape=(16, 128), strides=(128, 1)
        )
        with self.assertRaises(ValueError):
            p.opaque_compute(
                "invalid-mask",
                reads=(x,),
                capabilities=OpaqueCapabilities(prefix_read_counts=("n", "n")),
            )

    def test_unrepresentable_binding_keeps_unknown_storage(self):
        p = Pattern("oversized-input")
        x = p.region(
            "x",
            element_type="bf16",
            shape=(1, 2**62),
            strides=(2**62, 1),
            immutable=True,
            valid_count="n",
            dead_padding=True,
        )
        p.opaque_compute(
            "read",
            reads=(x,),
            capabilities=OpaqueCapabilities(prefix_read_counts=("n",)),
        )
        result = plan_storage_mlir(p.compile().analysis.mlir)["x"]
        self.assertIsNone(result.byte_count)
        self.assertIsNone(result.prefix_binding)


if __name__ == "__main__":
    unittest.main()
