"""Compiler-folded boundary cases for dynamic source extents and offsets."""

import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import aboleth  # Configure the plugin before loading Triton/LLVM.
from triton._C.libtriton import ir, passes
from aboleth.compiler import enabled_optimizations

SOURCE = (
    Path(__file__).resolve().parents[1] / "compiler/valid-extents.mlir"
).read_text()


def lowered_bytes(count, offset, requested, optimized):
    source = SOURCE.replace(
        "%count: i64, %offset: i64, %bytes: i32, %epoch: i64", "%epoch: i64"
    )
    source = source.replace(
        "    %true =",
        f"    %count = arith.constant {count} : i64\n"
        f"    %offset = arith.constant {offset} : i64\n"
        f"    %bytes = arith.constant {requested} : i32\n    %true =",
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir") as file:
        file.write(source)
        file.flush()
        context = ir.context()
        ir.load_dialects(context)
        module = ir.parse_mlir_module(file.name, context)
        manager = ir.pass_manager(context)
        passes.plugin.add_aboleth_materialize_valid_extents(
            manager, ["" if optimized else "unoptimized"]
        )
        passes.common.add_canonicalizer(manager)
        passes.common.add_cse(manager)
        manager.run(module, "valid-extents-constant-boundaries")
        output = module.str_nodebug()
    constants = dict(
        re.findall(r"(%[\w]+) = arith.constant (-?\d+) : i32", output)
    )
    operand = re.search(
        r"aboleth.put [^,\n]+, [^,\n]+, [^,\n]+, (%[\w]+)", output
    )
    assert operand is not None, output
    return int(constants[operand[1]])


class ValidExtentsTest(unittest.TestCase):
    def test_dynamic_count_and_address_boundaries(self):
        # count, byte offset, requested bytes, optimized bytes, generic bytes.
        cases = [
            (-9, 0, 128, 0, 128),
            (0, 0, 128, 0, 128),
            (1, 0, 128, 16, 128),
            (7, 0, 128, 112, 128),
            (8, 0, 128, 128, 128),
            (23, 0, 128, 128, 128),
            (2, 16, 128, 16, 112),
            (2, 32, 128, 0, 96),
            (8, 128, 128, 0, 0),
            (8, 144, 128, 0, 0),
            (8, -16, 128, 0, 0),
            (8, -(2**63), 128, 0, 0),
            (8, 0, -1, 0, 0),
            (8, 0, 5, 5, 5),
            (2**63 - 1, 0, 128, 128, 128),
            (-(2**63), 0, 128, 0, 128),
        ]
        for count, offset, requested, compact, generic in cases:
            for enabled, expected in [(True, compact), (False, generic)]:
                with self.subTest(
                    count=count,
                    offset=offset,
                    requested=requested,
                    enabled=enabled,
                ):
                    self.assertEqual(
                        lowered_bytes(count, offset, requested, enabled),
                        expected,
                    )

    def test_switch_is_part_of_the_public_compiler_controls(self):
        with patch.dict(os.environ, {"ABOLETH_PASSES": "-valid-extents"}):
            self.assertNotIn("valid-extents", enabled_optimizations())
        with patch.dict(os.environ, {"ABOLETH_PASSES": "+valid-extents"}):
            self.assertIn("valid-extents", enabled_optimizations())


if __name__ == "__main__":
    unittest.main()
