"""Issuer proofs must retain the exact launch domain that makes them legal."""

import tempfile
import unittest

import aboleth  # noqa: F401
from triton._C.libtriton import ir, passes


def module_text(expression, attributes=""):
    return (
        """module {
  tt.func public @domain(%table: !tt.ptr<i64>) {
    %peer = tt.get_program_id x : i32
    %lane = tt.get_program_id y : i32
    %c3 = arith.constant 3 : i32
    %c8 = arith.constant 8 : i32
    %negative = arith.constant -1 : i32
    %maximum = arith.constant 2147483647 : i32
"""
        + expression
        + """
    %ep = aboleth.endpoint %table[%index] """
        + attributes
        + """
        : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
}"""
    )


def run(source, *, analyze=True, lower=False):
    context = ir.context()
    ir.load_dialects(context)
    with tempfile.NamedTemporaryFile("w", suffix=".mlir") as file:
        file.write(source)
        file.flush()
        module = ir.parse_mlir_module(file.name, context)
    manager = ir.pass_manager(context)
    if analyze:
        passes.plugin.add_aboleth_analyze_issuers(manager)
    if lower:
        passes.plugin.add_aboleth_lower_rdma(manager)
    manager.run(module, "issuer-domain-test")
    return module.str_nodebug()


class IssuerDomainsTest(unittest.TestCase):
    def test_rotation_records_the_grid_extent_instead_of_assuming_it(self):
        text = run(module_text("""
    %sum = arith.addi %peer, %c3 : i32
    %index = arith.remsi %sum, %c8 : i32"""))
        self.assertIn("exclusiveAxis = 0 : i32", text)
        self.assertIn("exclusiveGridShape = array<i64: 8, 0, 0>", text)
        self.assertIn("exclusiveIssuer", text)

    def test_linearized_lanes_get_two_axes_and_a_checked_peer_extent(self):
        source = module_text("""
    %base = arith.muli %lane, %c8 : i32
    %index = arith.addi %base, %peer : i32""")
        proof = run(source)
        self.assertIn("exclusiveAxes = array<i32: 0, 1>", proof)
        self.assertIn("exclusiveGridShape = array<i64: 8, 0, 0>", proof)
        lowered = run(source, lower=True)
        self.assertIn("tt.get_num_programs x", lowered)
        self.assertIn("tt.get_num_programs y", lowered)
        self.assertIn("arith.constant 2147483648 : i64", lowered)
        self.assertIn("arith.cmpi ule", lowered)
        self.assertIn("tt.assert", lowered)
        self.assertNotIn("tt.assert", run(source, analyze=False, lower=True))

    def test_negative_and_wrapping_rotations_retain_shared_issuers(self):
        for offset in ["%negative", "%maximum"]:
            with self.subTest(offset=offset):
                text = run(module_text(f"""
    %sum = arith.addi %peer, {offset} : i32
    %index = arith.remsi %sum, %c8 : i32"""))
                self.assertNotIn("exclusive", text)

    def test_malformed_proof_metadata_is_rejected(self):
        for attributes in [
            "{exclusiveAxis = 3 : i32}",
            "{exclusiveAxis = -1 : i32}",
            "{exclusiveAxes = array<i32: 0, 0>}",
            "{exclusiveAxis = 0 : i32, exclusiveAxes = array<i32: 0, 1>}",
            "{exclusiveGridShape = array<i64: 8, 0, 0>}",
            "{exclusiveAxis = 0 : i32, exclusiveGridShape = array<i64: 8, 0>}",
            "{exclusiveAxis = 0 : i32, exclusiveGridShape = array<i64: -1, 0, 0>}",
            "{exclusiveAxis = 0 : i32, exclusiveGridShape = array<i64: 8, 2, 0>}",
        ]:
            with self.subTest(attributes=attributes):
                with self.assertRaises(RuntimeError):
                    run(
                        module_text(
                            "%index = arith.addi %peer, %c3 : i32", attributes
                        )
                    )


if __name__ == "__main__":
    unittest.main()
