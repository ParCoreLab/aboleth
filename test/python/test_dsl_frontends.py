"""The generated Tier-1 library bound into Triton and CuTe DSL kernels.

`aboleth.frontends.triton.tier1` and `aboleth.frontends.cute` link the
library's per-op bitcode into a DSL kernel. Without a GPU this pins the
binding contract: every op links at its exported ABI, dispatches on the
handle's kind bits, and inlines completely (no call to a Tier-1 symbol
survives in the PTX). The behavioural check is a hardware run of the dsl-*
phases of experiments/bench/matrix.sh.
"""

import os
import tempfile
import unittest

import aboleth  # noqa: F401  (configures the compiler plugin before Triton)
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as triton_compile

import aboleth.frontends.triton.tier1 as triton_tier1
from aboleth.device_library import generate_device_library
from aboleth.emit.tier1_header import op_slots

ARCH = "sm_80"
TARGET = GPUTarget("cuda", 80, 32)


@triton.jit
def exchange(endpoints, peer, dst, dst_key, src, src_key, flag, flag_key,
             my_flag, BYTES: tl.constexpr):
    if triton_tier1.tid() == 0:
        triton_tier1.put(endpoints, peer, dst, dst_key, src, src_key, BYTES)
        triton_tier1.put_inline(endpoints, peer, flag, flag_key, 7)
        triton_tier1.signal(endpoints, peer, flag, flag_key, 1)
        triton_tier1.put_signal(endpoints, peer, dst, dst_key, src, src_key,
                                BYTES, flag, flag_key, 2)
        triton_tier1.quiet(endpoints, peer)
        triton_tier1.wait_signal(my_flag, 2)


def residual_calls(ptx: str) -> list[str]:
    return [line for line in ptx.splitlines()
            if "call" in line and "aboleth_tier1" in line]


class DslFrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = generate_device_library(
            ARCH, output_dir=os.path.join(cls.directory.name, "tier1"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_op_slots_are_the_exported_abi(self):
        self.assertEqual(op_slots("quiet"), (("endpoint", "i64"),))
        self.assertEqual(len(op_slots("put_signal")), 10)
        self.assertEqual(len(op_slots("put_many_8_signal")), 5)
        with self.assertRaises(ValueError):
            op_slots("no_such_op")

    def test_triton_kernel_links_and_inlines_every_op(self):
        triton_tier1.bind(self.library)
        libraries = triton_tier1.extern_libs()
        self.assertEqual(
            set(libraries),
            {f"aboleth_{transport}_{op}"
             for transport, ops in self.library.op_bitcodes.items()
             for op in ops})
        compiled = triton_compile(
            ASTSource(exchange, signature={
                "endpoints": "*u64", "peer": "i32", "dst": "i64",
                "dst_key": "i32", "src": "i64", "src_key": "i32",
                "flag": "i64", "flag_key": "i32", "my_flag": "i64",
                "BYTES": "constexpr"}, constexprs={"BYTES": 64}),
            target=TARGET,
            options={"num_warps": 1, "extern_libs": libraries})
        ptx = compiled.asm["ptx"]
        self.assertEqual(residual_calls(ptx), [])
        # The generic library dispatches on the handle's kind bits.
        self.assertIn("and.b64", ptx)

    def test_cute_kernel_links_and_inlines_every_op(self):
        try:
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import make_fake_compact_tensor
        except ImportError:
            self.skipTest("CuTe DSL is not installed")
        if not hasattr(cute, "extern"):
            self.skipTest("this CuTe DSL has no cute.extern")
        import aboleth.frontends.cute as cute_tier1
        os.environ.setdefault("CUTE_DSL_ARCH", ARCH)
        cute_tier1.bind(self.library)

        @cute.kernel
        def kernel(endpoints: cute.Tensor, peer: cutlass.Int32,
                   dst: cutlass.Uint64, src: cutlass.Uint64,
                   flag: cutlass.Uint64, my_flag: cutlass.Uint64):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                cute_tier1.put_signal(endpoints, peer, dst, 1, src, 2, 64,
                                      flag, 3, 1)
                cute_tier1.quiet(endpoints, peer)
                cute_tier1.wait_signal(my_flag, 1)

        @cute.jit
        def launch(endpoints: cute.Tensor, peer: cutlass.Int32,
                   dst: cutlass.Uint64, src: cutlass.Uint64,
                   flag: cutlass.Uint64, my_flag: cutlass.Uint64):
            kernel(endpoints, peer, dst, src, flag, my_flag).launch(
                grid=[1, 1, 1], block=[32, 1, 1])

        compiled = cute.compile(
            launch, make_fake_compact_tensor(cutlass.Int64, (4,),
                                             assumed_align=8),
            1, 1 << 40, 1 << 41, 1 << 42, 1 << 43,
            options="--keep-ptx")
        ptx = compiled.__ptx__
        if isinstance(ptx, bytes):
            ptx = ptx.decode()
        self.assertEqual(residual_calls(ptx), [])


if __name__ == "__main__":
    unittest.main()
