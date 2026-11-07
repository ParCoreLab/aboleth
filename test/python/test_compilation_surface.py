"""What `aboleth.compilation` puts in the pipeline, and what importing Aboleth
costs. Nothing here launches a kernel: every compilation names its target, so
the tests run wherever Triton can build PTX."""

import contextlib
import io
import os
import pathlib
import subprocess
import sys
import unittest
from unittest import mock

import aboleth.language as aboleth
import triton
import triton.language as tl
from aboleth.compiler import (
    DEFAULT_INLINE_BYTES,
    MAX_INLINE_BYTES,
    _pattern_plan_path,
    _pipeline_key,
    compilation,
)
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as triton_compile

TARGET = GPUTarget("cuda", 80, 32)
WORDS = 2


@triton.jit
def staged_put(
    endpoint,
    destination,
    scratch,
    remote_signal,
    local_signal,
    source_key,
    destination_key,
    words: tl.constexpr,
):
    """Stage a small payload with scalar stores, then send the buffer.

    Exactly the shape aboleth-forward-inline-payloads rewrites into an inline
    command: the stores tile the transferred range and nothing else reaches it.
    """
    for index in tl.static_range(0, words):
        tl.store(scratch + index, tl.full((), index + 1, tl.uint64))
    transfer = aboleth.put(
        endpoint,
        destination,
        scratch,
        words * 8,
        source_key=source_key,
        destination_key=destination_key,
    )
    ready = aboleth.signal(
        endpoint,
        remote_signal,
        local_signal,
        transfer,
        source_key=source_key,
        destination_key=destination_key,
    )
    aboleth.wait(ready)


def scheduled_module(backend: str) -> str:
    """The kernel's IR after Aboleth's scheduling passes, before lowering."""
    source = ASTSource(
        staged_put,
        signature={
            "endpoint": "*i64",
            "destination": "*i8",
            "scratch": "*i64",
            "remote_signal": "*i64",
            "local_signal": "*i64",
            "source_key": "i32",
            "destination_key": "i32",
            "words": "constexpr",
        },
        constexprs={"words": WORDS},
    )
    dumped = io.StringIO()
    os.environ["ABOLETH_DUMP_SCHEDULED_TTIR"] = "1"
    # A cache hit skips the pipeline, and the pipeline is the subject here.
    always_compile = knobs.compilation.always_compile
    knobs.compilation.always_compile = True
    try:
        with contextlib.redirect_stdout(dumped):
            with compilation(backend):
                triton_compile(source, target=TARGET)
    finally:
        knobs.compilation.always_compile = always_compile
        os.environ.pop("ABOLETH_DUMP_SCHEDULED_TTIR", None)
    return dumped.getvalue()


class InlineForwardingBackendTest(unittest.TestCase):
    """Forwarding a payload into the command only happens where the command
    exists: every other lowering rejects the module it would produce."""

    def test_a_staged_put_compiles_for_proxy_unforwarded(self):
        scheduled = scheduled_module("proxy")

        self.assertIn("aboleth.post_put", scheduled)
        self.assertNotIn("aboleth.post_inline", scheduled)

    def test_the_same_kernel_is_forwarded_for_rdma_and_p2p(self):
        for backend in ("rdma", "p2p"):
            with self.subTest(backend=backend):
                scheduled = scheduled_module(backend)

                self.assertIn("aboleth.post_inline", scheduled)
                self.assertNotIn("aboleth.post_put", scheduled)

    def test_inline_bytes_cannot_exceed_the_wire_maximum(self):
        with self.assertRaisesRegex(ValueError, "972-byte wire maximum"):
            with compilation("rdma", inline_bytes=MAX_INLINE_BYTES + 1):
                pass

    def test_the_default_cap_is_one_work_queue_entry_block(self):
        self.assertEqual(DEFAULT_INLINE_BYTES, 28)
        self.assertLess(DEFAULT_INLINE_BYTES, MAX_INLINE_BYTES)

    def test_identical_pattern_plans_reuse_one_file(self):
        mlir = "module {}\n"
        first = _pattern_plan_path(mlir)
        second = _pattern_plan_path(mlir)

        self.assertEqual(first, second)
        self.assertEqual(
            pathlib.Path(first).read_text(encoding="utf-8"),
            mlir,
        )


class PipelineKeyTest(unittest.TestCase):
    def test_plugin_changes_invalidate_disk_and_memory_caches(self):
        with compilation("rdma"):
            with mock.patch("aboleth.compiler._plugin_hash", return_value="first"):
                first = _pipeline_key()
            with mock.patch("aboleth.compiler._plugin_hash", return_value="second"):
                second = _pipeline_key()
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_publication_pass_switch_invalidates_both_caches(self):
        # The merged compiler selects publication through ABOLETH_PASSES;
        # FORCE_STRONG_PUBLICATION was replaced by -plan-publication.
        with compilation("rdma"):
            with mock.patch.dict(os.environ, {"ABOLETH_PASSES": "+plan-publication"}):
                planned = _pipeline_key()
            with mock.patch.dict(os.environ, {"ABOLETH_PASSES": "-plan-publication"}):
                conservative = _pipeline_key()
        self.assertNotEqual(planned[0], conservative[0])
        self.assertNotEqual(planned[1], conservative[1])

    def test_emission_switches_reach_the_lowering(self):
        source = ASTSource(staged_put, signature={
            "endpoint": "*i64", "destination": "*i8", "scratch": "*i64",
            "remote_signal": "*i64", "local_signal": "*i64",
            "source_key": "i32", "destination_key": "i32", "words": "constexpr",
        }, constexprs={"words": WORDS})
        kernels = {}
        for passes in ("-invariant-descriptors,-vector-wqe-stores",
                       "+invariant-descriptors,+vector-wqe-stores"):
            with mock.patch.dict(os.environ, {"ABOLETH_PASSES": passes}):
                with compilation("rdma"):
                    kernels[passes[0]] = (
                        _pipeline_key(), triton_compile(source, target=TARGET))
        self.assertNotEqual(kernels["-"][0], kernels["+"][0])
        self.assertNotIn("ld.global.nc", kernels["-"][1].asm["ptx"])
        self.assertIn("ld.global.nc", kernels["+"][1].asm["ptx"])


class TargetEmissionPolicyTest(unittest.TestCase):
    def test_targets_share_the_generic_default(self):
        source = ASTSource(staged_put, signature={
            "endpoint": "*i64", "destination": "*i8", "scratch": "*i64",
            "remote_signal": "*i64", "local_signal": "*i64",
            "source_key": "i32", "destination_key": "i32", "words": "constexpr",
        }, constexprs={"words": WORDS})

        def compile_for(capability, switches, targets=None):
            with mock.patch.dict(os.environ, {"ABOLETH_PASSES": switches}), \
                    mock.patch.dict("aboleth.compiler.TARGET_OPTIMIZATIONS",
                                    targets or {}, clear=True):
                with compilation("rdma"):
                    return triton_compile(
                        source, target=GPUTarget("cuda", capability, 32)
                    ).asm["ptx"]

        self.assertIn("ld.global.nc", compile_for(100, ""))
        self.assertIn("ld.global.nc", compile_for(90, ""))
        self.assertNotIn("ld.global.nc",
                         compile_for(100, "-invariant-descriptors"))
        # A per-target default applies to its target only.
        scalar = {100: {"invariant-descriptors": False}}
        self.assertNotIn("ld.global.nc", compile_for(100, "", scalar))
        self.assertIn("ld.global.nc", compile_for(90, "", scalar))


class ImportSurfaceTest(unittest.TestCase):
    def test_importing_aboleth_needs_no_optional_packages(self):
        """The focused phase API must not require `triton_kernels`."""
        package = pathlib.Path(
            aboleth.__file__
        ).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import aboleth; "
                "print(aboleth.CompiledPhase.__name__); "
                "import sys; "
                "print('triton_kernels' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            env=environment,
        )

        self.assertEqual(
            completed.returncode,
            0,
            f"importing aboleth failed:\n{completed.stderr}",
        )
        reported = completed.stdout.split()
        self.assertEqual(reported[0], "CompiledPhase")
        self.assertEqual(
            reported[1],
            "False",
            "importing the focused Aboleth API loaded triton_kernels",
        )


if __name__ == "__main__":
    unittest.main()
