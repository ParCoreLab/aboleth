"""CPU-only proofs for Ionic's publication and issuer boundaries."""
import tempfile
import unittest
from unittest import mock

import aboleth.language as language
import triton
import triton.language as tl

import aboleth  # configure the plugin first
from triton._C.libtriton import ir, passes


def lower(body, *, detached=False, signature="", backend="ionic"):
    source = f"""module {{
  tt.func public @test(%endpoint: !tt.ptr<i64>, %srcptr: !tt.ptr<i8>,
      %dstptr: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %key: i32,
      %bytes: i32, %epoch: i64, %old: i64, %pred: i1{signature}) {{
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %src = aboleth.bind_local_region %srcptr, %key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %dst = aboleth.bind_remote_region %dstptr, %key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %ready = aboleth.bind_remote_region %flag, %key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    {body}
    tt.return
  }}
}}"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir") as file:
        file.write(source); file.flush()
        context = ir.context(); ir.load_dialects(context)
        module = ir.parse_mlir_module(file.name, context)
        manager = ir.pass_manager(context)
        passes.plugin.add_aboleth_form_batches(manager)
        if backend == "ionic":
            passes.plugin.add_aboleth_lower_ionic(
                manager, ["/aboleth/ionic.bc", "detached" if detached else ""])
        else:
            getattr(passes.plugin, "add_aboleth_lower_" + backend)(manager, [])
        manager.run(module, "ionic-contract")
        return module.str_nodebug()


PUT = """%t = aboleth.put %ep, %dst, %src, %bytes, %true
  : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token"""
SIGNAL = """%s = aboleth.signal %ep, %ready, %epoch, %t
  : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
  aboleth.wait %s : !aboleth.token"""


@triton.jit
def pipelined_epochs(endpoint, source, destination, flag, lkey, rkey, rounds):
    for index in tl.range(0, rounds):
        transfer = language.put(endpoint, destination, source, 64,
                                source_key=lkey, destination_key=rkey)
        ready = language.signal(endpoint, flag, (index + 1).to(tl.int64),
                                after=transfer, destination_key=rkey)
        language.wait(ready, iteration=index, reuse=16)
    language.drain(endpoint)


@triton.jit
def pipelined_epoch_chains(endpoint, source, destination, flag, lkey, rkey, rounds):
    for index in tl.range(0, rounds):
        epoch = (index + 1).to(tl.int64)
        t0 = language.put(endpoint, destination, source, 64,
                          source_key=lkey, destination_key=rkey)
        s0 = language.signal(endpoint, flag, epoch, after=t0, destination_key=rkey)
        t1 = language.put(endpoint, destination + 64, source + 64, 64,
                          source_key=lkey, destination_key=rkey)
        s1 = language.signal(endpoint, flag + 1, epoch, after=t1, destination_key=rkey)
        t2 = language.put(endpoint, destination + 128, source + 128, 64,
                          source_key=lkey, destination_key=rkey)
        s2 = language.signal(endpoint, flag + 2, epoch, after=t2, destination_key=rkey)
        t3 = language.put(endpoint, destination + 192, source + 192, 64,
                          source_key=lkey, destination_key=rkey)
        s3 = language.signal(endpoint, flag + 3, epoch, after=t3, destination_key=rkey)
        language.wait(s0, iteration=index, reuse=16)
        language.wait(s1, iteration=index, reuse=16)
        language.wait(s2, iteration=index, reuse=16)
        language.wait(s3, iteration=index, reuse=16)
    language.drain(endpoint)


class IonicLoweringTest(unittest.TestCase):
    def test_strong_signal_retires_payload_before_epoch(self):
        text = lower(PUT + "\n" + SIGNAL)
        symbols = ["aboleth_ionic_reserve_cta", "aboleth_ionic_put_cta",
                   "aboleth_ionic_publish_prefix_cta", "aboleth_ionic_inline_cta",
                   "aboleth_ionic_commit_cta", "aboleth_ionic_poll_cta"]
        offsets = [text.index(symbol) for symbol in symbols]
        self.assertEqual(offsets, sorted(offsets))
        self.assertNotIn("aboleth.post_", text)

    def test_detached_library_has_per_caller_effects(self):
        text = lower(PUT + "\n" + SIGNAL, detached=True)
        self.assertIn('symbol = "aboleth_ionic_reserve"', text)
        self.assertNotIn("_cta", text)

    def test_dynamic_predicate_reaches_private_scratch_legalization(self):
        text = lower(PUT.replace("%bytes, %true", "%bytes, %pred") + "\n" + SIGNAL)
        self.assertIn("aboleth_ionic_put_cta", text)
        self.assertIn("arith.extui", text)

    def test_explicit_checked_cas_is_rejected(self):
        with self.assertRaises(RuntimeError):
            lower("""%s = aboleth.post_signal %ep, %ready, %epoch expect %old
              : (!aboleth.endpoint, !aboleth.remote_region, i64, i64) -> !aboleth.token
              %done = aboleth.submit %ep, %s {batchSize = 1 : i32, deferCompletion}
              : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token""")

    def test_completed_prefix_predicate_controls_ordering_and_epoch(self):
        text = lower("""%s = aboleth.post_signal %ep, %ready, %epoch if %pred
          {completedPrefix} : (!aboleth.endpoint, !aboleth.remote_region, i64, i1) -> !aboleth.token
          %done = aboleth.submit %ep, %s {batchSize = 1 : i32, deferCompletion}
          : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token""")
        self.assertIn("scf.if", text)
        self.assertIn("arith.extui", text)
        self.assertIn("aboleth_ionic_publish_prefix_cta", text)
        self.assertIn("aboleth_ionic_inline_cta", text)

    def test_other_backends_reject_completed_prefix(self):
        body = """%s = aboleth.post_signal %ep, %ready, %epoch {completedPrefix}
          : (!aboleth.endpoint, !aboleth.remote_region, i64) -> !aboleth.token
          %done = aboleth.submit %ep, %s {batchSize = 1 : i32, deferCompletion}
          : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token"""
        lower(body.replace("{completedPrefix}", ""), backend="proxy")
        with self.assertRaises(RuntimeError):
            lower(body, backend="proxy")

    def test_pipelined_epochs_compile_with_publication_ablation(self):
        import os
        from pathlib import Path
        if not Path(aboleth.__file__).with_name("aboleth_ionic.bc").is_file():
            self.skipTest("HIP Ionic device bitcode is not built")
        from aboleth.compiler import compilation, OPTIMIZATIONS
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, compile as triton_compile
        signature = {"endpoint": "*i64", "source": "*i8", "destination": "*i8",
                     "flag": "*i64", "lkey": "i32", "rkey": "i32", "rounds": "i32"}
        all_off = ",".join("-" + name for name in OPTIMIZATIONS)
        for kernel in (pipelined_epochs, pipelined_epoch_chains):
            for passes in ("", all_off, "-elide-epochs", "-plan-publication", "-completion-planning"):
                with self.subTest(kernel=kernel.__name__, passes=passes), mock.patch.dict(os.environ, ABOLETH_PASSES=passes):
                    with compilation("ionic", queue_capacity=1023):
                        result = triton_compile(ASTSource(kernel, signature),
                                                target=GPUTarget("hip", "gfx950", 64),
                                                options={"num_warps": 1})
                    self.assertIn("aboleth_ionic_inline_cta", result.asm["ttir"])

    def test_drain_needs_no_unqualified_hardware_nop(self):
        text = lower("aboleth.drain %ep : !aboleth.endpoint")
        self.assertIn("aboleth_ionic_drain_cta", text)
        self.assertNotIn("aboleth_ionic_reserve", text)

    def test_receiver_uses_system_acquire_helper(self):
        text = lower("""%cell = aboleth.bind_local_region %flag, %key
          : (!tt.ptr<i64>, i32) -> !aboleth.local_region
          aboleth.wait_signal %cell, %epoch : (!aboleth.local_region, i64) -> ()""")
        self.assertIn("aboleth_ionic_wait_signal", text)

    def test_null_endpoint_queries_stay_inside_the_c_abi(self):
        import ctypes
        from pathlib import Path
        import torch
        if not torch.version.hip:
            self.skipTest("HIP runtime boundary")
        library = ctypes.CDLL(str(Path(aboleth.__file__).with_name("libaboleth_rdma_abi.so")))
        library.aboleth_runtime_last_error.restype = ctypes.c_char_p
        for suffix in ("capacity", "max_inline", "submit_mode", "gpu_resident_queues"):
            with self.subTest(query=suffix):
                function = getattr(library, "aboleth_runtime_local_endpoint_" + suffix)
                function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
                function.restype = ctypes.c_int
                value = ctypes.c_uint32()
                self.assertEqual(function(None, ctypes.byref(value)), -1)
                self.assertIn(b"endpoint", library.aboleth_runtime_last_error())

    def test_inline_forwarding_obeys_ionic_sixteen_byte_limit(self):
        from pathlib import Path
        if not Path(aboleth.__file__).with_name("aboleth_ionic.bc").is_file():
            self.skipTest("HIP Ionic device bitcode is not built")
        from test_compilation_surface import staged_put
        from aboleth.compiler import compilation
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, compile as triton_compile
        signature = {"endpoint": "*i64", "destination": "*i8", "scratch": "*i64",
                     "remote_signal": "*i64", "local_signal": "*i64",
                     "source_key": "i32", "destination_key": "i32", "words": "constexpr"}
        for words in (1, 2, 4):
            with self.subTest(bytes=words * 8):
                source = ASTSource(staged_put, signature=signature, constexprs={"words": words})
                with compilation("ionic"):
                    compiled = triton_compile(source, target=GPUTarget("hip", "gfx950", 64),
                                              options={"num_warps": 1})
                text = compiled.asm["ttir"]
                if words <= 2:
                    self.assertNotIn("aboleth_ionic_put_cta", text)
                else:
                    self.assertIn("aboleth_ionic_put_cta", text)
                self.assertIn("aboleth_ionic_inline_cta", text)


if __name__ == "__main__":
    unittest.main()
