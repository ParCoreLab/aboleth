import hashlib
import os
import pathlib
import re
import subprocess
import tempfile
import unittest
from dataclasses import replace

from aboleth import artifact as artifact_module
from aboleth.artifact import (
    compile_fragment_artifact,
    realize_bindable_epilogue,
    realize_standalone_kernel,
)
from aboleth.emit.devicify import (
    assert_single_public_entry,
    devicify_kernel,
    wrap_entry_point,
)
from aboleth.pattern import (
    KernelTransportArgument,
    KernelTransportProgram,
    KernelTransportReduction,
    KernelTransportSignal,
    KernelTransportTransfer,
)
from aboleth.planned_phase import (
    compile_planned_drain_artifact,
    compile_planned_fence_artifact,
    compile_planned_publish_artifact,
    compile_planned_put_artifact,
    planned_drain_program,
    planned_fence_program,
    planned_publish_program,
    planned_put_program,
)
from aboleth.transport_binding import invoke_manifest
from aboleth.world import device_library

# One transfer plus its publication: the smallest program the fragment pipeline
# accepts, in the slot order the artifact ABI exposes.
SLOTS = (
    ("endpoint", "i64"),
    ("destination", "i64"),
    ("source", "i64"),
    ("bytes", "i32"),
    ("source_key", "i32"),
    ("destination_key", "i32"),
    ("predicate", "i1"),
    ("remote_signal", "i64"),
    ("local_signal", "i64"),
    ("iteration", "i32"),
    ("iterations", "i32"),
)

PROGRAM = KernelTransportProgram(
    name="probe",
    operations=("payload",),
    kernel="probe_kernel",
    insertion_point="epilogue",
    arguments=tuple(
        KernelTransportArgument(name, mlir_type) for name, mlir_type in SLOTS
    ),
    transfers=(
        KernelTransportTransfer(
            operations=("payload",),
            destination="destination",
            source="source",
            byte_count="bytes",
            source_key="source_key",
            destination_key="destination_key",
            predicate="predicate",
            max_bytes=4096,
        ),
    ),
    signal=KernelTransportSignal(
        destination="remote_signal",
        source="local_signal",
        source_key="source_key",
        destination_key="destination_key",
        iteration="iteration",
        iterations="iterations",
    ),
    program_id=0,
    # The kind `realize_bindable_epilogue` demands: only a program whose
    # attachment was physicalized as a detached artifact may be compiled into
    # one, so the fixture declares it rather than relying on the default.
    artifact="bindable_epilogue",
)

ARCH = "sm_80"
SYMBOL = "aboleth_fragment_probe"


def built_device_library():
    """The device library this build emitted, whatever architecture that is."""
    root = pathlib.Path(__file__).resolve().parents[2]
    build = pathlib.Path(os.environ.get("ABOLETH_BUILD_DIR", root / "build"))
    built = sorted((build / "lib").glob("aboleth_rdma_*.bc"))
    if not built:
        return None
    return device_library(
        built[0].stem.removeprefix("aboleth_rdma_"), generated=False
    )


LIBRARY = built_device_library()


def ptxas_accepts(arch: str) -> bool:
    """Whether the available ptxas can assemble PTX targeting `arch`.

    CI's ptxas can lag the hardware this repo targets; skipping cleanly here
    gives the Blackwell test the same contract LIBRARY's own check gives the
    rest of the module, with no arch pinned into the skip condition itself.
    """
    with tempfile.TemporaryDirectory(prefix="aboleth-ptxas-probe-") as scratch:
        directory = pathlib.Path(scratch)
        source = directory / "probe.ptx"
        source.write_text(f".version 8.6\n.target {arch}\n.address_size 64\n")
        result = subprocess.run(
            (
                artifact_module._ptxas(),
                f"-arch={arch}",
                "-m64",
                "-c",
                str(source),
                "-o",
                str(directory / "probe.cubin"),
            ),
            capture_output=True,
            check=False,
            text=True,
        )
        return result.returncode == 0


SM100_SUPPORTED = ptxas_accepts("sm_100")


def lowered_program(
    program,
    *,
    completion_interval=1,
    device_bitcode=...,
) -> str:
    """The fragment IR the transport lowering produced for `program`.

    The pipeline intermediate `compile_fragment_artifact` feeds to Triton: past
    form-batches, so every protocol decision is already a device-library call
    or a descriptor word. `device_bitcode=None`
    selects the direct emitter in detached mode; the default targets the build's
    device library.
    """
    if device_bitcode is ...:
        if LIBRARY is None:
            raise unittest.SkipTest("CUDA RDMA device bitcode is not built")
        device_bitcode = pathlib.Path(LIBRARY.bitcode)
    with tempfile.TemporaryDirectory(prefix="aboleth-lowered-") as scratch:
        staging = pathlib.Path(scratch)
        plan = staging / "transport-program.mlir"
        plan.write_text(artifact_module._plan_module(program))
        module = staging / "fragment.mlir"
        module.write_text(
            artifact_module._fragment_module(
                program,
                SYMBOL,
                artifact_module._slots(program),
            )
        )
        return artifact_module._lower(
            module,
            plan,
            transport="rdma",
            completion_interval=completion_interval,
            lower_arguments=artifact_module._lower_arguments(
                program, "rdma", device_bitcode
            ),
        )


def device_symbols(text: str) -> set[str]:
    """Which transport device-library entry points the fragment calls."""
    return set(re.findall(r'symbol = "(aboleth_rdma_\w+)"', text))


def exported_parameters(artifact) -> int:
    """How many parameters the exported PTX function actually takes."""
    return len(
        set(
            re.findall(
                rf"{artifact.symbol}_param_(\d+)",
                artifact.ptx.read_text(),
            )
        )
    )


def reducing(operator: str, cardinality: int = 2, **overrides):
    """PROGRAM plus one reduce contract, which no fragment lowers yet."""
    return replace(
        PROGRAM,
        reductions=(
            KernelTransportReduction(
                operation="payload",
                operator=operator,
                cardinality=cardinality,
            ),
        ),
        **overrides,
    )


class ProgramContractTest(unittest.TestCase):
    """What a program must declare before any artifact is compiled from it.

    Every refusal precedes `_validated_inputs`, so none of it needs a built
    device library: the program's own contract is checked before the
    environment's.
    """

    def compile(self, program, **overrides):
        return compile_fragment_artifact(
            program,
            arch=ARCH,
            completion_interval=4,
            device_bitcode="unbuilt.bc",
            output_dir=tempfile.gettempdir(),
            **overrides,
        )

    def test_an_inline_program_cannot_become_a_detached_artifact(self):
        # The one kind physicalization lets through without asking the
        # neighbouring stages for permission.
        with self.assertRaises(ValueError) as raised:
            self.compile(replace(PROGRAM, artifact="inline_fragment"))
        message = str(raised.exception)
        self.assertIn("'inline_fragment'", message)
        self.assertIn("Artifact.BINDABLE_EPILOGUE", message)
        self.assertIn("realize_bindable_epilogue", message)
        self.assertIn("realize_standalone_kernel", message)

    def test_a_reduce_contract_is_refused_by_every_entry_point(self):
        for realize, program in (
            (self.compile, reducing("add")),
            (
                lambda program: realize_bindable_epilogue(
                    program,
                    arch=ARCH,
                    completion_interval=4,
                    device_bitcode="unbuilt.bc",
                    output_dir=tempfile.gettempdir(),
                ),
                reducing("add"),
            ),
            (
                lambda program: realize_standalone_kernel(
                    program,
                    arch=ARCH,
                    completion_interval=4,
                    device_bitcode="unbuilt.bc",
                    output_dir=tempfile.gettempdir(),
                ),
                reducing("add", artifact="standalone_kernel"),
            ),
        ):
            with self.subTest(realize=realize):
                with self.assertRaises(ValueError) as raised:
                    realize(program)
                message = str(raised.exception)
                self.assertIn("'payload'", message)
                self.assertIn("do not lower to fragments yet", message)

    def test_the_signature_separates_two_reduce_contracts(self):
        signatures = {
            artifact_module._signature(
                program,
                invoke_manifest(program),
                SYMBOL,
                ARCH,
                "rdma",
                4,
                "digest",
            )
            for program in (
                PROGRAM,
                reducing("add"),
                reducing("max"),
                reducing("add", cardinality=3),
            )
        }

        self.assertEqual(len(signatures), 4)

    def test_external_planned_put_has_no_publication_or_completion(self):
        program = planned_put_program(
            max_bytes=4096,
            cta_exclusive=True,
        )
        lowered = lowered_program(program)

        self.assertIsNone(invoke_manifest(program).signal)
        self.assertEqual(
            device_symbols(lowered),
            {
                "aboleth_rdma_post_write",
                "aboleth_rdma_reserve_planned",
            },
        )
        self.assertNotIn("aboleth_rdma_publish", lowered)
        self.assertNotIn("aboleth_rdma_poll", lowered)

    def test_planned_head_publication_is_its_own_fragment(self):
        program = planned_publish_program()
        lowered = lowered_program(program)

        self.assertEqual(program.command_count, 0)
        self.assertEqual(program.payload_bytes, 0)
        self.assertEqual(
            device_symbols(lowered),
            {"aboleth_rdma_publish_head"},
        )
        self.assertNotIn("aboleth_rdma_reserve", lowered)
        self.assertNotIn("aboleth_rdma_poll", lowered)

    def test_effect_boundary_fence_is_its_own_fragment(self):
        program = planned_fence_program()
        lowered = lowered_program(program)

        self.assertEqual(program.command_count, 0)
        self.assertEqual(program.payload_bytes, 0)
        self.assertEqual(device_symbols(lowered), set())
        self.assertIn("fence.acq_rel.sys", lowered)

    def test_capacity_reclamation_is_its_own_fragment(self):
        program = planned_drain_program()
        lowered = lowered_program(program)

        self.assertEqual(program.command_count, 0)
        self.assertEqual(
            device_symbols(lowered),
            {"aboleth_rdma_drain"},
        )


class DevicifyTest(unittest.TestCase):
    def test_two_public_entries_are_rejected(self):
        module = """
module {
  tt.func public @first(%a: i32) attributes {noinline = false} {
    tt.return
  }
  tt.func public @second(%a: i32) attributes {noinline = false} {
    tt.return
  }
}
"""
        with self.assertRaises(ValueError) as raised:
            assert_single_public_entry(module)
        message = str(raised.exception)
        self.assertIn("exactly one", message)
        self.assertIn("found 2", message)
        self.assertIn("first", message)
        self.assertIn("second", message)

    def test_no_public_entry_is_rejected(self):
        module = """
module {
  tt.func private @only(%a: i32) attributes {noinline = true} {
    tt.return
  }
}
"""
        with self.assertRaises(ValueError) as raised:
            assert_single_public_entry(module)
        self.assertIn("found 0", str(raised.exception))

    def test_a_non_kernel_definition_is_rejected(self):
        text = "define void @body(i32 %0) #0 {\n  ret void\n}\n"
        with self.assertRaises(ValueError) as raised:
            devicify_kernel(text, "body", 1)
        self.assertIn("ptx_kernel", str(raised.exception))

    def test_a_changed_parameter_count_is_rejected(self):
        # The guard that catches the ABI drifting away from the shim header.
        text = "define ptx_kernel void @body(i32 %0, i64 %1) #0 {\n  ret void\n}\n"
        with self.assertRaises(ValueError) as raised:
            devicify_kernel(text, "body", 3)
        message = str(raised.exception)
        self.assertIn("has 2 parameters", message)
        self.assertIn("fragment's 3", message)

    def test_trailing_scratch_pointers_are_dropped(self):
        text = (
            "define ptx_kernel void @body(i32 %0, "
            "ptr addrspace(1) %1 readnone captures(none), "
            "ptr addrspace(1) %2 readnone captures(none)) "
            "local_unnamed_addr #0 {\n  ret void\n}\n"
            'attributes #0 = { nounwind "nvvm.reqntid"="32" }\n'
        )
        result = devicify_kernel(text, "body", 1)
        self.assertIn("define void @body(i32 %0) #0 {", result)
        self.assertNotIn("nvvm.reqntid", result)
        self.assertNotIn("local_unnamed_addr", result)

    def test_thread_scoped_entry_has_no_thread_zero_guard(self):
        text = (
            "define void @body(i32 %0) #0 {\n  ret void\n}\n"
            "attributes #0 = { nounwind }\n"
        )
        result = wrap_entry_point(
            text,
            "body",
            issuer_scope="thread",
        )

        self.assertNotIn("read.ptx.sreg.tid.x", result)
        self.assertNotIn("%first", result)
        self.assertIn("call void @body_body(i32 %a0)", result)


class HipFragmentAbiTest(unittest.TestCase):
    def source(self):
        return (
            'target triple = "amdgcn-amd-amdhsa"\n'
            'define amdgpu_kernel void @body(i64 inreg %0, i32 inreg %1, '
            'ptr addrspace(1) inreg readnone captures(none) %2) #0 {\n'
            '  ret void\n}\n'
            'attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,64" }\n'
        )

    def test_callable_abi_drops_kernel_uniform_arguments_and_scratch(self):
        from aboleth.hip_artifact import devicify_hip
        result = devicify_hip(self.source(), "body", 2, "thread")
        self.assertNotIn("amdgpu_kernel", result)
        self.assertNotIn("inreg", result)
        self.assertNotIn("flat-work-group-size", result)
        self.assertIn("define i64 @body(i64 %a0, i32 %a1)", result)
        self.assertNotIn("%first", result)
        self.assertNotIn("llvm.nvvm", result)

    def test_cta_guard_uses_amd_thread_identity(self):
        from aboleth.hip_artifact import devicify_hip
        result = devicify_hip(self.source(), "body", 2, "cta")
        self.assertIn("llvm.amdgcn.workitem.id.x", result)
        self.assertIn("%first", result)
        self.assertNotIn("llvm.nvvm", result)

    def test_ionic_serializes_whole_operations_within_a_wave(self):
        from aboleth.hip_artifact import devicify_hip
        result = devicify_hip(self.source(), "body", 2, "thread", serialize_lanes=True)
        self.assertIn("llvm.amdgcn.ballot.i64", result)
        self.assertIn("%serial.mask = phi i64", result)
        self.assertLess(result.index("serial.post:"), result.index("call void @body_body"))
        self.assertLess(result.index("call void @body_body"), result.index("serial.join:"))
        self.assertNotIn("addrspace(3)", result)

    def test_shared_memory_cannot_escape_into_a_callable_function(self):
        from aboleth.hip_artifact import devicify_hip
        source = ('@global_smem = external addrspace(3) global [0 x i8]\n'
                  + self.source().replace('  ret void',
                    '  %loaded = load i64, ptr addrspace(3) @global_smem\n  ret void'))
        with self.assertRaisesRegex(ValueError, "shared memory"):
            devicify_hip(source, "body", 2, "thread")

    def test_explicit_checked_publications_are_rejected_on_hip(self):
        # Test the actual lowering, not just a string guard: unchecked strong
        # publication lowers, while an explicitly conditional update cannot
        # be turned into a completion-ordered unconditional write.
        from triton._C.libtriton import ir, passes
        for backend in ("p2p", "proxy"):
            for checked in (False, True):
                with self.subTest(backend=backend, checked=checked):
                    expected = " expect %old" if checked else ""
                    types = ", i64" if checked else ""
                    source = f"""module {{
  tt.func public @publish(%endpoint: !tt.ptr<i64>, %flag: !tt.ptr<i64>,
                          %key: i32, %epoch: i64, %old: i64) {{
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %flag, %key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal = aboleth.post_signal %ep, %dst, %epoch{expected}
      : (!aboleth.endpoint, !aboleth.remote_region, i64{types}) -> !aboleth.token
    %done = aboleth.submit %ep, %signal {{batchSize = 1 : i32, deferCompletion}}
      : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token
    tt.return
  }}
}}
"""
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir") as file:
                        file.write(source)
                        file.flush()
                        context = ir.context()
                        ir.load_dialects(context)
                        module = ir.parse_mlir_module(file.name, context)
                        manager = ir.pass_manager(context)
                        arguments = (["hip-test.bc"] if backend == "proxy" else
                                     ["", "", "hip-test.bc"])
                        getattr(passes.plugin, f"add_aboleth_lower_{backend}")(
                            manager, arguments)
                        if checked:
                            with self.assertRaises(RuntimeError):
                                manager.run(module, "hip-checked-publication")
                        else:
                            manager.run(module, "hip-unchecked-publication")
                            self.assertIn("aboleth.hip-device-library", module.str_nodebug())


class InlineEmissionFragmentTest(unittest.TestCase):
    """`device_bitcode=None`: the direct emitter in detached mode.

    No device library is linked or required — deliberately not gated on
    LIBRARY. The lowered fragment contains no extern library calls, its
    reservation is a per-thread inline atomic (a scalar tt.atomic_rmw's live
    result would make Triton allocate a shared-memory broadcast slot the
    artifact guard refuses), and no publishing store carries the tid.x==0
    predicate, which would let a caller's lane id decide whether the doorbell
    rings.
    """

    def test_lowered_fragment_has_no_library_calls(self):
        lowered = lowered_program(PROGRAM, device_bitcode=None)
        self.assertEqual(device_symbols(lowered), set())
        self.assertIn("atom.relaxed.gpu.global.add.u64", lowered)
        self.assertNotIn("tt.atomic_rmw", lowered)
        self.assertNotIn("setp.eq.u32 first, tid, 0", lowered)

    def test_detached_drain_reserves_per_thread(self):
        lowered = lowered_program(
            planned_drain_program(), device_bitcode=None
        )
        self.assertEqual(device_symbols(lowered), set())
        self.assertIn("atom.relaxed.gpu.global.add.u64", lowered)
        self.assertNotIn("setp.eq.u32 first, tid, 0", lowered)

    def test_artifact_compiles_without_a_device_library(self):
        # The shared-memory guard inside compile_fragment_artifact raises if
        # the fragment wants any; a successful compile is the assertion.
        with tempfile.TemporaryDirectory(
            prefix="aboleth-inline-emission-"
        ) as scratch:
            artifact = compile_fragment_artifact(
                PROGRAM,
                arch=ARCH,
                completion_interval=4,
                device_bitcode=None,
                output_dir=scratch,
            )
            for path in artifact.files:
                self.assertTrue(path.is_file(), path)
            self.assertIn(SYMBOL.encode(), artifact.bitcode.read_bytes())
            text = artifact.ptx.read_text()
            self.assertIn("atom.relaxed.gpu.global.add.u64", text)
            self.assertNotIn("aboleth_rdma_", text)


@unittest.skipIf(
    LIBRARY is None,
    "no aboleth_rdma_<arch>.bc in the build; configure with CUDA",
)
class FragmentArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory(prefix="aboleth-probe-")
        cls.artifact = compile_fragment_artifact(
            PROGRAM,
            arch=ARCH,
            completion_interval=4,
            device_bitcode=LIBRARY.bitcode,
            output_dir=cls._directory.name,
        )

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_every_artifact_file_exists(self):
        for path in self.artifact.files:
            self.assertTrue(path.is_file(), path)

    def test_symbol_and_arch_reach_the_file_names(self):
        self.assertEqual(self.artifact.symbol, SYMBOL)
        digest = hashlib.sha256(self.artifact.bitcode.read_bytes()).hexdigest()
        self.assertEqual(
            self.artifact.bitcode.name,
            f"probe.{ARCH}.{digest[:12]}.bc",
        )

    def test_bitcode_exports_the_fragment_symbol(self):
        self.assertIn(
            SYMBOL.encode(),
            self.artifact.bitcode.read_bytes(),
        )

    def test_ptx_exports_exactly_one_visible_function(self):
        text = self.artifact.ptx.read_text()
        visible = re.findall(
            r"^\.visible \.func\s+.*?([A-Za-z_][\w$]*)\($",
            text,
            re.MULTILINE,
        )
        self.assertEqual(visible, [SYMBOL])
        self.assertIn(f".target {ARCH}", text)

    def test_ptx_declares_at_least_the_mlx5_floor(self):
        # ptx84 is the floor the mlx5 inline asm needs; llc is free to declare
        # higher when an arch's own minimum exceeds it (see the sm_100 case
        # below), but never lower.
        match = re.search(
            r"^\.version (\d+)\.(\d+)$",
            self.artifact.ptx.read_text(),
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        self.assertGreaterEqual((int(match.group(1)), int(match.group(2))), (8, 4))

    def test_exported_parameters_match_the_slots(self):
        self.assertEqual(exported_parameters(self.artifact), len(SLOTS))

    def test_slots_survive_a_folded_completion_interval(self):
        # An interval of 1 leaves the iteration slots unused inside the
        # fragment, which is exactly when an optimizer is free to drop them.
        with tempfile.TemporaryDirectory() as directory:
            folded = compile_fragment_artifact(
                PROGRAM,
                arch=ARCH,
                completion_interval=1,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
            )
            self.assertEqual(exported_parameters(folded), len(SLOTS))

    def test_shim_declares_the_abi_in_slot_order(self):
        header = self.artifact.shim_header.read_text()
        self.assertIn(
            f'extern "C" __device__ std::uint64_t {SYMBOL}(',
            header,
        )
        self.assertIn(self.artifact.binding_signature, header)
        self.assertIn("-fcuda-short-ptr", header)
        self.assertNotIn("write_rdma", header)
        declaration = header.split(f"{SYMBOL}(", 1)[1].split(");", 1)[0]
        parameters = [
            line.strip().rstrip(",").split()[-1]
            for line in declaration.strip().splitlines()
        ]
        self.assertEqual(
            parameters,
            [name for name, _ in SLOTS],
        )
        # The i1 predicate slot widens to i32 in the call ABI; the bindings
        # struct keeps it a bool and the helper narrows it.
        self.assertIn("bool predicate;", header)
        self.assertIn("bindings.predicate ? 1u : 0u", header)

    def test_recompiling_reproduces_the_same_content(self):
        with tempfile.TemporaryDirectory() as directory:
            again = compile_fragment_artifact(
                PROGRAM,
                arch=ARCH,
                completion_interval=4,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
            )
            self.assertEqual(again.bitcode.name, self.artifact.bitcode.name)
            self.assertEqual(
                again.bitcode.read_bytes(),
                self.artifact.bitcode.read_bytes(),
            )
            self.assertEqual(
                again.binding_signature,
                self.artifact.binding_signature,
            )

    def test_a_changed_program_changes_the_signature_and_the_name(self):
        with tempfile.TemporaryDirectory() as directory:
            other = compile_fragment_artifact(
                PROGRAM,
                arch=ARCH,
                completion_interval=8,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
            )
            self.assertNotEqual(
                other.binding_signature,
                self.artifact.binding_signature,
            )
            self.assertNotEqual(
                other.bitcode.name,
                self.artifact.bitcode.name,
            )


@unittest.skipIf(
    LIBRARY is None,
    "no aboleth_rdma_<arch>.bc in the build; configure with CUDA",
)
class ExternalPlannedPutArtifactTest(unittest.TestCase):
    def test_compiles_one_externally_published_put(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = compile_planned_put_artifact(
                arch=ARCH,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
                max_bytes=4096,
                cta_exclusive=True,
            )

            for path in artifact.files:
                self.assertTrue(path.is_file(), path)
            self.assertEqual(exported_parameters(artifact), 7)
            self.assertIn(
                artifact.binding_signature,
                artifact.shim_header.read_text(),
            )
            self.assertIn(
                "Each calling thread issues its own transfer",
                artifact.shim_header.read_text(),
            )
            self.assertIn(
                "reserve_planned",
                artifact.shim_header.read_text(),
            )
            self.assertIn(
                "write_rdma",
                artifact.shim_header.read_text(),
            )
            publish_artifact = compile_planned_publish_artifact(
                arch=ARCH,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
            )
            fence_artifact = compile_planned_fence_artifact(
                arch=ARCH,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
            )
            drain_artifact = compile_planned_drain_artifact(
                arch=ARCH,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
            )
            for generated in (
                publish_artifact,
                fence_artifact,
                drain_artifact,
            ):
                for path in generated.files:
                    self.assertTrue(path.is_file(), path)


@unittest.skipIf(
    LIBRARY is None,
    "no aboleth_rdma_<arch>.bc in the build; configure with CUDA",
)
@unittest.skipUnless(SM100_SUPPORTED, "ptxas cannot assemble sm_100 PTX")
class BlackwellFragmentArtifactTest(unittest.TestCase):
    """sm_100's own PTX ISA minimum (8.6) already clears the mlx5 floor
    (8.4); pinning llc to the floor regardless is what broke this target."""

    def test_realizing_a_blackwell_fragment_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = compile_fragment_artifact(
                PROGRAM,
                arch="sm_100",
                completion_interval=4,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
            )
            for path in artifact.files:
                self.assertTrue(path.is_file(), path)
            text = artifact.ptx.read_text()
            self.assertIn(".target sm_100", text)
            match = re.search(r"^\.version (\d+)\.(\d+)$", text, re.MULTILINE)
            self.assertIsNotNone(match)
            self.assertGreaterEqual((int(match.group(1)), int(match.group(2))), (8, 6))


@unittest.skipIf(
    LIBRARY is None,
    "no aboleth_rdma_<arch>.bc in the build; configure with CUDA",
)
class DeviceLibraryTest(unittest.TestCase):
    def test_missing_architecture_names_what_is_built(self):
        with self.assertRaises(FileNotFoundError) as raised:
            device_library("sm_1", generated=False)
        message = str(raised.exception)
        self.assertIn("sm_1", message)
        self.assertIn("ABOLETH_RDMA_BITCODE_ARCHS", message)

    @unittest.skipIf(LIBRARY is None, "no device library in the build")
    def test_library_points_at_the_repository_headers(self):
        self.assertTrue((LIBRARY.include_dir / "aboleth" / "device.cuh").is_file())
        self.assertEqual(
            LIBRARY.archive.name,
            f"libaboleth_device_{LIBRARY.arch}.a",
        )


if __name__ == "__main__":
    unittest.main()
