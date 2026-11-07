"""Compile one transport fragment into a linkable device artifact.

This is the detached `bindable_epilogue` artifact: communication a Pattern
declared, compiled into a device function a CUDA or Triton caller can call. The
fragment is real compiler output — the same passes the in-kernel path runs — so
an artifact and an inlined fragment emit the same wire bytes;
`experiments/python/aboleth_fragment_differential.py` is the standing proof.

The fragment is compiled as a Triton kernel and rewritten into a device
function (see emit/devicify.py), and its exported ABI is scalar throughout —
pointer slots travel as device addresses, the i1 predicate slot widens to i32 —
so that both a CUDA caller and a Triton `extern_elementwise` call site can bind
the same symbol. Every thread of the calling block may call it; the transport
is issued once, by thread 0, which is where the in-kernel path also ends up.

The same pipeline, stopped before that rewrite, is the `standalone_kernel`
artifact: the module's public `tt.func` stays a kernel, so Triton's own cubin is
the deliverable and `realize_standalone_kernel` returns it with the launch
metadata the client needs to launch it on its own stream.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from ._plugin import configure_plugin
from .emit.devicify import (
    assert_single_public_entry,
    devicify_kernel,
    wrap_entry_point,
)
from .emit.shim_cuda import emit_cuda_shim
from .pattern import KernelTransportProgram
from .transport_binding import invoke_manifest

configure_plugin()

from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.compiler import compile as triton_compile


_SUPPORTED_TRANSPORTS = frozenset(("rdma", "p2p"))
# The pointee types the RDMA lowering demands: it loads descriptor words and
# the local epoch through them, and only ptrtoints the payload sides.
_DESCRIPTOR_POINTEE = "i64"
_PAYLOAD_POINTEE = "i8"
_ABI_TYPES = {"i1": "i32", "i32": "i32", "i64": "i64"}


@dataclass(frozen=True)
class FragmentArtifact:
    """One compiled fragment: the files, the symbol, and its ABI identity."""

    name: str
    arch: str
    symbol: str
    binding_signature: str
    bitcode: pathlib.Path
    relocatable: pathlib.Path
    ptx: pathlib.Path
    shim_header: pathlib.Path

    @property
    def files(self) -> tuple[pathlib.Path, ...]:
        return (
            self.bitcode,
            self.relocatable,
            self.ptx,
            self.shim_header,
        )


@dataclass(frozen=True)
class StandaloneKernel:
    """One compiled communication kernel: its files, symbol, and launch shape.

    `arguments` are the manifest's slots in order, typed as the exported ABI
    passes them; `scratch_parameters` counts the pointer parameters Triton
    appends after them, which a launcher must still supply.
    """

    name: str
    arch: str
    symbol: str
    binding_signature: str
    cubin: pathlib.Path
    ptx: pathlib.Path
    num_warps: int
    shared_bytes: int
    arguments: tuple[tuple[str, str], ...]
    scratch_parameters: int

    @property
    def files(self) -> tuple[pathlib.Path, ...]:
        return (self.cubin, self.ptx)


def compile_fragment_artifact(
    program: KernelTransportProgram,
    *,
    arch: str,
    transport: str = "rdma",
    completion_interval: int,
    device_bitcode: str | os.PathLike | None,
    output_dir: str | os.PathLike,
    name: str | None = None,
    max_registers: int | None = None,
    cuda_inline: bool = False,
    symbol: str | None = None,
    exclusive: bool = False,
    strict_registrations: bool = False,
    gpu_resident_queues: bool = False,
) -> FragmentArtifact:
    """Compile `program` into a device artifact under `output_dir`.

    The pass pipeline is the in-kernel one minus the enclosing kernel:
    materialize-transport-phase, materialize-completion, plan-publication,
    form-batches, lower-<transport>. With `device_bitcode` set, the lowering
    targets the transport device library; with `device_bitcode=None`, it runs
    the direct emitter in detached mode — real inline WQE construction whose
    reservation is a per-thread inline atomic and whose publishing stores are
    unpredicated, so the artifact stays free of any shared-memory contract
    with its caller either way (the guard below still verifies it).

    `max_registers` caps every linked device function so the fragment remains
    linkable from clients that compile entry kernels with `--maxrregcount`.

    `strict_registrations` lets publication planning make strong signals
    fenced writes; see `compiler.compilation`.

    `gpu_resident_queues` states that every endpoint the fragment will post
    to keeps its queues in GPU memory, so its doorbell publishes with a
    GPU-scope release (the `gpu-doorbell` optimization; see
    `compiler.compilation`).

    `cuda_inline` emits the same compiler-owned transport program as inline CUDA
    in the shim, preserving fusion when a client kernel can include Aboleth's
    device header. The relocatable artifact remains the Queue IR audit path.

    `device_bitcode` is the built aboleth_rdma_<arch>.bc (see
    `aboleth.device_library(generated=False)`); it is linked into the fragment's
    own module, so it need not match `arch`. Output names are content-addressed
    by the bitcode and code-shape options, so a stale artifact cannot alias a
    fresh one.
    """
    if arch.startswith("gfx"):
        from .hip_artifact import compile_hip_fragment
        return compile_hip_fragment(
            program, arch=arch, transport=transport,
            completion_interval=completion_interval, device_bitcode=device_bitcode,
            output_dir=output_dir, name=name, max_registers=max_registers,
            cuda_inline=cuda_inline, symbol=symbol, exclusive=exclusive,
            strict_registrations=strict_registrations)
    _require_detached(program)
    _require_no_reductions(program)
    completion_interval, device_bitcode = _validated_inputs(
        transport, completion_interval, device_bitcode
    )
    if max_registers is not None and max_registers < 1:
        raise ValueError("max_registers must be positive")
    manifest = invoke_manifest(program)
    name = name or program.name
    # `symbol` overrides the default for libraries with their own naming
    # discipline (the generated Tier-1 surface exports aboleth_tier1_*).
    symbol = symbol or f"aboleth_fragment_{_symbol_stem(name)}"
    slots = _slots(program)

    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aboleth-fragment-") as scratch:
        staging = pathlib.Path(scratch)
        compiled = _compile_module(
            program,
            symbol,
            slots,
            staging,
            arch=arch,
            transport=transport,
            completion_interval=completion_interval,
            device_bitcode=device_bitcode,
            exclusive=exclusive,
            strict_registrations=strict_registrations,
            gpu_resident_queues=gpu_resident_queues,
        )
        if compiled.metadata.shared:
            raise RuntimeError(
                f"fragment {name!r} wants {compiled.metadata.shared} bytes of "
                "shared memory; an artifact may not impose a shared-memory "
                "contract on its caller"
            )
        exported = staging / "fragment.ll"
        exported.write_text(wrap_entry_point(
            devicify_kernel(
                compiled.asm["llir"],
                symbol,
                len(slots),
            ),
            symbol,
            issuer_scope=program.issuer_scope,
        ))

        bitcode = staging / "fragment.bc"
        _run((_llvm_tool("opt"), "-passes=verify", "-o", bitcode, exported))
        configuration = (
            b""
            if max_registers is None and not cuda_inline
            else (
                f":max-registers={max_registers}:"
                f"cuda-inline={cuda_inline}"
            ).encode()
        ) + (b":inline-emission" if device_bitcode is None else b"") + (
            b":exclusive-issuer" if exclusive else b"")
        digest = hashlib.sha256(
            bitcode.read_bytes() + configuration
        ).hexdigest()
        stem = f"{_symbol_stem(name)}.{arch}.{digest[:12]}"

        artifact = FragmentArtifact(
            name=name,
            arch=arch,
            symbol=symbol,
            binding_signature=_signature(
                program, manifest, symbol, arch, transport,
                completion_interval, digest,
            ),
            bitcode=output / f"{stem}.bc",
            relocatable=output / f"{stem}.cubin",
            ptx=output / f"{stem}.ptx",
            shim_header=output / f"{stem}.cuh",
        )
        shutil.copyfile(bitcode, artifact.bitcode)
        llc = _llvm_tool("llc")
        _run((
            llc, "-march=nvptx64", f"-mcpu={arch}",
            *_ptx84_floor(llc, arch), exported, "-o", artifact.ptx,
        ))
        ptxas = [_ptxas(), f"-arch={arch}", "-m64", "-c"]
        if max_registers is not None:
            ptxas.append(f"--maxrregcount={max_registers}")
        ptxas.extend((
            artifact.ptx,
            "-o", artifact.relocatable,
        ))
        _run(tuple(ptxas))

    inline_body = (
        _cuda_inline_body(program)
        if cuda_inline
        else None
    )
    artifact.shim_header.write_text(emit_cuda_shim(
        name=name,
        symbol=symbol,
        arch=arch,
        binding_signature=artifact.binding_signature,
        slots=tuple(
            (slot, mlir_type, pointee is not None)
            for slot, mlir_type, pointee in slots
        ),
        issuer_scope=program.issuer_scope,
        inline_body=inline_body,
        inline_includes=(
            ("aboleth/Target/RdmaDevice.cuh",)
            if inline_body is not None
            else ()
        ),
    ))
    return artifact


def realize_bindable_epilogue(
    physicalized: KernelTransportProgram,
    *,
    arch: str,
    device_bitcode: str | os.PathLike | None,
    output_dir: str | os.PathLike,
    completion_interval: int,
    transport: str = "rdma",
    name: str | None = None,
    max_registers: int | None = None,
    cuda_inline: bool = False,
) -> FragmentArtifact:
    """Realize a `bindable_epilogue` attachment's program as its artifact."""
    _require_artifact(physicalized, "bindable_epilogue")
    return compile_fragment_artifact(
        physicalized,
        arch=arch,
        transport=transport,
        completion_interval=completion_interval,
        device_bitcode=device_bitcode,
        output_dir=output_dir,
        name=name,
        max_registers=max_registers,
        cuda_inline=cuda_inline,
    )


def realize_standalone_kernel(
    physicalized: KernelTransportProgram,
    *,
    arch: str,
    device_bitcode: str | os.PathLike | None,
    output_dir: str | os.PathLike,
    completion_interval: int,
    transport: str = "rdma",
    name: str | None = None,
) -> StandaloneKernel:
    """Realize a `standalone_kernel` program as a launchable kernel.

    The same pipeline `compile_fragment_artifact` runs, minus devicification:
    the module's public `tt.func` stays a kernel, so Triton's own cubin is the
    deliverable and its metadata is the launch contract. Shared memory is fine
    here — a standalone kernel imposes no contract on a caller's block.
    """
    _require_artifact(physicalized, "standalone_kernel")
    _require_no_reductions(physicalized)
    completion_interval, device_bitcode = _validated_inputs(
        transport, completion_interval, device_bitcode
    )
    name = name or physicalized.name
    symbol = f"aboleth_kernel_{_symbol_stem(name)}"
    slots = _slots(physicalized)

    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aboleth-kernel-") as scratch:
        compiled = _compile_module(
            physicalized,
            symbol,
            slots,
            pathlib.Path(scratch),
            arch=arch,
            transport=transport,
            completion_interval=completion_interval,
            device_bitcode=device_bitcode,
        )
        metadata = compiled.metadata
        cubin = compiled.asm["cubin"]
        ptx = compiled.asm["ptx"]

    if metadata.name != symbol:
        raise RuntimeError(
            f"Triton named the entry point {metadata.name!r}, not the "
            f"requested {symbol!r}; the launch metadata would not find it"
        )
    parameters = _entry_parameters(ptx, symbol)
    if parameters < len(slots):
        raise RuntimeError(
            f"kernel {symbol!r} exports {parameters} parameters, fewer than "
            f"the manifest's {len(slots)} slots"
        )
    digest = hashlib.sha256(cubin).hexdigest()
    stem = f"{_symbol_stem(name)}.{arch}.{digest[:12]}"
    kernel = StandaloneKernel(
        name=name,
        arch=arch,
        symbol=symbol,
        binding_signature=_signature(
            physicalized, invoke_manifest(physicalized), symbol, arch,
            transport, completion_interval, digest,
        ),
        cubin=output / f"{stem}.cubin",
        ptx=output / f"{stem}.ptx",
        num_warps=metadata.num_warps,
        shared_bytes=metadata.shared,
        arguments=tuple(
            (slot, _ABI_TYPES[mlir_type])
            for slot, mlir_type, _ in slots
        ),
        scratch_parameters=parameters - len(slots),
    )
    kernel.cubin.write_bytes(cubin)
    kernel.ptx.write_text(ptx)
    return kernel


def _require_artifact(
    program: KernelTransportProgram,
    kind: str,
) -> None:
    if program.artifact != kind:
        raise ValueError(
            f"program {program.name!r} carries artifact kind "
            f"{program.artifact!r}, not {kind!r}; physicalize the attachment "
            f"with Artifact.{kind.upper()} to realize it this way"
        )


# The kinds physicalization gates on the neighbouring stages' capabilities.
_DETACHED_KINDS = ("bindable_epilogue", "standalone_kernel")


def _require_detached(program: KernelTransportProgram) -> None:
    """Refuse to detach a program whose neighbours never permitted it.

    `inline_fragment` is the one kind physicalization lets through ungated,
    because an inline fragment is emitted into a kernel the client already
    compiles through Aboleth. Compiling one into a standalone file is exactly
    the artifact those gates exist to withhold, so the kind is checked here
    too and not only in `realize_bindable_epilogue`.
    """
    if program.artifact not in _DETACHED_KINDS:
        raise ValueError(
            f"program {program.name!r} carries artifact kind "
            f"{program.artifact!r}, which no detached artifact may be "
            "compiled from; physicalize the attachment with "
            "Artifact.BINDABLE_EPILOGUE or Artifact.STANDALONE_KERNEL and "
            "realize it through realize_bindable_epilogue or "
            "realize_standalone_kernel"
        )


def _require_no_reductions(program: KernelTransportProgram) -> None:
    """Refuse a program whose reduce contracts the fragment would drop.

    A reduce contract reaches neither `KernelTransportProgram.to_mlir` nor the
    fragment pipeline, so two programs differing only in their operator or
    cardinality would compile to the same bytes under the same signature.
    Until reductions lower, refusing is the only answer that does not lie.
    """
    if program.reductions:
        raise ValueError(
            f"program {program.name!r} declares reduce contracts for "
            + ", ".join(
                repr(reduction.operation)
                for reduction in program.reductions
            )
            + "; reduce contracts do not lower to fragments yet, so the "
            "artifact would drop them silently"
        )


def _validated_inputs(
    transport: str,
    completion_interval: int,
    device_bitcode: str | os.PathLike | None,
) -> tuple[int, pathlib.Path | None]:
    if transport not in _SUPPORTED_TRANSPORTS:
        raise ValueError(
            f"transport {transport!r} has no fragment device library; "
            f"expected one of {', '.join(sorted(_SUPPORTED_TRANSPORTS))}"
        )
    completion_interval = int(completion_interval)
    if completion_interval <= 0:
        raise ValueError("completion interval must be positive")
    if device_bitcode is None:
        # Detached inline emission: the lowering runs the direct emitter and
        # links no device library.
        return completion_interval, None
    if transport != "rdma":
        raise ValueError(
            f"transport {transport!r} has no device library; compile its "
            "fragments with device_bitcode=None"
        )
    device_bitcode = pathlib.Path(device_bitcode).resolve()
    if not device_bitcode.is_file():
        raise FileNotFoundError(
            f"transport device library {device_bitcode} is not built"
        )
    return completion_interval, device_bitcode


def _compile_module(
    program: KernelTransportProgram,
    symbol: str,
    slots: tuple[tuple[str, str, str | None], ...],
    staging: pathlib.Path,
    *,
    arch: str,
    transport: str,
    completion_interval: int,
    device_bitcode: pathlib.Path | None,
    exclusive: bool = False,
    strict_registrations: bool = False,
    gpu_resident_queues: bool = False,
):
    """Materialize, lower, and compile the fragment module inside `staging`."""
    plan = staging / "transport-program.mlir"
    plan.write_text(_plan_module(program))
    module = staging / "fragment.mlir"
    source = _fragment_module(program, symbol, slots)
    assert_single_public_entry(source)
    module.write_text(source)

    lowered = staging / "fragment.ttir"
    lowered.write_text(_lower(
        module,
        plan,
        transport=transport,
        completion_interval=completion_interval,
        strict_registrations=strict_registrations,
        lower_arguments=_lower_arguments(
            program, transport, device_bitcode, exclusive,
            gpu_resident_queues, arch=arch,
        ),
    ))
    options = {"num_warps": 1}
    if device_bitcode is not None:
        options["extern_libs"] = (
            (f"aboleth_{transport}", str(device_bitcode)),
        )
    return triton_compile(
        str(lowered),
        target=GPUTarget("cuda", _capability(arch), 32),
        options=options,
    )


def _entry_parameters(ptx: str, symbol: str) -> int:
    """How many parameters the compiled entry point actually takes."""
    return len(set(re.findall(
        rf"{re.escape(symbol)}_param_(\d+)",
        ptx,
    )))


def _symbol_stem(name: str) -> str:
    return "".join(
        character if character.isalnum() else "_"
        for character in name
    )


def _capability(arch: str) -> int:
    # sm_90a and sm_100a name the same capability as sm_90/sm_100 to Triton;
    # the suffix only reaches llc and ptxas.
    match = re.fullmatch(r"sm_(\d+)[a-z]*", arch)
    if match is None:
        raise ValueError(f"expected an sm_NN architecture, got {arch!r}")
    return int(match.group(1))


def _cuda_inline_body(program: KernelTransportProgram) -> str:
    endpoint = (
        "    const auto& endpoint = *reinterpret_cast<\n"
        "        const ::aboleth::abi::RdmaEndpointDescriptor*>(\n"
        "            bindings.endpoint);\n"
    )
    rdma = "::aboleth::cuda::rdma"
    if program.system_fence:
        return (
            "    if (bindings.predicate)\n"
            "        __threadfence_system();"
        )
    if program.drain_endpoint:
        return endpoint + (
            f"    {rdma}::drain(endpoint, nullptr);"
        )
    if program.publish_head:
        return endpoint + (
            "    const auto head =\n"
            f"        {rdma}::load_acquire(\n"
            "            reinterpret_cast<const std::uint64_t*>(\n"
            "                endpoint.producer_index_address));\n"
            "    const auto published =\n"
            f"        {rdma}::load_acquire(\n"
            f"            {rdma}::completion_state(endpoint, 2));\n"
            "    if (!bindings.only_if_advanced || head > published)\n"
            f"        {rdma}::publish_head(\n"
            "            endpoint, nullptr, head);"
        )
    if (
        program.external_publication
        and program.planned_reservation
        and len(program.transfers) == 1
        and program.signal is None
    ):
        transfer = program.transfers[0]
        cta_exclusive = (
            "true"
            if program.reservation_scope == "cta"
            else "false"
        )
        return endpoint + (
            f"    if (!bindings.{transfer.predicate})\n"
            "        return;\n"
            f"    const auto index = {rdma}::reserve_planned(\n"
            f"        endpoint, {cta_exclusive}, 1);\n"
            f"    {rdma}::write_rdma(\n"
            "        endpoint,\n"
            "        index,\n"
            "        reinterpret_cast<const void*>(\n"
            f"            bindings.{transfer.source}),\n"
            f"        bindings.{transfer.source_key},\n"
            f"        bindings.{transfer.destination},\n"
            f"        bindings.{transfer.destination_key},\n"
            f"        bindings.{transfer.byte_count},\n"
            "        /*request_completion=*/false);"
        )
    raise ValueError(
        f"program {program.name!r} has no inline CUDA lowering"
    )


def _slots(
    program: KernelTransportProgram,
) -> tuple[tuple[str, str, str | None], ...]:
    """(name, mlir type, tt.ptr pointee or None) in manifest slot order.

    A program that states its own pointer classification (a raw-body program
    has no transfers or signal to derive one from) carries it as
    `pointer_pointees`: slot name -> pointee type, absent for scalars.
    """
    pointees = getattr(program, "pointer_pointees", None)
    if pointees is not None:
        stray = set(pointees) - {
            argument.name for argument in program.arguments
        }
        if stray:
            raise ValueError(
                "pointer_pointees names slots the program does not "
                "declare: " + ", ".join(sorted(stray))
            )
        result = []
        for argument in program.arguments:
            pointee = pointees.get(argument.name)
            if pointee is not None and argument.mlir_type != "i64":
                raise ValueError(
                    f"pointer slot {argument.name!r} must be i64, "
                    f"got {argument.mlir_type}"
                )
            result.append((argument.name, argument.mlir_type, pointee))
        return tuple(result)
    descriptors = {"endpoint"}
    if program.signal is not None:
        descriptors.update((
            program.signal.destination,
            program.signal.source,
        ))
    payloads = {
        pointer
        for transfer in program.transfers
        for pointer in (transfer.destination, transfer.source)
    }
    result = []
    for argument in program.arguments:
        if argument.name in descriptors:
            pointee = _DESCRIPTOR_POINTEE
        elif argument.name in payloads:
            pointee = _PAYLOAD_POINTEE
        else:
            pointee = None
        if pointee is not None and argument.mlir_type != "i64":
            raise ValueError(
                f"pointer slot {argument.name!r} must be i64, "
                f"got {argument.mlir_type}"
            )
        result.append((argument.name, argument.mlir_type, pointee))
    return tuple(result)


def _plan_module(program: KernelTransportProgram) -> str:
    return "\n".join((
        "module {",
        "  aboleth.transport_program {",
        program.to_mlir(),
        "    ub.unreachable",
        "  } {",
        f'    name = "{program.name}-fragment"',
        "  }",
        "}",
        "",
    ))


def _fragment_module(
    program: KernelTransportProgram,
    symbol: str,
    slots: tuple[tuple[str, str, str | None], ...],
) -> str:
    """The module the passes materialize the fragment into.

    One public function, which Triton compiles as a kernel and devicify then
    rewrites into the exported device function. Every parameter is a scalar of
    the exported ABI; pointers and the i1 predicate are reconstructed inside,
    so the artifact stays callable from CUDA and from Triton alike.
    """
    parameters = ", ".join(
        f"%{slot}: {_ABI_TYPES[mlir_type]}"
        for slot, mlir_type, _ in slots
    )
    body = []
    operands = []
    operand_types = []
    for slot, mlir_type, pointee in slots:
        if pointee is not None:
            body.append(
                f"    %{slot}_pointer = tt.int_to_ptr %{slot} "
                f": i64 -> !tt.ptr<{pointee}>"
            )
            operands.append(f"%{slot}_pointer")
            operand_types.append(f"!tt.ptr<{pointee}>")
        elif mlir_type == "i1":
            body.append(
                f"    %{slot}_flag = arith.cmpi ne, %{slot}, %zero : i32"
            )
            operands.append(f"%{slot}_flag")
            operand_types.append("i1")
        else:
            operands.append(f"%{slot}")
            operand_types.append(mlir_type)
    return "\n".join((
        "module {",
        f"  tt.func public @{symbol}({parameters}) "
        "attributes {noinline = false} {",
        "    %zero = arith.constant 0 : i32",
        f"    %program = arith.constant {program.program_id} : i32",
        *body,
        '    "aboleth.invoke_transport_phase"(%program, '
        f'{", ".join(operands)})',
        f'      : (i32, {", ".join(operand_types)}) -> ()',
        "    tt.return",
        "  }",
        "}",
        "",
    ))


def _lower_arguments(
    program,
    transport: str,
    device_bitcode: pathlib.Path | None,
    exclusive: bool = False,
    gpu_resident_queues: bool = False,
    *,
    arch: str | None = None,
) -> list[str]:
    """The transport lowering's pass arguments.

    RDMA: named options (`aboleth.compiler.rdma_lowering_options`); a
    `bitcode-path` selects library-call mode, `detached` the direct emitter
    with caller-agnostic emission, and `exclusive` states the
    single-issuer-per-endpoint contract (no reservation atomic, no
    publication-order spin), and `gpu_resident_queues` a GPU-scope doorbell.
    P2P has no library — always detached here — and
    takes its copy mode from the program (`p2p_copy_mode`: '', 'scalar',
    'wide', or 'warp').
    """
    if transport == "rdma":
        from .compiler import enabled_optimizations, rdma_lowering_options
        if device_bitcode is None:
            return rdma_lowering_options(
                enabled_optimizations(
                    capability=_capability(arch) if arch is not None else None
                ),
                detached=True, exclusive=exclusive,
                gpu_resident_queues=gpu_resident_queues)
        if exclusive:
            raise ValueError(
                "exclusive-issuer emission needs the direct emitter "
                "(device_bitcode=None)")
        return [f"bitcode-path={device_bitcode}"]
    if transport == "p2p":
        return [getattr(program, "p2p_copy_mode", ""), "detached"]
    raise ValueError(f"transport {transport!r} has no fragment lowering")


def _lower(
    module: pathlib.Path,
    plan: pathlib.Path,
    *,
    transport: str,
    completion_interval: int,
    lower_arguments: list[str],
    strict_registrations: bool = False,
) -> str:
    context = ir.context()
    ir.load_dialects(context)
    parsed = ir.parse_mlir_module(str(module), context)
    manager = ir.pass_manager(context)
    passes.plugin.add_aboleth_materialize_transport_phase(
        manager,
        [str(plan)],
    )
    # Completion placement has to see the materialized fragment and precede
    # batch formation, which fuses the reservations it plans around.
    passes.plugin.add_aboleth_materialize_completion(
        manager,
        [str(completion_interval)],
    )
    passes.plugin.add_aboleth_plan_publication(
        manager,
        [transport, "strict" if strict_registrations else ""],
    )
    passes.plugin.add_aboleth_form_batches(manager)
    getattr(passes.plugin, f"add_aboleth_lower_{transport}")(
        manager, lower_arguments
    )
    manager.run(parsed, "aboleth-fragment")
    return parsed.str_nodebug()



def _signature(
    program: KernelTransportProgram,
    manifest,
    symbol: str,
    arch: str,
    transport: str,
    completion_interval: int,
    digest: str,
) -> str:
    payload = {
        "name": program.name,
        "symbol": symbol,
        "arch": arch,
        "transport": transport,
        "completion_interval": completion_interval,
        "program_id": manifest.program_id,
        "slots": [
            (argument.name, argument.mlir_type)
            for argument in manifest.arguments
        ],
        "transfers": [
            {
                "operations": list(transfer.operations),
                "destination": transfer.destination,
                "source": transfer.source,
                "byte_count": transfer.byte_count,
                "source_key": transfer.source_key,
                "destination_key": transfer.destination_key,
                "predicate": transfer.predicate,
                "max_bytes": transfer.max_bytes,
            }
            for transfer in manifest.transfers
        ],
        # The manifest carries no reduce contract, and neither does the
        # fragment: two programs differing only in their operator would
        # otherwise share one signature the day reductions start lowering.
        # Nothing can be compiled with one yet, so this is the cheap moment to
        # put the field in.
        "reductions": [
            {
                "operation": reduction.operation,
                "operator": reduction.operator,
                "cardinality": reduction.cardinality,
                "cardinality_source": reduction.cardinality_source,
            }
            for reduction in program.reductions
        ],
        "signal": (
            {
                "destination": manifest.signal.destination,
                "source": manifest.signal.source,
                "source_key": manifest.signal.source_key,
                "destination_key": manifest.signal.destination_key,
                "iteration": manifest.signal.iteration,
                "iterations": manifest.signal.iterations,
            }
            if manifest.signal is not None
            else None
        ),
        "external_publication": program.external_publication,
        "system_fence": program.system_fence,
        "drain_endpoint": program.drain_endpoint,
        "publish_head": program.publish_head,
        "planned_reservation": program.planned_reservation,
        "reservation_scope": program.reservation_scope,
        "issuer_scope": program.issuer_scope,
        "bitcode": digest,
    }
    # A raw-body program's shape lives in its body text, not in transfers or
    # flags; entered only when present so every prior signature is unchanged.
    raw_body = getattr(program, "body", None)
    if raw_body is not None:
        payload["raw_body"] = raw_body
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_dir() -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parents[2]
    return pathlib.Path(
        os.environ.get("ABOLETH_BUILD_DIR", root / "build")
    )


def _cache_variable(name: str) -> str | None:
    cache = _build_dir() / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text().splitlines():
        variable, _, value = line.partition("=")
        if variable.split(":")[0] == name:
            return value
    return None


def _llvm_tool(name: str) -> str:
    """opt, llc, and clang++ come from the LLVM distribution that built
    libtriton; the build records two of them and the rest sit beside."""
    configured = os.environ.get(f"ABOLETH_LLVM_{name.upper()}")
    if configured:
        return configured
    recorded = _cache_variable("ABOLETH_LLVM_OPT")
    if recorded:
        candidate = pathlib.Path(recorded).parent / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"{name} not found; set ABOLETH_LLVM_{name.upper()} or configure "
            "the build with -DABOLETH_LLVM_OPT="
        )
    return found


@functools.lru_cache(maxsize=None)
def _llc_default_ptx_isa(llc: str, arch: str) -> tuple[int, int]:
    """The `.version` llc emits for `arch` with no -mattr pinning one.

    llc knows each target's own PTX ISA minimum better than a hardcoded
    sm_NN-to-version table would (and such a table would go stale on new
    chips), so probing a trivial module is how `_ptx84_floor` below finds it.
    """
    result = subprocess.run(
        (llc, "-march=nvptx64", f"-mcpu={arch}", "-o", "-", "-"),
        input='target triple = "nvptx64-nvidia-cuda"\n',
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llc failed probing the default PTX ISA for {arch}: "
            f"{result.stderr.strip()}"
        )
    match = re.search(r"\.version (\d+)\.(\d+)", result.stdout)
    if match is None:
        raise RuntimeError(
            f"llc's probe output for {arch} has no .version directive"
        )
    return int(match.group(1)), int(match.group(2))


def _ptx84_floor(llc: str, arch: str) -> tuple[str, ...]:
    """-mattr=+ptx84 flags, but only where `arch`'s own minimum falls short.

    ptx84 is the floor the mlx5 inline asm needs; ptxas accepts any newer PTX
    ISA than the one the module declares. Pinning it unconditionally is what
    broke sm_100, whose minimum is 8.6 — llc refuses to emit PTX below a
    target's own floor.
    """
    if _llc_default_ptx_isa(llc, arch) >= (8, 4):
        return ()
    return ("-mattr=+ptx84",)


def _ptxas() -> str:
    from triton import knobs

    return knobs.nvidia.ptxas.path


def _run(command) -> None:
    result = subprocess.run(
        [str(argument) for argument in command],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{pathlib.Path(str(command[0])).name} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
