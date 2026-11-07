"""HIP fragment ABI realization; the transport program still lowers in MLIR."""
from pathlib import Path
import hashlib
import re
import shutil
import tempfile

from .emit.devicify import _find_define, strip_metadata, wrap_entry_point
from .emit.shim_cuda import emit_cuda_shim


def devicify_hip(text, symbol, parameters, issuer_scope, *, serialize_lanes=False):
    match = _find_define(text, symbol)
    if "amdgpu_kernel" not in match.group(0):
        raise ValueError("HIP fragment must start with the amdgpu_kernel ABI")
    arguments = [argument.strip() for argument in match.group("args").split(",")]
    while len(arguments) > parameters and "readnone captures(none)" in arguments[-1]:
        arguments.pop()
    if len(arguments) != parameters:
        raise ValueError(f"HIP fragment has {len(arguments)} ABI slots, expected {parameters}")
    arguments = [re.sub(r"\binreg\s+", "", argument) for argument in arguments]
    text = (text[:match.start()] + f"define void @{symbol}({', '.join(arguments)}) "
            + match.group("post").replace("local_unnamed_addr", "").strip() + " {"
            + text[match.end():])
    text = strip_metadata(text)
    if text.count("@global_smem") > 1:
        raise ValueError("HIP detached fragment must not use shared memory")
    text = re.sub(r"^@global_smem = .*\n", "", text, flags=re.MULTILINE)
    text = text.replace("__hip_cuid_", "__hip_cuid_" + symbol + "_")
    for attribute in ("amdgpu-flat-work-group-size", "amdgpu-implicitarg-num-bytes",
                      "uniform-work-group-size"):
        text = re.sub(r'\s*"' + attribute + r'"="[^"]*"', "", text)
    # Scalar exported arguments and an i64 return match both HIP C++ and
    # Triton extern calls. CTA scope adds an AMD thread-0 guard explicitly.
    text = wrap_entry_point(text, symbol, issuer_scope="thread")
    if issuer_scope == "cta":
        intrinsic = "llvm.amdgcn.workitem.id.x"
        if f"@{intrinsic}(" not in text:
            text += f"\ndeclare i32 @{intrinsic}()\n"
        needle = f"  call void @{symbol}_body("
        offset = text.index(needle, text.index(f"define i64 @{symbol}("))
        end = text.index("\n", offset)
        text = (text[:offset]
                + f"  %tid = call i32 @{intrinsic}()\n"
                + "  %first = icmp eq i32 %tid, 0\n"
                + "  br i1 %first, label %post, label %done\npost:\n"
                + text[offset:end] + "\n  br label %done\ndone:"
                + text[end:])
    elif issuer_scope != "thread":
        raise ValueError(f"unknown HIP issuer scope {issuer_scope!r}")
    if serialize_lanes and issuer_scope == "thread":
        # A wave cannot spin in reserve while another of its lanes holds the
        # lock until commit: it must execute one caller's whole op first.
        # Iterate the entry EXEC mask, so the usual single-caller case takes
        # one turn. Other waves/CTAs serialize through the endpoint lock.
        text = re.sub(r'\s*"amdgpu-no-workitem-id-x"', "", text)
        wrapper = _find_define(text, symbol)
        text = text[:wrapper.end()] + "\nentry:" + text[wrapper.end():]
        offset = text.index(f"  call void @{symbol}_body(", wrapper.end())
        end = text.index("\n", offset)
        invocation = text[offset:end]
        text = (text[:offset]
                + "  %serial.tid = call i32 @llvm.amdgcn.workitem.id.x()\n"
                + "  %serial.lane = and i32 %serial.tid, 63\n"
                + "  %serial.active = call i64 @llvm.amdgcn.ballot.i64(i1 true)\n"
                + "  br label %serial.loop\nserial.loop:\n"
                + "  %serial.mask = phi i64 [%serial.active, %entry], [%serial.next, %serial.join]\n"
                + "  %serial.first64 = call i64 @llvm.cttz.i64(i64 %serial.mask, i1 true)\n"
                + "  %serial.first = trunc i64 %serial.first64 to i32\n"
                + "  %serial.mine = icmp eq i32 %serial.lane, %serial.first\n"
                + "  br i1 %serial.mine, label %serial.post, label %serial.join\nserial.post:\n"
                + invocation + "\n  br label %serial.join\nserial.join:\n"
                + "  %serial.less = sub i64 %serial.mask, 1\n"
                + "  %serial.next = and i64 %serial.mask, %serial.less\n"
                + "  %serial.done = icmp eq i64 %serial.next, 0\n"
                + "  br i1 %serial.done, label %serial.exit, label %serial.loop\nserial.exit:"
                + text[end:])
        for declaration in ("declare i32 @llvm.amdgcn.workitem.id.x()",
                            "declare i64 @llvm.amdgcn.ballot.i64(i1)",
                            "declare i64 @llvm.cttz.i64(i64, i1 immarg)"):
            name = declaration.split("@", 1)[1].split("(", 1)[0]
            if not re.search(r"^declare .*@" + re.escape(name) + r"\(", text, re.MULTILINE):
                text += "\n" + declaration + "\n"
    return text


def compile_hip_fragment(program, *, arch, transport, completion_interval,
                         device_bitcode, output_dir, name=None,
                         max_registers=None, cuda_inline=False, symbol=None,
                         exclusive=False, strict_registrations=False):
    from . import artifact as common
    from .transport_binding import invoke_manifest
    from triton import compile as triton_compile
    from triton.backends.compiler import GPUTarget
    if arch != "gfx950" or transport not in ("p2p", "proxy", "ionic"):
        raise ValueError("HIP fragments currently support gfx950 p2p/proxy/ionic")
    if max_registers is not None:
        raise ValueError("HIP fragment register ceilings are not implemented; use None")
    if device_bitcode is not None or cuda_inline:
        raise ValueError("HIP fragments use native linked helpers, without CUDA inline emission")
    common._require_detached(program)
    common._require_no_reductions(program)
    if completion_interval < 1:
        raise ValueError("completion interval must be positive")
    bitcode = Path(__file__).with_name("aboleth_hip.bc")
    if not bitcode.is_file():
        raise FileNotFoundError("build the HIP runtime device bitcode first")
    name = name or program.name
    symbol = symbol or f"aboleth_fragment_{common._symbol_stem(name)}"
    slots = common._slots(program)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aboleth-hip-fragment-") as scratch:
        staging = Path(scratch)
        plan = staging / "plan.mlir"
        plan.write_text(common._plan_module(program))
        module = staging / "fragment.mlir"
        module.write_text(common._fragment_module(program, symbol, slots))
        lowered = staging / "fragment.ttir"
        arguments = ([str(bitcode), "detached"] if transport == "proxy" else
                     [getattr(program, "p2p_copy_mode", ""), "detached", str(bitcode)])
        libraries = [("aboleth_hip", str(bitcode))]
        if transport == "ionic":
            ionic = Path(__file__).with_name("aboleth_ionic.bc")
            if not ionic.is_file():
                raise FileNotFoundError("build Ionic device bitcode first")
            libraries.append(("aboleth_ionic", str(ionic)))
            arguments = [str(ionic), "detached"]
        lowered.write_text(common._lower(
            module, plan, transport=transport, completion_interval=completion_interval,
            lower_arguments=arguments, strict_registrations=strict_registrations))
        compiled = triton_compile(str(lowered), target=GPUTarget("hip", arch, 64),
                                 options={"num_warps": 1,
                                          "extern_libs": tuple(libraries)})
        if compiled.metadata.shared:
            raise RuntimeError(f"HIP detached fragment needs {compiled.metadata.shared} bytes LDS")
        exported = staging / "fragment.ll"
        exported.write_text(devicify_hip(compiled.asm["llir"], symbol, len(slots),
                                         program.issuer_scope,
                                         serialize_lanes=transport == "ionic"))
        verified = staging / "fragment.bc"
        common._run((common._llvm_tool("opt"), "-passes=verify", "-o", verified, exported))
        digest = hashlib.sha256(verified.read_bytes()).hexdigest()
        stem = f"{common._symbol_stem(name)}.{arch}.{digest[:12]}"
        artifact = common.FragmentArtifact(
            name=name, arch=arch, symbol=symbol,
            binding_signature=common._signature(
                program, invoke_manifest(program), symbol, arch, transport,
                completion_interval, digest),
            bitcode=output / f"{stem}.bc", relocatable=output / f"{stem}.o",
            ptx=output / f"{stem}.s", shim_header=output / f"{stem}.cuh")
        shutil.copyfile(verified, artifact.bitcode)
        shutil.copyfile(exported, output / f"{stem}.ll")
        for kind, destination in (("obj", artifact.relocatable), ("asm", artifact.ptx)):
            common._run((common._llvm_tool("llc"), "-march=amdgcn", f"-mcpu={arch}",
                         f"-filetype={kind}", exported, "-o", destination))
    header = emit_cuda_shim(
        name=name, symbol=symbol, arch=arch, binding_signature=artifact.binding_signature,
        slots=tuple((slot, type_, pointee is not None) for slot, type_, pointee in slots),
        issuer_scope=program.issuer_scope)
    start = header.index("// Link one of the artifacts")
    end = header.index("// Generated by", start)
    header = (header[:start] + "// Link with hipcc --hip-device-lib-path=<directory> "
              "--hip-device-lib=<fragment>.bc.\n//\n" + header[end:])
    artifact.shim_header.write_text(header)
    return artifact
