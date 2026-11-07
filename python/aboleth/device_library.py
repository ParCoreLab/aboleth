"""Generate the Tier-1 device library from the compiler.

`generate_device_library` compiles every Tier-1 op program (python/aboleth/
tier1.py) with detached inline emission, links the per-op bitcode into one
`aboleth_tier1_<arch>.bc`, archives relocatable objects into
`libaboleth_tier1_<arch>.a` for nvcc `-rdc` clients, and emits the generated
`aboleth/device.cuh` bindings header. Generation needs Triton, the LLVM tools
that built it, and (for the archive) nvcc — but no GPU and no connected
World: the default artifact is geography-free and contains RDMA+P2P;
`World.device_library(specialize=True)` bakes the connected world's complete
peer-to-transport geography into the bindings (uniform worlds naturally fold
to one backend).

`python -m aboleth.device_library --arch sm_90a --output-dir DIR` is the
build-system entry point.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Mapping

from .artifact import FragmentArtifact, compile_fragment_artifact, _llvm_tool
from .emit.tier1_header import ABI_VERSION, emit_tier1_header
from .tier1 import (
    DEFAULT_PUT_MANY_SIZES,
    TIER1_MAX_REGISTERS,
    tier1_programs,
    tier1_symbol,
)

DEFAULT_TRANSPORTS = ("rdma", "p2p")
HIP_DEFAULT_TRANSPORTS = ("p2p", "proxy")
_SUPPORTED_TRANSPORTS = (*DEFAULT_TRANSPORTS, "proxy", "ionic")
PeerTransports = tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class GeneratedDeviceLibrary:
    """One generated Tier-1 library build: files, ops, and identity.

    `op_bitcodes` is the primary bitcode surface: one module per (transport,
    op), which is what Triton `extern_libs` (multiple entries), `cute.extern`
    (per-symbol sources), and clang `-mlink-builtin-bitcode` (repeatable,
    only-needed semantics) each consume directly. `bitcode` is the
    llvm-linked convenience single file, present only where a real
    `llvm-link` exists — Triton's trimmed LLVM distribution ships none, and
    its own linker binding links only-needed, which would drop every
    unreferenced op.

    `ops` and `op_bitcodes` are keyed transport -> op.
    """

    arch: str
    transports: tuple[str, ...]
    peer_transports: PeerTransports | None
    bitcode: pathlib.Path | None
    op_bitcodes: Mapping[str, Mapping[str, pathlib.Path]]
    archive: pathlib.Path | None
    include_dir: pathlib.Path
    header: pathlib.Path
    manifest: pathlib.Path
    digest: str
    ops: Mapping[str, Mapping[str, str]]
    endpoints_per_peer: int = 1


def _nvcc() -> str | None:
    configured = os.environ.get("ABOLETH_NVCC")
    if configured:
        return configured
    return shutil.which("nvcc")


def _llvm_link() -> str | None:
    try:
        return _llvm_tool("llvm-link")
    except FileNotFoundError:
        return None


def _run(command, cwd=None) -> None:
    result = subprocess.run(
        [str(argument) for argument in command],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{pathlib.Path(str(command[0])).name} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _normalize_configuration(
    transports: tuple[str, ...],
    peer_transports: PeerTransports | None,
) -> tuple[tuple[str, ...], PeerTransports | None]:
    """Canonicalize one generic or geography-specialized build request."""
    transports = tuple(transports)
    if not transports or len(set(transports)) != len(transports):
        raise ValueError("transports must be a non-empty set")
    unknown = set(transports) - set(_SUPPORTED_TRANSPORTS)
    if unknown:
        raise ValueError(
            "transport " + ", ".join(repr(name) for name in sorted(unknown))
            + " has no generated Tier-1 library yet; expected only "
            + ", ".join(_SUPPORTED_TRANSPORTS)
        )
    # A transport set has one identity independent of caller ordering.
    transports = tuple(
        name for name in _SUPPORTED_TRANSPORTS if name in transports
    )
    if peer_transports is None:
        return transports, None
    if not peer_transports:
        raise ValueError("peer_transports must name at least one peer")
    normalized = []
    for entry in peer_transports:
        try:
            peer, transport = entry
        except (TypeError, ValueError) as error:
            raise ValueError(
                "peer_transports entries must be (rank, transport) pairs"
            ) from error
        if isinstance(peer, bool) or not isinstance(peer, int) or peer < 0:
            raise ValueError("peer transport ranks must be non-negative ints")
        if not isinstance(transport, str):
            raise ValueError("peer transport names must be strings")
        normalized.append((peer, transport))
    normalized.sort()
    if len({peer for peer, _ in normalized}) != len(normalized):
        raise ValueError("peer_transports cannot name one rank twice")
    geography_transports = {transport for _, transport in normalized}
    if geography_transports != set(transports):
        raise ValueError(
            "peer_transports must use every and only compiled transport; "
            f"geography has {sorted(geography_transports)}, library has "
            f"{list(transports)}"
        )
    return transports, tuple(normalized)


def library_identity(
    *,
    arch: str,
    transports: tuple[str, ...],
    put_many_sizes: tuple[int, ...],
    completion_interval: int,
    max_registers: int,
    peer_transports: PeerTransports | None = None,
    exclusive: bool = False,
    endpoints_per_peer: int = 1,
    strict_registrations: bool = False,
    gpu_resident_queues: bool = False,
) -> str:
    """The content key for one library build, computable before compiling.

    Folds the op program bodies (via tier1_programs), the options, the ABI
    version, and the compiler plugin hash — a rebuilt plugin or an edited op
    program can never alias a cached library.
    """
    from .artifact import _capability
    from .compiler import _plugin_hash, enabled_optimizations

    hip = arch.startswith("gfx")
    if hip and transports == DEFAULT_TRANSPORTS:
        transports = HIP_DEFAULT_TRANSPORTS
    if hip and max_registers == TIER1_MAX_REGISTERS:
        max_registers = None  # The NVIDIA register ceiling has no HIP equivalent.
    transports, peer_transports = _normalize_configuration(
        transports, peer_transports
    )
    programs = {
        f"{transport}:{op}":
            getattr(program, "body", None) or program.to_mlir()
        for transport in transports
        for op, program in tier1_programs(
            transport, put_many_sizes=put_many_sizes,
            hip_device_library=(str(pathlib.Path(__file__).with_name("aboleth_hip.bc"))
                                if hip else None)
        ).items()
    }
    payload = json.dumps(
        {
            "abi": ABI_VERSION,
            "arch": arch,
            "transports": list(transports),
            "peer_transports": peer_transports,
            "put_many_sizes": list(put_many_sizes),
            "completion_interval": completion_interval,
            "max_registers": max_registers,
            "exclusive": exclusive,
            "endpoints_per_peer": endpoints_per_peer,
            "strict_registrations": strict_registrations,
            "plugin": _plugin_hash(),
            "rdma_emission": sorted(
                enabled_optimizations(capability=_capability(arch))
                & ({"invariant-descriptors", "vector-wqe-stores"}
                   | ({"gpu-doorbell"} if gpu_resident_queues else set()))),
            "programs": programs,
        },
        sort_keys=True,
    )
    if hip:
        native = pathlib.Path(__file__).with_name("aboleth_hip.bc")
        payload += hashlib.sha256(native.read_bytes()).hexdigest()
        if "ionic" in transports:
            ionic = pathlib.Path(__file__).with_name("aboleth_ionic.bc")
            payload += hashlib.sha256(ionic.read_bytes()).hexdigest()
        for emitter in ("hip_artifact.py", "emit/devicify.py", "emit/tier1_header.py"):
            payload += hashlib.sha256(pathlib.Path(__file__).with_name(
                "hip_artifact.py").parent.joinpath(emitter).read_bytes()).hexdigest()
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_device_library(
    arch: str,
    *,
    output_dir: str | os.PathLike,
    transports: tuple[str, ...] = DEFAULT_TRANSPORTS,
    peer_transports: PeerTransports | None = None,
    put_many_sizes: tuple[int, ...] = DEFAULT_PUT_MANY_SIZES,
    completion_interval: int = 1,
    max_registers: int = TIER1_MAX_REGISTERS,
    require_archive: bool = False,
    exclusive: bool = False,
    endpoints_per_peer: int = 1,
    strict_registrations: bool = False,
    gpu_resident_queues: bool = False,
) -> GeneratedDeviceLibrary:
    """Generate the Tier-1 library for `arch` under `output_dir`.

    By default the artifact contains both RDMA and P2P and dispatches on the
    tagged endpoint handle. `peer_transports=((rank, transport), ...)` creates
    a geography specialization: mixed builds dispatch on peer rank (which a
    constant/template peer folds away), never on a runtime transport tag.
    Passing one transport creates the naturally branch-free uniform case.
    `endpoints_per_peer` bakes the connect-time fan-out stride into the
    header's endpoint-table indexing (see emit_tier1_header).
    `strict_registrations` builds strong signals as fenced writes, legal only
    when every registration they publish into is strictly ordered; the
    default publishes with a responder compare-and-swap.
    `gpu_resident_queues` builds the rdma ops' doorbell as a GPU-scope release
    (the `gpu-doorbell` optimization), legal only when every rdma endpoint the
    library posts to keeps its queues in GPU memory (`World.device_library`
    states it from the connected endpoints).
    """
    hip = arch.startswith("gfx")
    if hip and transports == DEFAULT_TRANSPORTS:
        transports = HIP_DEFAULT_TRANSPORTS
    if hip and max_registers == TIER1_MAX_REGISTERS:
        max_registers = None  # The NVIDIA register ceiling has no HIP equivalent.
    transports, peer_transports = _normalize_configuration(
        transports, peer_transports
    )
    if endpoints_per_peer < 1:
        raise ValueError("endpoints_per_peer must be >= 1")

    output = pathlib.Path(output_dir)
    artifacts_dir = output / "artifacts"
    include_dir = output / "include"
    header = include_dir / "aboleth" / "device.cuh"
    output.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    header.parent.mkdir(parents=True, exist_ok=True)

    digest = library_identity(
        arch=arch,
        transports=transports,
        put_many_sizes=put_many_sizes,
        completion_interval=completion_interval,
        max_registers=max_registers,
        peer_transports=peer_transports,
        exclusive=exclusive,
        endpoints_per_peer=endpoints_per_peer,
        strict_registrations=strict_registrations,
        gpu_resident_queues=gpu_resident_queues,
    )

    artifacts: dict[str, dict[str, FragmentArtifact]] = {}
    for transport in transports:
        artifacts[transport] = {}
        for op, program in tier1_programs(
            transport, put_many_sizes=put_many_sizes,
            hip_device_library=(str(pathlib.Path(__file__).with_name("aboleth_hip.bc"))
                                if hip else None)
        ).items():
            artifacts[transport][op] = compile_fragment_artifact(
                program,
                arch=arch,
                transport=transport,
                completion_interval=completion_interval,
                device_bitcode=None,
                output_dir=artifacts_dir,
                max_registers=max_registers,
                symbol=tier1_symbol(op, transport),
                exclusive=exclusive and transport == "rdma",
                strict_registrations=strict_registrations,
                gpu_resident_queues=(gpu_resident_queues
                                     and transport == "rdma"),
            )

    every = [
        artifact
        for per_transport in artifacts.values()
        for artifact in per_transport.values()
    ]

    bitcode: pathlib.Path | None = output / f"aboleth_tier1_{arch}.bc"
    linker = _llvm_link()
    if linker is None and hip:
        candidate = pathlib.Path(os.environ.get("ROCM_PATH", "/opt/rocm/core-7.14")) / "lib/llvm/bin/llvm-link"
        linker = str(candidate) if candidate.is_file() else None
    if linker is None:
        bitcode = None
    else:
        _run((
            linker,
            *(artifact.bitcode for artifact in every),
            "-o", bitcode,
        ))

    archive: pathlib.Path | None = output / f"libaboleth_tier1_{arch}.a"
    nvcc = None if hip else _nvcc()
    if hip and require_archive:
        raise ValueError("HIP Tier 1 links the combined device bitcode; archive output is not implemented")
    if nvcc is None:
        if require_archive:
            raise FileNotFoundError(
                "nvcc not found; set ABOLETH_NVCC or drop require_archive"
            )
        archive = None
    else:
        virtual = arch.replace("sm_", "compute_")
        objects = []
        for transport, per_transport in artifacts.items():
            for op, artifact in per_transport.items():
                # nvcc, not ptxas: nvlink cannot search an archive of bare
                # cubins — members must be host objects carrying relocatable
                # device code (see cmake/AbolethBitcode.cmake).
                member = artifacts_dir / f"{transport}.{op}.{arch}.o"
                # -fPIC so the archive links into shared objects as well as
                # executables; the device code is unaffected.
                _run((
                    nvcc, f"-gencode=arch={virtual},code={arch}",
                    "-rdc=true", "-dc", "-Xcompiler", "-fPIC",
                    artifact.ptx, "-o", member,
                ))
                objects.append(member)
        archive.unlink(missing_ok=True)
        _run(("ar", "qcs", archive, *objects))

    symbols = {
        transport: {
            op: artifact.symbol for op, artifact in per_transport.items()
        }
        for transport, per_transport in artifacts.items()
    }
    header.write_text(emit_tier1_header(
        arch=arch,
        transports=transports,
        symbols=symbols,
        put_many_sizes=put_many_sizes,
        library_digest=digest[:12],
        peer_transports=peer_transports,
        endpoints_per_peer=endpoints_per_peer,
    ))

    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(
        {
            "abi": ABI_VERSION,
            "digest": digest,
            "arch": arch,
            "transports": list(transports),
            "peer_transports": peer_transports,
            "put_many_sizes": list(put_many_sizes),
            "completion_interval": completion_interval,
            "max_registers": max_registers,
            "exclusive": exclusive,
            "endpoints_per_peer": endpoints_per_peer,
            "gpu_resident_queues": gpu_resident_queues,
            "bitcode": bitcode.name if bitcode else None,
            "archive": archive.name if archive else None,
            "header": str(header.relative_to(output)),
            "ops": {
                transport: {
                    op: {
                        "symbol": artifact.symbol,
                        "binding_signature": artifact.binding_signature,
                        "bitcode": artifact.bitcode.name,
                    }
                    for op, artifact in per_transport.items()
                }
                for transport, per_transport in artifacts.items()
            },
        },
        indent=2,
        sort_keys=True,
    ) + "\n")

    return GeneratedDeviceLibrary(
        arch=arch,
        transports=transports,
        peer_transports=peer_transports,
        bitcode=bitcode,
        op_bitcodes={
            transport: {
                op: artifact.bitcode
                for op, artifact in per_transport.items()
            }
            for transport, per_transport in artifacts.items()
        },
        archive=archive,
        include_dir=include_dir,
        header=header,
        manifest=manifest,
        digest=digest,
        ops=symbols,
        endpoints_per_peer=endpoints_per_peer,
    )


def _load(output: pathlib.Path) -> GeneratedDeviceLibrary | None:
    manifest = output / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        recorded = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return None
    header = output / recorded["header"]
    bitcode = (
        output / recorded["bitcode"] if recorded.get("bitcode") else None
    )
    archive = (
        output / recorded["archive"] if recorded.get("archive") else None
    )
    try:
        peer_transports = recorded.get("peer_transports")
        if peer_transports is not None:
            peer_transports = tuple(
                (int(peer), str(transport))
                for peer, transport in peer_transports
            )
        op_bitcodes = {
            transport: {
                op: output / "artifacts" / entry["bitcode"]
                for op, entry in per_transport.items()
            }
            for transport, per_transport in recorded["ops"].items()
        }
        ops = {
            transport: {
                op: entry["symbol"]
                for op, entry in per_transport.items()
            }
            for transport, per_transport in recorded["ops"].items()
        }
    except (KeyError, TypeError, ValueError, AttributeError):
        # A manifest from before the per-transport nesting: rebuild.
        return None
    if not header.is_file():
        return None
    if any(
        not path.is_file()
        for per_transport in op_bitcodes.values()
        for path in per_transport.values()
    ):
        return None
    if bitcode is not None and not bitcode.is_file():
        return None
    if archive is not None and not archive.is_file():
        return None
    return GeneratedDeviceLibrary(
        arch=recorded["arch"],
        transports=tuple(recorded["transports"]),
        peer_transports=peer_transports,
        bitcode=bitcode,
        op_bitcodes=op_bitcodes,
        archive=archive,
        include_dir=output / "include",
        header=header,
        manifest=manifest,
        digest=recorded["digest"],
        ops=ops,
        endpoints_per_peer=int(recorded.get("endpoints_per_peer", 1)),
    )


def cached_device_library(
    arch: str,
    *,
    cache_root: str | os.PathLike | None = None,
    **options,
) -> GeneratedDeviceLibrary:
    """The generated library from a digest-keyed cache, building on miss.

    The cache key is `library_identity` — computable before compiling — so a
    stale build can never alias a fresh one; a hit is returned without
    invoking the compiler at all.
    """
    if arch.startswith("gfx"):
        options.setdefault("transports", HIP_DEFAULT_TRANSPORTS)
        options.setdefault("max_registers", None)
    if cache_root is None:
        root = pathlib.Path(__file__).resolve().parents[2]
        build = pathlib.Path(
            os.environ.get("ABOLETH_BUILD_DIR", root / "build")
        )
        cache_root = build / "lib" / "tier1"
    cache_root = pathlib.Path(cache_root)
    digest = library_identity(
        arch=arch,
        transports=options.get("transports", DEFAULT_TRANSPORTS),
        peer_transports=options.get("peer_transports"),
        put_many_sizes=options.get(
            "put_many_sizes", DEFAULT_PUT_MANY_SIZES
        ),
        completion_interval=options.get("completion_interval", 1),
        max_registers=options.get("max_registers", TIER1_MAX_REGISTERS),
        exclusive=options.get("exclusive", False),
        endpoints_per_peer=options.get("endpoints_per_peer", 1),
        strict_registrations=options.get("strict_registrations", False),
        gpu_resident_queues=options.get("gpu_resident_queues", False),
    )
    output = cache_root / f"{arch}-{digest[:12]}"
    # CMake builds the default mixed artifact at the stable per-arch path so
    # build/install consumers can find it without knowing its content digest.
    # Accept it only when its manifest proves the exact requested identity.
    prebuilt = _load(cache_root / arch)
    if prebuilt is not None and prebuilt.digest == digest:
        return prebuilt
    cached = _load(output)
    if cached is not None and cached.digest == digest:
        return cached
    # The manifest is written after every artifact. A completed cache remains
    # readable without a lock; cold writers serialize and recheck after waiting.
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / f".{arch}-{digest[:12]}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _load(output)
        if cached is not None and cached.digest == digest:
            return cached
        return generate_device_library(arch, output_dir=output, **options)


def _parse_peer_transports(text: str) -> PeerTransports:
    entries = []
    try:
        for entry in text.split(","):
            if not entry:
                continue
            peer, transport = entry.split(":", 1)
            entries.append((int(peer), transport))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated rank:transport pairs"
        ) from error
    return tuple(entries)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Tier-1 device library"
    )
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--transports",
        default=",".join(DEFAULT_TRANSPORTS),
        help="comma-separated transports to compile (rdma,p2p)",
    )
    parser.add_argument(
        "--peer-transports",
        type=_parse_peer_transports,
        default=None,
        help="optional geography specialization as rank:transport pairs "
             "(for example 0:p2p,2:rdma,3:p2p)",
    )
    parser.add_argument(
        "--put-many",
        default=",".join(str(size) for size in DEFAULT_PUT_MANY_SIZES),
        help="comma-separated put_many batch variants",
    )
    parser.add_argument("--completion-interval", type=int, default=1)
    parser.add_argument("--max-registers", type=int,
                        default=TIER1_MAX_REGISTERS)
    parser.add_argument(
        "--require-archive", action="store_true",
        help="fail rather than skip the archive when nvcc is missing",
    )
    parser.add_argument(
        "--exclusive", action="store_true",
        help="compile the rdma ops under the exclusive-issuer contract "
             "(one issuing thread per endpoint: no reservation atomic, no "
             "publication-order spin)",
    )
    parser.add_argument(
        "--endpoints-per-peer", type=int, default=1,
        help="bake the connect-time endpoint fan-out stride: the header "
             "indexes a peer-major [world x N] endpoint table and CTA b "
             "drives slot b %% N",
    )
    parser.add_argument(
        "--gpu-resident-queues", action="store_true",
        help="every rdma endpoint keeps its queues in GPU memory: publish "
             "the doorbell with a GPU-scope release",
    )
    arguments = parser.parse_args()
    library = generate_device_library(
        arguments.arch,
        output_dir=arguments.output_dir,
        transports=tuple(
            transport
            for transport in arguments.transports.split(",")
            if transport
        ),
        peer_transports=arguments.peer_transports,
        put_many_sizes=tuple(
            int(size)
            for size in arguments.put_many.split(",")
            if size
        ),
        completion_interval=arguments.completion_interval,
        max_registers=arguments.max_registers,
        require_archive=arguments.require_archive,
        exclusive=arguments.exclusive,
        endpoints_per_peer=arguments.endpoints_per_peer,
        gpu_resident_queues=arguments.gpu_resident_queues,
    )
    if library.bitcode is not None:
        print(f"generated {library.bitcode}")
    for per_transport in library.op_bitcodes.values():
        for path in per_transport.values():
            print(f"generated {path}")
    if library.archive is not None:
        print(f"generated {library.archive}")
    print(f"generated {library.header}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
