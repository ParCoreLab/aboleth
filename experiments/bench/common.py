"""Seed of the Tier-1 benchmark harness (workstream 1).

Shared by ladder.py and alltoall.py: size-ladder parsing, world/rank setup,
percentiles, the runtime-nvcc shape compiler with compile-time capture, and
the table/JSON emitters. Workstream 2 grows this into python/aboleth/bench/;
keep it under ~300 lines and free of benchmark-specific logic.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "experiments" / "bench"
KERNELS_DIR = BENCH_DIR / "kernels"

OP_CODES = {"put": 0, "put+signal": 1, "put_inline": 2}

# CUDA-binary arms and the transports every arm accepts (first = default).
BINARY_ARMS = {"aboleth-cpp": "bench_tier1_aboleth",
               "nvshmem": "bench_tier1_nvshmem",
               # The same binary built against the DSL arms' NVSHMEM 3.8.0
               # (bench_tier1_nvshmem.cu, BENCH_NVSHMEM_COLLECTIVE_THREADS).
               "nvshmem38": "bench_tier1_nvshmem38",
               "nccl-device": "bench_tier1_nccl",
               "nccl-host": "bench_tier1_nccl_host"}
# The generic Aboleth arm defaults to the compiler-generated mixed RDMA+P2P
# library. The handwritten facade remains an explicit RDMA-only oracle; the
# proxy arm has its own adapter.
ARM_TRANSPORTS = {"aboleth-generic": ("rdma", "proxy", "p2p", "auto"),
                  "aboleth-triton": ("auto", "rdma", "p2p"),
                  "aboleth-specialized": ("rdma", "p2p", "auto"),
                  "aboleth-cpp": ("rdma", "proxy"),
                  "nvshmem": ("ibgda", "ibrc"),
                  "nvshmem38": ("ibgda", "ibrc"),
                  # *-lsa: same-node peers over NVLink (LSA), GIN for the
                  # rest; plain gdaki/proxy is the pure-GIN shape.
                  "nccl-device": ("gdaki", "proxy", "gdaki-lsa",
                                  "proxy-lsa"),
                  "nccl-host": ("host",),
                  "cute-tier1": ("rdma", "p2p", "auto"),
                  "cute-nvshmem": ("ibgda", "ibrc"),
                  "triton-tier1": ("rdma", "p2p", "auto"),
                  "triton-nvshmem": ("ibgda", "ibrc"),
                  "tdist-nvshmem": ("ibgda", "ibrc"),
                  # cuTile reaches peers only through peer-mapped arrays.
                  "cutile-p2p": ("p2p",),
                  # W4 (experiments/dsl/dispatch.py): a cuTile pack, then
                  # Aboleth's Tier-2 send or NCCL's all-to-all.
                  "cutile-tier2": ("auto", "rdma", "p2p"),
                  "cutile-nccl": ("auto",)}

# Arms whose benchmark kernels a module compiles in-process, with tier2.py's
# launcher protocol (pingpong, windowed, alltoall); the module path is
# relative to experiments/. DSL_ARMS are the plain kernels transcribed into
# another DSL (experiments/dsl/), of which DSL_TIER1_ARMS link the generated
# library. A module may also bring its own world (make_world, library_meta),
# as the NVSHMEM arms do.
KERNEL_ARMS = {"aboleth-triton": "bench/tier2",
               "cute-tier1": "dsl/cute/cute_tier1",
               "cute-nvshmem": "dsl/cute/cute_nvshmem",
               "triton-tier1": "dsl/triton_dsl/triton_tier1",
               "triton-nvshmem": "dsl/triton_dsl/triton_nvshmem",
               "tdist-nvshmem": "dsl/tdist/tdist_nvshmem",
               "cutile-p2p": "dsl/cutile/cutile_p2p"}
DSL_ARMS = ("cute-tier1", "cute-nvshmem", "triton-tier1", "triton-nvshmem",
            "tdist-nvshmem", "cutile-p2p")
DSL_TIER1_ARMS = ("cute-tier1", "triton-tier1")
# The issue forms each DSL arm's library ships, as the CUDA adapters
# declare them (bench_kernels.cuh): a warp-collective put (warp), its fused
# signal (warp-signal), several issuing threads per CTA (issuers), and a send
# queue the library reclaims by itself, so windows may stream (stream).
# triton-nvshmem is a diagnostic (NVSHMEM's bitcode on the series' Triton
# through our shim) and keeps the plain per-thread forms.
DSL_ISSUE = {"cute-tier1": {"warp", "issuers"},
             "triton-tier1": {"warp", "issuers"},
             "cute-nvshmem": {"warp", "warp-signal", "issuers", "stream",
                              "collective"},
             "tdist-nvshmem": {"warp", "warp-signal", "issuers", "stream"},
             "triton-nvshmem": set(), "cutile-p2p": set()}
# Arms that run under another interpreter: Triton-distributed's environment
# (its own Triton, which cannot share a process with the series' Triton and
# the Aboleth plugin). The variable names the interpreter.
ARM_PYTHON = {"tdist-nvshmem": "ABOLETH_TDIST_PYTHON",
              "tdist-ll": "ABOLETH_TDIST_PYTHON",
              "tdist-normal": "ABOLETH_TDIST_PYTHON"}


def kernel_module(arm: str):
    """The launcher module of an in-process kernel arm, or None."""
    path = KERNEL_ARMS.get(arm)
    if path is None:
        return None
    import importlib
    directory, _, name = (ROOT / "experiments" / path).as_posix().rpartition("/")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module(name)

_SUFFIXES = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}


def parse_size(text: str) -> int:
    """nccl-tests size syntax: '256', '4K', '64M', '1G' (1024-based)."""
    text = str(text).strip()
    suffix = text[-1].upper() if text and text[-1].isalpha() else ""
    number = text[:-1] if suffix else text
    if suffix not in _SUFFIXES or not number:
        raise SystemExit(f"bench: bad size {text!r} (use e.g. 8, 4K, 64M, 1G)")
    return int(float(number) * _SUFFIXES[suffix])


def size_ladder(minbytes: int, maxbytes: int, factor: int, stepbytes: int) -> list[int]:
    if minbytes > maxbytes:
        raise SystemExit(f"bench: min size {minbytes} > max size {maxbytes}")
    sizes, size = [], minbytes
    while size <= maxbytes:
        sizes.append(size)
        size = size * factor if factor > 1 else size + stepbytes
        if factor <= 1 and stepbytes <= 0:
            break
    return sizes


def add_common_flags(parser) -> None:
    parser.add_argument("-w", "--warmup", type=int, default=50)
    parser.add_argument("-n", "--iters", type=int, default=50)
    parser.add_argument("-b", "--minbytes", default=None)
    parser.add_argument("-e", "--maxbytes", default=None)
    parser.add_argument("-f", "--stepfactor", type=int, default=2)
    parser.add_argument("-i", "--stepbytes", default="0")
    parser.add_argument("--op", default="put",
                        help="comma-separated: put, put+signal, put_inline")
    parser.add_argument("--arm", default="aboleth-generic")
    parser.add_argument("--arms", default=None,
                        help="comma-separated arm list for --interleave")
    parser.add_argument("--interleave", action="store_true",
                        help="run --arms --repeats times, one joint JSON")
    parser.add_argument("--repeats", type=int, default=2,
                        help="interleaved passes over --arms: forward, "
                             "reverse, then seeded random orders")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm-timeout", type=float, default=1200,
                        help="seconds before an interleaved arm is killed "
                             "and recorded as timed out")
    parser.add_argument("--transport", default=None)
    parser.add_argument("--ctas", type=int, default=1)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stream", choices=("off", "on"), default="off",
                        help="windowed bandwidth, libraries that reclaim "
                             "their own queue (nvshmem, nccl-device) and "
                             "Tier 2 (the compiler's self-reclaim): post "
                             "every window back to back and complete once "
                             "at the end (2610.01380's message-rate "
                             "method); off completes every window")
    parser.add_argument("--issuers", type=int, default=1,
                        help="windowed bandwidth: threads per CTA that "
                             "issue the CTA's window concurrently (1, or a "
                             "multiple of 32 dividing --window); each posts "
                             "window/issuers puts, one quiet per window")
    parser.add_argument("--puts", type=int, default=1)
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="2 sizes, 5 iters, correctness only")
    parser.add_argument("--local-ranks", type=int, default=2)
    parser.add_argument("--library", choices=("handwritten", "generated"),
                        default="generated",
                        help="which Tier-1 device library the aboleth arms "
                             "compile against")
    parser.add_argument("--put-many", choices=("auto", "off"),
                        default="auto",
                        help="auto: the generated library batches the --puts "
                             "sweep through put_many (one doorbell); off "
                             "keeps the per-put loop")
    parser.add_argument("--put-issue", choices=("thread", "warp"),
                        default="thread",
                        help="warp: kernels keep the first warp alive and "
                             "issue puts through the generated put_warp "
                             "(cooperative copy on p2p, lane-elected WQE on "
                             "rdma); thread keeps the lane-0 put")
    parser.add_argument("--fuse-signal", choices=("off", "on"),
                        default="off",
                        help="on: the payload put(s) and the epoch share "
                             "one batched submission (generated "
                             "put_signal/put_many_signal — one doorbell "
                             "instead of two); off keeps put-then-signal")
    parser.add_argument("--gin-contexts", type=int, default=0,
                        help="nccl-device: at most this many GIN contexts "
                             "(0: one per peer); CTAs share them "
                             "round-robin")
    parser.add_argument("--bake", choices=("off", "on"), default="off",
                        help="Tier 1: on compiles the swept size, put count "
                             "and op into the kernel (Tier 2's compiler "
                             "always does); off takes them at run time, as "
                             "the library binaries do")
    parser.add_argument("--batch-window", choices=("off", "on"),
                        default="off",
                        help="windowed bandwidth: on posts each window as one "
                             "hand-batched library call (Tier 1 "
                             "put_many_flush, NCCL aggregated requests: one "
                             "doorbell); off issues each put on its own, as "
                             "the plain API does. Tier 2 batches by itself")
    parser.add_argument("--collective", choices=("off", "on"),
                        default="off",
                        help="nvshmem all-to-all: on runs NVSHMEM's own "
                             "collective (nvshmemx_alltoallmem_block) instead "
                             "of the shared kernel")
    parser.add_argument("--graph-rounds", type=int, default=0,
                        help="nccl-host: capture this many rounds per CUDA "
                             "graph and time graph launches (0: per-round "
                             "host timing)")
    parser.add_argument("--env", action="append", default=[],
                        metavar="NAME=VALUE",
                        help="set a library environment variable for this "
                             "run (repeatable; an arm token passes it as "
                             "env=NAME=VALUE)")
    parser.add_argument("--small-inline", choices=("off", "on"),
                        default="off",
                        help="on: generated-header puts of exactly 8/16/24 "
                             "aligned bytes ride inline in the WQE (no NIC "
                             "gather of the source buffer)")
    parser.add_argument("--exclusive", choices=("off", "on"),
                        default="off",
                        help="on: compile the generated rdma ops under the "
                             "exclusive-issuer contract (one issuing thread "
                             "per endpoint: no reservation atomic, no "
                             "publication-order spin)")
    parser.add_argument("--doorbell-scope", choices=("auto", "system"),
                        default="auto",
                        help="Tier 1: auto publishes the rdma doorbell with "
                             "a GPU-scope release when every rdma endpoint "
                             "keeps its queues in GPU memory (as Tier 2 "
                             "does); system keeps the system-scope release")
    parser.add_argument("--endpoints-per-peer", type=int, default=1,
                        help="connect the world with N endpoints per peer "
                             "(peer-major table, CTA b drives slot b %% N): "
                             "multi-CTA rdma grids get per-CTA endpoints, "
                             "so quiet's single-consumer rule holds per CTA")


def generated_library(exclusive: bool = False,
                      endpoints_per_peer: int = 1,
                      gpu_resident_queues: bool = False):
    """The generic mixed RDMA+P2P library for this GPU."""
    from aboleth.device_library import cached_device_library
    from aboleth.world import normalize_arch

    return cached_device_library(
        normalize_arch(device_arch()), exclusive=exclusive,
        endpoints_per_peer=endpoints_per_peer,
        gpu_resident_queues=gpu_resident_queues)


def library_options(args, world=None,
                    bootstrap=None) -> tuple[tuple, tuple, tuple, dict]:
    """(include_dirs, extra_flags, extra_objects, meta) for --library.

    The `aboleth-specialized` arm bakes the connected world's complete
    peer-to-transport geography (`world.device_library(specialize=True)`). A
    uniform world folds to one backend; a mixed world emits rank-specialized
    bindings over both. Plain `--library generated` remains generic RDMA+P2P.
    Generation is digest-cached; rank 0 builds first, after which other ranks
    either reuse that identity or build rank-specific mixed geographies in
    distinct digest directories.
    """
    specialized = getattr(args, "arm", "") == "aboleth-specialized"
    module = kernel_module(getattr(args, "arm", ""))
    if module is not None and hasattr(module, "library_meta"):
        return (), (), (), module.library_meta()
    if getattr(args, "arm", "") == "aboleth-triton":
        # Tier 2 compiles the communication into the benchmark kernel.
        return (), (), (), {"library": "tier2",
                            "passes": os.environ.get("ABOLETH_PASSES", "")}
    if (not specialized
            and getattr(args, "library", "generated") != "generated"):
        return (), (), (), {"library": "handwritten"}

    exclusive = getattr(args, "exclusive", "off") == "on"
    fan_out = int(getattr(args, "endpoints_per_peer", 1))
    # The doorbell's release scope, as World.device_library picks it.
    gpu_queues = (getattr(args, "doorbell_scope", "auto") == "auto"
                  and world is not None and world.gpu_resident_queues)

    def build():
        if specialized:
            if world is None:
                raise SystemExit(
                    "bench: aboleth-specialized resolves its library from "
                    "a connected world")
            # A connected world knows its own fan-out.
            return world.device_library(
                specialize=True, exclusive=exclusive,
                gpu_resident_queues=gpu_queues)
        return generated_library(exclusive=exclusive,
                                 endpoints_per_peer=fan_out,
                                 gpu_resident_queues=gpu_queues)

    if bootstrap is not None and bootstrap.rank != 0:
        bootstrap.barrier("bench:tier1-library")
        library = build()
    else:
        library = build()
        if bootstrap is not None:
            bootstrap.barrier("bench:tier1-library")
    if library.archive is None:
        raise SystemExit(
            "bench: the generated library has no archive (nvcc was missing "
            "at generation); regenerate with nvcc on PATH")
    meta = {"library": "specialized" if specialized else "generated",
            "library_digest": library.digest[:12],
            "library_transports": list(library.transports),
            "library_exclusive": exclusive,
            "library_doorbell_scope": "gpu" if gpu_queues else "system",
            "library_endpoints_per_peer": library.endpoints_per_peer}
    if library.peer_transports is not None:
        meta["library_peer_transports"] = [
            {"peer": peer, "transport": transport}
            for peer, transport in library.peer_transports
        ]
    if getattr(args, "arm", "") in DSL_TIER1_ARMS:
        # The same library, linked as per-op bitcode into the DSL's kernels.
        kernel_module(args.arm).bind(library)
        return (), (), (), {**meta, "frontend": args.arm.split("-")[0]}
    return ((library.include_dir,), ("-rdc=true",), (library.archive,),
            meta)


def check_tier2_flags(args) -> None:
    """The Tier-1 issue knobs are compiler decisions in a Tier-2 kernel."""
    if getattr(args, "arm", "") in DSL_ARMS:
        check_dsl_flags(args)
        return
    if getattr(args, "arm", "") != "aboleth-triton":
        return
    for flag, default in (("put_issue", "thread"), ("fuse_signal", "off"),
                          ("small_inline", "off"), ("exclusive", "off"),
                          ("doorbell_scope", "auto"), ("l2_flush", False)):
        if getattr(args, flag, default) != default:
            raise SystemExit(
                f"bench: --{flag.replace('_', '-')} is a Tier-1 knob; the "
                "aboleth-triton arm leaves it to the compiler")
    if getattr(args, "issuers", 1) != 1:
        raise SystemExit(
            "bench: --issuers: a Tier-2 kernel's issuers are its programs "
            "(--ctas with --endpoints-per-peer)")
    args.put_many = "off"


def check_dsl_flags(args) -> None:
    """The DSL arms transcribe the per-call kernels: one issuing thread per
    CTA, fused put+signal (and Tier 1's exclusive build) as options, and the
    issue forms the arm's library ships (DSL_ISSUE)."""
    for flag, default in (("small_inline", "off"), ("batch_window", "off"),
                          ("bake", "off"), ("l2_flush", False)):
        if getattr(args, flag, default) != default:
            raise SystemExit(
                f"bench: --{flag.replace('_', '-')} has no {args.arm} "
                "transcription")
    forms = DSL_ISSUE.get(args.arm, set())
    warp = getattr(args, "put_issue", "thread") == "warp"
    if warp and "warp" not in forms:
        raise SystemExit(f"bench: {args.arm} has no warp-collective put")
    if (warp and getattr(args, "fuse_signal", "off") == "on"
            and "warp-signal" not in forms):
        raise SystemExit(f"bench: {args.arm} has no warp-collective "
                         "put-with-signal (--fuse-signal on needs "
                         "--put-issue thread)")
    if warp and getattr(args, "mode", None) in ("latency", "bandwidth"):
        raise SystemExit("bench: the DSL arms issue warp-collective puts in "
                         "the all-to-all only (as the series' arms)")
    if getattr(args, "issuers", 1) != 1 and "issuers" not in forms:
        raise SystemExit(f"bench: --issuers has no {args.arm} transcription")
    if getattr(args, "collective", "off") == "on":
        if "collective" not in forms:
            raise SystemExit(f"bench: {args.arm} ships no all-to-all "
                             "collective")
        if warp or getattr(args, "fuse_signal", "off") == "on":
            raise SystemExit("bench: --collective on is the library's own "
                             "all-to-all (no issue flags)")
    if getattr(args, "stream", "off") != "off" and "stream" not in forms:
        raise SystemExit(f"bench: --stream needs a self-reclaiming library "
                         f"({args.arm})")
    if getattr(args, "issuers", 1) != 1 and args.arm in DSL_TIER1_ARMS \
            and getattr(args, "exclusive", "off") == "on":
        raise SystemExit("bench: --issuers > 1 shares the CTA's endpoint; "
                         "the exclusive-issuer build forbids it")
    # The DSL kernels take sizes, put counts and the op as compile-time
    # constants (as Tier 2 does); the record says so. The CUDA arms bake
    # them only with --bake on (a labelled reference arm in the DSL phases).
    args.bake = "constexpr"
    if (args.arm in DSL_TIER1_ARMS
            and getattr(args, "library", "generated") != "generated"):
        raise SystemExit(f"bench: {args.arm} links the generated library")
    for flag, default in (("exclusive", "off"), ("doorbell_scope", "auto")):
        if (args.arm not in DSL_TIER1_ARMS
                and getattr(args, flag, default) != default):
            raise SystemExit(f"bench: --{flag.replace('_', '-')} is a "
                             f"Tier-1 build ({args.arm})")
    args.put_many = "off"


def resolve_transport(args) -> None:
    """Fill args.transport from the flag, ABOLETH_FORCE_TRANSPORT, or the
    arm's default, and refuse transports the arm cannot drive."""
    allowed = ARM_TRANSPORTS.get(args.arm)
    if allowed is None:
        raise SystemExit(f"bench: unknown --arm {args.arm!r}")
    issuers = getattr(args, "issuers", 1)
    if issuers != 1 and (issuers % 32 or issuers > 1024
                         or args.window % issuers):
        raise SystemExit(
            f"bench: --issuers {issuers}: 1, or a multiple of 32 (at most "
            f"1024) that divides --window {args.window}")
    # Per-run library settings: the binaries and the in-process arms both
    # read them from the environment (setenv soft-defaults yield to them).
    for setting in getattr(args, "env", ()):
        name, separator, value = setting.partition("=")
        if not separator or not name:
            raise SystemExit(f"bench: --env {setting!r} is not NAME=VALUE")
        os.environ[name] = value
    transport = (args.transport
                 or os.environ.get("ABOLETH_FORCE_TRANSPORT") or allowed[0])
    if transport not in allowed:
        raise SystemExit(
            f"bench: {args.arm} supports transports {'/'.join(allowed)}, "
            f"not {transport!r}")
    if (transport in ("p2p", "auto")
            and args.arm != "aboleth-specialized"
            and getattr(args, "library", "generated") != "generated"):
        raise SystemExit(
            f"bench: --transport {transport} needs --library generated "
            "(the hand-written Tier-1 facade drives rdma only)")
    args.transport = transport


def binary_path(name: str) -> Path:
    build = Path(os.environ.get("ABOLETH_BUILD_DIR", ROOT / "build"))
    return build / "bin" / name


def run_binary_arm(binary: Path, argv: list[str], *, ranks: int,
                   json_out: str | None = None) -> int:
    """Run a CUDA benchmark arm: under an external launcher (RANK set) exec
    one rank in place; otherwise spawn `ranks` local copies with a fresh
    rendezvous port. Streams rank 0's output through, honors `SKIP:` lines,
    and copies the arm's final JSON line to json_out."""
    if not binary.exists():
        print(f"SKIP: {binary.name} is not built (arm gated off in CMake?)")
        return 0
    if os.environ.get("RANK") is not None:
        return subprocess.run([str(binary), *argv]).returncode
    port = _free_store_port()
    processes = []
    for rank in range(ranks):
        environment = {**os.environ, "MASTER_ADDR": "127.0.0.1",
                       "MASTER_PORT": str(port), "RANK": str(rank),
                       "WORLD_SIZE": str(ranks), "LOCAL_RANK": str(rank)}
        processes.append(subprocess.Popen(
            [str(binary), *argv], env=environment, text=True,
            stdout=subprocess.PIPE if rank == 0 else subprocess.DEVNULL))
    output, _ = processes[0].communicate()
    code = processes[0].returncode
    for process in processes[1:]:
        process.wait()
        code = code or process.returncode
    sys.stdout.write(output)
    record = None
    for line in output.splitlines():
        if line.startswith("{"):
            record = line
    if json_out and record:
        parsed = json.loads(record)
        parsed.setdefault("node", {}).update(provenance())
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(parsed, sort_keys=True) + "\n")
    if any(line.startswith("SKIP:") for line in output.splitlines()):
        return 0
    return code


def driver_argv(args, extra: list[str] = ()) -> list[str]:
    """Re-render the uniform flags for a child driver invocation (used by
    the interleaved compare). --arm/--arms/--json-out are the caller's."""
    argv = ["-w", str(args.warmup), "-n", str(args.iters),
            "--puts", str(args.puts), "--window", str(args.window),
            "--op", args.op, "--local-ranks", str(args.local_ranks)]
    if args.minbytes:
        argv += ["-b", str(args.minbytes)]
    if args.maxbytes:
        argv += ["-e", str(args.maxbytes)]
    if args.stepfactor != 2:
        argv += ["-f", str(args.stepfactor)]
    if args.transport:
        argv += ["--transport", args.transport]
    if args.l2_flush:
        argv += ["--l2-flush"]
    if args.smoke:
        argv += ["--smoke"]
    if getattr(args, "library", "generated") != "generated":
        argv += ["--library", args.library]
    if getattr(args, "put_many", "auto") != "auto":
        argv += ["--put-many", args.put_many]
    if parse_size(args.stepbytes):
        argv += ["-i", str(args.stepbytes)]
    for flag in ("ctas", "endpoints_per_peer", "put_issue", "fuse_signal",
                 "small_inline", "exclusive", "doorbell_scope"):
        argv += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
    return argv + list(extra)


def _free_store_port() -> int:
    """A free port below the ephemeral range (32768 up) for a rendezvous
    store. A port the kernel hands out for binding to 0 is ephemeral: once
    released it can become an outgoing connection's source port before the
    store binds it again (EADDRINUSE). Below the range only explicit binds
    compete."""
    import random
    for port in random.Random().sample(range(20000, 32000), 64):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit("bench: no free rendezvous port in 20000-31999")


def run_interleaved(args, script: Path, extra: list[str] = ()) -> int:
    """A/B..B/A over --arms: each arm runs as a child driver invocation;
    the joint record keeps every run's record with its ordering index. An
    arm token may pin its transport as `arm:transport` (e.g. nvshmem:ibrc)
    and, for the aboleth arms, its device library as
    `arm:transport:library` (e.g. aboleth-generic:rdma:generated), and
    pass any further fields as flags (`aboleth-generic:auto::fuse-signal=on`
    adds `--fuse-signal on`), so each arm runs its best configuration.
    Under an external launcher every rank runs the same loop; ports are
    offset per child so rendezvous sockets never collide."""
    import random
    arms = [token.strip() for token in (args.arms or args.arm).split(",")]
    shuffled = random.Random(args.seed)
    order = []
    for repeat in range(args.repeats):
        if repeat == 0:
            order += arms
        elif repeat == 1:
            order += arms[::-1]
        else:
            order += shuffled.sample(arms, len(arms))
    child_argv = driver_argv(args, extra)
    lead = os.environ.get("RANK") in (None, "0")
    runs, code = [], 0
    for index, token in enumerate(order):
        parts = token.split(":")
        arm = parts[0]
        transport = parts[1] if len(parts) > 1 else ""
        library = parts[2] if len(parts) > 2 else ""
        python = sys.executable
        if arm in ARM_PYTHON:
            python = os.environ.get(ARM_PYTHON[arm])
            if not python:
                raise SystemExit(f"bench: {arm} runs under ${ARM_PYTHON[arm]}"
                                 " (its own environment); set it")
        command = [python, str(script), "--arm", arm, *child_argv]
        if transport:
            command += ["--transport", transport]
        if library:
            command += ["--library", library]
        for flag in parts[3:]:
            name, _, value = flag.partition("=")
            command += [f"--{name}", value] if value else [f"--{name}"]
        environment = dict(os.environ)
        if environment.get("RANK") is not None and \
                environment.get("MASTER_PORT"):
            environment["MASTER_PORT"] = str(
                int(environment["MASTER_PORT"]) + 1 + index)
        if lead:
            print(f"# interleave {index + 1}/{len(order)}: {token}",
                  flush=True)
        # A rank that never arrives stalls every bootstrap waiting for it,
        # and NVSHMEM's waits forever: kill the arm's whole process group
        # (the driver and the binary it runs) and go on to the next.
        child = subprocess.Popen(command, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True,
                                 env=environment, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = child.communicate(timeout=args.arm_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            stdout, stderr = child.communicate()
        if lead:
            sys.stdout.write(stdout)
            sys.stderr.write(stderr)
            if timed_out:
                print(f"# interleave {index + 1}: {token} killed after "
                      f"{args.arm_timeout:.0f} s", flush=True)
        code = code or child.returncode or int(timed_out)
        record = None
        for line in stdout.splitlines():
            if line.startswith("{"):
                # The record is one JSON line, but a library may print on
                # the same line after it (NVSHMEM at finalize): take the
                # first object.
                try:
                    record = json.JSONDecoder().raw_decode(line)[0]
                except ValueError:
                    continue
        if record is not None:
            record.setdefault("node", {}).update(provenance())
        runs.append({"arm": token, "order": index, "record": record,
                     "exit": "timeout" if timed_out else child.returncode})
    # Under srun --kill-on-bad-exit a failing non-lead rank would kill the
    # lead before it writes the joint record; failures are in the record.
    if not lead:
        return 0
    if lead:
        emit({"suite": "tier1", "benchmark": script.stem,
              "interleaved": True, "ordering": order, "seed": args.seed,
              "provenance": provenance(), "runs": runs},
             args.json_out)
    return code


def binary_argv(args, bench: str, mode: str) -> list[str]:
    """Map the uniform driver flags onto a CUDA arm's argv."""
    argv = ["--bench", bench, "--mode", mode, "--transport", args.transport,
            "-w", str(args.warmup), "-n", str(args.iters),
            "--puts", str(args.puts), "--window", str(args.window),
            "--op", args.op, "--ctas", str(args.ctas),
            "--put-issue", getattr(args, "put_issue", "thread"),
            "--fuse-signal", getattr(args, "fuse_signal", "off")]
    if getattr(args, "issuers", 1) > 1:
        argv += ["--issuers", str(args.issuers)]
    if getattr(args, "stream", "off") == "on":
        argv += ["--stream", "on"]
    if getattr(args, "gin_contexts", 0):
        argv += ["--gin-contexts", str(args.gin_contexts)]
    if getattr(args, "graph_rounds", 0):
        argv += ["--graph-rounds", str(args.graph_rounds)]
    if getattr(args, "batch_window", "off") == "on":
        argv += ["--batch-window", "on"]
    if getattr(args, "collective", "off") == "on":
        argv += ["--collective", "on"]
    if args.minbytes:
        argv += ["-b", str(args.minbytes)]
    if args.maxbytes:
        argv += ["-e", str(args.maxbytes)]
    if args.stepfactor != 2:
        argv += ["-f", str(args.stepfactor)]
    if parse_size(args.stepbytes):
        argv += ["-i", str(args.stepbytes)]
    if args.smoke:
        argv += ["--smoke"]
    return argv


def resolve_sweep(args, default_min: str, default_max: str) -> list[int]:
    sizes = size_ladder(parse_size(args.minbytes or default_min),
                        parse_size(args.maxbytes or default_max),
                        args.stepfactor, parse_size(args.stepbytes))
    if args.smoke:
        args.iters, args.warmup = 5, 2
        sizes = [sizes[0], sizes[-1]] if len(sizes) > 1 else sizes
    return sizes


def run_ranks(args, rank_main) -> int:
    """Run rank_main on every rank: directly when RANK is already in the
    environment (srun / multinode-launch.sh), else by spawning
    --local-ranks child processes with a loopback TCP store rendezvous."""
    if os.environ.get("RANK") is not None:
        return rank_main(args) or 0
    import multiprocessing

    import torch
    if args.local_ranks < 2:
        raise SystemExit("bench: need at least 2 ranks")
    if args.local_ranks > torch.cuda.device_count():
        raise SystemExit(
            f"bench: --local-ranks {args.local_ranks} exceeds the "
            f"{torch.cuda.device_count()} visible GPUs")
    port = _free_store_port()
    context = multiprocessing.get_context("spawn")
    processes = []
    for rank in range(args.local_ranks):
        environment = {
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
            "RANK": str(rank), "WORLD_SIZE": str(args.local_ranks),
            "LOCAL_RANK": str(rank),
        }
        process = context.Process(
            target=_rank_entry, args=(environment, args, rank_main))
        process.start()
        processes.append(process)
    code = 0
    for process in processes:
        process.join()
        code = code or (process.exitcode or 0)
    return code


def _rank_entry(environment, args, rank_main) -> None:
    os.environ.update(environment)
    raise SystemExit(rank_main(args) or 0)


def arena(world, nbytes: int, device):
    """The benchmark arena: zeroed device memory, from the world's own
    allocator when it has one (a symmetric heap)."""
    import torch
    allocate = getattr(world, "allocate_arena", None)
    if allocate is not None:
        return allocate(nbytes)
    return torch.zeros(nbytes, dtype=torch.uint8, device=device)


def make_world(args):
    """World + bootstrap for this rank."""
    import torch

    module = kernel_module(args.arm)
    if module is not None and hasattr(module, "make_world"):
        return module.make_world(args)
    import aboleth
    from aboleth.bootstrap import TcpStoreBootstrap
    if args.transport not in ("rdma", "proxy", "p2p", "auto"):
        raise SystemExit(
            f"bench: --transport {args.transport!r} is not a Tier-1 "
            "transport (rdma, proxy, p2p, auto)")
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)
    # The library's device footprint: free memory before World creation
    # less free memory once connected, before any benchmark buffer.
    free_before = torch.cuda.mem_get_info()[0]
    # Large cold-cache worlds (32 ranks JIT-compiling shape libraries, or
    # the specialized arm generating rank-specific mixed libraries) can
    # legitimately outlast the default window.
    bootstrap = TcpStoreBootstrap.from_env(
        timeout=float(os.environ.get("ABOLETH_BOOTSTRAP_TIMEOUT_S", "300")))
    fan_out = int(getattr(args, "endpoints_per_peer", 1))
    if fan_out > 1 and (getattr(args, "library", "generated") != "generated"
                        and getattr(args, "arm", "") != "aboleth-specialized"):
        raise SystemExit(
            "bench: --endpoints-per-peer > 1 needs --library generated "
            "(the hand-written header indexes one endpoint per peer)")
    world = aboleth.World(
        bootstrap.rank, bootstrap.world_size, device=device, bootstrap=bootstrap)
    world.connect_all(args.transport, endpoints_per_peer=fan_out)
    free_after = torch.cuda.mem_get_info()[0]
    world.bench_memory = {"free_before": free_before,
                          "free_after": free_after,
                          "library_bytes": free_before - free_after}
    return world, bootstrap


def kernel_attributes(handle, which: int, threads: int, ctas: int) -> dict:
    """Registers, spill and shared bytes of a compiled shape library's
    kernel (0 ping-pong, 1 bandwidth, 2 all-to-all)."""
    registers, local, shared = (ctypes.c_int(), ctypes.c_int(),
                                ctypes.c_int())
    code = handle.bench_kernel_attributes(
        which, ctypes.byref(registers), ctypes.byref(local),
        ctypes.byref(shared))
    if code:
        raise RuntimeError(f"bench: cudaFuncGetAttributes failed ({code})")
    return {"registers": registers.value, "local_bytes": local.value,
            "shared_bytes": shared.value, "threads": threads, "ctas": ctas}


def clamp_sizes(sizes: list[int], per_size_bytes) -> tuple[list[int], list[str]]:
    """Drop sizes whose buffers exceed free GPU memory — loudly, never
    silently (the 64MiB-per-peer x world 8 trap)."""
    import torch
    free, _total = torch.cuda.mem_get_info()
    budget = int(free * 0.8)
    kept, notes = [], []
    for size in sizes:
        need = per_size_bytes(size)
        if need <= budget:
            kept.append(size)
        else:
            notes.append(f"size {size} dropped: needs {need} B, "
                         f"budget {budget} B (free {free} B)")
    for note in notes:
        print(f"# clamp: {note}")
    if not kept:
        raise SystemExit("bench: every size exceeds free GPU memory")
    return kept, notes


_PATTERN_CHUNK = 256 << 20


def _pattern_base(device):
    import torch
    return ((torch.arange(256, device=device) * 17 + 3) & 0xFF).to(torch.uint8)


# The salt of the pattern a timed launch moves (the correctness launch uses
# 0): receivers are zeroed in between, so a check after timing sees only
# what the timed rounds delivered.
TIMED_SALT = 0xA5


def pattern_fill_gpu(view, salt: int = 0) -> None:
    """In-place (i*17+3)&0xff ^ salt fill without a full-size index tensor
    (the pattern has period 256, so it broadcasts). The view must start at a
    256-aligned position of the pattern domain (offset 0 in practice); salt
    makes alltoall slices sender-distinct so misrouting is caught."""
    base = _pattern_base(view.device) ^ (salt & 0xFF)
    whole = view.numel() // 256 * 256
    if whole:
        view[:whole].view(-1, 256)[:] = base
    if whole < view.numel():
        view[whole:] = base[: view.numel() - whole]


def pattern_wrong_gpu(view, salt: int = 0) -> int:
    """Mismatch count against the (salted) pattern, chunked to bound
    temporaries."""
    base = _pattern_base(view.device) ^ (salt & 0xFF)
    wrong, position = 0, 0
    while position < view.numel():
        chunk = view[position:position + _PATTERN_CHUNK]
        whole = chunk.numel() // 256 * 256
        if whole:
            wrong += int((chunk[:whole].view(-1, 256) != base).sum().item())
        if whole < chunk.numel():
            wrong += int((chunk[whole:]
                          != base[: chunk.numel() - whole]).sum().item())
        position += _PATTERN_CHUNK
    return wrong


def submit_label(transport: str) -> str:
    if transport in ("proxy", "p2p"):
        return transport
    if transport == "auto":
        return "mixed"
    return os.environ.get("ABOLETH_RDMA_SUBMIT", "probe")


def port_counters(hca: str | None, port: int = 1) -> dict[str, int]:
    """The HCA port's counters and hw_counters (sysfs), {} if unreadable."""
    values: dict[str, int] = {}
    if not hca:
        return values
    base = Path("/sys/class/infiniband") / hca / "ports" / str(port)
    for group in ("counters", "hw_counters"):
        directory = base / group
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            try:
                values[f"{group}/{entry.name}"] = int(entry.read_text())
            except (OSError, ValueError):
                pass
    return values


def percentiles(samples_us: list[float]) -> dict:
    if len(samples_us) < 2:
        value = samples_us[0] if samples_us else float("nan")
        return {"p50_us": value, "p95_us": value, "p99_us": value,
                "mean_us": value}
    marks = statistics.quantiles(samples_us, n=100, method="inclusive")
    return {"p50_us": marks[49], "p95_us": marks[94], "p99_us": marks[98],
            "mean_us": statistics.fmean(samples_us)}


def _nvcc() -> str:
    explicit = os.environ.get("ABOLETH_NVCC")
    if explicit:
        return explicit
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(variable)
        if base and (Path(base) / "bin" / "nvcc").exists():
            return str(Path(base) / "bin" / "nvcc")
    if Path("/usr/local/cuda/bin/nvcc").exists():
        return "/usr/local/cuda/bin/nvcc"
    found = shutil.which("nvcc")
    if found:
        return found
    raise SystemExit("bench: nvcc not found (set ABOLETH_NVCC or CUDA_HOME)")


def default_cache_dir() -> Path:
    return Path(os.environ.get("ABOLETH_BENCH_CACHE", ROOT / ".bench-cache"))


def compile_shape_library(source: str, *, arch: str, cache_dir: Path,
                          tag: str = "", include_dirs: tuple = (),
                          extra_flags: tuple = (),
                          extra_objects: tuple = (),
                          ) -> tuple[ctypes.CDLL, dict]:
    """nvcc-compile a rendered benchmark TU into a cached .so and dlopen it.
    The returned dict is the per-variant compile record.

    `include_dirs` are searched BEFORE the tree's include/, which is how the
    generated Tier-1 header shadows the hand-written facade at the same
    include path. `extra_objects` (archives, relocatable objects) join the
    link, and `extra_flags` (e.g. -rdc=true) the compile; all three are part
    of the cache key."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-std=c++17", f"-arch={arch}", *extra_flags]
    for directory in include_dirs:
        flags += ["-I", str(directory)]
    flags += ["-I", str(ROOT / "include"), "-I", str(BENCH_DIR),
              "-I", str(KERNELS_DIR), "-Xcompiler", "-fPIC", "-shared"]
    objects = [str(path) for path in extra_objects]
    # The unit includes the shared kernels and adapters: their text is part
    # of the key, or an edited header reuses a stale library.
    headers = "".join(path.read_text()
                      for path in sorted(KERNELS_DIR.glob("*"))
                      + [BENCH_DIR / "bench_common.h"])
    key = hashlib.sha256(
        "\0".join([source, headers, *flags, *objects]).encode()
    ).hexdigest()[:24]
    library = cache_dir / f"bench_{tag}_{key}.so"
    record = {"cache_hit": library.exists(),
              "first_call_s": 0.0, "cached_call_s": 0.0}
    start = time.perf_counter()
    if not library.exists():
        # Cold caches are filled by every rank at once (shared GPFS tree):
        # compile a private copy and publish both files atomically, so no
        # rank ever reads a half-written unit or dlopens a partial .so.
        unit = cache_dir / f"bench_{tag}_{key}.cu"
        private = cache_dir / f".{os.getpid()}_{key}.cu"
        private.write_text(source)
        scratch = cache_dir / f".{os.getpid()}_{key}.so"
        subprocess.run(
            [_nvcc(), *flags, str(private), *objects, "-o", str(scratch)],
            check=True)
        os.replace(scratch, library)
        os.replace(private, unit)
        record["first_call_s"] = time.perf_counter() - start
        handle = ctypes.CDLL(str(library))
    else:
        handle = ctypes.CDLL(str(library))
        record["cached_call_s"] = time.perf_counter() - start
    return handle, record


def device_arch() -> str:
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def node_info(world=None) -> dict:
    import torch
    info = {"host": socket.gethostname(),
            "gpus": torch.cuda.get_device_name()}
    if world is not None:
        info["hca"] = getattr(world, "hca", None)
    return {**info, **provenance()}


# Environment that changes what an arm does; recorded with every result.
_RECORDED_ENVIRONMENT = ("ABOLETH_", "NCCL_", "NVSHMEM_", "TRITON_",
                         "CUDA_VISIBLE_DEVICES", "SLURM_JOB_NODELIST")


def provenance() -> dict:
    """Commit, platform tag, library versions and the relevant environment.

    The commit comes from ABOLETH_COMMIT when the tree was copied without
    .git; `--dirty` is only known where git is.
    """
    commit = os.environ.get("ABOLETH_COMMIT")
    if commit is None:
        try:
            commit = subprocess.run(
                ["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                capture_output=True, text=True,
                timeout=10).stdout.strip() or None
        except OSError:
            commit = None
    versions = {}
    for module in ("torch", "triton"):
        try:
            versions[module] = __import__(module).__version__
        except ImportError:
            pass
    try:
        import torch
        versions["cuda_runtime"] = torch.version.cuda
        versions["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.split("\n")[0].strip() or None
    except (ImportError, OSError):
        pass
    for package in ("nvidia-nccl-cu12", "nvidia-nvshmem-cu12"):
        try:
            from importlib.metadata import version
            versions[package] = version(package)
        except Exception:
            pass
    return {
        "commit": commit,
        "platform": os.environ.get("ABOLETH_PLATFORM", "unlabeled"),
        "versions": versions,
        "environment": {key: value for key, value in sorted(os.environ.items())
                        if key.startswith(_RECORDED_ENVIRONMENT)},
    }


def methodology(mode: str, args) -> dict:
    return {
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iters,
        # Staged endpoints place their queues per this knob (gpu = DEVX
        # queues in device memory with probe-fallback to host).
        "staged_queues": os.environ.get(
            "ABOLETH_RDMA_STAGED_QUEUES", "gpu"),
        "timer": ("device %globaltimer per-iteration" if mode == "latency"
                  else "device %globaltimer, first CTA start to last CTA "
                       "end"),
        "latency_definition":
            "full round trip: payload put(s)+signal, signal ack back",
        "percentile_method": "statistics.quantiles inclusive",
        "gbps_definition": "decimal GB/s over 1024-based sizes",
        "busbw_factor": "alltoall (n-1)/n; point-to-point 1.0",
        "caveats": [
            "put_inline is 8B-only on the Tier-1 facade",
            "latency-mode algbw = one-way payload / full round-trip time",
        ],
    }


_COLUMNS = ("size", "count", "p50_us", "p95_us", "p99_us", "mean_us",
            "algbw_GBs", "busbw_GBs", "wrong")


def table_print(meta: dict, rows: list[dict]) -> None:
    print("# " + " ".join(f"{key}={value}" for key, value in meta.items()))
    print("# " + "  ".join(f"{name:>9}" for name in _COLUMNS))
    for row in rows:
        cells = []
        for name in _COLUMNS:
            value = row.get(name.replace("_GBs", "_gbps"), row.get(name, 0))
            cells.append(f"{value:>9.2f}" if isinstance(value, float)
                         else f"{value:>9}")
        print("  " + "  ".join(cells))


def emit(record: dict, json_out: str | None) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line)
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(line + "\n")
