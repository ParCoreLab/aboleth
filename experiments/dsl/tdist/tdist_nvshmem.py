"""The Triton-distributed arm: triton_bench.py's kernels in its language.

What a Triton-distributed user writes: the kernels call its
`libshmem_device` primitives (NVSHMEM's device API, linked as bitcode by
its compiler) and are compiled by `triton_dist.jit` on its own Triton
(upstream 3.7.1 + its patches and plugin). The kernels are triton_bench.py,
shared with every Triton arm, instantiated with the policy below, so this
arm and triton-tier1 differ in the library and its compiler, not in the
benchmark. The policy is comm_nvshmem.cuh call for call: putmem_nbi,
fence + signal_op(SET) for a strong signal, putmem_signal_nbi for the fused
one (fenced behind earlier puts), the warp-collective
putmem(_signal)_nbi_warp, a PE-wide quiet, signal_wait_until(GE).

Runs in Triton-distributed's environment (envs/tdist), whose Triton cannot
share a process with the series' Triton and the Aboleth plugin: nothing
here imports `aboleth` (common.ARM_PYTHON selects the interpreter; the
world is ../nvshmem_world.py, set up as bench_tier1_nvshmem.cu).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _plugin_path() -> str:
    # Its kernels import triton before triton_dist, so its plugin is named
    # in the environment before Triton loads (its scripts/setenv.sh).
    spec = importlib.util.find_spec("triton_dist")
    return str(Path(list(spec.submodule_search_locations)[0]) / "lib"
               / "libtriton_dist.so")


os.environ["TRITON_PLUGIN_PATHS"] = _plugin_path()
# NVSHMEM_HOME: its scripts/setenv.sh points it at this environment's pip
# NVSHMEM (3.6.5, the host library loaded here). Unset, triton_dist takes
# `nvidia.nvshmem`, which the base path's regular `nvidia` package shadows
# (NVSHMEM 3.4.5): the bitcode and the wrapper's device code would not match
# the host library.
os.environ.setdefault("NVSHMEM_HOME", str(importlib.metadata.distribution(
    "nvidia-nvshmem-cu12").locate_file("nvidia/nvshmem")))
# IBGDA: Triton-distributed's documented route is its NVSHMEM wrapper
# (TRITON_DIST_SHMEM_WRAPPER: the device calls compiled by nvcc against
# NVSHMEM's CUDA device library and linked into the kernel; its bitcode path
# has no IBGDA and segfaults in IBGDA init). The transport is known from the
# command line before Triton loads.
_argv = sys.argv
if "--transport" in _argv and _argv[_argv.index("--transport") + 1:][:1] == [
        "ibgda"]:
    os.environ.setdefault("TRITON_DIST_SHMEM_WRAPPER", "1")
    os.environ.setdefault("NVSHMEM_IBGDA_SUPPORT", "1")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import triton_dist  # noqa: E402
from triton_dist.jit import TritonDistJITFunction, _install_triton_dist_hook  # noqa: E402
from triton_dist.language.extra import libshmem_device as shmem  # noqa: E402

import nvshmem.bindings.nvshmem as nvshmem_bindings  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "triton_dsl"))
sys.path.insert(0, str(HERE.parent))
import nvshmem_world  # noqa: E402
import triton_launch  # noqa: E402

SIGNAL_SET = tl.constexpr(shmem.NVSHMEM_SIGNAL_SET)
CMP_GE = tl.constexpr(shmem.NVSHMEM_CMP_GE)


@triton.jit
def _bytes(address):
    return address.to(tl.pointer_type(tl.int8))


@triton.jit
def _word(address):
    return address.to(tl.pointer_type(tl.uint64))


# The policy: comm_nvshmem.cuh, in Triton-distributed's primitives. Keys and
# the endpoint table do not exist in NVSHMEM; a remote address is the
# symmetric local one.

@triton.jit
def put(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    shmem.putmem_nbi(_bytes(dst), _bytes(src), nbytes, peer)


@triton.jit
def put_warp(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    shmem.putmem_nbi_warp(_bytes(dst), _bytes(src), nbytes, peer)


@triton.jit
def put_inline(endpoints, peer, dst, dst_key, value):
    # Triton-distributed exposes no uint64_p; Triton resolves the name even
    # where a constexpr removes the call, so it exists and refuses to trace.
    tl.static_assert(False, "tdist-nvshmem: no put_inline (uint64_p)")


@triton.jit
def signal(endpoints, peer, flag, flag_key, epoch):
    shmem.fence()
    shmem.signal_op(_word(flag), epoch, SIGNAL_SET, peer)


@triton.jit
def put_signal(endpoints, peer, dst, dst_key, src, src_key, nbytes, flag,
               flag_key, epoch):
    shmem.putmem_signal_nbi(_bytes(dst), _bytes(src), nbytes, _word(flag),
                            epoch, SIGNAL_SET, peer)


@triton.jit
def put_signal_warp(endpoints, peer, dst, dst_key, src, src_key, nbytes,
                    flag, flag_key, epoch):
    shmem.putmem_signal_nbi_warp(_bytes(dst), _bytes(src), nbytes,
                                 _word(flag), epoch, SIGNAL_SET, peer)


@triton.jit
def fence():
    shmem.fence()


@triton.jit
def quiet(endpoints, peer):
    shmem.quiet()


@triton.jit
def wait_signal(flag, epoch):
    shmem.signal_wait_until(_word(flag), CMP_GE, epoch)


def make_world(args):
    return nvshmem_world.make_world(args)


def _commit() -> str | None:
    source = Path(triton_dist.__file__).resolve().parents[2]
    try:
        return subprocess.run(["git", "-C", str(source), "rev-parse",
                               "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def library_meta() -> dict:
    return {**nvshmem_world.library_meta("triton-distributed"),
            "shmem_wrapper": os.environ.get("TRITON_DIST_SHMEM_WRAPPER", "0"),
            "nvshmem_home": os.environ.get("NVSHMEM_HOME"),
            "triton_dist_commit": _commit(),
            "triton_version": triton.__version__}


def _module_init(compiled) -> None:
    nvshmem_bindings.cumodule_init(compiled.module)


_STATE: dict = {}


def _arm():
    if "arm" not in _STATE:
        bench = triton_launch.load(sys.modules[__name__],
                                   quiet_is_global=True, name="tdist")
        # triton_dist.jit, applied to the shared kernels: its compiler hook
        # (the distributed lowering and NVSHMEM's bitcode, linked when a
        # kernel names it) and its launch wrapper.
        _install_triton_dist_hook()
        for name in ("pingpong", "windowed", "alltoall"):
            setattr(bench, name, TritonDistJITFunction(getattr(bench, name)))
        _STATE["arm"] = triton_launch.Arm(
            bench, launch_options=dict, after_compile=_module_init)
    return _STATE["arm"]


def pingpong(setup, size, puts, opcode):
    if opcode == 2:
        raise SystemExit("tdist-nvshmem: Triton-distributed has no uint64_p "
                         "(put_inline)")
    return _arm().pingpong(setup, size, puts, opcode)


def windowed(setup, size):
    return _arm().windowed(setup, size)


def alltoall(*arguments, **options):
    return _arm().alltoall(*arguments, **options)
