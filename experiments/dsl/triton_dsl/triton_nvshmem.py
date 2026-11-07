"""The Triton NVSHMEM arm: triton_bench.py's kernels over NVSHMEM's bitcode.

What Triton-distributed does underneath its `libshmem_device` primitives:
NVSHMEM's device bitcode linked into the Triton kernel through
`extern_libs`, the loaded module initialized with nvshmemx_cumodule_init.
Here on the series' upstream Triton, so the kernel and compiler are those of
the triton-tier1 arm and only the library differs. The policy is
comm_nvshmem.cuh call for call (as cute_nvshmem.py), through a scalar-ABI
shim (nvshmem_shim.ll) that inlines away. The world is NVSHMEM's own
(../nvshmem_world.py).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import aboleth  # noqa: F401  (configures the compiler plugin before Triton)
import triton
import triton.language as tl
from triton.language import core

import nvshmem.bindings.nvshmem as nvshmem_bindings
from nvshmem.core.init_fini import find_device_bitcode_library

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nvshmem_world  # noqa: E402
import triton_launch  # noqa: E402

SIGNAL_SET = tl.constexpr(nvshmem_world.SIGNAL_SET)
CMP_GE = tl.constexpr(nvshmem_world.CMP_GE)

_I32, _I64 = core.int32, core.int64


def _call(symbol: str, args: list, types: tuple, _semantic):
    cast = [core.cast(value, kind, _semantic=_semantic)
            for value, kind in zip(args, types)]
    return core.extern_elementwise(
        "dsl_nvshmem", "", cast, {types: (symbol, core.int64)},
        is_pure=False, _semantic=_semantic)


@core.extern
def _putmem_nbi(dst, src, nbytes, pe, _semantic=None):
    return _call("dsl_nvshmem_putmem_nbi", [dst, src, nbytes, pe],
                 (_I64, _I64, _I64, _I32), _semantic)


@core.extern
def _putmem_nbi_block(dst, src, nbytes, pe, _semantic=None):
    return _call("dsl_nvshmemx_putmem_nbi_block", [dst, src, nbytes, pe],
                 (_I64, _I64, _I64, _I32), _semantic)


@core.extern
def _putmem_signal_nbi(dst, src, nbytes, flag, value, op, pe,
                       _semantic=None):
    return _call("dsl_nvshmem_putmem_signal_nbi",
                 [dst, src, nbytes, flag, value, op, pe],
                 (_I64, _I64, _I64, _I64, _I64, _I32, _I32), _semantic)


@core.extern
def _uint64_p(dst, value, pe, _semantic=None):
    return _call("dsl_nvshmem_uint64_p", [dst, value, pe],
                 (_I64, _I64, _I32), _semantic)


@core.extern
def _signal_op(flag, value, op, pe, _semantic=None):
    return _call("dsl_nvshmemx_signal_op", [flag, value, op, pe],
                 (_I64, _I64, _I32, _I32), _semantic)


@core.extern
def _fence(_semantic=None):
    return _call("dsl_nvshmem_fence", [], (), _semantic)


@core.extern
def _quiet(_semantic=None):
    return _call("dsl_nvshmem_quiet", [], (), _semantic)


@core.extern
def _signal_wait_until(flag, cmp, value, _semantic=None):
    return _call("dsl_nvshmem_signal_wait_until", [flag, cmp, value],
                 (_I64, _I32, _I64), _semantic)


# The policy: comm_nvshmem.cuh, transcribed. Keys and the endpoint table do
# not exist in NVSHMEM; a remote address is the symmetric local one.

@triton.jit
def put(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    _putmem_nbi(dst, src, nbytes, peer)


@triton.jit
def put_block(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    """Block-collective put (every thread of the CTA): NVSHMEM's
    nvshmemx_putmem_nbi_block, as Triton-distributed's kernels send."""
    _putmem_nbi_block(dst, src, nbytes, peer)


@triton.jit
def put_inline(endpoints, peer, dst, dst_key, value):
    _uint64_p(dst, value, peer)


@triton.jit
def signal(endpoints, peer, flag, flag_key, epoch):
    _fence()
    _signal_op(flag, epoch, SIGNAL_SET, peer)


@triton.jit
def put_signal(endpoints, peer, dst, dst_key, src, src_key, nbytes, flag,
               flag_key, epoch):
    _putmem_signal_nbi(dst, src, nbytes, flag, epoch, SIGNAL_SET, peer)


@triton.jit
def fence():
    _fence()


@triton.jit
def quiet(endpoints, peer):
    _quiet()


@triton.jit
def wait_signal(flag, epoch):
    _signal_wait_until(flag, CMP_GE, epoch)


def _shim_bitcode() -> str:
    """nvshmem_shim.ll assembled with the LLVM that built Triton (cached)."""
    from aboleth.artifact import _llvm_tool
    source = HERE / "nvshmem_shim.ll"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    output = Path(triton.knobs.cache.dir) / f"dsl_nvshmem_shim_{digest}.bc"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([_llvm_tool("opt"), str(source), "-o", str(output)],
                       check=True)
    return str(output)


def make_world(args):
    return nvshmem_world.make_world(args)


def library_meta() -> dict:
    return {**nvshmem_world.library_meta("triton"),
            "nvshmem_bitcode": find_device_bitcode_library()}


def _module_init(compiled) -> None:
    nvshmem_bindings.cumodule_init(compiled.module)


_STATE: dict = {}


def _arm():
    if "arm" not in _STATE:
        bench = triton_launch.load(sys.modules[__name__], quiet_is_global=True,
                                   name="nvshmem")
        # Shim first: the linker links only what is needed, in order.
        libraries = {"dsl_nvshmem": _shim_bitcode(),
                     "libnvshmem_device": find_device_bitcode_library()}
        _STATE["arm"] = triton_launch.Arm(
            bench, launch_options=lambda: {"extern_libs": libraries},
            after_compile=_module_init)
    return _STATE["arm"]


def pingpong(setup, size, puts, opcode):
    return _arm().pingpong(setup, size, puts, opcode)


def windowed(setup, size):
    return _arm().windowed(setup, size)


def alltoall(*arguments, **options):
    return _arm().alltoall(*arguments, **options)
