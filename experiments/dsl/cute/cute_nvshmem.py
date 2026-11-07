"""The CuTe NVSHMEM arm: cute_bench.py's kernels over NVSHMEM4Py's CuTe API.

What a CuTe DSL user does today for in-kernel communication: NVSHMEM4Py's
documented CuTe device API (`nvshmem.core.device.cute`: tensor views over
the symmetric heap), the NVSHMEM device library linked as bitcode through
`cute.extern`. The policy below is comm_nvshmem.cuh call for call (the CUDA
NVSHMEM arm): put_nbi (bytes), fence + signal_op(SET) for a strong signal,
put_signal_nbi for the fused one (fenced behind earlier puts), their warp
forms, p (a uint64 scalar), a PE-wide quiet, signal_wait(GE). Each lowers to
the same NVSHMEM device call as the CUDA arm's (`int8_put_nbi` is
`putmem_nbi` by the byte). The world is NVSHMEM's own (../nvshmem_world.py);
ladder.py and alltoall.py run the same sweeps, checks and records over it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import aboleth  # noqa: F401  (configures the compiler plugin before Triton)
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint64

import nvshmem.core
import nvshmem.core.device.cute as nv
import nvshmem.core.interop.cute as nvshmem_cute

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import cute_launch  # noqa: E402
import nvshmem_world  # noqa: E402
from nvshmem_world import CMP_GE, SIGNAL_SET  # noqa: E402


# The policy: comm_nvshmem.cuh, transcribed. Keys and the endpoint table do
# not exist in NVSHMEM; a remote address is the symmetric local one, viewed
# as a tensor of its bytes (payloads) or one uint64 (flags).

@cute.jit
def _bytes(address, nbytes):
    return cute.make_tensor(
        cute.make_ptr(cutlass.Int8, Uint64(address), cute.AddressSpace.gmem),
        cute.make_layout(nbytes))


@cute.jit
def _word(address):
    return cute.make_tensor(
        cute.make_ptr(cutlass.Uint64, Uint64(address),
                      cute.AddressSpace.gmem, assumed_align=8),
        cute.make_layout(1))


@cute.jit
def put(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    nv.put_nbi(_bytes(dst, nbytes), _bytes(src, nbytes), Int32(peer))


@cute.jit
def put_warp(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    nv.put_nbi_warp(_bytes(dst, nbytes), _bytes(src, nbytes), Int32(peer))


@cute.jit
def put_inline(endpoints, peer, dst, dst_key, value):
    nv.p(_word(dst), Uint64(value), Int32(peer))


@cute.jit
def signal(endpoints, peer, flag, flag_key, epoch):
    nv.fence()
    nv.signal_op(_word(flag), Uint64(epoch), Int32(SIGNAL_SET), Int32(peer))


@cute.jit
def put_signal(endpoints, peer, dst, dst_key, src, src_key, nbytes, flag,
               flag_key, epoch):
    nv.put_signal_nbi(_bytes(dst, nbytes), _bytes(src, nbytes), _word(flag),
                      Uint64(epoch), Int32(SIGNAL_SET), Int32(peer))


@cute.jit
def put_signal_warp(endpoints, peer, dst, dst_key, src, src_key, nbytes,
                    flag, flag_key, epoch):
    nv.put_signal_nbi_warp(_bytes(dst, nbytes), _bytes(src, nbytes),
                           _word(flag), Uint64(epoch), Int32(SIGNAL_SET),
                           Int32(peer))


@cute.jit
def alltoall_block(dst, src, nbytes):
    """NVSHMEM's own all-to-all (nvshmemx_alltoallmem_block's typed form)
    over the world team: `nbytes` is every PE's slices together."""
    nv.alltoall_block(nvshmem.core.Teams.TEAM_WORLD, _bytes(dst, nbytes),
                      _bytes(src, nbytes))


@cute.jit
def fence():
    nv.fence()


@cute.jit
def quiet(endpoints, peer):
    nv.quiet()


@cute.jit
def wait_signal(flag, epoch):
    nv.signal_wait(_word(flag), Int32(CMP_GE), Uint64(epoch))


def make_world(args):
    world, bootstrap = nvshmem_world.make_world(args)
    world.finalizers.append(nvshmem_cute.cleanup_cute)
    return world, bootstrap


def library_meta() -> dict:
    return nvshmem_world.library_meta("cute")


def _compile(function, *arguments):
    compiled, kernel = nvshmem_cute.cute_compile_helper(function, *arguments)
    nvshmem_world.WORLD[0].kernels.append(kernel)
    return compiled


_ARM = cute_launch.Arm(sys.modules[__name__], quiet_is_global=True,
                       compile_kernel=_compile, key="nvshmem")


def pingpong(setup, size, puts, opcode):
    return _ARM.pingpong(setup, size, puts, opcode)


def windowed(setup, size):
    return _ARM.windowed(setup, size)


def alltoall(*arguments, **options):
    return _ARM.alltoall(*arguments, **options)
