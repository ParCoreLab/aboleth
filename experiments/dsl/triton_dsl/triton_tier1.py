"""The Triton Tier-1 arm: triton_bench.py's kernels over the generated library.

The generated Tier-1 library (generic RDMA+P2P, kind-bit dispatch) linked
into ordinary Triton kernels through `extern_libs`
(`aboleth.frontends.triton.tier1`): the same mechanism as the triton-nvshmem
arm, which links NVSHMEM's device bitcode instead, so the two differ only in
the library. The drivers own world setup, checks and reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import aboleth  # noqa: F401  (configures the compiler plugin before Triton)

import aboleth.frontends.triton.tier1 as tier1

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triton_launch  # noqa: E402

_ARM: list = []


def bind(library) -> None:
    """Bind the generated library the arm links (one per process)."""
    if _ARM:
        if _ARM[0][0] != library.digest:
            raise RuntimeError("triton_tier1: one library per process")
        return
    tier1.bind(library)
    bench = triton_launch.load(tier1, quiet_is_global=False, name="tier1")
    libraries = tier1.extern_libs()
    _ARM.append((library.digest, triton_launch.Arm(
        bench, launch_options=lambda: {"extern_libs": libraries})))


def pingpong(setup, size, puts, opcode):
    return _ARM[0][1].pingpong(setup, size, puts, opcode)


def windowed(setup, size):
    return _ARM[0][1].windowed(setup, size)


def alltoall(*arguments, **options):
    return _ARM[0][1].alltoall(*arguments, **options)
