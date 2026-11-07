"""The CuTe Tier-1 arm: cute_bench.py's kernels over the generated library.

The CuTe counterpart of the `aboleth-generic` arm: the same generated Tier-1
library (generic RDMA+P2P, kind-bit dispatch), linked into CuTe DSL kernels
as bitcode through `aboleth.frontends.cute` instead of into nvcc kernels
through the header. The drivers own world setup, checks and reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import aboleth  # noqa: F401  (configures the compiler plugin before Triton)

import aboleth.frontends.cute as tier1

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cute_launch  # noqa: E402

_ARM: list = []


def bind(library) -> None:
    """Bind the generated library the arm links (one per process)."""
    if _ARM:
        if _ARM[0].key != library.digest:
            raise RuntimeError("cute_tier1: one library per process")
        return
    tier1.bind(library)
    _ARM.append(cute_launch.Arm(tier1, key=library.digest))


def pingpong(setup, size, puts, opcode):
    return _ARM[0].pingpong(setup, size, puts, opcode)


def windowed(setup, size):
    return _ARM[0].windowed(setup, size)


def alltoall(*arguments, **options):
    return _ARM[0].alltoall(*arguments, **options)
