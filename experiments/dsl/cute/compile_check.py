"""Compile cute_bench.py's kernels against every CuTe policy, without a GPU.

    CUTE_DSL_ARCH=sm_90a python experiments/dsl/cute/compile_check.py \
        [--library DIR] [--policy tier1,nvshmem]

Traces and compiles the ping-pong (each op, fused and not), the windowed
kernel and the all-to-all for each policy, and reports whether any extern
call survived inlining in the PTX. A login-node check before GPU time.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aboleth  # noqa: F401,E402

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import make_fake_compact_tensor  # noqa: E402

import cute_bench  # noqa: E402


def fake(dtype, n):
    return make_fake_compact_tensor(dtype, (n,), assumed_align=8)


def residual_calls(compiled, prefix: str) -> list[str]:
    ptx = getattr(compiled, "__ptx__", None) or ""
    if isinstance(ptx, bytes):
        ptx = ptx.decode()
    return [line.strip() for line in ptx.splitlines()
            if "call" in line and prefix in line]


def check(policy, name: str, prefix: str, quiet_is_global: bool,
          forms=frozenset()) -> int:
    """`forms`: the arm's issue forms beyond one thread (common.DSL_ISSUE)."""
    cute_bench.use(policy, quiet_is_global)
    endpoints = fake(cutlass.Int64, 8)
    samples = fake(cutlass.Int64, 64)
    spans = fake(cutlass.Int64, 32)
    barrier = fake(cutlass.Int32, 1)
    failures = 0
    cases = []
    for op, fuse, puts in ((0, False, 1), (0, True, 1), (0, True, 4),
                           (1, False, 2), (2, False, 1)):
        for leader in (True, False):
            cases.append((f"pingpong op={op} fuse={fuse} puts={puts} "
                          f"leader={leader}", cute_bench.pingpong,
                          (endpoints, 1, 1 << 40, 7, 1 << 41, 9, 1 << 42, 11,
                           1 << 43, 10, 100, samples,
                           64, puts, op, leader, fuse)))
    for issuers, stream in ((1, 0), (32, 0), (128, 0), (128, 1)):
        if (issuers > 1 and "issuers" not in forms) or \
                (stream and "stream" not in forms):
            continue
        cases.append((f"windowed issuers={issuers} stream={stream}",
                      cute_bench.windowed,
                      (endpoints, 1, 1 << 40, 7, 1 << 41, 9, 128, 100, 1024,
                       samples, spans, 1, barrier, 4, stream,
                       8 if issuers > 1 else 4096, issuers)))
    tables = [fake(cutlass.Int64, 3), fake(cutlass.Int32, 3),
              fake(cutlass.Int64, 3), fake(cutlass.Int64, 3),
              fake(cutlass.Int32, 3), fake(cutlass.Int64, 3)]
    for fuse in (False, True):
        for warp in (False, True):
            if warp and ("warp" not in forms
                         or (fuse and "warp-signal" not in forms)):
                continue
            cases.append((f"alltoall fuse={fuse} warp={warp}",
                          cute_bench.alltoall,
                          (endpoints, 1, tables[0], tables[1], tables[2], 9,
                           tables[3], tables[4], tables[5], 10, 100, samples,
                           barrier, 3, 65536 if warp else 256, fuse, warp)))
    if "collective" in forms:
        cases.append(("alltoall collective", cute_bench.alltoall_collective,
                      (1 << 40, 1 << 41, 10, 100, samples, 256, 8)))
    for label, function, arguments in cases:
        try:
            compiled = cute.compile(function, *arguments,
                                    options="--keep-ptx --keep-cubin")
        except Exception as error:  # noqa: BLE001 - report every case
            failures += 1
            print(f"FAIL {name} {label}: {type(error).__name__}: {error}")
            continue
        calls = residual_calls(compiled, prefix)
        print(f"ok   {name} {label}"
              + (f"  residual calls: {calls[:3]}" if calls else ""))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="generated Tier-1 library dir")
    parser.add_argument("--policy", default="tier1,nvshmem")
    args = parser.parse_args()
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")
    failures = 0
    policies = args.policy.split(",")
    if "tier1" in policies:
        import aboleth.frontends.cute as tier1
        from aboleth.device_library import cached_device_library, \
            generate_device_library
        library = (generate_device_library("sm_90a",
                                           output_dir=Path(args.library))
                   if args.library
                   else cached_device_library("sm_90a"))
        tier1.bind(library)
        failures += check(tier1, "tier1", "aboleth_tier1", False,
                          {"warp", "issuers"})
    if "nvshmem" in policies:
        import cute_nvshmem
        failures += check(cute_nvshmem, "nvshmem", "nvshmem", True,
                          {"warp", "warp-signal", "issuers", "stream",
                           "collective"})
    print("COMPILE-CHECK", "FAILED" if failures else "OK", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
