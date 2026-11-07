"""Compile triton_bench.py's kernels for every Triton library arm, no GPU.

    python experiments/dsl/triton_dsl/compile_check.py [--policy tier1,nvshmem]

Compiles the ping-pong (each op, fused and not), the windowed kernel and the
all-to-all for sm_90a with each arm's extern_libs, and reports registers,
spills and any library call that survived inlining in the PTX.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "bench"))

import aboleth  # noqa: F401,E402

from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402
from triton.compiler import compile as triton_compile  # noqa: E402

import triton_launch  # noqa: E402

TARGET = GPUTarget("cuda", 90, 32)
PINGPONG = {"endpoints": "*u64", "peer": "i32", "dst": "i64", "dst_key": "i32",
            "src": "i64", "src_key": "i32", "peer_flag": "i64",
            "flag_key": "i32", "my_flag": "i64", "warmup": "i32",
            "iters": "i32", "samples": "*i64", "BYTES": "constexpr",
            "PUTS": "constexpr", "OP": "constexpr", "LEADER": "constexpr",
            "FUSE": "constexpr"}
WINDOWED = {"endpoints": "*u64", "peer": "i32", "dst": "i64", "dst_key": "i32",
            "src": "i64", "src_key": "i32", "window": "i32", "iters": "i32",
            "slots": "i64", "samples": "*i64", "spans": "*i64",
            "record": "i32", "barrier": "*i32", "stream": "i32",
            "BYTES": "constexpr", "ISSUERS": "constexpr"}
ALLTOALL = {"endpoints": "*u64", "rank": "i32", "dst": "*i64",
            "dst_key": "*i32", "src": "*i64", "src_key": "i32",
            "flag": "*i64", "flag_key": "*i32", "my_flag": "*i64",
            "warmup": "i32", "iters": "i32", "samples": "*i64",
            "barrier": "*i32", "BYTES": "constexpr", "FUSE": "constexpr",
            "WARP": "constexpr"}


def check(bench, libraries: dict, name: str, prefix: str,
          forms=frozenset()) -> int:
    """`forms`: the arm's issue forms beyond one thread (common.DSL_ISSUE)."""
    cases = []
    for op, fuse, puts in ((0, False, 1), (0, True, 1), (0, True, 4),
                           (1, False, 2), (2, False, 1)):
        for leader in (True, False):
            cases.append((f"pingpong op={op} fuse={fuse} puts={puts} "
                          f"leader={leader}", bench.pingpong, PINGPONG,
                          {"BYTES": 64, "PUTS": puts, "OP": op,
                           "LEADER": leader, "FUSE": fuse}))
    cases.append(("windowed", bench.windowed, WINDOWED,
                  {"BYTES": 4096, "ISSUERS": 1}))
    if "issuers" in forms:
        for issuers in (32, 128):
            cases.append((f"windowed issuers={issuers}", bench.windowed,
                          WINDOWED, {"BYTES": 8, "ISSUERS": issuers}))
    for fuse in (False, True):
        for warp in (False, True):
            if warp and ("warp" not in forms
                         or (fuse and "warp-signal" not in forms)):
                continue
            cases.append((f"alltoall fuse={fuse} warp={warp}", bench.alltoall,
                          ALLTOALL, {"BYTES": 65536 if warp else 256,
                                     "FUSE": fuse, "WARP": warp}))
    failures = 0
    for label, function, signature, constants in cases:
        warps = max(constants.get("ISSUERS", 1), 32) // 32
        try:
            compiled = triton_compile(
                ASTSource(function, signature=signature,
                          constexprs=constants),
                target=TARGET,
                options={"num_warps": warps, "extern_libs": libraries})
        except Exception as error:  # noqa: BLE001 - report every case
            failures += 1
            print(f"FAIL {name} {label}: {type(error).__name__}: "
                  f"{str(error)[:2000]}")
            continue
        calls = [line.strip() for line in compiled.asm["ptx"].splitlines()
                 if "call" in line and prefix in line]
        print(f"ok   {name} {label}"
              + (f"  residual calls: {calls[:2]}" if calls else ""))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="tier1,nvshmem")
    args = parser.parse_args()
    policies = args.policy.split(",")
    failures = 0
    if "tier1" in policies:
        import aboleth.frontends.triton.tier1 as tier1
        from aboleth.device_library import cached_device_library
        tier1.bind(cached_device_library("sm_90a"))
        bench = triton_launch.load(tier1, quiet_is_global=False,
                                   name="tier1")
        failures += check(bench, tier1.extern_libs(), "tier1",
                          "aboleth_tier1", {"warp", "issuers"})
    if "nvshmem" in policies:
        import triton_nvshmem
        bench = triton_launch.load(triton_nvshmem, quiet_is_global=True,
                                   name="nvshmem")
        from nvshmem.core.init_fini import find_device_bitcode_library
        try:
            nvshmem_bc = find_device_bitcode_library()
        except Exception:  # noqa: BLE001 - no device on the login node
            import nvidia.nvshmem
            nvshmem_bc = str(Path(list(nvidia.nvshmem.__path__)[0]) / "lib"
                             / "libnvshmem_device_sm_90.bc")
        failures += check(bench, {"dsl_nvshmem": triton_nvshmem._shim_bitcode(),
                                  "libnvshmem_device": nvshmem_bc},
                          "nvshmem", "nvshmem")
    print("COMPILE-CHECK", "FAILED" if failures else "OK", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
