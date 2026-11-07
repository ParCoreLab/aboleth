"""Compile triton_bench.py's kernels for the Triton-distributed arm, no GPU.

    envs/tdist/bin/python experiments/dsl/tdist/compile_check.py

triton_dsl/compile_check.py's cases (ping-pong, windowed with one and several
issuers, all-to-all thread- and warp-issued, fused and not) with the
tdist-nvshmem policy, compiled for sm_90a by Triton-distributed's compiler
(its hook links NVSHMEM's bitcode). Runs in Triton-distributed's
environment, so it imports no `aboleth`.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "triton_dsl"))

import tdist_nvshmem  # noqa: E402  (names its plugin before Triton loads)

from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402
from triton.compiler import compile as triton_compile  # noqa: E402
from triton_dist.jit import get_shmem_extern_lib  # noqa: E402

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


def main() -> int:
    tdist_nvshmem._install_triton_dist_hook()
    bench = triton_launch.load(tdist_nvshmem, quiet_is_global=True,
                               name="tdist")
    libraries = get_shmem_extern_lib()
    cases = []
    for op, fuse, puts in ((0, False, 1), (0, True, 1), (0, True, 4),
                           (1, False, 2)):
        for leader in (True, False):
            cases.append((f"pingpong op={op} fuse={fuse} puts={puts} "
                          f"leader={leader}", bench.pingpong, PINGPONG,
                          {"BYTES": 64, "PUTS": puts, "OP": op,
                           "LEADER": leader, "FUSE": fuse}))
    for issuers in (1, 32, 128):
        cases.append((f"windowed issuers={issuers}", bench.windowed,
                      WINDOWED, {"BYTES": 8, "ISSUERS": issuers}))
    for fuse in (False, True):
        for warp in (False, True):
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
            print(f"FAIL tdist {label}: {type(error).__name__}: "
                  f"{str(error)[:2000]}")
            continue
        calls = sorted({line.split("(")[0].split()[-1]
                        for line in compiled.asm["ptx"].splitlines()
                        if "call" in line and "nvshmem" in line})
        print(f"ok   tdist {label}"
              + (f"  calls: {calls[:4]}" if calls else ""))
    print("COMPILE-CHECK", "FAILED" if failures else "OK", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
