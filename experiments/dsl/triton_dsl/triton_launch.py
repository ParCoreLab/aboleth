"""Launchers for triton_bench.py's kernels, shared by the Triton library arms.

`load(policy, ...)` instantiates triton_bench.py with an arm's policy as its
COMM (one module, and so one set of jit functions and caches, per arm).
`Arm` drives the instance with tier2.py's launcher protocol, so ladder.py and
alltoall.py run it unchanged. Each kernel is compiled with `warmup` before
its first launch, so an arm can initialize the loaded module first
(`after_compile`: NVSHMEM's cumodule_init); its registers, spills, shared
bytes and compile time land in the driver's kernel records.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch
import triton.language as tl

HERE = Path(__file__).resolve().parent


def load(policy, *, quiet_is_global: bool, name: str,
         source: str = "triton_bench.py"):
    """Instantiate a policy-generic kernel module (triton_bench.py, or W4's
    dispatch_kernels.py) with `policy` as its COMM."""
    spec = importlib.util.spec_from_file_location(
        f"{Path(source).stem}_{name}", HERE / source)
    module = importlib.util.module_from_spec(spec)
    module.COMM = policy
    module.QUIET_IS_GLOBAL = tl.constexpr(quiet_is_global)
    module.HAS_FENCE = tl.constexpr(hasattr(policy, "fence"))
    module.FENCE = getattr(policy, "fence", None)
    module.PUT_WARP = getattr(policy, "put_warp", None)
    module.HAS_PUT_SIGNAL_WARP = tl.constexpr(
        hasattr(policy, "put_signal_warp"))
    module.PUT_SIGNAL_WARP = getattr(policy, "put_signal_warp", None)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Arm:
    def __init__(self, bench, *, launch_options, after_compile=None):
        self.bench = bench
        self.launch_options = launch_options
        self.after_compile = after_compile

    def _prepare(self, records, name, size, ctas, kernel, grid, arguments,
                 constants, threads: int = 32):
        options = {"num_warps": threads // 32, **self.launch_options(),
                   **constants}
        began = time.perf_counter()
        compiled = kernel.warmup(*arguments, grid=grid, **options)
        elapsed = time.perf_counter() - began
        compiled._init_handles()
        if self.after_compile is not None:
            self.after_compile(compiled)
        records.append({
            "kernel": name, "size": size, "registers": compiled.n_regs,
            "spills": compiled.n_spills,
            "shared_bytes": compiled.metadata.shared, "threads": threads,
            "ctas": ctas, "compile_s": elapsed,
            "cubin_bytes": len(compiled.asm.get("cubin", b""))})
        return options

    def pingpong(self, setup, size: int, puts: int, opcode: int):
        constants = {"BYTES": size, "PUTS": puts, "OP": opcode,
                     "LEADER": setup.leader,
                     "FUSE": setup.args.fuse_signal == "on"}
        fixed = (setup.table.endpoint_handles, setup.peer,
                 setup.remote_data.address, setup.remote_data.rkey,
                 setup.local_data.address, setup.local_data.lkey,
                 setup.remote_flags.address + setup.rank * 8,
                 setup.remote_flags.rkey,
                 setup.local_flags.address + setup.peer * 8)
        state = {}

        def run(warmup: int, iters: int) -> None:
            arguments = (*fixed, warmup, iters, setup.samples)
            if "options" not in state:
                state["options"] = self._prepare(
                    setup.kernels, "pingpong", size, 1, self.bench.pingpong,
                    (1,), arguments, constants)
            self.bench.pingpong[(1,)](*arguments, **state["options"])
            torch.cuda.synchronize()
        return run

    def windowed(self, setup, size: int):
        barrier = torch.zeros(1, dtype=torch.int32, device=setup.device)
        issuers = setup.args.issuers
        stream = int(setup.args.stream == "on")
        state = {}

        def run(iters: int, record: bool, ctas: int) -> None:
            barrier.zero_()
            arguments = (setup.table.endpoint_handles, setup.peer,
                         setup.remote_data.address, setup.remote_data.rkey,
                         setup.local_data.address, setup.local_data.lkey,
                         setup.args.window, iters, setup.data_bytes // size,
                         setup.samples, setup.spans, int(record), barrier,
                         stream)
            if "options" not in state:
                state["options"] = self._prepare(
                    setup.kernels, "bandwidth", size, setup.args.ctas,
                    self.bench.windowed, (ctas,), arguments,
                    {"BYTES": size, "ISSUERS": issuers},
                    threads=max(issuers, 32))
            self.bench.windowed[(ctas,)](*arguments, **state["options"])
        return run

    def alltoall(self, world, table, source, source_key, flags, size,
                 stride, receive, samples, barrier, records, drain=False,
                 args=None):
        rank, world_size = world.rank, world.world_size
        peers = [peer for peer in range(world_size) if peer != rank]
        device = samples.device

        def column(values, dtype=torch.int64):
            return torch.tensor(values, dtype=dtype, device=device)

        remote_data = [table.remote(peer, "data") for peer in peers]
        remote_flags = [table.remote(peer, "flags") for peer in peers]
        local_flags = table.local("flags")
        tables = (
            column([r.address + receive + rank * stride
                    for r in remote_data]),
            column([r.rkey for r in remote_data], torch.int32),
            column([source + peer * stride for peer in peers]),
        )
        flag_tables = (
            column([r.address + rank * 8 for r in remote_flags]),
            column([r.rkey for r in remote_flags], torch.int32),
            column([local_flags.address + peer * 8 for peer in peers]),
        )
        constants = {"BYTES": size,
                     "FUSE": args is not None and args.fuse_signal == "on",
                     "WARP": args is not None and args.put_issue == "warp"}
        grid = (len(peers),)
        state = {}

        def run(warmup: int, iters: int) -> None:
            barrier.zero_()
            arguments = (table.endpoint_handles, rank, *tables, source_key,
                         *flag_tables, warmup, iters, samples, barrier)
            if "options" not in state:
                state["options"] = self._prepare(
                    records, "alltoall", size, len(peers),
                    self.bench.alltoall, grid, arguments, constants)
            self.bench.alltoall[grid](*arguments, **state["options"])
            torch.cuda.synchronize()
        return run
