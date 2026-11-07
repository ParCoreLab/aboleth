#!/usr/bin/env python3
"""Tier-1 point-to-point ladder: persistent ping-pong latency and windowed
one-way bandwidth over the Tier-1 device library, sweeping message sizes
nccl-tests-style. Two ranks; rank 0 leads and reports."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

LATENCY_DEFAULTS = ("8", "64K")
BANDWIDTH_DEFAULTS = ("4K", "256M")

_PINGPONG_ARGTYPES = [
    ctypes.c_uint64, ctypes.c_uint64,               # endpoints, doorbell_locks
    ctypes.c_int, ctypes.c_int,                     # peer, leader
    ctypes.c_uint64, ctypes.c_uint32,               # dst address/key
    ctypes.c_uint64, ctypes.c_uint32,               # src address/key
    ctypes.c_uint64, ctypes.c_uint32,               # peer_flag address/key
    ctypes.c_uint64,                                # my_flag pointer
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int,  # bytes, puts, op
    ctypes.c_uint32, ctypes.c_uint32,               # warmup, iters
    ctypes.c_uint64,                                # samples pointer
    ctypes.c_uint64, ctypes.c_uint32,               # l2 pointer/bytes
    ctypes.c_int,                                   # threads
    ctypes.c_uint64,                                # put_many descs pointer
]

_BW_ARGTYPES = [
    ctypes.c_uint64, ctypes.c_uint64,               # endpoints, doorbell_locks
    ctypes.c_int,                                   # peer
    ctypes.c_uint64, ctypes.c_uint32,               # dst address/key
    ctypes.c_uint64, ctypes.c_uint32,               # src address/key
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,  # bytes, window, iters
    ctypes.c_uint64, ctypes.c_uint64,               # arena_bytes, samples ptr
    ctypes.c_int,                                   # ctas
    ctypes.c_uint64,                                # put_many descs pointer
    ctypes.c_uint64,                                # spans pointer
    ctypes.c_uint32,                                # issuers per CTA
]

# p2p and auto (mixed) worlds ride the same adapter as rdma: the generated
# header dispatches per peer on the handle's kind bits, so the bench code
# has no per-transport branch to make.
_ADAPTERS = {"rdma": ("comm_aboleth.cuh", "AbolethComm"),
             "p2p": ("comm_aboleth.cuh", "AbolethComm"),
             "auto": ("comm_aboleth.cuh", "AbolethComm"),
             "proxy": ("comm_aboleth_proxy.cuh", "AbolethProxyComm")}


def render_prelude(transport: str, size: int, puts: int, opcode: int,
                   warp: bool = False, fuse: bool = False,
                   bw_flush: bool = False,
                   small_inline: bool = False, bake: bool = False) -> str:
    # Unbaked (the default), the kernel takes the size, put count and op at
    # run time, as the library binaries' kernels do; baking them is what
    # Tier 2's compiler does, so Tier 1 bakes only on request.
    header, comm = _ADAPTERS[transport]
    if not bake:
        size, puts, opcode = 0, 0, -1
    return (
        # Before the adapter include: the generated header reads it.
        f"#define ABOLETH_TIER1_SMALL_INLINE {1 if small_inline else 0}\n"
        f'#include "{header}"\n'
        f"#define BENCH_COMM {comm}\n"
        f"#define BENCH_BAKED_BYTES {size}u\n"
        f"#define BENCH_BAKED_PUTS {puts}u\n"
        f"#define BENCH_BAKED_OP {opcode}\n"
        f"#define BENCH_BAKED_WARP_PUTS {1 if warp else 0}\n"
        f"#define BENCH_BAKED_FUSE_SIGNAL {1 if fuse else 0}\n"
        f"#define BENCH_BAKED_BW_FLUSH {1 if bw_flush else 0}\n"
        f'#include "launch_aboleth.cu"\n')


class _Setup:
    """Everything the sweeps share: world, arena, region table, launchers."""

    def __init__(self, args, world, bootstrap, data_bytes: int):
        import torch
        self.args, self.world, self.bootstrap = args, world, bootstrap
        self.rank, self.world_size = bootstrap.rank, bootstrap.world_size
        self.peer, self.leader = 1 - self.rank, self.rank == 0
        self.device = torch.device(f"cuda:{world.device}")
        self.data_bytes = data_bytes
        flags_offset = (data_bytes + 7) // 8 * 8
        self.arena = common.arena(world, flags_offset + 8 * self.world_size,
                                  self.device)
        registration = world.register(self.arena)
        self.table = world.exchange_regions({
            "data": world.region("data", registration,
                                 offset=0, length=data_bytes),
            "flags": world.region("flags", registration, offset=flags_offset,
                                  length=8 * self.world_size),
        })
        self.flags_view = self.arena[flags_offset:].view(torch.int64)
        self.endpoints_ptr = self.table.endpoint_handles.data_ptr()
        self.local_data = self.table.local("data")
        self.remote_data = self.table.remote(self.peer, "data")
        self.local_flags = self.table.local("flags")
        self.remote_flags = self.table.remote(self.peer, "flags")
        self.samples = torch.zeros(max(args.iters, 1),
                                   dtype=torch.int64, device=self.device)
        # Per-CTA [start, end] device timestamps of a bandwidth launch.
        self.spans = torch.zeros(2 * max(args.ctas, 1), dtype=torch.int64,
                                 device=self.device)
        self.kernels = []
        self.locks = (torch.zeros(self.world_size, dtype=torch.int32,
                                  device=self.device)
                      if args.ctas > 1 else None)
        (self.include_dirs, self.extra_flags, self.extra_objects,
         self.library_meta) = common.library_options(
            args, world, bootstrap)
        # Descriptor scratch for the generated composite put path: global
        # memory the kernel stages PutDesc records into (32 bytes each).
        self.put_descs = (
            torch.zeros(args.puts * 32, dtype=torch.uint8,
                        device=self.device)
            if (self.library_meta.get("library") in
                ("generated", "specialized")
                and args.put_many == "auto" and args.puts > 1)
            else None)
        self.compile_records = []
        self._step = 0

    def barrier(self) -> None:
        self._step += 1
        self.bootstrap.barrier(f"ladder:{self._step}")

    def library(self, size: int, puts: int, opcode: int, tag: str,
                bw_flush: bool = False):
        handle, record = common.compile_shape_library(
            render_prelude(self.args.transport, size, puts, opcode,
                           warp=self.args.put_issue == "warp",
                           fuse=self.args.fuse_signal == "on",
                           bw_flush=bw_flush,
                           small_inline=self.args.small_inline == "on",
                           bake=self.args.bake == "on"),
            arch=common.device_arch(), cache_dir=common.default_cache_dir(),
            tag=tag, include_dirs=self.include_dirs,
            extra_flags=self.extra_flags, extra_objects=self.extra_objects)
        self.compile_records.append(record)
        return handle

    def pingpong(self, size: int, puts: int, opcode: int):
        """run(warmup, iters) for one latency shape."""
        import torch
        module = common.kernel_module(self.args.arm)
        if module is not None:
            return module.pingpong(self, size, puts, opcode)
        handle = self.library(size, puts, opcode, f"pp_{self.args.transport}")
        self.kernels.append({"kernel": "pingpong", "size": size,
                             **common.kernel_attributes(handle, 0, 32, 1)})
        fn = handle.bench_launch_pingpong
        fn.restype, fn.argtypes = ctypes.c_int, _PINGPONG_ARGTYPES
        l2 = self.l2_buffer
        descs = self.put_descs

        def run(warmup: int, iters: int) -> None:
            code = fn(self.endpoints_ptr, 0, self.peer, int(self.leader),
                      self.remote_data.address, self.remote_data.rkey,
                      self.local_data.address, self.local_data.lkey,
                      self.remote_flags.address + self.rank * 8,
                      self.remote_flags.rkey,
                      self.local_flags.address + self.peer * 8,
                      size, puts, opcode, warmup, iters,
                      self.samples.data_ptr(),
                      l2.data_ptr() if l2 is not None else 0,
                      l2.numel() if l2 is not None else 0,
                      256 if l2 is not None else 32,
                      descs.data_ptr() if descs is not None else 0)
            if code:
                raise RuntimeError(f"ladder: launch failed, cudaError {code}")
            torch.cuda.synchronize()
        return run

    def share_wrong(self, tag: str, wrong: int) -> int:
        gathered = self.bootstrap.all_gather(tag, str(wrong).encode())
        return max(int(value.decode()) for value in gathered.values())


def run_latency(setup: _Setup) -> tuple[list[dict], int]:
    import torch
    args, arena = setup.args, setup.arena
    ops = [op.strip() for op in args.op.split(",")]
    for op in ops:
        if op not in common.OP_CODES:
            raise SystemExit(f"ladder: unknown --op {op!r}")
    sizes = common.resolve_sweep(args, *LATENCY_DEFAULTS)
    setup.l2_buffer = (torch.zeros(256 << 20, dtype=torch.uint8,
                                   device=setup.device)
                       if args.l2_flush else None)
    rows, failures = [], 0

    for op in ops:
        opcode = common.OP_CODES[op]
        # The facade's put_inline carries exactly 8 bytes from registers.
        for size in ([8] if op == "put_inline" else sizes):
            puts = 1 if op == "put_inline" else args.puts
            payload = size * puts
            run = setup.pingpong(size, puts, opcode)

            # Correctness before timing (the tree's convention): the leader
            # sends the pattern once, the follower asserts on it.
            if setup.leader:
                common.pattern_fill_gpu(arena[:payload])
            else:
                arena[:payload].zero_()
            setup.flags_view.zero_()
            torch.cuda.synchronize()
            setup.barrier()
            run(0, 1)
            setup.barrier()
            wrong = 0
            if not setup.leader:
                if op == "put_inline":
                    wrong = int(setup.flags_view[setup.peer].item() != 1)
                else:
                    wrong = common.pattern_wrong_gpu(arena[:payload])
            wrong = setup.share_wrong(f"wrong:{op}:{size}", wrong)
            failures += wrong
            if wrong:
                print(f"FAIL: {op} size {size}: {wrong} wrong",
                      file=sys.stderr)

            # The timed rounds move a differently salted pattern into a
            # zeroed receiver, so the check after timing sees only what
            # they delivered.
            if setup.leader:
                common.pattern_fill_gpu(arena[:payload], common.TIMED_SALT)
            else:
                arena[:payload].zero_()
            setup.flags_view.zero_()
            torch.cuda.synchronize()
            setup.barrier()
            run(args.warmup, args.iters)
            setup.barrier()
            # Correctness after timing: every round's epoch arrived, and the
            # payload holds the timed rounds' pattern.
            after = int(setup.flags_view[setup.peer].item()
                        != args.warmup + args.iters)
            if not setup.leader and op != "put_inline":
                after += common.pattern_wrong_gpu(arena[:payload],
                                                  common.TIMED_SALT)
            after = setup.share_wrong(f"wrong-after:{op}:{size}", after)
            if after:
                print(f"FAIL: {op} size {size}: {after} wrong after timing",
                      file=sys.stderr)
            wrong += after
            failures += after

            if setup.leader:
                raw = setup.samples.tolist()
                samples_us = [value / 1000.0 for value in raw]
                row = {"op": op, "size": size, "count": puts,
                       **common.percentiles(samples_us), "wrong": wrong,
                       "samples_ns": raw}
                gbps = (payload / (row["mean_us"] * 1e-6) / 1e9
                        if row["mean_us"] else 0.0)
                row["algbw_gbps"] = row["busbw_gbps"] = gbps
                rows.append(row)
    return rows, failures


def run_bandwidth(setup: _Setup, sizes: list[int]) -> tuple[list[dict], int]:
    import torch
    args, arena = setup.args, setup.arena
    rows, failures = [], 0

    # Hand-batched windows (--batch-window on): each window is one batched
    # submission carrying its own retirement (put_many_flush) — one doorbell
    # + one CQE where the per-put loop pays a doorbell per put plus quiet's
    # second posting sequence. Off by default: it is the batching Tier 2's
    # compiler derives by itself, so Tier 1 uses it only as a labelled arm,
    # as NCCL does its aggregated requests. Only meaningful on queue-backed
    # transports; p2p windows have no doorbell to fuse.
    kernel_arm = common.kernel_module(args.arm)
    bw_flush = (
        setup.library_meta.get("library") in ("generated", "specialized")
        and args.batch_window == "on"
        and args.put_many == "auto"
        and args.put_issue != "warp"
        and args.transport != "p2p")
    bw_descs = (
        torch.zeros(args.ctas * args.window * 32, dtype=torch.uint8,
                    device=setup.device)
        if bw_flush else None)

    def launch(fn, size, iters, samples_ptr, ctas) -> None:
        spans = setup.spans.data_ptr() if samples_ptr else 0
        if kernel_arm is not None:
            fn(iters, bool(samples_ptr), ctas)
            return
        code = fn(setup.endpoints_ptr,
                  setup.locks.data_ptr() if setup.locks is not None else 0,
                  setup.peer,
                  setup.remote_data.address, setup.remote_data.rkey,
                  setup.local_data.address, setup.local_data.lkey,
                  size, args.window, iters, setup.data_bytes,
                  samples_ptr, ctas,
                  bw_descs.data_ptr() if bw_descs is not None else 0,
                  spans, args.issuers)
        if code:
            raise RuntimeError(f"ladder: launch failed, cudaError {code}")

    for size in sizes:
        checked = args.window * size
        if kernel_arm is not None:
            fn = kernel_arm.windowed(setup, size)
        else:
            handle = setup.library(size, 0, 0, f"bw_{args.transport}",
                                   bw_flush=bw_flush)
            setup.kernels.append({
                "kernel": "bandwidth", "size": size,
                **common.kernel_attributes(
                    handle, 1, args.issuers if args.issuers > 1 else 32,
                    args.ctas)})
            fn = handle.bench_launch_windowed_bw
            fn.restype, fn.argtypes = ctypes.c_int, _BW_ARGTYPES

        # Correctness: one single-CTA iteration covers [0, window*size) once;
        # quiet means the RC ack landed, so a host barrier suffices.
        if setup.leader:
            common.pattern_fill_gpu(arena[:checked])
        else:
            arena[:checked].zero_()
        torch.cuda.synchronize()
        setup.barrier()
        if setup.leader:
            launch(fn, size, 1, 0, 1)
            torch.cuda.synchronize()
        setup.barrier()
        wrong = 0
        if not setup.leader:
            wrong = common.pattern_wrong_gpu(arena[:checked])
        wrong = setup.share_wrong(f"bw-wrong:{size}", wrong)
        failures += wrong
        if wrong:
            print(f"FAIL: bandwidth size {size}: {wrong} wrong",
                  file=sys.stderr)

        # The timed windows move a differently salted pattern into a
        # receiver zeroed after the warmup: the check after timing sees only
        # what the timed launch delivered, in every slot it touched (every
        # CTA's and issuer's windows wrap the arena from slot 0).
        slots = setup.data_bytes // size
        touched = min(slots, args.ctas * args.window * args.iters) * size
        if setup.leader:
            common.pattern_fill_gpu(arena[:slots * size], common.TIMED_SALT)
        torch.cuda.synchronize()
        setup.barrier()
        span_ns = 0
        if setup.leader:
            launch(fn, size, args.warmup, 0, args.ctas)   # warmup
            torch.cuda.synchronize()
        setup.barrier()
        if not setup.leader:
            arena[:touched].zero_()
            torch.cuda.synchronize()
        setup.barrier()
        if setup.leader:
            setup.spans.zero_()
            launch(fn, size, args.iters, setup.samples.data_ptr(), args.ctas)
            torch.cuda.synchronize()
            # The device clock, as for every arm: first start to last end.
            spans = setup.spans.view(-1, 2)
            span_ns = int(spans[:, 1].max() - spans[:, 0].min())
        setup.barrier()
        after = 0
        if not setup.leader:
            after = common.pattern_wrong_gpu(arena[:touched],
                                             common.TIMED_SALT)
        after = setup.share_wrong(f"bw-wrong-after:{size}", after)
        if after:
            print(f"FAIL: bandwidth size {size}: {after} wrong after timing",
                  file=sys.stderr)
        wrong += after
        failures += after

        if setup.leader:
            puts = args.ctas * args.window * args.iters
            gbps = puts * size / span_ns if span_ns else 0.0
            raw = setup.samples.tolist()
            samples_us = [value / 1000.0 for value in raw]
            row = {"op": "put", "size": size, "count": args.window,
                   **common.percentiles(samples_us), "wrong": wrong,
                   "samples_ns": raw, "span_ns": span_ns, "ctas": args.ctas,
                   "issuers": args.issuers,
                   "window": args.window, "iters": args.iters,
                   "mops": puts / span_ns * 1e3 if span_ns else 0.0}
            row["algbw_gbps"] = row["busbw_gbps"] = gbps
            rows.append(row)
    return rows, failures


def rank_main(args) -> int:
    common.check_tier2_flags(args)
    world, bootstrap = common.make_world(args)
    if bootstrap.world_size != 2:
        raise SystemExit("ladder: exactly 2 ranks (use alltoall.py for more)")
    if args.put_issue == "warp":
        # put_warp exists only in the generated header, and the warp path
        # drives payload puts only — protocol ops stay on lane 0.
        if (args.library != "generated"
                and args.arm != "aboleth-specialized"):
            raise SystemExit(
                "ladder: --put-issue warp needs --library generated "
                "(put_warp is a generated-header op)")
        if {op.strip() for op in args.op.split(",")} != {"put"}:
            raise SystemExit(
                "ladder: --put-issue warp supports --op put only")
        args.put_many = "off"

    generated = (args.library == "generated"
                 or args.arm == "aboleth-specialized")
    if args.fuse_signal == "on":
        # put_signal/put_many_signal are generated-header ops, and the fused
        # kernels drive lane-issued puts only.
        if not generated:
            raise SystemExit(
                "ladder: --fuse-signal on needs --library generated")
        if args.put_issue == "warp":
            raise SystemExit(
                "ladder: --fuse-signal on composes with "
                "--put-issue thread only")
        if {op.strip() for op in args.op.split(",")} - {"put"}:
            raise SystemExit(
                "ladder: --fuse-signal on supports --op put only")
    if args.small_inline == "on" and not generated:
        raise SystemExit(
            "ladder: --small-inline on needs --library generated")
    if args.exclusive == "on" and not generated:
        raise SystemExit(
            "ladder: --exclusive on needs --library generated")
    if (args.stream != "off" and args.arm not in common.DSL_ARMS
            and args.arm != "aboleth-triton"):
        # (A DSL arm streams only over a self-reclaiming library:
        # common.check_dsl_flags. Tier 2 streams through the compiler's
        # queue reclamation.)
        raise SystemExit("ladder: --stream: a Tier-1 put waits for room only "
                         "a quiet frees; amortize it with a larger --window")
    if args.issuers > 1:
        # Several threads of a CTA share its endpoint: the reservation
        # atomic and the publication-order ticket (the non-exclusive
        # build) are what make that safe, and the window must fit the
        # queue (Tier-1 puts wait for room only a quiet reclaims).
        if args.mode != "bandwidth":
            raise SystemExit("ladder: --issuers drives the windowed "
                             "bandwidth kernel only")
        if args.exclusive == "on":
            raise SystemExit("ladder: --issuers > 1 shares the CTA's "
                             "endpoint; the exclusive-issuer build forbids it")
        if args.put_issue == "warp" or args.batch_window == "on":
            raise SystemExit("ladder: --issuers > 1 composes with "
                             "lane-issued per-put windows only")

    clamp_notes: list[str] = []
    if (args.ctas > args.endpoints_per_peer and args.transport != "p2p"
            and not args.arm.endswith("-nvshmem")):
        # Two CTAs draining one endpoint's CQ deadlock: the Tier-1 quiet is
        # a single-consumer poll. Multi-CTA on rdma needs per-CTA endpoints:
        # --endpoints-per-peer N fans the world out so CTA b owns slot b % N
        # and up to N CTAs run safely. A pure-p2p world has no queue at all
        # — quiet is vacuous — so its CTA count stands.
        note = (f"ctas clamped {args.ctas} -> {args.endpoints_per_peer}: "
                "the Tier-1 quiet is single-consumer per endpoint; raise "
                "--endpoints-per-peer to fan out")
        print(f"# clamp: {note}")
        clamp_notes.append(note)
        args.ctas = args.endpoints_per_peer
    if args.mode == "latency":
        sizes = common.resolve_sweep(args, *LATENCY_DEFAULTS)
        data_bytes = max(max(sizes) * args.puts, 8)
    else:
        sizes = common.resolve_sweep(args, *BANDWIDTH_DEFAULTS)
        sizes, sizes_notes = common.clamp_sizes(
            sizes, lambda size: args.window * size + (1 << 20))
        clamp_notes += sizes_notes
        data_bytes = args.window * max(sizes)

    setup = _Setup(args, world, bootstrap, data_bytes)
    if args.mode == "latency":
        rows, failures = run_latency(setup)
    else:
        rows, failures = run_bandwidth(setup, sizes)

    setup.barrier()
    setup.table.close()
    if setup.leader:
        submit = common.submit_label(args.transport)
        meta = {"suite": "tier1", "benchmark": "ladder", "mode": args.mode,
                "arm": args.arm, "transport": args.transport,
                "submit_mode": submit, "world": setup.world_size,
                "ctas": args.ctas, "issuers": args.issuers,
                **setup.library_meta,
                "put_many": (setup.put_descs is not None
                             or (args.mode == "bandwidth"
                                 and args.batch_window == "on"
                                 and args.put_many == "auto" and generated
                                 and args.put_issue != "warp"
                                 and args.transport != "p2p")),
                "batch_window": args.batch_window,
                "put_issue": args.put_issue,
                "fuse_signal": args.fuse_signal,
                "small_inline": args.small_inline,
                "exclusive": args.exclusive,
                "endpoints_per_peer": args.endpoints_per_peer}
        common.table_print(meta, rows)
        compiled = [r for r in setup.compile_records if not r["cache_hit"]]
        cached = [r for r in setup.compile_records if r["cache_hit"]]
        common.emit({
            **meta, "world_size": setup.world_size,
            "flags": {"warmup": args.warmup, "iters": args.iters,
                      "puts": args.puts, "window": args.window,
                      "l2_flush": args.l2_flush, "smoke": args.smoke,
                      "op": args.op},
            "node": common.node_info(world),
            "platform": {
                "cluster": os.environ.get("SLURM_CLUSTER_NAME", "local")},
            "methodology": common.methodology(args.mode, args),
            "compile": {
                "first_call_s": (compiled[0]["first_call_s"]
                                 if compiled else 0.0),
                "cached_call_s": (cached[0]["cached_call_s"]
                                  if cached else 0.0),
                "variants_compiled": len(compiled),
                "total_compile_s": sum(r["first_call_s"] for r in compiled),
            },
            "kernels": setup.kernels,
            "memory": world.bench_memory,
            "results": rows, "clamped": clamp_notes,
            "skipped": False, "skip_reason": None,
        }, args.json_out)
    world.close()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_flags(parser)
    parser.add_argument("--mode", choices=("latency", "bandwidth"),
                        default="latency")
    args = parser.parse_args()
    if args.interleave or args.arms:
        return common.run_interleaved(args, Path(__file__).resolve(),
                                      ["--mode", args.mode])
    common.resolve_transport(args)
    if args.arm in common.BINARY_ARMS:
        return common.run_binary_arm(
            common.binary_path(common.BINARY_ARMS[args.arm]),
            common.binary_argv(args, "ladder", args.mode),
            ranks=2, json_out=args.json_out)
    return common.run_ranks(args, rank_main)


if __name__ == "__main__":
    sys.exit(main())
