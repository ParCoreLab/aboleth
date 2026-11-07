#!/usr/bin/env python3
"""Tier-1 symmetric all-to-all: every rank puts a per-peer slice to every
peer, signals, and waits for all inbound signals; the round completes when
the whole exchange has. Sizes are per-peer; algbw follows the nccl-tests
convention (n*S per rank), busbw = algbw*(n-1)/n. Rank 0 reports."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from ladder import render_prelude

ALLTOALL_DEFAULTS = ("1K", "64M")

_A2A_ARGTYPES = [
    ctypes.c_uint64, ctypes.c_uint64,     # endpoints, doorbell_locks
    ctypes.c_int, ctypes.c_int,           # rank, world
    ctypes.c_uint64, ctypes.c_uint64,     # dsts, srcs (device tables)
    ctypes.c_uint64, ctypes.c_uint64,     # peer_flags, my_flags (device tables)
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,  # bytes, warmup, iters
    ctypes.c_uint64, ctypes.c_uint64,     # samples, barrier_state
]


def _salt(sender: int, receiver: int) -> int:
    return (sender * 29 + receiver * 13) & 0xFF



def _device_table(records: list[bytes], device):
    import torch
    blob = b"".join(records)
    return torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(device)


def rank_main(args) -> int:
    import torch
    generated = (args.library == "generated"
                 or args.arm == "aboleth-specialized")
    for flag in ("fuse_signal", "small_inline", "exclusive"):
        if getattr(args, flag) == "on" and not generated:
            raise SystemExit(
                f"alltoall: --{flag.replace('_', '-')} on needs "
                "--library generated")
    if args.put_issue == "warp" and args.arm not in common.DSL_ARMS:
        # put_warp exists only in the generated header; the warp path
        # drives payload puts only, so the fused epoch cannot compose.
        # (A DSL arm's warp forms are its library's: check_dsl_flags.)
        if not generated:
            raise SystemExit(
                "alltoall: --put-issue warp needs --library generated "
                "(put_warp is a generated-header op)")
        if args.fuse_signal == "on":
            raise SystemExit(
                "alltoall: --fuse-signal on composes with "
                "--put-issue thread only")
    if args.completion == "drain" and args.arm != "aboleth-triton":
        raise SystemExit("alltoall: --completion drain is a Tier-2 "
                         "diagnostic (--arm aboleth-triton)")
    if args.quiet_every != 1 and (
            args.arm not in ("aboleth-generic", "aboleth-specialized")
            or args.quiet_every < 1):
        raise SystemExit("alltoall: --quiet-every N (N >= 1) is a Tier-1 "
                         "diagnostic of the CUDA arms")
    common.check_tier2_flags(args)
    world, bootstrap = common.make_world(args)
    rank, world_size = bootstrap.rank, bootstrap.world_size
    (include_dirs, extra_flags, extra_objects,
     library_meta) = common.library_options(args, world, bootstrap)
    peers = [peer for peer in range(world_size) if peer != rank]

    sizes = common.resolve_sweep(args, *ALLTOALL_DEFAULTS)
    sizes, clamp_notes = common.clamp_sizes(
        sizes, lambda size: 2 * world_size * size + (1 << 20))
    stride = max(sizes)
    recv_offset = world_size * stride
    flags_offset = 2 * world_size * stride
    device = torch.device(f"cuda:{world.device}")
    arena = common.arena(world, flags_offset + 8 * world_size, device)
    registration = world.register(arena)
    table = world.exchange_regions({
        "data": world.region("data", registration,
                             offset=0, length=flags_offset),
        "flags": world.region("flags", registration, offset=flags_offset,
                              length=8 * world_size),
    })
    flags_view = arena[flags_offset:].view(torch.int64)
    local_data = table.local("data")
    local_flags = table.local("flags")

    # Per-peer tables, indexed by CTA lane (peer order with self skipped);
    # offsets stride by the max size, so the tables are size-independent.
    dsts, srcs, peer_flags, my_flags = [], [], [], []
    for peer in peers:
        remote_data = table.remote(peer, "data")
        remote_flags = table.remote(peer, "flags")
        dsts.append(struct.pack(
            "<QI4x", remote_data.address + recv_offset + rank * stride,
            remote_data.rkey))
        srcs.append(struct.pack(
            "<QI4x", local_data.address + peer * stride, local_data.lkey))
        peer_flags.append(struct.pack(
            "<QI4x", remote_flags.address + rank * 8, remote_flags.rkey))
        my_flags.append(struct.pack("<Q", local_flags.address + peer * 8))
    tables = {name: _device_table(records, device)
              for name, records in (("dsts", dsts), ("srcs", srcs),
                                    ("peer_flags", peer_flags),
                                    ("my_flags", my_flags))}
    samples = torch.zeros(max(args.iters, 1), dtype=torch.int64, device=device)
    barrier_state = torch.zeros(1, dtype=torch.int32, device=device)

    def shape_launcher(size: int):
        handle, record = common.compile_shape_library(
            f"#define BENCH_QUIET_EVERY {args.quiet_every}u\n"
            + render_prelude(args.transport, size, 0, 0,
                             warp=args.put_issue == "warp",
                             fuse=args.fuse_signal == "on",
                             small_inline=args.small_inline == "on",
                             bake=args.bake == "on"),
            arch=common.device_arch(), cache_dir=common.default_cache_dir(),
            tag=f"a2a_{args.transport}", include_dirs=include_dirs,
            extra_flags=extra_flags, extra_objects=extra_objects)
        compile_records.append(record)
        kernels.append({"kernel": "alltoall", "size": size,
                        **common.kernel_attributes(
                            handle, 2, 32, world_size - 1)})
        fn = handle.bench_launch_alltoall
        fn.restype, fn.argtypes = ctypes.c_int, _A2A_ARGTYPES

        def launch(warmup: int, iters: int) -> None:
            code = fn(table.endpoint_handles.data_ptr(), 0, rank, world_size,
                      tables["dsts"].data_ptr(), tables["srcs"].data_ptr(),
                      tables["peer_flags"].data_ptr(),
                      tables["my_flags"].data_ptr(),
                      size, warmup, iters, samples.data_ptr(),
                      barrier_state.data_ptr())
            if code:
                raise RuntimeError(f"alltoall: launch failed, cudaError {code}")
            torch.cuda.synchronize()
        return launch

    step = 0

    def barrier() -> None:
        nonlocal step
        step += 1
        bootstrap.barrier(f"a2a:{step}")

    compile_records, kernels, rows, failures = [], [], [], 0
    # A library's own all-to-all collective in a DSL (--collective on) takes
    # contiguous slices (slice p at p * size, the layout NVSHMEM's
    # alltoallmem takes, as the CUDA collective arm stages them) and
    # publishes no epochs.
    collective = (args.arm in common.DSL_ARMS
                  and getattr(args, "collective", "off") == "on")
    for size in sizes:
        layout = size if collective else stride
        kernel_arm = common.kernel_module(args.arm)
        if kernel_arm is not None:
            extra = {"args": args} if args.arm in common.DSL_ARMS else {}
            launch = kernel_arm.alltoall(
                world, table, local_data.address, local_data.lkey,
                flags_view, size, layout, recv_offset, samples,
                barrier_state, kernels,
                drain=args.completion == "drain", **extra)
        else:
            launch = shape_launcher(size)

        def stage(salt: int) -> None:
            for peer in peers:
                common.pattern_fill_gpu(
                    arena[peer * layout: peer * layout + size],
                    _salt(rank, peer) ^ salt)
                base = recv_offset + peer * layout
                arena[base: base + size].zero_()

        # Correctness before timing: sender-salted slices catch misrouting,
        # not just corruption.
        stage(0)
        flags_view.zero_()
        torch.cuda.synchronize()
        barrier()
        launch(0, 1)
        barrier()
        wrong = 0
        for peer in peers:
            base = recv_offset + peer * layout
            wrong += common.pattern_wrong_gpu(
                arena[base: base + size], _salt(peer, rank))
        wrong = max(int(value.decode()) for value in bootstrap.all_gather(
            f"a2a-wrong:{size}", str(wrong).encode()).values())
        failures += wrong
        if wrong:
            print(f"FAIL: alltoall size {size}: {wrong} wrong",
                  file=sys.stderr)

        stage(common.TIMED_SALT)
        flags_view.zero_()
        torch.cuda.synchronize()
        # ABOLETH_PORT_COUNTERS: this rank's HCA port and hw counters
        # around the timed launch, summed over ranks into the row.
        counting = bool(os.environ.get("ABOLETH_PORT_COUNTERS"))
        hca = getattr(world, "hca", None) if counting else None
        before = common.port_counters(hca) if counting else {}
        barrier()
        launch(args.warmup, args.iters)
        barrier()
        counter_delta = {}
        if counting:
            after_counters = common.port_counters(hca)
            local = {name: after_counters[name] - value
                     for name, value in before.items()
                     if name in after_counters
                     and after_counters[name] != value}
            for payload in bootstrap.all_gather(
                    f"a2a-counters:{size}",
                    json.dumps(local).encode()).values():
                for name, value in json.loads(payload.decode()).items():
                    counter_delta[name] = counter_delta.get(name, 0) + value
        # Correctness after timing: every inbound epoch arrived and every
        # zeroed slice holds its sender's timed pattern.
        after = 0 if collective else sum(
            int(flags_view[peer].item() != args.warmup + args.iters)
            for peer in peers)
        for peer in peers:
            base = recv_offset + peer * layout
            after += common.pattern_wrong_gpu(
                arena[base: base + size],
                _salt(peer, rank) ^ common.TIMED_SALT)
        after = max(int(value.decode()) for value in bootstrap.all_gather(
            f"a2a-wrong-after:{size}", str(after).encode()).values())
        if after:
            print(f"FAIL: alltoall size {size}: {after} wrong after timing",
                  file=sys.stderr)
        wrong += after
        failures += after

        raw = samples.tolist()
        samples_us = [value / 1000.0 for value in raw]
        mean_us = sum(samples_us) / len(samples_us)
        rank_means = [float(value.decode())
                      for _, value in sorted(bootstrap.all_gather(
                          f"a2a-mean:{size}", str(mean_us).encode()).items())]
        if rank == 0:
            row = {"op": "put", "size": size, "count": world_size - 1,
                   **common.percentiles(samples_us), "wrong": wrong,
                   "samples_ns": raw}
            # Conservative: bandwidth from the slowest rank's mean round.
            slowest_s = max(rank_means) * 1e-6
            algbw = world_size * size / slowest_s / 1e9 if slowest_s else 0.0
            row["algbw_gbps"] = algbw
            row["busbw_gbps"] = algbw * (world_size - 1) / world_size
            row["rank_mean_us"] = rank_means
            if counter_delta:
                row["port_counters"] = counter_delta
            rows.append(row)

    barrier()
    table.close()
    if rank == 0:
        submit = common.submit_label(args.transport)
        meta = {"suite": "tier1", "benchmark": "alltoall", "mode": "alltoall",
                "arm": args.arm, "transport": args.transport,
                "submit_mode": submit, "world": world_size,
                "ctas": world_size - 1, **library_meta,
                "put_issue": args.put_issue,
                "fuse_signal": args.fuse_signal,
                "small_inline": args.small_inline,
                "exclusive": args.exclusive,
                "bake": args.bake,
                "completion": args.completion,
                "quiet_every": args.quiet_every,
                "collective": getattr(args, "collective", "off")}
        common.table_print(meta, rows)
        compiled = [r for r in compile_records if not r["cache_hit"]]
        cached = [r for r in compile_records if r["cache_hit"]]
        methodology = common.methodology("alltoall", args)
        methodology["timer"] = "device %globaltimer per-round on rank 0"
        methodology["caveats"].append(
            "percentiles are rank 0's rounds; algbw/busbw use the slowest "
            "rank's mean round (rank_mean_us records all)")
        common.emit({
            **meta, "world_size": world_size,
            "flags": {"warmup": args.warmup, "iters": args.iters,
                      "smoke": args.smoke},
            "node": common.node_info(world),
            "platform": {
                "cluster": os.environ.get("SLURM_CLUSTER_NAME", "local")},
            "methodology": methodology,
            "compile": {
                "first_call_s": (compiled[0]["first_call_s"]
                                 if compiled else 0.0),
                "cached_call_s": (cached[0]["cached_call_s"]
                                  if cached else 0.0),
                "variants_compiled": len(compiled),
                "total_compile_s": sum(r["first_call_s"] for r in compiled),
            },
            "kernels": kernels,
            "memory": world.bench_memory,
            "results": rows, "clamped": clamp_notes,
            "skipped": False, "skip_reason": None,
        }, args.json_out)
    world.close()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_flags(parser)
    parser.add_argument("--completion", choices=("wait", "drain"),
                        default="wait",
                        help="aboleth-triton diagnostic: drain also "
                             "retires each round as Tier 1's quiet does (a "
                             "signalled no-op, its own doorbell and poll) "
                             "before the wait on the batch's own completion")
    parser.add_argument("--quiet-every", type=int, default=1,
                        help="Tier-1 diagnostic (aboleth-generic/-specialized):"
                             " quiet every Nth round and the last instead of "
                             "every round, to price the per-round retirement "
                             "(a labelled arm, never Tier 1's result)")
    args = parser.parse_args()
    if args.interleave or args.arms:
        return common.run_interleaved(args, Path(__file__).resolve())
    common.resolve_transport(args)
    if args.arm in common.BINARY_ARMS:
        return common.run_binary_arm(
            common.binary_path(common.BINARY_ARMS[args.arm]),
            common.binary_argv(args, "alltoall", "alltoall"),
            ranks=args.local_ranks, json_out=args.json_out)
    return common.run_ranks(args, rank_main)


if __name__ == "__main__":
    sys.exit(main())
