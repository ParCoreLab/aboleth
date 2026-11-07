#!/usr/bin/env python3
"""dsl-moe: MoE dispatch and combine, each family's arms against the
implementations its developers ship.

    python experiments/dsl/moe/moe.py --arm aboleth-moe --model qwen3-235b \
        --tokens 1,8,32,64,128 [--bf16] [-n 200 -w 20]
    ... --arms aboleth-moe:auto,tdist-ll:ibrc --interleave --repeats 3

Every arm runs moe_common's contract (inputs, expert function, checks,
timer); an arm module (arm_*.py, ARMS) adapts one implementation to it:

    ctx = module.open(args, spec)       # world, library init; ctx.bootstrap
    cell = ctx.cell(tokens)             # per token count, the library's
                                        # best buffers for it
    handle = cell.dispatch(x, ids, w)   # timed
    counts = cell.counts(handle)        # rows per local expert (check)
    y = cell.expert(handle)             # f_e on what arrived (untimed)
    out = cell.combine(handle, y)       # timed
    cell.close(); ctx.close()

Each token count is checked (counts exact, combined output within FP8's
error, moe_common.check_output) before and after timing over several
routings, then timed on one routing; records follow the bench drivers'
(one JSON line; rows per token count with dispatch and combine p50s).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "bench"))

import common  # noqa: E402
import moe_common  # noqa: E402

# arm -> (module, transports)
ARMS = {"aboleth-moe": ("arm_aboleth", ("auto",)),
        "aboleth-moe-fused": ("arm_aboleth", ("auto",)),
        "tdist-ll": ("arm_tdist", ("ibrc", "ibgda")),
        "tdist-normal": ("arm_tdist", ("ibrc",)),
        "deepep": ("arm_deepep", ("gin",)),
        "cutile-tier2-moe": ("arm_cutile", ("auto",)),
        "cutile-nccl-moe": ("arm_cutile", ("auto",))}
CHECK_STEPS = 3


def rank_main(args) -> int:
    import torch
    if os.environ.get("ABOLETH_MOE_DEBUG"):
        # A stalled rank dumps every thread's stack (and exits) after the
        # deadline, so a hang names its call.
        import faulthandler
        faulthandler.dump_traceback_later(
            float(os.environ.get("ABOLETH_MOE_STALL_S", "240")), exit=True)
    module = importlib.import_module(ARMS[args.arm][0])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    hidden, experts, topk = moe_common.MODELS[args.model]
    spec = moe_common.Spec(hidden, experts, topk, world_size, rank,
                           fp8=not args.bf16)
    ctx = module.open(args, spec)
    bootstrap = ctx.bootstrap
    timer = moe_common.Timer(bootstrap, rank, world_size, ctx.device)
    device = torch.device("cuda", ctx.device)
    rows, failures, kernels = [], 0, []
    step = [0]

    def sync(tag):
        step[0] += 1
        bootstrap.barrier(f"moe:{tag}:{step[0]}")

    def worst(tag, value):
        return max(int(v.decode()) for v in bootstrap.all_gather(
            f"moe:{tag}:{step[0]}", str(value).encode()).values())

    def round_trip(cell, x, ids, weights):
        handle = cell.dispatch(x, ids, weights)
        counts = cell.counts(handle)
        out = cell.combine(handle, cell.expert(handle))
        return counts, out

    for tokens in [int(t) for t in args.tokens.split(",")]:
        cell = ctx.cell(tokens)
        wrong = 0
        for phase in ("before", "after"):
            for check in range(CHECK_STEPS):
                routing = check + (CHECK_STEPS if phase == "after" else 0)
                x, ids, weights = moe_common.inputs(spec, rank, tokens,
                                                    routing)
                sync(f"check:{tokens}")
                counts, out = round_trip(cell, x.to(device), ids.to(device),
                                         weights.to(device))
                torch.cuda.synchronize()
                expected = moe_common.reference_counts(spec, tokens, routing)
                bad = int((counts.cpu().to(torch.int64)
                           != expected).sum())
                bad += moe_common.check_output(out, x, ids, weights,
                                               spec.fp8)
                if bad and os.environ.get("ABOLETH_MOE_DEBUG"):
                    moe_common.diagnose(rank, tokens, counts, expected, out,
                                        x, ids, weights)
                bad = worst(f"wrong:{tokens}", bad)
                if bad:
                    print(f"FAIL: tokens {tokens} routing {routing}: {bad} "
                          "wrong", file=sys.stderr)
                wrong += bad
            if phase == "before":
                # Timed: one routing, warmup then iters calls of each.
                x, ids, weights = (t.to(device) for t in moe_common.inputs(
                    spec, rank, tokens, 1000))
                sync(f"timed:{tokens}")
                dispatch_events, combine_events = [], []
                for index in range(args.warmup + args.iters):
                    handle, d0, d1 = timer.bracket(
                        lambda: cell.dispatch(x, ids, weights))
                    y = cell.expert(handle)
                    _, c0, c1 = timer.bracket(lambda: cell.combine(handle, y))
                    if index >= args.warmup:
                        dispatch_events.append((d0, d1))
                        combine_events.append((c0, c1))
                torch.cuda.synchronize()
                dispatch_ms = [a.elapsed_time(b) for a, b in dispatch_events]
                combine_ms = [a.elapsed_time(b) for a, b in combine_events]
        failures += wrong
        kernels += cell.resources()
        # Conservative across ranks: each call's slowest rank.
        gathered = bootstrap.all_gather(
            f"moe:spans:{tokens}",
            json.dumps([dispatch_ms, combine_ms]).encode())
        per_rank = [json.loads(value.decode())
                    for _, value in sorted(gathered.items())]
        dispatch_max = [max(r[0][i] for r in per_rank)
                        for i in range(len(dispatch_ms))]
        combine_max = [max(r[1][i] for r in per_rank)
                       for i in range(len(combine_ms))]
        if rank == 0:
            dispatch = moe_common.summarize(dispatch_max)
            combine = moe_common.summarize(combine_max)
            rows.append({
                "op": "dispatch+combine", "size": tokens, "count": tokens,
                "p50_us": dispatch["p50_us"] + combine["p50_us"],
                "dispatch": dispatch, "combine": combine,
                "dispatch_rank0_p50_us": moe_common.summarize(
                    dispatch_ms)["p50_us"],
                "combine_rank0_p50_us": moe_common.summarize(
                    combine_ms)["p50_us"],
                "wrong": wrong, **cell.describe()})
        cell.close()
    sync("done")
    if rank == 0:
        meta = {"suite": "dsl", "benchmark": "moe", "mode": "moe",
                "arm": args.arm, "transport": args.transport,
                "submit_mode": common.submit_label(args.transport),
                "world": world_size, "model": args.model, "hidden": hidden,
                "experts": experts, "topk": topk,
                "dtype": "bf16" if args.bf16 else "fp8",
                "quant_group": moe_common.GROUP, **ctx.meta}
        print(f"# {' '.join(f'{k}={v}' for k, v in meta.items())}")
        print("#   tokens  dispatch_p50  combine_p50   wrong")
        for row in rows:
            print(f"{row['size']:>10} {row['dispatch']['p50_us']:>13.2f} "
                  f"{row['combine']['p50_us']:>12.2f} {row['wrong']:>7}")
        timing = ("CUDA events per call (dispatch and combine apart), each "
                  f"after a one-element NCCL all-reduce and a "
                  f"{timer.sleep_us:.0f} us device sleep; p50 over calls "
                  "of the slowest rank per call")
        common.emit({**meta, "world_size": world_size,
                     "flags": {"warmup": args.warmup, "iters": args.iters,
                               "tokens": args.tokens,
                               "check_routings": 2 * CHECK_STEPS},
                     "node": common.node_info(None),
                     "platform": {"cluster": os.environ.get(
                         "SLURM_CLUSTER_NAME", "local")},
                     "methodology": {"timer": timing},
                     "kernels": kernels, "memory": ctx.memory,
                     "results": rows, "skipped": False}, args.json_out)
    ctx.close()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_flags(parser)
    parser.add_argument("--model", default="qwen3-235b",
                        choices=sorted(moe_common.MODELS))
    parser.add_argument("--tokens", default="1,8,32,64,128",
                        help="tokens per rank, comma-separated")
    parser.add_argument("--bf16", action="store_true",
                        help="dispatch BF16 (default: FP8, group 128)")
    args = parser.parse_args()
    extra = ["--model", args.model, "--tokens", args.tokens]
    if args.bf16:
        extra.append("--bf16")
    if args.interleave or args.arms:
        return common.run_interleaved(args, Path(__file__).resolve(), extra)
    if args.arm not in ARMS:
        raise SystemExit(f"moe: unknown --arm {args.arm!r} "
                         f"({', '.join(ARMS)})")
    args.transport = args.transport or ARMS[args.arm][1][0]
    if args.transport not in ARMS[args.arm][1]:
        raise SystemExit(f"moe: {args.arm} runs {ARMS[args.arm][1]}")
    for setting in args.env:
        name, _, value = setting.partition("=")
        os.environ[name] = value
    if os.environ.get("RANK") is None:
        raise SystemExit("moe: run under a launcher (RANK, WORLD_SIZE, "
                         "LOCAL_RANK, MASTER_ADDR, MASTER_PORT)")
    return rank_main(args)


if __name__ == "__main__":
    sys.exit(main())
