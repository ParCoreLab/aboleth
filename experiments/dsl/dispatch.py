"""W4: FP8 MoE dispatch (paper Listing 2's shape) with each library.

    python experiments/dsl/dispatch.py --arm triton-tier1 --tokens 1,8,64 \
        [--hidden 4096 --topk 8 --experts 128] [-n 200 -w 20]
    ... --arms triton-tier1:auto,triton-nvshmem:ibgda,aboleth-triton:auto \
        --interleave --repeats 5

Each rank routes its tokens to experts (seeded, top-k distinct experts,
experts striped over ranks), and one program per peer packs and quantizes
that peer's rows and sends them (triton_dsl/dispatch_kernels.py): the
triton-tier1 and triton-nvshmem arms through a library's calls, the
aboleth-triton arm through Aboleth operations. The cutile-* arms pack in a
cuTile kernel and send in a second kernel (cutile/cutile_dispatch.py),
since a tile kernel cannot communicate. A round is pack + send for
every peer, ended by a grid barrier; lane 0 times it on the device clock.
Before and after timing, every rank checks every sender's records bitwise
(FP8 bytes, scales, token index) and counts against a host quantization of
that sender's regenerated tokens. Records follow alltoall.py's.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))
sys.path.insert(0, str(HERE / "triton_dsl"))

import common  # noqa: E402

GROUP = 128
ARMS = ("triton-tier1", "triton-nvshmem", "aboleth-triton", "cutile-tier2",
        "cutile-nccl")


def record_bytes(hidden: int) -> int:
    return (hidden + 4 * (hidden // GROUP) + 4 + 15) // 16 * 16


def routes(rank: int, tokens: int, hidden: int, topk: int, experts: int,
           world_size: int):
    import torch
    generator = torch.Generator().manual_seed(7919 * rank + tokens)
    ids = torch.stack([torch.randperm(experts, generator=generator)[:topk]
                       for _ in range(tokens)]) if tokens else \
        torch.empty((0, topk), dtype=torch.int64)
    x = torch.randn((tokens, hidden), generator=generator).to(torch.bfloat16)
    dest = (ids // (experts // world_size)).to(torch.int32)
    return x, dest


def expected(x, dest, peer: int, hidden: int):
    """The records `peer` receives: quantized rows, scales, token indices."""
    import torch
    rows = [t for t in range(x.shape[0]) if bool((dest[t] == peer).any())]
    if not rows:
        return rows, None, None
    values = x[rows].float().view(len(rows), hidden // GROUP, GROUP)
    amax = values.abs().clamp_min(1e-4).amax(dim=2)
    scale = amax / 448.0
    quantized = (values / scale[..., None]).clamp(-448.0, 448.0).to(
        torch.float8_e4m3fn).view(len(rows), hidden)
    return rows, quantized, scale


def compare(arena, recv_offset: int, counts, rank: int, world_size: int,
            tokens: int, args, maxn: int) -> int:
    import torch
    rb, hidden = record_bytes(args.hidden), args.hidden
    groups = hidden // GROUP
    wrong = 0
    for source in range(world_size):
        if source == rank:
            continue
        x, dest = routes(source, tokens, hidden, args.topk, args.experts,
                         world_size)
        rows, quantized, scale = expected(x, dest, rank, hidden)
        if int(counts[source].item()) != len(rows):
            wrong += 1 + len(rows)
            continue
        if not rows:
            continue
        base = recv_offset + source * maxn * rb
        records = arena[base: base + len(rows) * rb].view(len(rows), rb).cpu()
        got_q = records[:, :hidden].view(torch.float8_e4m3fn)
        got_scale = records[:, hidden: hidden + 4 * groups].contiguous() \
            .view(torch.float32)
        got_token = records[:, hidden + 4 * groups: hidden + 4 * groups + 4] \
            .contiguous().view(torch.int32).flatten()
        wrong += int((got_q.view(torch.uint8)
                      != quantized.view(torch.uint8)).any(dim=1).sum())
        wrong += int((got_scale != scale).any(dim=1).sum())
        wrong += int((got_token != torch.tensor(rows,
                                                dtype=torch.int32)).sum())
    return wrong


def library_launcher(args, world, table, arena_info, records):
    """triton-tier1 / triton-nvshmem: one launch, one program per peer."""
    import torch
    import triton_launch
    if args.arm == "triton-tier1":
        import aboleth.frontends.triton.tier1 as policy
        options = {"extern_libs": policy.extern_libs()}
        after = None
        global_quiet = False
    else:
        import triton_nvshmem as policy
        from nvshmem.core.init_fini import find_device_bitcode_library
        options = {"extern_libs": {
            "dsl_nvshmem": policy._shim_bitcode(),
            "libnvshmem_device": find_device_bitcode_library()}}
        after = policy._module_init
        global_quiet = True
    kernels = triton_launch.load(policy, quiet_is_global=global_quiet,
                                 name=args.arm.split("-")[1],
                                 source="dispatch_kernels.py")
    kernels.PAYLOAD = (kernels.payload_block if hasattr(policy, "put_block")
                       else kernels.payload_warp)
    rank, world_size = world.rank, world.world_size
    peers = [p for p in range(world_size) if p != rank]
    device = arena_info["device"]
    rb, maxn = arena_info["rb"], arena_info["maxn"]

    def column(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device=device)

    data = [table.remote(p, "data") for p in peers]
    cells = [table.remote(p, "counts") for p in peers]
    flags = [table.remote(p, "flags") for p in peers]
    local, local_flags = table.local("data"), table.local("flags")
    fixed = {
        "recv": column([r.address + arena_info["recv"] + rank * maxn * rb
                        for r in data]),
        "recv_key": column([r.rkey for r in data], torch.int32),
        "cell": column([r.address + rank * 8 for r in cells]),
        "cell_key": column([r.rkey for r in cells], torch.int32),
        "flag": column([r.address + rank * 8 for r in flags]),
        "flag_key": column([r.rkey for r in flags], torch.int32),
        "my_flag": column([local_flags.address + p * 8 for p in peers]),
    }
    state = {}

    def run(x, dest, tokens, warmup, iters, samples, barrier):
        arguments = (table.endpoint_handles, rank, x, dest, tokens,
                     local.address, local.lkey, fixed["recv"],
                     fixed["recv_key"], fixed["cell"], fixed["cell_key"],
                     fixed["flag"], fixed["flag_key"], fixed["my_flag"],
                     warmup, iters, samples, barrier)
        constants = {"H": args.hidden, "TOPK": args.topk, "GROUP": GROUP,
                     "RB": rb, "MAXN": maxn}
        if "options" not in state:
            state["options"] = {**options, "num_warps": 4, **constants}
            compiled = kernels.dispatch.warmup(*arguments, grid=(len(peers),),
                                               **state["options"])
            compiled._init_handles()
            if after is not None:
                after(compiled)
            records.append({"kernel": "dispatch", "registers": compiled.n_regs,
                            "spills": compiled.n_spills,
                            "shared_bytes": compiled.metadata.shared,
                            "threads": 128, "ctas": len(peers)})
        barrier.zero_()
        kernels.dispatch[(len(peers),)](*arguments, **state["options"])
        torch.cuda.synchronize()
    return run


def tier2_launcher(args, world, table, arena_info, records):
    """aboleth-triton: per transport partition, as tier2.py's all-to-all."""
    import torch
    import triton_launch
    from aboleth.planning import plan_transport_partitions
    kernels = triton_launch.load(None, quiet_is_global=False, name="tier2",
                                 source="dispatch_kernels.py")
    rank, world_size = world.rank, world.world_size
    local = table.local("data")
    handles = table.endpoint_handles.view(torch.int64)
    launches = plan_transport_partitions(world.transport_geography)
    partitions = []
    for index, spec in enumerate(launches):
        selection = torch.tensor(spec.peers, device=handles.device)
        partitions.append((spec, handles[selection].contiguous(),
                           torch.cuda.current_stream() if index == 0
                           else torch.cuda.Stream()))
    rb, maxn = arena_info["rb"], arena_info["maxn"]
    state = {"recorded": False}

    def launch_all(arguments, warmup, iters):
        x, dest, tokens, samples, barrier = arguments
        for index, (spec, endpoints, stream) in enumerate(partitions):
            with torch.cuda.stream(stream), \
                    world.compilation(peers=spec.peers):
                compiled = kernels.dispatch_tier2[(spec.grid_extent,)](
                    endpoints, rank, x, dest, tokens, local.address,
                    local.lkey, table.regions("data"),
                    table.regions("counts"), table.regions("flags"),
                    arena_info["flags_view"], warmup, iters, samples, barrier,
                    H=args.hidden, TOPK=args.topk, GROUP=GROUP, RB=rb,
                    MAXN=maxn, RECEIVE=arena_info["recv"], RANK=rank,
                    PROGRAMS=world_size - 1, RECORD=index == 0,
                    PEERS=spec.peers, num_warps=4)
            if not state["recorded"]:
                records.append({"kernel": f"dispatch[{spec.backend}]",
                                "registers": compiled.n_regs,
                                "spills": compiled.n_spills,
                                "shared_bytes": compiled.metadata.shared,
                                "threads": 128, "ctas": spec.grid_extent})
            if iters == 0:
                torch.cuda.synchronize()
        state["recorded"] = True

    def run(x, dest, tokens, warmup, iters, samples, barrier):
        arguments = (x, dest, tokens, samples, barrier)
        if "loaded" not in state:
            barrier.zero_()
            launch_all(arguments, 0, 0)
            state["loaded"] = True
        barrier.zero_()
        torch.cuda.synchronize()
        launch_all(arguments, warmup, iters)
        torch.cuda.synchronize()
    return run


def rank_main(args) -> int:
    import torch
    if args.arm not in ARMS:
        raise SystemExit(f"dispatch: --arm is one of {', '.join(ARMS)}")
    common.check_tier2_flags(args)
    world, bootstrap = common.make_world(args)
    rank, world_size = bootstrap.rank, bootstrap.world_size
    if args.arm.startswith("cutile"):
        import importlib.metadata
        library_meta = {"library": args.arm.split("-")[1],
                        "frontend": "cutile",
                        "cuda-tile": importlib.metadata.version("cuda-tile")}
    else:
        _, _, _, library_meta = common.library_options(args, world,
                                                       bootstrap)
    tokens_list = [int(t) for t in args.tokens.split(",")]
    maxn = max(max(tokens_list), 1)
    rb = record_bytes(args.hidden)
    recv = world_size * maxn * rb
    counts_offset = 2 * world_size * maxn * rb
    flags_offset = counts_offset + 8 * world_size
    device = torch.device(f"cuda:{world.device}")
    arena = common.arena(world, flags_offset + 8 * world_size, device)
    registration = world.register(arena)
    table = world.exchange_regions({
        "data": world.region("data", registration, offset=0,
                             length=counts_offset),
        "counts": world.region("counts", registration, offset=counts_offset,
                               length=8 * world_size),
        "flags": world.region("flags", registration, offset=flags_offset,
                              length=8 * world_size),
    })
    counts = arena[counts_offset: flags_offset].view(torch.int64)
    flags = arena[flags_offset:].view(torch.int64)
    arena_info = {"device": device, "rb": rb, "maxn": maxn, "recv": recv,
                  "flags_view": flags, "counts_view": counts, "arena": arena}
    kernels: list = []
    if args.arm.startswith("cutile"):
        sys.path.insert(0, str(HERE / "cutile"))
        import cutile_dispatch
        build = cutile_dispatch.launcher
    elif args.arm == "aboleth-triton":
        build = tier2_launcher
    else:
        build = library_launcher
    launcher = build(args, world, table, arena_info, kernels)
    samples = torch.zeros(max(args.iters, 1), dtype=torch.int64,
                          device=device)
    barrier = torch.zeros(1, dtype=torch.int32, device=device)
    step = [0]

    def sync():
        step[0] += 1
        bootstrap.barrier(f"w4:{step[0]}")

    rows, failures = [], 0
    for tokens in tokens_list:
        x, dest = routes(rank, tokens, args.hidden, args.topk, args.experts,
                         world_size)
        x, dest = x.to(device), dest.to(device)
        wrong = 0
        for phase, (warmup, iters) in (("check", (0, 1)),
                                       ("timed", (args.warmup, args.iters))):
            arena[recv: counts_offset].zero_()
            counts.zero_()
            flags.zero_()
            torch.cuda.synchronize()
            sync()
            launcher(x, dest, tokens, warmup, iters, samples, barrier)
            sync()
            bad = compare(arena, recv, counts, rank, world_size, tokens,
                          args, maxn)
            if args.arm != "cutile-nccl":   # NCCL publishes no epochs
                bad += sum(int(flags[p].item() != warmup + iters)
                           for p in range(world_size) if p != rank)
            bad = max(int(v.decode()) for v in bootstrap.all_gather(
                f"w4-wrong:{tokens}:{phase}", str(bad).encode()).values())
            if bad:
                print(f"FAIL: dispatch tokens {tokens} ({phase}): {bad} "
                      "wrong", file=sys.stderr)
            wrong += bad
        failures += wrong
        raw = samples.tolist()[:args.iters]
        samples_us = [value / 1000.0 for value in raw]
        if rank == 0:
            rows.append({"op": "dispatch", "size": tokens, "count": tokens,
                         **common.percentiles(samples_us), "wrong": wrong,
                         "samples_ns": raw,
                         "bytes_per_record": rb})
    sync()
    table.close()
    if rank == 0:
        meta = {"suite": "dsl", "benchmark": "dispatch", "mode": "dispatch",
                "arm": args.arm, "transport": args.transport,
                "submit_mode": common.submit_label(args.transport),
                "world": world_size, "hidden": args.hidden,
                "topk": args.topk, "experts": args.experts,
                "quant_group": GROUP, **library_meta}
        common.table_print(meta, rows)
        common.emit({**meta, "world_size": world_size,
                     "flags": {"warmup": args.warmup, "iters": args.iters,
                               "tokens": args.tokens},
                     "node": common.node_info(world),
                     "platform": {"cluster": os.environ.get(
                         "SLURM_CLUSTER_NAME", "local")},
                     "kernels": kernels, "memory": world.bench_memory,
                     "results": rows, "skipped": False}, args.json_out)
    world.close()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_flags(parser)
    parser.add_argument("--tokens", default="1,8,64",
                        help="tokens per rank, comma-separated")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    args = parser.parse_args()
    extra = ["--tokens", args.tokens, "--hidden", str(args.hidden),
             "--topk", str(args.topk), "--experts", str(args.experts)]
    if args.interleave or args.arms:
        return common.run_interleaved(args, Path(__file__).resolve(), extra)
    common.resolve_transport(args)
    return common.run_ranks(args, rank_main)


if __name__ == "__main__":
    sys.exit(main())
