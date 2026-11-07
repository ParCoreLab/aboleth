"""MoE dispatch with packing fused into the send program, against separate
pack and send kernels (paper Listing 2's shape against the measured path).

One NativeMoESlot per token count and endpoint profile serves both pack
arms (same buffers, plans and endpoints); `fuse_pack` selects the arm. The
profiles (--profiles) are "calibrated" (World.endpoint_profiles: cached or
measured once), "fallback" (none: one lane, one sender warp) or identities
to pin, joined by "+"; each gets its own slot on disjoint endpoints. A pack
arm ending in "+par" runs each transport class's send program on its own
forked stream, joined before the awaits: a measurement of what one launch for
both classes could buy, not a mode Aboleth has (it assigns no streams). Each count is first
checked: every arm's combined output must be bitwise equal and, for BF16,
equal to a CPU reference of the same expert. Then dispatch and combine are
captured as CUDA graphs per arm and timed with events around each replay,
all arms interleaved over --runs runs. Prints and writes per-rank JSON.

    torchrun-style env (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR/PORT):
    python aboleth_moe_fused_pack.py --out DIR [--fp8] [--tokens 1,16,256]
        [--profiles fallback,calibrated]
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from aboleth.bootstrap import TcpStoreBootstrap
from aboleth.moe.compiled_plan import MoESpec, compile_moe_plan
from aboleth.moe.native import NativeMoESlot
from aboleth.world import World

LANES = 8


class ConcurrentSlot(NativeMoESlot):
    """With `concurrent`, the send programs of a phase run on forked streams."""

    concurrent = False
    _streams = ()

    def _transport(self, phase, *, active=1, num_tokens=0):
        calls = self.calls[phase]
        if not self.concurrent or len(calls) < 2:
            return super()._transport(
                phase, active=active, num_tokens=num_tokens)
        if len(self._streams) < len(calls):
            self._streams = tuple(torch.cuda.Stream() for _ in calls)
        main = torch.cuda.current_stream()
        ready = torch.cuda.Event()
        ready.record(main)
        joins = []
        for call, stream in zip(calls, self._streams):
            stream.wait_event(ready)
            with torch.cuda.stream(stream):
                self._send(phase, call, active, num_tokens)
            joins.append(torch.cuda.Event())
            joins[-1].record(stream)
        for join in joins:
            main.wait_event(join)
        self._await(phase, active)


def routes(rank, n, hidden, top_k, experts, step):
    generator = torch.Generator().manual_seed(1000 * rank + step)
    ids = torch.empty((n, top_k), dtype=torch.int64)
    for row in range(n):
        ids[row] = torch.randperm(experts, generator=generator)[:top_k]
    # Multiples of 1/8: the expert's coefficient sums are exact, so the GPU
    # and CPU reduction orders cannot round differently.
    weights = torch.randint(1, 8, (n, top_k), generator=generator) / 8
    tokens = torch.randn((n, hidden), generator=generator).to(torch.bfloat16)
    return tokens, ids, weights


def expert(slot, prepared, ranks, experts):
    # Expert e scales its input by (e + 1); weights combine them.
    x, _, ids, weights = prepared
    local = ids // (experts // ranks) == slot.rank
    coefficients = torch.where(local, (ids + 1) * weights, 0).sum(dim=1)
    value = torch.where(
        local.any(dim=1)[:, None], x.float() * coefficients[:, None], 0
    )
    slot.expert_output.copy_(value.to(torch.bfloat16))
    return slot.combine(slot.expert_output)


def reference(tokens, ids, weights, ranks, experts):
    total = torch.zeros_like(tokens, dtype=torch.float32)
    for owner in range(ranks):
        coefficients = torch.where(
            ids // (experts // ranks) == owner, (ids + 1) * weights, 0
        ).sum(dim=1)
        total += (
            (tokens.float() * coefficients[:, None]).to(torch.bfloat16).float()
        )
    return total.to(torch.bfloat16)


def canonical(slot, ranks):
    """Received records per source rank, independent of their slot order:
    (count, sorted (token, selected, experts, weight bits, payload hash))."""
    s = slot.spec
    b = slot.buffers
    counts = b["received_counts"].cpu().tolist()
    metadata = b["received_metadata"].cpu()
    payload = slot.expert_input.view(torch.uint8).cpu()
    image = []
    for source in range(ranks):
        count = int(counts[source])
        records = []
        for record in range(max(0, min(count, s.capacity))):
            row = source * s.capacity + record
            words = metadata[row].tolist()
            selected = words[1]
            records.append(
                (
                    words[0],
                    selected,
                    tuple(words[2 : 2 + selected]),
                    tuple(words[2 + s.top_k : 2 + s.top_k + selected]),
                    hash(payload[row].numpy().tobytes()),
                )
            )
        image.append((count, sorted(records)))
    return image


def capture(slot, tokens, ids, weights):
    dispatch, combine = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(dispatch, stream=stream):
        slot.dispatch(tokens, ids, weights)
    with torch.cuda.graph(combine, stream=stream):
        slot.combine(slot.expert_output)
    return dispatch, combine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", default="1,4,16,64,256,1024")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--profiles", default="calibrated")
    parser.add_argument("--pack", default="separate,fused")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--keep-going", action="store_true",
                        help="record gate failures and go on (diagnosis only)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    ranks = int(os.environ["WORLD_SIZE"])
    device = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    os.environ.setdefault(
        "TRITON_CACHE_DIR", str(args.out / f"cache-rank{rank}")
    )
    sizes = [int(value) for value in args.tokens.split(",")]
    profile_arms = args.profiles.split(",")
    # "separate", "fused", or "fused:WxC" (the fused programs' warps and
    # widest row chunk, a measurement of NativeMoESlot.fused_tile), each
    # optionally with "+par" (forked send streams).
    packs = []
    for value in args.pack.split(","):
        value, concurrent = value.removesuffix("+par"), value.endswith("+par")
        kind, _, tile = value.partition(":")
        if kind == "separate":
            packs.append((False, concurrent))
        else:
            tile = tuple(int(v) for v in tile.split("x")) if tile else True
            packs.append((tile, concurrent))
    bootstrap = TcpStoreBootstrap.from_env(timeout=300)
    world = World(rank, ranks, device=device, bootstrap=bootstrap)
    world.connect_all(
        "auto",
        remote_transport="rdma",
        endpoints_per_peer=len(sizes) * len(profile_arms) * LANES,
    )
    facts = world.endpoint_registration(LANES).plan_facts()
    resolved = {}
    for name in profile_arms:
        if name == "fallback":
            resolved[name] = None
        elif name == "calibrated":
            resolved[name] = world.endpoint_profiles()
        else:
            resolved[name] = world.endpoint_profiles(pin=tuple(name.split("+")))
    capacities = {
        kind: min(
            world.endpoint(peer).capacity
            for peer, backend in world.transport_geography
            if backend == kind
        )
        for kind in {backend for _, backend in world.transport_geography}
    }
    experts, top_k, hidden = args.experts, args.top_k, args.hidden
    owners = torch.arange(experts, device=device, dtype=torch.int32) // (
        experts // ranks
    )
    placeholder = ((rank + 1) % ranks) * (experts // ranks)
    quant = (
        dict(value_type="f8E4M3FN", scale_group=128, quant_amax_floor=1e-10)
        if args.fp8
        else {}
    )
    record = dict(
        rank=rank,
        ranks=ranks,
        fp8=args.fp8,
        hidden=hidden,
        top_k=top_k,
        experts=experts,
        facts=facts,
        profiles={
            name: None if value is None else value.record()
            for name, value in resolved.items()
        },
        passes=os.environ.get("ABOLETH_PASSES", ""),
        commit=os.environ.get("ABOLETH_COMMIT", ""),
        cells=[],
    )
    arms = [(name, fused) for name in profile_arms for fused in packs]

    def label(arm):
        fused, concurrent = arm[1]
        suffix = "+par" if concurrent else ""
        if fused is False:
            return f"{arm[0]}/separate{suffix}"
        tile = "" if fused is True else ":" + "x".join(map(str, fused))
        return f"{arm[0]}/fused{tile}{suffix}"

    def select(slot, pack):
        fused, slot.concurrent = pack
        slot.fuse_pack = fused is not False
        slot.fused_tile = fused if isinstance(fused, tuple) else NativeMoESlot.fused_tile

    for index, n in enumerate(sizes):
        slots = {}
        for offset, name in enumerate(profile_arms):
            plan = compile_moe_plan(
                MoESpec(n, hidden, top_k, experts, ranks, **quant),
                geography=world.transport_geography,
                capacities=capacities,
                available_lanes=LANES,
                profiles=() if resolved[name] is None else resolved[name].profiles,
                **facts,
            )
            slots[name] = ConcurrentSlot(
                world,
                plan,
                owners=owners,
                placeholder=placeholder,
                tag=f"fused-{n}-{name}",
                endpoint_base=(index * len(profile_arms) + offset) * LANES,
                endpoint_capacity=LANES,
            )
        ConcurrentSlot.bind_input_domain(list(slots.values()))
        for name, fused in arms:
            select(slots[name], fused)
            slots[name].warmup_transport()
        # Correctness: every arm on the same routes, several steps.
        for step in range(4):
            tokens, ids, weights = (
                t.to(device)
                for t in routes(rank, n, hidden, top_k, experts, step)
            )
            inputs = tuple(t.cpu().clone() for t in (tokens, ids, weights))
            outputs, received = {}, {}
            for arm in arms:
                slot = slots[arm[0]]
                select(slot, arm[1])
                # Only the artifact's own storage: after a dispatch,
                # `buffers` also holds the caller's borrowed inputs.
                for name, tensor in slot._owned_buffers.items():
                    if name != "owners":
                        tensor.fill_(
                            float("nan")
                            if tensor.dtype.is_floating_point
                            else -12345
                        )
                torch.cuda.synchronize()
                bootstrap.barrier(f"check-{n}-{step}-{label(arm)}")
                prepared = slot.dispatch(tokens, ids, weights)
                torch.cuda.synchronize()
                received[arm] = canonical(slot, ranks)
                outputs[arm] = expert(slot, prepared, ranks, experts).cpu()
                assert slot.buffers["overflow"].item() == 0
                torch.cuda.synchronize()
                bootstrap.barrier(f"checked-{n}-{step}-{label(arm)}")
            failures = []
            if not args.fp8:
                expected = reference(*inputs, ranks, experts)
                for arm in arms:
                    if not torch.equal(outputs[arm], expected):
                        rows = (outputs[arm] != expected).any(dim=1)
                        failures.append(
                            f"{label(arm)} differs from the reference in rows "
                            f"{rows.nonzero().flatten().tolist()[:8]}"
                        )
            first = arms[0]
            for arm in arms[1:]:
                for source in range(ranks):
                    if received[first][source] != received[arm][source]:
                        a, b = received[first][source], received[arm][source]
                        failures.append(
                            f"records from rank {source}: {label(first)} count "
                            f"{a[0]} {label(arm)} count {b[0]}; first differing "
                            f"record "
                            f"{next((x, y) for x, y in zip(a[1], b[1]) if x != y) if a[0] == b[0] else '-'}"
                        )
                if not torch.equal(
                    outputs[first].view(torch.int16),
                    outputs[arm].view(torch.int16),
                ):
                    failures.append(f"{label(arm)} output differs from {label(first)}")
            if failures:
                print(
                    f"rank {rank} n={n} step={step}: " + "; ".join(failures),
                    flush=True,
                )
            if not args.keep_going:
                assert not failures, (n, step)
            record.setdefault("failures", []).extend(f"n={n} step={step}: {f}" for f in failures)
        # Timing: fixed routes, graphs per arm, arms interleaved per run
        # (rotated, so each arm takes each position).
        tokens, ids, weights = (
            t.to(device) for t in routes(rank, n, hidden, top_k, experts, 99)
        )
        graphs = {}
        for arm in arms:
            select(slots[arm[0]], arm[1])
            torch.cuda.synchronize()
            bootstrap.barrier(f"capture-{n}-{label(arm)}")
            graphs[arm] = capture(slots[arm[0]], tokens, ids, weights)
        samples = {arm: {"dispatch": [], "combine": []} for arm in arms}
        for run in range(args.runs):
            shift = run % len(arms)
            for arm in arms[shift:] + arms[:shift]:
                dispatch, combine = graphs[arm]
                events = [
                    torch.cuda.Event(enable_timing=True)
                    for _ in range(3 * args.iters)
                ]
                torch.cuda.synchronize()
                bootstrap.barrier(f"run-{n}-{run}-{label(arm)}")
                for _ in range(args.warmup):
                    dispatch.replay()
                    combine.replay()
                for i in range(args.iters):
                    events[3 * i].record()
                    dispatch.replay()
                    events[3 * i + 1].record()
                    combine.replay()
                    events[3 * i + 2].record()
                torch.cuda.synchronize()
                d = [
                    events[3 * i].elapsed_time(events[3 * i + 1]) * 1e3
                    for i in range(args.iters)
                ]
                c = [
                    events[3 * i + 1].elapsed_time(events[3 * i + 2]) * 1e3
                    for i in range(args.iters)
                ]
                samples[arm]["dispatch"].append(statistics.median(d))
                samples[arm]["combine"].append(statistics.median(c))
        cell = dict(
            tokens=n,
            launches={
                name: {
                    phase: [
                        dict(
                            backend=p.backend,
                            peers=len(p.peers),
                            lanes=p.lanes,
                            warps=p.sender_warps,
                            profile=p.profile_identity,
                        )
                        for p, _ in launches
                    ]
                    for phase, launches in slot.launches.items()
                }
                for name, slot in slots.items()
            },
            arms={label(arm): samples[arm] for arm in arms},
            registers_last_arm={
                name: {
                    f"{phase}/{backend}": kernel.n_regs
                    for (phase, backend), kernel in slot.compiled_kernels.items()
                }
                for name, slot in slots.items()
            },
        )
        record["cells"].append(cell)
        if rank == 0:
            medians = "  ".join(
                f"{label(arm)} {statistics.median(samples[arm]['dispatch']):8.2f}"
                f"/{statistics.median(samples[arm]['combine']):.2f}"
                for arm in arms
            )
            print(f"n={n:5d} dispatch/combine us: {medians}", flush=True)
        (args.out / f"rank-{rank}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        torch.cuda.synchronize()
        bootstrap.barrier(f"done-{n}")
        for slot in slots.values():
            slot.close()
    record["status"] = "FAIL" if record.get("failures") else "PASS"
    (args.out / f"rank-{rank}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    bootstrap.barrier("complete")
    world.close()
    bootstrap.close()


if __name__ == "__main__":
    main()
