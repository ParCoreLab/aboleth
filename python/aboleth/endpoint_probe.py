"""Bounded endpoint probe: how a platform's queues scale with lanes and warps.

The runtime measures; plan-endpoints decides. `measure` is collective over a
connected World and runs only when no rank has a profile for the platform
(endpoint_profiles.resolve_endpoint_profiles). Each rank sends to one partner
over the probed backend (RDMA: the same local index on the next host; P2P:
the next GPU on the same host), so sends form a permutation and every link
carries one stream each way. Per lane and iteration: one put of the lane's
slice, a strong signal after it, and a sender wait, i.e. sender-completed
useful throughput, as in the B200 study the seed profiles came from. Only
semantic operations; the compiler plans the rest. A trial lasts as long as
its slowest rank (endpoint_profiles.world_rates).
"""

import json
import time

import torch
import triton
import triton.language as tl

import aboleth.language as ops
from .endpoint_profiles import PROBE_LANES, PROBE_WARPS, fit_sender_warps, world_rates

FIT_BYTES = (64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20)
HOLDOUT_BYTES = (512 << 10, 2 << 20, 8 << 20)
TRIALS = 5  # measured, after one warm-up trial
TRIAL_BYTES = 32 << 20  # at most, per lane set and trial
ITERATIONS = (4, 32)


@triton.jit(do_not_specialize=["partner", "iterations"])
def _send(endpoints, payload, payload_key, receive, flags, starts, partner,
          chunk, iterations):
    lane = tl.program_id(0)
    endpoint = ops.endpoint(endpoints, lane)
    source = ops.region(payload, payload_key)
    destination = ops.remote(receive, partner, tl.uint8)
    flag = ops.remote(flags, partner, tl.int64)
    offset = lane * chunk
    first = tl.load(starts + lane)
    for index in range(iterations):
        data = ops.put(endpoint, destination.at(offset), source.at(offset), chunk)
        ready = ops.signal(endpoint, flag.at(lane), first + index + 1, after=data)
        ops.wait(ready)


def partners(world, backend):
    """(send to, receive from) for this rank over `backend`, None if absent."""
    topology = world.topology
    hosts = sorted({info.hostname for info in topology.values()})
    ranks = {
        host: sorted(r for r, info in topology.items() if info.hostname == host)
        for host in hosts
    }
    host = topology[world.rank].hostname
    local = ranks[host].index(world.rank)
    if backend == "rdma":
        at = hosts.index(host)
        after, before = ranks[hosts[(at + 1) % len(hosts)]], ranks[hosts[at - 1]]
        to = after[local] if len(hosts) > 1 and local < len(after) else None
        source = before[local] if len(hosts) > 1 and local < len(before) else None
    else:
        group = ranks[host]
        many = len(group) > 1
        to = group[(local + 1) % len(group)] if many else None
        source = group[local - 1] if many else None
    kinds = dict(world.transport_geography)
    return (
        to if kinds.get(to) == backend else None,
        source if kinds.get(source) == backend else None,
    )


def _gather(world, tag, value):
    gathered = world.bootstrap.all_gather(tag, json.dumps(value).encode())
    return [json.loads(gathered[rank]) for rank in range(world.world_size)]


def _pattern(size, rank, device):
    values = torch.arange(size, device=device, dtype=torch.int32) * 7 + rank * 13
    return values.remainder(251).to(torch.uint8)


def measure(world, backends, tag):
    """{backend: {"samples", "world", "seconds"} or {"reason"}}; collective."""
    return {backend: _measure(world, backend, f"{tag}.{backend}") for backend in backends}


def _measure(world, backend, tag):
    began = time.monotonic()
    to, source = partners(world, backend)
    images = _gather(world, tag + ".partners", [to, source, world.endpoints_per_peer])
    if all(image[0] is None for image in images):
        return dict(reason=f"no rank has a {backend} partner")
    widths = tuple(w for w in PROBE_LANES[backend] if w <= min(i[2] for i in images))
    if len(widths) < 2:
        return dict(reason=f"{widths[-1]} endpoint per peer; nothing to fan out")
    device = f"cuda:{world.device}"
    largest = max(FIT_BYTES + HOLDOUT_BYTES)
    tensors = dict(
        payload=_pattern(largest, world.rank, device),
        receive=torch.zeros(largest, device=device, dtype=torch.uint8),
        flags=torch.zeros(widths[-1], device=device, dtype=torch.int64),
    )
    registrations = {name: world.register(t) for name, t in tensors.items()}
    table = world.exchange_regions(
        {name: world.region(name, registrations[name], t) for name, t in tensors.items()},
        tag=tag,
    )
    try:
        return _sweep(world, backend, tag, table, tensors, widths, to, source, began)
    finally:
        table.close()
        for registration in registrations.values():
            registration.close()


def _sweep(world, backend, tag, table, tensors, widths, to, source, began):
    starts = torch.zeros(widths[-1], device=tensors["flags"].device, dtype=torch.int64)
    cursors = [0] * widths[-1]
    endpoints = None
    if to is not None:
        handles = table.endpoint_handles.view(torch.int64)
        endpoints = (
            handles.view(world.world_size, world.endpoints_per_peer)[to, : widths[-1]]
            .contiguous()
            .view(torch.uint64)
        )
    key = table.local("payload").lkey
    remote_receive, remote_flags = table.regions("receive"), table.regions("flags")

    def launch(lanes, warps, size, iterations):
        _send[(lanes,)](endpoints, tensors["payload"], key, remote_receive,
                        remote_flags, starts, to, size // lanes, iterations,
                        num_warps=warps)

    # Compile every variant before timing; a failure stops every rank.
    compile_began, failure = time.monotonic(), ""
    warps_needed = PROBE_WARPS if backend == "p2p" else (1,)
    if to is not None:
        try:
            with world.compilation(peers=(to,)):
                for warps in warps_needed:
                    launch(1, warps, FIT_BYTES[0], 0)
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001
            failure = f"{type(error).__name__}: {error}"
    compile_seconds = time.monotonic() - compile_began
    failures = [f for f in _gather(world, tag + ".compiled", failure) if f]
    if failures:
        raise RuntimeError(f"probe kernel did not compile: {failures[0]}")

    def sweep(sizes, ordered, lanes_warps, label):
        """Rows [bytes, order, width, per-rank us] over the sending ranks."""
        local = []
        for size in sizes:
            for order in (0, 1):
                _gather(world, f"{tag}.{label}.{size}.{order}", 0)
                for width in ordered if order == 0 else ordered[::-1]:
                    times = []
                    if to is not None:
                        lanes, warps = lanes_warps(width)
                        iterations = max(ITERATIONS[0], min(ITERATIONS[1], TRIAL_BYTES // size))
                        events = []
                        with world.compilation(peers=(to,)):
                            for _ in range(TRIALS + 1):
                                begin = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                begin.record()
                                launch(lanes, warps, size, iterations)
                                end.record()
                                starts[:lanes] += iterations
                                events.append((begin, end))
                        torch.cuda.synchronize()
                        for lane in range(lanes):
                            cursors[lane] += iterations * (TRIALS + 1)
                        times = [b.elapsed_time(e) * 1e3 / iterations for b, e in events[1:]]
                    local.append([size, order, width, times])
        gathered = _gather(world, f"{tag}.{label}.samples", local)
        return [
            [size, order, width, [image[i][3] for image in gathered if image[i][3]]]
            for i, (size, order, width, _) in enumerate(local)
        ]

    samples = {}
    warps = 1
    if backend == "p2p":
        samples["warps"] = sweep(FIT_BYTES, PROBE_WARPS, lambda w: (1, w), "warps")
        warps = fit_sender_warps(world_rates(
            {(s, o, w): ranks for s, o, w, ranks in samples["warps"]}))
        samples["sender_warps"] = warps
    rows = sweep(sorted(FIT_BYTES + HOLDOUT_BYTES), widths, lambda w: (w, warps), "lanes")
    samples["fit"] = [row for row in rows if row[0] in FIT_BYTES]
    samples["holdout"] = [row for row in rows if row[0] in HOLDOUT_BYTES]

    # The last cell wrote every byte of the largest size; check the whole
    # receive buffer and every lane's epoch against the sender's cursors.
    torch.cuda.synchronize()
    cursor_images = _gather(world, tag + ".cursors", cursors)
    problem = ""
    if source is not None:
        largest = max(FIT_BYTES + HOLDOUT_BYTES)
        expected = _pattern(largest, source, tensors["receive"].device)
        if not torch.equal(tensors["receive"], expected):
            problem = "payload mismatch"
        elif tensors["flags"].tolist() != cursor_images[source]:
            problem = f"epochs {tensors['flags'].tolist()} != {cursor_images[source]}"
    problems = [(rank, p) for rank, p in enumerate(_gather(world, tag + ".checked", problem)) if p]
    if problems:
        raise RuntimeError(f"probe check failed on rank {problems[0][0]}: {problems[0][1]}")
    seconds = max(_gather(world, tag + ".seconds", time.monotonic() - began))
    return dict(
        samples=samples,
        seconds=seconds,
        world=dict(
            ranks=world.world_size,
            hosts=len({info.hostname for info in world.topology.values()}),
            senders=sum(1 for image in cursor_images if any(image)),
            widths=list(widths),
            compile_seconds=max(_gather(world, tag + ".compile", compile_seconds)),
            trials=TRIALS,
        ),
    )
