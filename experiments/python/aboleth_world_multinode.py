"""Multi-node World smoke: one process per rank over the TCP bootstrap.

Stage A brings up a uniform RDMA World and proves the plumbing without a
compiled datapath: every peer endpoint connects, a registered region is
exchanged, the peer tables carry real addresses and rkeys, and the ranks agree
on the host they landed on.

Stage B compiles the datapath and drives a put+signal ring: every rank writes a
payload to its successor and publishes one signal epoch, then waits for its
predecessor's epoch and checks the bytes. The payload offset moves each round,
so a lowering that folds a dynamic address into a constant is caught here. The
signal cell advances by exactly one per round, which is what
docs/COMPILER_SEMANTICS.md requires of a strong publication: on a transport
with remote atomics an out-of-sequence epoch is a compare-and-swap that misses,
and a missed compare completes successfully while writing nothing.

Ranks come from the environment (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT,
LOCAL_RANK); scripts/multinode-launch.sh sets them from SLURM. `--local-ranks`
spawns that many ranks on this host instead, which is what the ctest entry
uses.

Exit status: 0 pass, 3 a stage-B verification failure (the datapath ran and the
bytes or the epoch were wrong), 1 anything else.

Detected failures stay collective-symmetric: every rank still reaches the
verdict gather and the exit barrier, so all ranks exit together. A hard
exception escaping between collectives desyncs the bootstrap's sequence
counter instead, and peers then block until the store timeout — bounded by
--timeout (twice in the worst case), by the local spawn deadline, and by
srun --kill-on-bad-exit, but not graceful. Keep new checks in the
soft-failure style.
"""

import argparse
import multiprocessing
import os
import pathlib
import socket
import sys
import time
import traceback

import aboleth as _aboleth  # configure the compiler plugin before Triton
import torch
import triton
import triton.language as tl

import aboleth.language as aboleth
from aboleth.bootstrap import TcpStoreBootstrap
from aboleth.runtime import Transport
from aboleth.world import World


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE_BYTES = 4096
CHUNK_BYTES = 256
VERIFY_FAILURE_STATUS = 3
# Rank 0 hosts the store the others are attached to, so dropping it ends the
# server. The exit barrier returns once every peer has published its key, which
# is a moment before the slowest peer has finished reading rank 0's; this is
# the grace that covers that gap.
STORE_DRAIN_SECONDS = 0.5


@triton.jit(do_not_specialize=["byte_offset"])
def ring_kernel(
    endpoint,
    destination_address,
    source,
    remote_signal_address,
    local_signal,
    data_source_lkey,
    data_destination_rkey,
    signal_source_lkey,
    signal_destination_rkey,
    byte_offset,
    byte_count: tl.constexpr,
):
    destination = destination_address.to(
        tl.pointer_type(tl.uint8)
    )
    remote_signal = remote_signal_address.to(
        tl.pointer_type(tl.int64)
    )
    transfer = aboleth.put(
        endpoint,
        destination + byte_offset,
        source + byte_offset,
        byte_count,
        source_key=data_source_lkey,
        destination_key=data_destination_rkey,
    )
    ready = aboleth.signal(
        endpoint,
        remote_signal,
        local_signal,
        transfer,
        source_key=signal_source_lkey,
        destination_key=signal_destination_rkey,
    )
    aboleth.wait(ready)


@triton.jit(do_not_specialize=["warmup_rounds", "measured_rounds"])
def ring_benchmark_kernel(
    endpoint,
    destination_address,
    source,
    remote_signal_address,
    local_signal,
    incoming_signal,
    elapsed_ns,
    data_source_lkey,
    data_destination_rkey,
    signal_source_lkey,
    signal_destination_rkey,
    warmup_rounds,
    measured_rounds,
    transfer_count: tl.constexpr,
    bytes_per_transfer: tl.constexpr,
    submission_per_transfer: tl.constexpr,
    operation_at_a_time: tl.constexpr,
    completion_batch_transfers: tl.constexpr,
):
    destination = destination_address.to(tl.pointer_type(tl.uint8))
    remote_signal = remote_signal_address.to(tl.pointer_type(tl.int64))
    total_rounds = warmup_rounds + measured_rounds
    start = tl.extra.cuda.globaltimer()
    for round_index in tl.range(0, total_rounds):
        if round_index == warmup_rounds:
            start = tl.extra.cuda.globaltimer()
        if submission_per_transfer:
            completion_group = aboleth.begin_group(endpoint)
        transfer = None
        for transfer_index in tl.static_range(0, transfer_count):
            offset = transfer_index * bytes_per_transfer
            transfer = aboleth.put(
                endpoint,
                destination + offset,
                source + offset,
                bytes_per_transfer,
                source_key=data_source_lkey,
                destination_key=data_destination_rkey,
                after=transfer,
            )
            if operation_at_a_time or submission_per_transfer:
                epoch = (
                    round_index * transfer_count
                    + transfer_index
                    + 1
                )
                tl.store(local_signal, epoch.to(tl.int64))
                if submission_per_transfer:
                    aboleth.signal(
                        endpoint,
                        remote_signal,
                        local_signal,
                        transfer,
                        source_key=signal_source_lkey,
                        destination_key=signal_destination_rkey,
                        group=completion_group,
                    )
                else:
                    ready = aboleth.signal(
                        endpoint,
                        remote_signal,
                        local_signal,
                        transfer,
                        source_key=signal_source_lkey,
                        destination_key=signal_destination_rkey,
                    )
                    aboleth.wait(ready)
                    aboleth.wait_signal(incoming_signal, epoch)
                transfer = None
            elif completion_batch_transfers > 0:
                if (
                    transfer_index + 1
                ) % completion_batch_transfers == 0:
                    batches_per_round = (
                        transfer_count
                        // completion_batch_transfers
                    )
                    epoch = (
                        round_index
                        * batches_per_round
                        + transfer_index
                        // completion_batch_transfers
                        + 1
                    )
                    tl.store(
                        local_signal,
                        epoch.to(tl.int64),
                    )
                    ready = aboleth.signal(
                        endpoint,
                        remote_signal,
                        local_signal,
                        transfer,
                        source_key=signal_source_lkey,
                        destination_key=signal_destination_rkey,
                    )
                    aboleth.wait(ready)
                    aboleth.wait_signal(
                        incoming_signal,
                        epoch,
                    )
                    transfer = None
        if submission_per_transfer:
            aboleth.wait_signal(incoming_signal, epoch)
            aboleth.wait_group(completion_group)
        elif (
            not operation_at_a_time
            and completion_batch_transfers == 0
        ):
            epoch = round_index + 1
            tl.store(local_signal, epoch.to(tl.int64))
            ready = aboleth.signal(
                endpoint,
                remote_signal,
                local_signal,
                transfer,
                source_key=signal_source_lkey,
                destination_key=signal_destination_rkey,
            )
            aboleth.wait(ready)
            aboleth.wait_signal(incoming_signal, epoch)
    end = tl.extra.cuda.globaltimer()
    tl.store(elapsed_ns, end - start)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Aboleth multi-node World smoke (one process per rank)"
        )
    )
    parser.add_argument(
        "--stage",
        choices=("a", "b", "c", "all"),
        default="all",
    )
    parser.add_argument(
        "--local-ranks",
        type=int,
        default=0,
        help=(
            "spawn this many ranks on this host instead of reading "
            "RANK/WORLD_SIZE from the environment"
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--transfers", type=int, default=16)
    parser.add_argument("--bytes", type=int, default=256)
    parser.add_argument(
        "--submission-mode",
        choices=("phase", "submission", "operation"),
        default="phase",
        help=(
            "compile all transfers into one phase submission or force one "
            "submission or full completion per transfer"
        ),
    )
    parser.add_argument(
        "--completion-batch-transfers",
        type=int,
        default=0,
        help=(
            "in phase mode, force one publication/completion per "
            "N transfers instead of using the compiler's terminal "
            "phase deadline"
        ),
    )
    return parser.parse_args(argv)


def emit(line: str) -> None:
    """Write a line in one call: ranks share a stdout, and a line written in
    pieces interleaves with another rank's."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def pattern(rank: int, round_index: int, device: int):
    """The bytes rank `rank` sends in round `round_index`."""
    return (
        (
            torch.arange(
                CHUNK_BYTES,
                device=f"cuda:{device}",
                dtype=torch.int64,
            )
            + rank * 31
            + round_index * 29
            + 7
        )
        & 0xFF
    ).to(torch.uint8)


# ---------------------------------------------------------------------------
# Stage A — the World itself, with no compiled datapath
# ---------------------------------------------------------------------------


def check_endpoints(world: World) -> list[str]:
    failures = []
    expected = set(range(world.world_size)) - {world.rank}
    if set(world.endpoints) != expected:
        failures.append(
            f"endpoints {sorted(world.endpoints)} do not cover "
            f"peers {sorted(expected)}"
        )
    if world.transport != Transport.RDMA:
        failures.append(
            f"world transport is {world.transport}, not rdma"
        )
    for peer in sorted(expected & set(world.endpoints)):
        endpoint = world.endpoint(peer)
        if endpoint.peer != peer:
            failures.append(
                f"endpoint for rank {peer} reports peer "
                f"{endpoint.peer}"
            )
        if endpoint.transport != Transport.RDMA:
            failures.append(
                f"endpoint for rank {peer} is "
                f"{endpoint.transport}, not rdma"
            )
        if endpoint.device != world.device:
            failures.append(
                f"endpoint for rank {peer} is on GPU "
                f"{endpoint.device}, not {world.device}"
            )
        if endpoint.device_handle.data_ptr() == 0:
            failures.append(
                f"endpoint for rank {peer} has no device handle"
            )
        producer, consumer, status = endpoint.state()
        if status != 0 or producer != consumer:
            failures.append(
                f"endpoint for rank {peer} is not quiet: "
                f"state={(producer, consumer, status)}"
            )
    return failures


def check_region_tables(world: World, table, name: str) -> list[str]:
    failures = []
    addresses = table.addresses(name).cpu().tolist()
    rkeys = table.rkeys(name).cpu().tolist()
    handles = table.endpoint_handles.cpu().tolist()
    for peer in range(world.world_size):
        if addresses[peer] == 0:
            failures.append(
                f"rank {peer} published a null address for "
                f"region {name!r}"
            )
        if peer == world.rank:
            continue
        if rkeys[peer] == 0:
            failures.append(
                f"rank {peer} published a null rkey for "
                f"region {name!r}"
            )
        if handles[peer] == 0:
            failures.append(
                f"rank {peer} has a null endpoint handle"
            )
    return failures


def run_stage_a(world: World, bootstrap) -> list[str]:
    """Return the local failures; every collective runs either way."""
    rank = world.rank
    failures = list(check_endpoints(world))
    if not failures:
        emit(
            f"stage-a: endpoints ok rank={rank} "
            f"peers={sorted(world.endpoints)} "
            f"transport={world.transport.value} hca={world.hca}"
        )

    probe = torch.full(
        (PROBE_BYTES,),
        rank & 0xFF,
        dtype=torch.uint8,
        device=f"cuda:{world.device}",
    )
    registration = world.register(probe)
    region = world.region("probe", registration, probe)
    table = world.exchange_regions({"probe": region}, tag="stage-a")
    if region.length != PROBE_BYTES or region.lkey == 0:
        failures.append(
            f"local region is {region.length} bytes with lkey "
            f"{region.lkey}"
        )
    else:
        emit(
            f"stage-a: regions ok rank={rank} "
            f"bytes={region.length} lkey={region.lkey:#x}"
        )

    table_failures = check_region_tables(world, table, "probe")
    failures.extend(table_failures)
    if not table_failures:
        emit(
            f"stage-a: peer-tables ok rank={rank} "
            f"names={list(table.names)} "
            f"peers={world.world_size - 1}"
        )

    hostname = socket.gethostname()
    gathered = bootstrap.all_gather(
        "aboleth.multinode.hosts",
        hostname.encode("utf-8"),
    )
    hosts = {
        peer: payload.decode("utf-8")
        for peer, payload in sorted(gathered.items())
    }
    topology_failures = []
    for peer, host in hosts.items():
        if not host:
            topology_failures.append(f"rank {peer} sent no hostname")
        elif world.topology[peer].hostname != host:
            topology_failures.append(
                f"rank {peer} bootstrapped as {host} but its "
                f"topology bundle says "
                f"{world.topology[peer].hostname}"
            )
    failures.extend(topology_failures)
    if not topology_failures:
        emit(f"stage-a: hostnames ok rank={rank} map={hosts}")
    return failures


# ---------------------------------------------------------------------------
# Stage B — the compiled put+signal ring
# ---------------------------------------------------------------------------


def run_stage_b(world: World, bootstrap, args) -> list[str]:
    rank = world.rank
    device = world.device
    successor = (rank + 1) % world.world_size
    predecessor = (rank - 1) % world.world_size
    arena_bytes = max(PROBE_BYTES, args.rounds * CHUNK_BYTES)
    # Each round advances the successor's cell by exactly one, which is what
    # a strong publication requires (docs/COMPILER_SEMANTICS.md): on a
    # transport with remote atomics an out-of-sequence epoch is a
    # compare-and-swap that misses, and a missed compare writes nothing while
    # completing successfully. The payload offset moves every round, which a
    # lowering that folded a dynamic address into a constant would miss.
    emit(
        f"stage-b: ring rank={rank} -> {successor} "
        f"(waiting on {predecessor}) rounds={args.rounds} "
        f"chunk={CHUNK_BYTES}B arena={arena_bytes}B"
    )

    send = torch.zeros(
        arena_bytes,
        dtype=torch.uint8,
        device=f"cuda:{device}",
    )
    receive = torch.zeros(
        arena_bytes,
        dtype=torch.uint8,
        device=f"cuda:{device}",
    )
    signal_out = torch.zeros(
        1,
        dtype=torch.int64,
        device=f"cuda:{device}",
    )
    signal_in = torch.zeros(
        1,
        dtype=torch.int64,
        device=f"cuda:{device}",
    )
    tensors = {
        "send": send,
        "receive": receive,
        "signal_out": signal_out,
        "signal_in": signal_in,
    }
    regions = {}
    for name, tensor in tensors.items():
        regions[name] = world.region(
            name,
            world.register(tensor),
            tensor,
        )
    table = world.exchange_regions(regions, tag="stage-b")

    remote_receive = table.remote(successor, "receive")
    remote_signal = table.remote(successor, "signal_in")
    endpoint = world.endpoint(successor)
    arguments = (
        endpoint.device_handle,
        remote_receive.address,
        send,
        remote_signal.address,
        signal_out,
        table.local("send").lkey,
        remote_receive.rkey,
        table.local("signal_out").lkey,
        remote_signal.rkey,
    )

    # A round waits for a peer that may still be compiling its own first
    # launch, so the per-round budget is generous but always bounded.
    round_seconds = min(args.timeout, 120.0)
    failures = []
    bootstrap.barrier("aboleth.multinode.stage-b.start")
    with world.compilation():
        for round_index in range(args.rounds):
            epoch = round_index + 1
            offset = round_index * CHUNK_BYTES
            send[offset:offset + CHUNK_BYTES].copy_(
                pattern(rank, round_index, device)
            )
            signal_out.fill_(epoch)
            ring_kernel[(1,)](
                *arguments,
                offset,
                byte_count=CHUNK_BYTES,
                num_warps=1,
            )
            torch.cuda.synchronize(device)

            # The ring is not lock-stepped: the predecessor publishes and moves
            # on, so by the time this reads the cell it may already hold a
            # later epoch. Waiting for "at least" rather than "exactly" is
            # what makes the observation independent of that skew; the exact
            # count is checked once at the end. Every epoch up to this one is
            # ordered after its own payload, so the round's bytes have landed.
            deadline = time.monotonic() + round_seconds
            observed = int(signal_in.cpu().item())
            while observed < epoch and time.monotonic() < deadline:
                time.sleep(0.001)
                observed = int(signal_in.cpu().item())
            if observed < epoch:
                failures.append(
                    f"round {round_index}: rank {predecessor}'s "
                    f"signal cell read {observed}, short of epoch "
                    f"{epoch}, after {round_seconds:g}s"
                )
                break

            expected = pattern(predecessor, round_index, device)
            landed = receive[offset:offset + CHUNK_BYTES]
            if not torch.equal(landed, expected):
                mismatched = int((landed != expected).sum().item())
                failures.append(
                    f"round {round_index}: {mismatched}/"
                    f"{CHUNK_BYTES} bytes from rank {predecessor} "
                    f"are wrong at offset {offset}"
                )
                break
            state = endpoint.state()
            if state[2] != 0:
                failures.append(
                    f"round {round_index}: endpoint state {state} "
                    "reports an error"
                )
                break

    if not failures:
        # One epoch per round and no more: a cell that stopped short would
        # have hung the wait above, and one that ran ahead means a
        # publication landed that no round issued.
        final = int(signal_in.cpu().item())
        if final != args.rounds:
            failures.append(
                f"rank {predecessor} published {final} epochs, not "
                f"{args.rounds}"
            )
    return failures


# ---------------------------------------------------------------------------
# Stage C — timed generated phase batching over the physical fabric
# ---------------------------------------------------------------------------


def run_stage_c(world: World, bootstrap, args) -> list[str]:
    rank = world.rank
    device = world.device
    successor = (rank + 1) % world.world_size
    predecessor = (rank - 1) % world.world_size
    data_bytes = args.transfers * args.bytes
    send = (
        (
            torch.arange(
                data_bytes,
                dtype=torch.int64,
                device=f"cuda:{device}",
            )
            * 17
            + rank
            + 3
        )
        & 0xFF
    ).to(torch.uint8)
    receive = torch.zeros_like(send)
    signal_out = torch.zeros(
        1, dtype=torch.int64, device=f"cuda:{device}"
    )
    signal_in = torch.zeros_like(signal_out)
    elapsed_ns = torch.zeros_like(signal_out)

    tensors = {
        "send": send,
        "receive": receive,
        "signal_out": signal_out,
        "signal_in": signal_in,
    }
    regions = {
        name: world.region(name, world.register(tensor), tensor)
        for name, tensor in tensors.items()
    }
    table = world.exchange_regions(regions, tag="stage-c")
    remote_receive = table.remote(successor, "receive")
    remote_signal = table.remote(successor, "signal_in")
    endpoint = world.endpoint(successor)
    arguments = (
        endpoint.device_handle,
        remote_receive.address,
        send,
        remote_signal.address,
        signal_out,
        signal_in,
        elapsed_ns,
        table.local("send").lkey,
        remote_receive.rkey,
        table.local("signal_out").lkey,
        remote_signal.rkey,
        args.warmup_rounds,
        args.rounds,
    )

    with world.compilation():
        ring_benchmark_kernel.warmup(
            *arguments,
            transfer_count=args.transfers,
            bytes_per_transfer=args.bytes,
            submission_per_transfer=(
                args.submission_mode == "submission"
            ),
            operation_at_a_time=(
                args.submission_mode == "operation"
            ),
            completion_batch_transfers=(
                args.completion_batch_transfers
            ),
            num_warps=1,
            grid=(1,),
        )
    bootstrap.barrier("aboleth.multinode.stage-c.start")
    with world.compilation():
        ring_benchmark_kernel[(1,)](
            *arguments,
            transfer_count=args.transfers,
            bytes_per_transfer=args.bytes,
            submission_per_transfer=(
                args.submission_mode == "submission"
            ),
            operation_at_a_time=(
                args.submission_mode == "operation"
            ),
            completion_batch_transfers=(
                args.completion_batch_transfers
            ),
            num_warps=1,
        )
    torch.cuda.synchronize(device)

    failures = []
    expected = (
        (
            torch.arange(
                data_bytes,
                dtype=torch.int64,
                device=f"cuda:{device}",
            )
            * 17
            + predecessor
            + 3
        )
        & 0xFF
    ).to(torch.uint8)
    if not torch.equal(receive, expected):
        failures.append(
            f"rank {rank}: received payload does not match rank {predecessor}"
        )
    publications_per_round = (
        args.transfers
        if args.submission_mode
        in ("submission", "operation")
        else (
            (
                args.transfers
                + args.completion_batch_transfers
                - 1
            )
            // args.completion_batch_transfers
            if args.completion_batch_transfers
            else 1
        )
    )
    expected_epoch = (
        args.warmup_rounds + args.rounds
    ) * publications_per_round
    if int(signal_in.cpu().item()) != expected_epoch:
        failures.append(
            f"rank {rank}: observed epoch {int(signal_in.cpu().item())}, "
            f"expected {expected_epoch}"
        )
    state = endpoint.state()
    if state[2] != 0:
        failures.append(f"rank {rank}: endpoint reports an error: {state}")
    elapsed = int(elapsed_ns.cpu().item())
    emit(
        "stage-c: "
        f"rank={rank} predecessor={predecessor} successor={successor} "
        f"submission_mode={args.submission_mode} "
        f"completion_batch={args.completion_batch_transfers} "
        f"transfers={args.transfers} bytes={args.bytes} "
        f"rounds={args.rounds} ns_per_round={elapsed / args.rounds:.3f} "
        f"put_mops={args.rounds * args.transfers / elapsed * 1000:.6f}"
    )
    return failures


# ---------------------------------------------------------------------------
# One rank
# ---------------------------------------------------------------------------


def agree(bootstrap, stage: str, failures: list[str]) -> list[int]:
    """Gather the per-rank verdict so every rank leaves the same way."""
    gathered = bootstrap.all_gather(
        f"aboleth.multinode.{stage}.verdict",
        b"ok" if not failures else b"fail",
    )
    return sorted(
        peer
        for peer, payload in gathered.items()
        if payload != b"ok"
    )


def report(stage: str, rank: int, failures, failed_ranks) -> None:
    if failures:
        emit(
            f"stage-{stage}: FAIL rank={rank} "
            + "; ".join(failures)
        )
    elif failed_ranks:
        emit(
            f"stage-{stage}: FAIL rank={rank} ranks "
            f"{failed_ranks} reported a failure"
        )


def run_rank(args) -> int:
    local_rank = os.environ.get("LOCAL_RANK")
    if not local_rank:
        emit(
            "multinode: FAIL LOCAL_RANK is unset: every rank picks its "
            "GPU from LOCAL_RANK (scripts/multinode-launch.sh sets it "
            "from SLURM_LOCALID)"
        )
        return 1
    device = int(local_rank)
    torch.cuda.set_device(device)

    bootstrap = TcpStoreBootstrap.from_env(timeout=args.timeout)
    rank = bootstrap.rank
    world_size = bootstrap.world_size
    status = 0
    world = None
    try:
        world = World(
            rank,
            world_size,
            device=device,
            bootstrap=bootstrap,
        )
        # A uniform world: one transport for every peer, which is what a
        # single compilation context can lower.
        world.connect_all(transport="rdma")
        run_b = args.stage in ("b", "all")
        run_c = args.stage in ("c", "all")
        if args.stage in ("a", "all"):
            failures = run_stage_a(world, bootstrap)
            failed_ranks = agree(bootstrap, "a", failures)
            report("a", rank, failures, failed_ranks)
            if failed_ranks:
                status = 1
                run_b = False
                run_c = False
        if run_b:
            failures = run_stage_b(world, bootstrap, args)
            failed_ranks = agree(bootstrap, "b", failures)
            report("b", rank, failures, failed_ranks)
            if failed_ranks:
                status = VERIFY_FAILURE_STATUS
            else:
                emit(
                    f"stage-b: PASS rank={rank} rounds={args.rounds}"
                )
        if run_c and status == 0:
            failures = run_stage_c(world, bootstrap, args)
            failed_ranks = agree(bootstrap, "c", failures)
            report("c", rank, failures, failed_ranks)
            if failed_ranks:
                status = VERIFY_FAILURE_STATUS
            else:
                emit(
                    f"stage-c: PASS rank={rank} rounds={args.rounds}"
                )
    except Exception:
        traceback.print_exc()
        emit(
            f"multinode: FAIL rank={rank} raised; peers will time out "
            "at their next collective"
        )
        status = 1
    finally:
        try:
            bootstrap.barrier("aboleth.multinode.exit")
            if rank == 0:
                time.sleep(STORE_DRAIN_SECONDS)
        except Exception:
            traceback.print_exc()
            status = status or 1
        finally:
            try:
                if world is not None:
                    world.close()
            finally:
                bootstrap.close()
    return status


# ---------------------------------------------------------------------------
# Self-spawn: N ranks on this host
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def local_entry(environment: dict, args) -> None:
    os.environ.update(environment)
    sys.exit(run_rank(args))


def spawn_local(args) -> int:
    visible = torch.cuda.device_count()
    if args.local_ranks < 2:
        emit("multinode: FAIL --local-ranks needs at least two ranks")
        return 1
    if args.local_ranks > visible:
        emit(
            f"multinode: FAIL --local-ranks {args.local_ranks} exceeds "
            f"the {visible} visible GPUs"
        )
        return 1

    port = free_port()
    context = multiprocessing.get_context("spawn")
    processes = {}
    try:
        for rank in range(args.local_ranks):
            environment = {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
                "WORLD_SIZE": str(args.local_ranks),
                "LOCAL_RANK": str(rank),
                # Concurrent ranks compile the same kernel at the same
                # moment; a cache directory each keeps them out of one
                # another's way.
                "TRITON_CACHE_DIR": str(
                    ROOT / ".triton" / "multinode-cache" / f"rank{rank}"
                ),
                "TORCHINDUCTOR_CACHE_DIR": str(
                    ROOT / ".torch-cache" / f"multinode-rank{rank}"
                ),
            }
            process = context.Process(
                target=local_entry,
                args=(environment, args),
            )
            process.start()
            processes[rank] = process

        deadline = time.monotonic() + args.timeout + 120.0
        for rank, process in processes.items():
            process.join(
                timeout=max(1.0, deadline - time.monotonic())
            )
            if process.is_alive():
                emit(
                    f"multinode: FAIL rank {rank} did not exit within "
                    f"{args.timeout + 120:g}s"
                )
    finally:
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        for process in processes.values():
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)

    codes = {
        rank: process.exitcode
        for rank, process in processes.items()
    }
    emit(f"multinode: local ranks exited {codes}")
    if all(code == 0 for code in codes.values()):
        emit("aboleth_world_multinode=PASS")
        return 0
    emit("aboleth_world_multinode=FAIL")
    if all(
        code in (0, VERIFY_FAILURE_STATUS)
        for code in codes.values()
    ):
        return VERIFY_FAILURE_STATUS
    return 1


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        emit("multinode: FAIL --rounds must be positive")
        return 1
    if args.timeout <= 0:
        emit("multinode: FAIL --timeout must be positive")
        return 1
    if args.warmup_rounds < 0:
        emit("multinode: FAIL --warmup-rounds cannot be negative")
        return 1
    if args.transfers < 1 or args.transfers > 256:
        emit("multinode: FAIL --transfers must be between 1 and 256")
        return 1
    if args.bytes < 1:
        emit("multinode: FAIL --bytes must be positive")
        return 1
    if (
        args.completion_batch_transfers < 0
        or args.completion_batch_transfers
        > args.transfers
    ):
        emit(
            "multinode: FAIL --completion-batch-transfers must "
            "be zero or no greater than --transfers"
        )
        return 1
    if (
        args.completion_batch_transfers
        and args.submission_mode != "phase"
    ):
        emit(
            "multinode: FAIL --completion-batch-transfers "
            "requires --submission-mode phase"
        )
        return 1
    if (
        args.completion_batch_transfers
        and args.transfers
        % args.completion_batch_transfers
    ):
        emit(
            "multinode: FAIL --completion-batch-transfers must "
            "divide --transfers"
        )
        return 1
    if args.local_ranks:
        return spawn_local(args)
    return run_rank(args)


if __name__ == "__main__":
    sys.exit(main())
