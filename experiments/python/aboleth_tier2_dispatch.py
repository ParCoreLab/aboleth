"""Tier-2 fused FP8 dispatch: the paper's Listing 2, run end to end.

One program per peer quantizes that peer's tokens chunk by chunk, stages each
chunk into a send slot and puts it, sends the token count from a register,
and publishes the epoch. The compiler sees the whole kernel: the chunks, the
count, and the flag become one submission, the count travels inside the
command, the endpoint has a single issuer, the staging stores are ordered
before the submission, and completion is due only as often as the send slots
recycle.

Ranks run on this host (`--ranks`), over peer memory or over the NIC
(`--transport rdma`, with ABOLETH_RDMA_SUBMIT=staged where the GPU may not
ring doorbells). Every receiver checks each epoch's counts, flags, and
dequantized rows against a reference. ABOLETH_DUMP_SCHEDULED_TTIR=1 prints the
Queue IR each kernel compiles to.

Exit status: 0 pass, 1 failure.
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
from aboleth.world import World

ROOT = pathlib.Path(__file__).resolve().parents[2]
H = 128
BLOCK = 16
CHUNKS = 4
SLOTS = 4
CHUNK_BYTES = BLOCK * H + BLOCK * 4


@triton.jit(do_not_specialize=["epoch"])
def dispatch(x, counts, endpoints, send_ptr, send_key, recv_table,
             recv_n_table, flag_table, epoch, RANK: tl.constexpr,
             WORLD: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
             CHUNKS: tl.constexpr, SLOTS: tl.constexpr):
    peer = tl.program_id(0)
    if peer != RANK:
        CHUNK: tl.constexpr = BLOCK * H + BLOCK * 4
        ep = aboleth.endpoint(endpoints, peer)  # one issuer
        send = aboleth.region(send_ptr, send_key)
        recv = aboleth.remote(recv_table, peer)
        recv_n = aboleth.remote(recv_n_table, peer, tl.int32)
        flag = aboleth.remote(flag_table, peer, tl.int64)
        n = tl.load(counts + peer)  # tokens for peer
        slot = epoch % SLOTS
        columns = tl.arange(0, H)
        t = None
        for c in tl.static_range(CHUNKS):  # unrolled
            rows = c * BLOCK + tl.arange(0, BLOCK)
            live = c * BLOCK < n
            tokens = tl.load(
                x + (peer * CHUNKS * BLOCK + rows)[:, None] * H
                + columns[None, :],
                mask=(rows < n)[:, None], other=0.0,
            ).to(tl.float32)
            scale = tl.maximum(tl.max(tl.abs(tokens), axis=1), 1e-12) / 448.0
            q = (tokens / scale[:, None]).to(tl.float8e4nv)
            src = send.at(((slot * WORLD + peer) * CHUNKS + c) * CHUNK)
            tl.store(src.pointer + tl.arange(0, BLOCK)[:, None] * H
                     + columns[None, :], q.to(tl.uint8, bitcast=True))
            scales = (src.pointer + BLOCK * H).to(
                tl.pointer_type(tl.float32), bitcast=True)
            tl.store(scales + tl.arange(0, BLOCK), scale)
            t = aboleth.put(ep, recv.at((RANK * CHUNKS + c) * CHUNK), src,
                            CHUNK, predicate=live, after=t)
        t = aboleth.put_inline(ep, recv_n.at(RANK), n, after=t)  # register
        s = aboleth.signal(ep, flag.at(RANK), epoch, after=t)
        aboleth.wait(s, iteration=epoch, reuse=SLOTS)


@triton.jit(do_not_specialize=["epoch"])
def await_peers(flags, epoch, RANK: tl.constexpr):
    peer = tl.program_id(0)
    if peer != RANK:
        aboleth.wait_signal(flags + peer, epoch)


@triton.jit
def drain(endpoints, RANK: tl.constexpr):
    peer = tl.program_id(0)
    if peer != RANK:
        aboleth.drain(aboleth.endpoint(endpoints, peer))


def tokens_for(sender: int, epoch: int, world: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(sender * 1_000_003 + epoch)
    return torch.randn(world, CHUNKS * BLOCK, H, generator=generator,
                       dtype=torch.float32).to(torch.bfloat16)


def count_for(sender: int, receiver: int, epoch: int) -> int:
    return (sender * 7 + receiver * 3 + epoch * 5) % (CHUNKS * BLOCK + 1)


def check(rank: int, world: int, epoch: int, receive, counts, flags):
    failures = []
    for peer in range(world):
        if peer == rank:
            continue
        expected_count = count_for(peer, rank, epoch)
        if int(counts[peer]) != expected_count:
            failures.append(f"epoch {epoch}: count from {peer} is "
                            f"{int(counts[peer])}, not {expected_count}")
        if int(flags[peer]) != epoch:
            failures.append(f"epoch {epoch}: flag from {peer} is "
                            f"{int(flags[peer])}")
        rows = tokens_for(peer, epoch, world)[rank, :expected_count].float()
        chunks = receive[peer].cpu()
        payload = chunks[:, :BLOCK * H].reshape(CHUNKS * BLOCK, H)
        scales = chunks[:, BLOCK * H:].contiguous().view(torch.float32)
        received = (payload[:expected_count].view(torch.float8_e4m3fn).float()
                    * scales.reshape(-1)[:expected_count, None])
        error = (received - rows).abs()
        bound = 0.0725 * rows.abs().amax(dim=1, keepdim=True) + 1e-6
        if bool((error > bound).any()):
            failures.append(f"epoch {epoch}: {int((error > bound).sum())} "
                            f"values from {peer} are wrong")
    return failures


def run_rank(args) -> int:
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    bootstrap = TcpStoreBootstrap.from_env(timeout=args.timeout)
    rank, world_size = bootstrap.rank, bootstrap.world_size
    failures = []
    world = None
    try:
        world = World(rank, world_size, device=device, bootstrap=bootstrap)
        world.connect_all(transport=args.transport)
        cuda = f"cuda:{device}"
        tensors = {
            "send": torch.zeros(SLOTS * world_size * CHUNKS * CHUNK_BYTES,
                                dtype=torch.uint8, device=cuda),
            "receive": torch.zeros(world_size, CHUNKS, CHUNK_BYTES,
                                   dtype=torch.uint8, device=cuda),
            "counts": torch.zeros(world_size, dtype=torch.int32, device=cuda),
            "flags": torch.zeros(world_size, dtype=torch.int64, device=cuda),
        }
        regions = {name: world.region(name, world.register(tensor), tensor)
                   for name, tensor in tensors.items()}
        table = world.exchange_regions(regions, tag="tier2-dispatch")
        send_key = table.local("send").lkey
        constants = dict(RANK=rank, WORLD=world_size, H=H, BLOCK=BLOCK,
                         CHUNKS=CHUNKS, SLOTS=SLOTS, num_warps=4)
        bootstrap.barrier("tier2-dispatch.start")
        with world.compilation():
            for epoch in range(1, args.epochs + 1):
                x = tokens_for(rank, epoch, world_size).to(cuda)
                counts = torch.tensor(
                    [count_for(rank, peer, epoch)
                     for peer in range(world_size)],
                    dtype=torch.int32, device=cuda)
                dispatch[(world_size,)](
                    x, counts, table.endpoint_handles, tensors["send"],
                    send_key, table.regions("receive"),
                    table.regions("counts"), table.regions("flags"), epoch,
                    **constants)
                await_peers[(world_size,)](tensors["flags"], epoch, RANK=rank)
                torch.cuda.synchronize(device)
                failures += check(rank, world_size, epoch, tensors["receive"],
                                  tensors["counts"].cpu(),
                                  tensors["flags"].cpu())
                if failures:
                    break
                # The next epoch overwrites what every receiver just checked.
                bootstrap.barrier(f"tier2-dispatch.epoch{epoch}")
            drain[(world_size,)](table.endpoint_handles, RANK=rank)
            torch.cuda.synchronize(device)
    except Exception:
        traceback.print_exc()
        failures.append("raised")
    for failure in failures:
        print(f"rank {rank}: {failure}", flush=True)
    try:
        bootstrap.barrier("tier2-dispatch.exit")
    finally:
        if world is not None:
            world.close()
        bootstrap.close()
    return 1 if failures else 0


def local_entry(environment: dict, args) -> None:
    os.environ.update(environment)
    sys.exit(run_rank(args))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--transport", default="p2p", choices=("p2p", "rdma"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.ranks < 2 or args.ranks > torch.cuda.device_count():
        print(f"need 2..{torch.cuda.device_count()} ranks", flush=True)
        return 1
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    processes = []
    for rank in range(args.ranks):
        environment = {
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
            "RANK": str(rank), "WORLD_SIZE": str(args.ranks),
            "LOCAL_RANK": str(rank),
            "TRITON_CACHE_DIR": str(ROOT / ".triton" / "tier2" / f"rank{rank}"),
        }
        process = context.Process(target=local_entry, args=(environment, args))
        process.start()
        processes.append(process)
    deadline = time.monotonic() + args.timeout + 120.0
    for process in processes:
        process.join(timeout=max(1.0, deadline - time.monotonic()))
        if process.is_alive():
            process.kill()
            process.join()
    codes = [process.exitcode for process in processes]
    passed = all(code == 0 for code in codes)
    print(f"aboleth_tier2_dispatch={'PASS' if passed else 'FAIL'} "
          f"transport={args.transport} exits={codes}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
