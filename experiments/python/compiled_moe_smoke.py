"""Changing-route graph smoke for the compiler-owned MoE path.

Run directly on two local GPUs, or under torchrun with --internode on two hosts.
The scalar expert oracle is independent of dispatch/return implementation.
"""

import argparse
import hashlib
import os
import socket

import aboleth  # Configure the compiler plugin before importing Triton.
import torch
from aboleth.bootstrap import TcpStoreBootstrap
from aboleth.moe.native import NativeMoEFamily


def worker(rank, port, internode=False):
    device = int(os.environ.get("LOCAL_RANK", rank))
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="2",
        MASTER_ADDR=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        MASTER_PORT=str(port),
    )
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    bootstrap = TcpStoreBootstrap.from_env(timeout=90)
    family = NativeMoEFamily(
        rank=rank,
        ranks=2,
        device=device,
        bootstrap=bootstrap,
        capacity=16,
        hidden=256,
        top_k=2,
        experts=16,
        slots=2,
        owners=tuple(e // 8 for e in range(16)),
        internode=internode,
        tag="compiled-moe-smoke",
    )

    def values(n, step):
        tokens = (
            (torch.arange(n * 256).reshape(n, 256) + rank + step) % 8
        ).to(torch.bfloat16)
        first = (torch.arange(n) + 3 * step + rank) % 16
        ids = torch.stack((first, (first + 5) % 16), dim=1).to(torch.int32)
        if step % 7 == 3:
            ids[:, 1] = -1
        weights = torch.full((n, 2), 0.5, dtype=torch.float32)
        # Independent scalar-expert oracle: expert e multiplies by (e+1).
        coefficient = torch.where(
            ids >= 0, (ids.float() + 1) * weights, 0
        ).sum(dim=1)
        expected = (tokens.float() * coefficient[:, None]).to(torch.bfloat16)
        return tokens, ids, weights, expected

    def operation(handle, x, ids, w):
        received, counts, expert_ids, expert_weights = handle.dispatch(
            x, ids, w
        )
        local = (expert_ids >= 0) & (expert_ids // 8 == rank)
        coefficient = torch.where(
            local, (expert_ids.float() + 1) * expert_weights, 0
        ).sum(dim=1)
        result = torch.where(
            coefficient[:, None] != 0,
            received.float() * coefficient[:, None],
            0,
        ).to(torch.bfloat16)
        return handle.combine(result).clone()

    graphs = {}
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for n in [0, 1, 7, 16]:
        for slot in range(2):
            handle = family.select(max(1, n), slot)
            x, ids, w, _ = values(n, 0)
            inputs = [v.cuda() for v in (x, ids, w)]
            operation(handle, *inputs)
            torch.cuda.synchronize()
            bootstrap.barrier(f"warm-{n}-{slot}")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=streams[slot]):
                result = operation(handle, *inputs)
            torch.cuda.synchronize()
            bootstrap.barrier(f"capture-{n}-{slot}")
            graphs[n, slot] = (graph, handle, inputs, result)
    digest = hashlib.sha256()
    checked = 0
    observations = []
    for step in range(24):
        n = [0, 1, 7, 16][step % 4]
        cpu = values(n, step)
        expected = cpu[3]
        for slot in range(2):
            _, _, inputs, _ = graphs[n, slot]
            for tensor, value in zip(inputs, cpu[:3], strict=True):
                tensor.copy_(value)
        torch.cuda.synchronize()
        bootstrap.barrier(f"inputs-{step}")
        for slot in range(2):
            graph, _, _, _ = graphs[n, slot]
            with torch.cuda.stream(streams[slot]):
                graph.replay()
        torch.cuda.synchronize()
        for slot in range(2):
            _, handle, _, result = graphs[n, slot]
            actual = result.cpu()
            assert torch.equal(actual, expected), (
                step,
                slot,
                n,
                actual,
                expected,
            )
            assert int(handle.buffers["overflow"].item()) == 0
            digest.update(actual.view(torch.uint16).numpy().tobytes())
            checked += actual.numel()
            observations.append(
                dict(step=step, slot=slot, n=n, capacity=handle.spec.capacity)
            )
        bootstrap.barrier(f"checked-{step}")
    bootstrap.barrier("done")
    family.close()
    bootstrap.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internode", action="store_true")
    args = parser.parse_args()
    if "RANK" in os.environ:
        assert int(os.environ["WORLD_SIZE"]) == 2
        worker(
            int(os.environ["RANK"]),
            int(os.environ["MASTER_PORT"]),
            args.internode,
        )
    else:
        assert not args.internode and torch.cuda.device_count() >= 2
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        torch.multiprocessing.spawn(worker, args=(port,), nprocs=2, join=True)
    print("compiled_moe_smoke=PASS")


if __name__ == "__main__":
    main()
