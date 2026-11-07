import argparse
import json

import torch
import triton
import triton.language as tl

import aboleth.language as aboleth
from aboleth.runtime import Runtime


@triton.jit(
    do_not_specialize=[
        "warmup_iterations",
        "measured_iterations",
    ]
)
def exchange_kernel(
    endpoint,
    remote_tokens,
    local_tokens,
    remote_indices,
    local_indices,
    remote_experts,
    local_experts,
    remote_count,
    local_count,
    remote_dispatch_ready,
    local_dispatch_ready,
    dispatch_value,
    remote_results,
    local_results,
    remote_returned_indices,
    local_received_indices,
    remote_returned_count,
    local_received_count,
    remote_return_ready,
    local_return_ready,
    return_value,
    elapsed_ns,
    token_lkey,
    token_rkey,
    index_lkey,
    index_rkey,
    expert_lkey,
    expert_rkey,
    count_lkey,
    count_rkey,
    dispatch_signal_lkey,
    dispatch_signal_rkey,
    result_lkey,
    result_rkey,
    return_index_lkey,
    return_index_rkey,
    return_count_lkey,
    return_count_rkey,
    return_signal_lkey,
    return_signal_rkey,
    warmup_iterations,
    measured_iterations,
    hidden_size: tl.constexpr,
):
    total_iterations = warmup_iterations + measured_iterations
    start = tl.extra.cuda.globaltimer()
    for iteration in tl.range(0, total_iterations):
        epoch = iteration + 1
        if iteration == warmup_iterations:
            start = tl.extra.cuda.globaltimer()

        count = tl.load(local_count).to(tl.int32)
        has_tokens = count > 0
        token_bytes = count * hidden_size * 4
        metadata_bytes = count * 4
        transfer = aboleth.put(
            endpoint,
            remote_tokens,
            local_tokens,
            token_bytes,
            source_key=token_lkey,
            destination_key=token_rkey,
            predicate=has_tokens,
        )
        transfer = aboleth.put(
            endpoint,
            remote_indices,
            local_indices,
            metadata_bytes,
            source_key=index_lkey,
            destination_key=index_rkey,
            predicate=has_tokens,
            after=transfer,
        )
        transfer = aboleth.put(
            endpoint,
            remote_experts,
            local_experts,
            metadata_bytes,
            source_key=expert_lkey,
            destination_key=expert_rkey,
            predicate=has_tokens,
            after=transfer,
        )
        transfer = aboleth.put(
            endpoint,
            remote_count,
            local_count,
            8,
            source_key=count_lkey,
            destination_key=count_rkey,
            after=transfer,
        )
        tl.store(dispatch_value, epoch)
        ready = aboleth.signal(
            endpoint,
            remote_dispatch_ready,
            dispatch_value,
            transfer,
            source_key=dispatch_signal_lkey,
            destination_key=dispatch_signal_rkey,
        )
        aboleth.wait(ready)
        aboleth.wait_signal(local_dispatch_ready, epoch)

        return_count = tl.load(local_received_count).to(tl.int32)
        has_results = return_count > 0
        result_bytes = return_count * hidden_size * 4
        return_index_bytes = return_count * 4
        returned = aboleth.put(
            endpoint,
            remote_results,
            local_results,
            result_bytes,
            source_key=result_lkey,
            destination_key=result_rkey,
            predicate=has_results,
        )
        returned = aboleth.put(
            endpoint,
            remote_returned_indices,
            local_received_indices,
            return_index_bytes,
            source_key=return_index_lkey,
            destination_key=return_index_rkey,
            predicate=has_results,
            after=returned,
        )
        returned = aboleth.put(
            endpoint,
            remote_returned_count,
            local_received_count,
            8,
            source_key=return_count_lkey,
            destination_key=return_count_rkey,
            after=returned,
        )
        tl.store(return_value, epoch)
        returned_ready = aboleth.signal(
            endpoint,
            remote_return_ready,
            return_value,
            returned,
            source_key=return_signal_lkey,
            destination_key=return_signal_rkey,
        )
        aboleth.wait(returned_ready)
        aboleth.wait_signal(local_return_ready, epoch)

    end = tl.extra.cuda.globaltimer()
    tl.store(elapsed_ns, end - start)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--transport",
        choices=("rdma", "proxy"),
        default="rdma",
    )
    return parser.parse_args()


def allocate(rank: int, tokens: int, hidden: int):
    device = f"cuda:{rank}"
    return {
        "send_tokens": torch.full(
            (tokens, hidden),
            float(rank + 1),
            device=device,
            dtype=torch.float32,
        ),
        "send_indices": torch.arange(
            tokens,
            device=device,
            dtype=torch.int32,
        ),
        "send_experts": torch.zeros(
            tokens,
            device=device,
            dtype=torch.int32,
        ),
        "send_count": torch.full(
            (1,),
            tokens,
            device=device,
            dtype=torch.int64,
        ),
        "recv_tokens": torch.zeros(
            tokens,
            hidden,
            device=device,
            dtype=torch.float32,
        ),
        "recv_indices": torch.zeros(
            tokens,
            device=device,
            dtype=torch.int32,
        ),
        "recv_experts": torch.zeros(
            tokens,
            device=device,
            dtype=torch.int32,
        ),
        "recv_count": torch.zeros(
            1,
            device=device,
            dtype=torch.int64,
        ),
        "dispatch_ready": torch.zeros(
            1,
            device=device,
            dtype=torch.int64,
        ),
        "dispatch_value": torch.zeros(
            1,
            device=device,
            dtype=torch.int64,
        ),
        "returned_results": torch.zeros(
            tokens,
            hidden,
            device=device,
            dtype=torch.float32,
        ),
        "returned_indices": torch.zeros(
            tokens,
            device=device,
            dtype=torch.int32,
        ),
        "returned_count": torch.zeros(
            1,
            device=device,
            dtype=torch.int64,
        ),
        "return_ready": torch.zeros(
            1,
            device=device,
            dtype=torch.int64,
        ),
        "return_value": torch.zeros(
            1,
            device=device,
            dtype=torch.int64,
        ),
        "elapsed_ns": torch.zeros(
            1,
            device=device,
            dtype=torch.int64,
        ),
    }


def register(runtime: Runtime, tensors):
    return {
        name: runtime.register(tensor)
        for name, tensor in tensors.items()
        if name != "elapsed_ns"
    }


def keys(endpoint, local, remote):
    return (
        endpoint.local_buffer(local).lkey,
        endpoint.remote_buffer(remote).rkey,
    )


def kernel_arguments(endpoint, local, remote, local_reg, remote_reg):
    key_names = (
        ("send_tokens", "recv_tokens"),
        ("send_indices", "recv_indices"),
        ("send_experts", "recv_experts"),
        ("send_count", "recv_count"),
        ("dispatch_value", "dispatch_ready"),
        ("recv_tokens", "returned_results"),
        ("recv_indices", "returned_indices"),
        ("recv_count", "returned_count"),
        ("return_value", "return_ready"),
    )
    key_values = []
    for local_name, remote_name in key_names:
        key_values.extend(
            keys(
                endpoint,
                local_reg[local_name],
                remote_reg[remote_name],
            )
        )
    return (
        endpoint.device_handle,
        remote["recv_tokens"],
        local["send_tokens"],
        remote["recv_indices"],
        local["send_indices"],
        remote["recv_experts"],
        local["send_experts"],
        remote["recv_count"],
        local["send_count"],
        remote["dispatch_ready"],
        local["dispatch_ready"],
        local["dispatch_value"],
        remote["returned_results"],
        local["recv_tokens"],
        remote["returned_indices"],
        local["recv_indices"],
        remote["returned_count"],
        local["recv_count"],
        remote["return_ready"],
        local["return_ready"],
        local["return_value"],
        local["elapsed_ns"],
        *key_values,
    )


def main() -> None:
    args = parse_args()
    tensors = (
        allocate(0, args.remote_tokens, args.hidden),
        allocate(1, args.remote_tokens, args.hidden),
    )
    with Runtime() as runtime:
        registered = (
            register(runtime, tensors[0]),
            register(runtime, tensors[1]),
        )
        with runtime.create_endpoint_pair(
            0,
            1,
            transport=args.transport,
        ) as pair:
            forward, reverse = pair.endpoints
            arguments = (
                kernel_arguments(
                    forward,
                    tensors[0],
                    tensors[1],
                    registered[0],
                    registered[1],
                ),
                kernel_arguments(
                    reverse,
                    tensors[1],
                    tensors[0],
                    registered[1],
                    registered[0],
                ),
            )
            with forward.compilation():
                for rank in range(2):
                    with torch.cuda.device(rank):
                        compiled = exchange_kernel.warmup(
                            *arguments[rank],
                            args.warmup_iterations,
                            args.iterations,
                            hidden_size=args.hidden,
                            num_warps=1,
                            grid=(1,),
                        )
                        compiled._init_handles()
                with torch.cuda.device(0):
                    exchange_kernel[(1,)](
                        *arguments[0],
                        args.warmup_iterations,
                        args.iterations,
                        hidden_size=args.hidden,
                        num_warps=1,
                    )
                with torch.cuda.device(1):
                    exchange_kernel[(1,)](
                        *arguments[1],
                        args.warmup_iterations,
                        args.iterations,
                        hidden_size=args.hidden,
                        num_warps=1,
                    )
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)

            assert pair.status() == (0, 0)
            total_iterations = (
                args.warmup_iterations + args.iterations
            )
            if args.transport == "proxy":
                expected_state = (
                    2 * total_iterations,
                ) * 4
            else:
                expected_state = (
                    9 * total_iterations,
                    2 * total_iterations,
                    9 * total_iterations,
                    2 * total_iterations,
                )
            assert pair.queue_state() == expected_state

    for rank in range(2):
        expected = tensors[rank]["send_tokens"]
        torch.testing.assert_close(
            tensors[rank]["returned_results"],
            expected,
            rtol=0,
            atol=0,
        )
    elapsed = max(
        tensors[0]["elapsed_ns"].cpu().item(),
        tensors[1]["elapsed_ns"].cpu().item(),
    )
    result = {
        "backend": "aboleth",
        "remote_tokens": args.remote_tokens,
        "hidden_size": args.hidden,
        "iterations": args.iterations,
        "transport": args.transport,
        "total_ms": elapsed / args.iterations / 1_000_000,
        "status": "PASS",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
