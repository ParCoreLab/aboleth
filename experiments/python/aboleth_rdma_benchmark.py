import argparse
import json

import torch
import triton
import triton.language as tl

import aboleth.language as aboleth
from aboleth.compiler import compilation
from aboleth.runtime import Runtime


@triton.jit
def benchmark_producer(
    endpoint,
    local_data,
    remote_data,
    local_ready_value,
    remote_ready_signal,
    acknowledgment_signal,
    elapsed_ns,
    source_lkey,
    destination_rkey,
    warmup_rounds,
    measured_rounds,
    transfer_count: tl.constexpr,
    bytes_per_transfer: tl.constexpr,
):
    total_rounds = warmup_rounds + measured_rounds
    start = tl.extra.cuda.globaltimer()
    for round_index in tl.range(0, total_rounds):
        if round_index == warmup_rounds:
            start = tl.extra.cuda.globaltimer()
        round_value = round_index + 1
        transfer = None
        for index in tl.static_range(0, transfer_count):
            offset = index * bytes_per_transfer
            transfer = aboleth.put(
                endpoint,
                remote_data + offset,
                local_data + offset,
                bytes_per_transfer,
                source_key=source_lkey,
                destination_key=destination_rkey,
                after=transfer,
            )
        tl.store(local_ready_value, round_value.to(tl.int64))
        ready = aboleth.signal(
            endpoint,
            remote_ready_signal,
            local_ready_value,
            transfer,
            source_key=source_lkey,
            destination_key=destination_rkey,
        )
        aboleth.wait(ready)
        aboleth.wait_signal(
            acknowledgment_signal,
            round_value,
        )
    end = tl.extra.cuda.globaltimer()
    tl.store(elapsed_ns, end - start)


@triton.jit
def benchmark_consumer(
    reverse_endpoint,
    ready_signal,
    local_ack_value,
    remote_ack_signal,
    source_lkey,
    destination_rkey,
    warmup_rounds,
    measured_rounds,
):
    total_rounds = warmup_rounds + measured_rounds
    for round_index in tl.range(0, total_rounds):
        round_value = round_index + 1
        aboleth.wait_signal(ready_signal, round_value)
        tl.store(local_ack_value, round_value.to(tl.int64))
        acknowledgment = aboleth.signal(
            reverse_endpoint,
            remote_ack_signal,
            local_ack_value,
            source_key=source_lkey,
            destination_key=destination_rkey,
        )
        aboleth.wait(acknowledgment)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10_000)
    parser.add_argument("--warmup-rounds", type=int, default=100)
    parser.add_argument("--transfers", type=int, default=32)
    parser.add_argument("--bytes", type=int, default=256)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    assert args.rounds > 0
    assert args.warmup_rounds >= 0
    assert 0 < args.transfers <= 256
    assert args.bytes > 0
    assert torch.cuda.device_count() >= 2

    data_bytes = args.transfers * args.bytes
    allocation_bytes = data_bytes + 16
    allocation0 = torch.zeros(
        allocation_bytes,
        dtype=torch.uint8,
        device="cuda:0",
    )
    allocation1 = torch.zeros(
        allocation_bytes,
        dtype=torch.uint8,
        device="cuda:1",
    )
    data0 = allocation0[:data_bytes]
    acknowledgment0 = allocation0[
        data_bytes:data_bytes + 8
    ].view(torch.int64)
    ready_value0 = allocation0[
        data_bytes + 8:data_bytes + 16
    ].view(torch.int64)
    data1 = allocation1[:data_bytes]
    ready1 = allocation1[
        data_bytes:data_bytes + 8
    ].view(torch.int64)
    acknowledgment_value1 = allocation1[
        data_bytes + 8:data_bytes + 16
    ].view(torch.int64)

    pattern = (
        (torch.arange(data_bytes, device="cuda:0") * 17 + 3) & 0xFF
    ).to(torch.uint8)
    data0.copy_(pattern)

    elapsed_ns_device = torch.zeros(
        1,
        dtype=torch.int64,
        device="cuda:0",
    )

    runtime = Runtime()
    buffer0 = runtime.register(allocation0)
    buffer1 = runtime.register(allocation1)
    pair = runtime.create_endpoint_pair(
        0,
        1,
        transport="rdma",
    )
    forward, reverse = pair.endpoints
    forward_local = forward.local_buffer(buffer0)
    forward_remote = forward.remote_buffer(buffer1)
    reverse_local = reverse.local_buffer(buffer1)
    reverse_remote = reverse.remote_buffer(buffer0)

    try:
        with torch.cuda.device(1):
            compiled_consumer = benchmark_consumer.warmup(
                reverse.device_handle,
                ready1,
                acknowledgment_value1,
                acknowledgment0,
                reverse_local.lkey,
                reverse_remote.rkey,
                args.warmup_rounds,
                args.rounds,
                num_warps=1,
                grid=(1,),
            )
            assert "aboleth." not in compiled_consumer.asm["ttir"]

        with torch.cuda.device(0):
            compiled_producer = benchmark_producer.warmup(
                forward.device_handle,
                data0,
                data1,
                ready_value0,
                ready1,
                acknowledgment0,
                elapsed_ns_device,
                forward_local.lkey,
                forward_remote.rkey,
                args.warmup_rounds,
                args.rounds,
                transfer_count=args.transfers,
                bytes_per_transfer=args.bytes,
                num_warps=1,
                grid=(1,),
            )
            assert "aboleth." not in compiled_producer.asm["ttir"]

        with torch.cuda.device(1):
            benchmark_consumer[(1,)](
                reverse.device_handle,
                ready1,
                acknowledgment_value1,
                acknowledgment0,
                reverse_local.lkey,
                reverse_remote.rkey,
                args.warmup_rounds,
                args.rounds,
                num_warps=1,
            )

        with torch.cuda.device(0):
            benchmark_producer[(1,)](
                forward.device_handle,
                data0,
                data1,
                ready_value0,
                ready1,
                acknowledgment0,
                elapsed_ns_device,
                forward_local.lkey,
                forward_remote.rkey,
                args.warmup_rounds,
                args.rounds,
                transfer_count=args.transfers,
                bytes_per_transfer=args.bytes,
                num_warps=1,
            )

        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        torch.testing.assert_close(
            data1.cpu(),
            pattern.cpu(),
            rtol=0,
            atol=0,
        )
        total_rounds = (
            args.warmup_rounds + args.rounds
        )
        assert ready1.cpu().item() == total_rounds
        assert acknowledgment0.cpu().item() == total_rounds

        states = pair.queue_state()
        assert states == (
            total_rounds * (args.transfers + 1),
            total_rounds,
            total_rounds,
            total_rounds,
        )
        assert pair.status() == (0, 0)

        elapsed_ns = elapsed_ns_device.cpu().item()
        round_trip_ns = elapsed_ns / args.rounds
        seconds = elapsed_ns / 1_000_000_000
        payload_bytes = args.rounds * args.transfers * args.bytes
        result = {
            "backend": "aboleth",
            "rounds": args.rounds,
            "warmup_rounds": args.warmup_rounds,
            "transfers": args.transfers,
            "bytes_per_transfer": args.bytes,
            "transport": "rdma",
            "elapsed_ns": elapsed_ns,
            "round_trip_ns": round_trip_ns,
            "put_mops": args.rounds * args.transfers / seconds / 1_000_000,
            "effective_payload_gbps": payload_bytes * 8 / seconds / 1_000_000_000,
        }
        print(json.dumps(result, sort_keys=True))
    finally:
        runtime.close()


def main() -> None:
    with compilation("rdma"):
        run()


if __name__ == "__main__":
    main()
