import argparse

import torch
import triton
import triton.language as tl
from cuda.bindings import runtime as cudart

import aboleth.language as aboleth
from aboleth.compiler import compilation
from aboleth_rdma_abi_test import (
    DESCRIPTOR_WORDS,
    EndpointPair,
    load_runtime,
)


BYTES_PER_TRANSFER = 256
BUFFER_SLOTS = 2


@triton.jit
def persistent_producer(
    endpoint,
    local_data,
    remote_data,
    local_ready_value,
    remote_ready_signal,
    acknowledgment_signal,
    source_lkey,
    destination_rkey,
    rounds,
    transfer_count: tl.constexpr,
    bytes_per_transfer: tl.constexpr,
    batch_bytes: tl.constexpr,
    block_size: tl.constexpr,
    buffer_slots: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    for round_index in tl.range(0, rounds):
        round_value = round_index + 1
        if round_index >= buffer_slots:
            aboleth.wait_signal(
                acknowledgment_signal,
                round_value - buffer_slots,
            )

        slot = round_index & (buffer_slots - 1)
        slot_offset = slot * batch_bytes
        pattern = (
            (offsets * 17 + round_value * 29 + 3) & 0xFF
        ).to(tl.uint8)
        tl.store(
            local_data + slot_offset + offsets,
            pattern,
            mask=offsets < batch_bytes,
        )

        transfer = None
        for index in tl.static_range(0, transfer_count):
            transfer_offset = (
                slot_offset +
                index * bytes_per_transfer
            )
            transfer = aboleth.put(
                endpoint,
                remote_data + transfer_offset,
                local_data + transfer_offset,
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

    aboleth.wait_signal(acknowledgment_signal, rounds)


@triton.jit
def persistent_consumer(
    reverse_endpoint,
    received_data,
    ready_signal,
    local_ack_value,
    remote_ack_signal,
    mismatch_count,
    source_lkey,
    destination_rkey,
    rounds,
    transfer_count: tl.constexpr,
    bytes_per_transfer: tl.constexpr,
    batch_bytes: tl.constexpr,
    block_size: tl.constexpr,
    buffer_slots: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    for round_index in tl.range(0, rounds):
        round_value = round_index + 1
        aboleth.wait_signal(ready_signal, round_value)

        slot = round_index & (buffer_slots - 1)
        slot_offset = slot * batch_bytes
        values = tl.load(
            received_data + slot_offset + offsets,
            mask=offsets < batch_bytes,
            other=0,
        )
        expected = (
            (offsets * 17 + round_value * 29 + 3) & 0xFF
        ).to(tl.uint8)
        mismatches = tl.sum(
            (values != expected) &
            (offsets < batch_bytes)
        )
        tl.atomic_add(mismatch_count, mismatches)

        tl.store(local_ack_value, round_value.to(tl.int64))
        acknowledgment = aboleth.signal(
            reverse_endpoint,
            remote_ack_signal,
            local_ack_value,
            source_key=source_lkey,
            destination_key=destination_rkey,
        )
        aboleth.wait(acknowledgment)


def cuda_check(result, operation: str) -> None:
    error = result[0]
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {error}")


def enable_peer(source: int, destination: int) -> None:
    cuda_check(
        cudart.cudaSetDevice(source),
        f"cudaSetDevice({source})",
    )
    result = cudart.cudaDeviceEnablePeerAccess(
        destination,
        0,
    )
    if result[0] not in (
        cudart.cudaError_t.cudaSuccess,
        cudart.cudaError_t.cudaErrorPeerAccessAlreadyEnabled,
    ):
        cuda_check(result, "cudaDeviceEnablePeerAccess")


def disable_peer(source: int, destination: int) -> None:
    cudart.cudaSetDevice(source)
    result = cudart.cudaDeviceDisablePeerAccess(
        destination
    )
    if result[0] not in (
        cudart.cudaError_t.cudaSuccess,
        cudart.cudaError_t.cudaErrorPeerAccessNotEnabled,
    ):
        cuda_check(result, "cudaDeviceDisablePeerAccess")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10_000)
    parser.add_argument("--transfers", type=int, default=32)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    assert args.rounds > 0
    assert 0 < args.transfers <= 256
    assert torch.cuda.device_count() >= 2
    assert torch.cuda.can_device_access_peer(0, 1)
    assert torch.cuda.can_device_access_peer(1, 0)

    enable_peer(0, 1)
    enable_peer(1, 0)

    batch_bytes = args.transfers * BYTES_PER_TRANSFER
    data_bytes = BUFFER_SLOTS * batch_bytes
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

    descriptor0 = torch.empty(
        DESCRIPTOR_WORDS,
        dtype=torch.uint64,
        device="cuda:0",
    )
    descriptor1 = torch.empty(
        DESCRIPTOR_WORDS,
        dtype=torch.uint64,
        device="cuda:1",
    )
    mismatch_count = torch.zeros(
        1,
        dtype=torch.int32,
        device="cuda:1",
    )

    library = load_runtime()
    endpoints = EndpointPair(
        library,
        descriptor0,
        allocation0,
        descriptor1,
        allocation1,
    )

    try:
        forward_lkey, forward_rkey = endpoints.keys(0)
        reverse_lkey, reverse_rkey = endpoints.keys(1)
        with torch.cuda.device(1):
            compiled_consumer = persistent_consumer.warmup(
                descriptor1,
                data1,
                ready1,
                acknowledgment_value1,
                acknowledgment0,
                mismatch_count,
                reverse_lkey,
                reverse_rkey,
                args.rounds,
                transfer_count=args.transfers,
                bytes_per_transfer=BYTES_PER_TRANSFER,
                batch_bytes=batch_bytes,
                block_size=batch_bytes,
                buffer_slots=BUFFER_SLOTS,
                num_warps=4,
                grid=(1,),
            )
            assert "aboleth." not in compiled_consumer.asm["ttir"]
            persistent_consumer[(1,)](
                descriptor1,
                data1,
                ready1,
                acknowledgment_value1,
                acknowledgment0,
                mismatch_count,
                reverse_lkey,
                reverse_rkey,
                args.rounds,
                transfer_count=args.transfers,
                bytes_per_transfer=BYTES_PER_TRANSFER,
                batch_bytes=batch_bytes,
                block_size=batch_bytes,
                buffer_slots=BUFFER_SLOTS,
                num_warps=4,
            )

        with torch.cuda.device(0):
            compiled_producer = persistent_producer.warmup(
                descriptor0,
                data0,
                data1,
                ready_value0,
                ready1,
                acknowledgment0,
                forward_lkey,
                forward_rkey,
                args.rounds,
                transfer_count=args.transfers,
                bytes_per_transfer=BYTES_PER_TRANSFER,
                batch_bytes=batch_bytes,
                block_size=batch_bytes,
                buffer_slots=BUFFER_SLOTS,
                num_warps=4,
                grid=(1,),
            )
            assert "aboleth." not in compiled_producer.asm["ttir"]
            # Time the whole persistent-producer span; every round performs
            # the transfers, one ordered signal, sender completion, and the
            # reverse acknowledgment, so elapsed/rounds is the round trip —
            # the same definition the compiled and NVSHMEM harnesses report.
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            persistent_producer[(1,)](
                descriptor0,
                data0,
                data1,
                ready_value0,
                ready1,
                acknowledgment0,
                forward_lkey,
                forward_rkey,
                args.rounds,
                transfer_count=args.transfers,
                bytes_per_transfer=BYTES_PER_TRANSFER,
                batch_bytes=batch_bytes,
                block_size=batch_bytes,
                buffer_slots=BUFFER_SLOTS,
                num_warps=4,
            )
            end_event.record()

        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        elapsed_ms = start_event.elapsed_time(end_event)

        mismatches = mismatch_count.cpu().item()
        ready_value = ready1.cpu().item()
        acknowledgment_value = acknowledgment0.cpu().item()
        states = endpoints.state()
        expected_forward_producer = (
            args.rounds * (args.transfers + 1)
        )
        expected_reverse_producer = args.rounds
        expected_states = (
            expected_forward_producer,
            args.rounds,
            expected_reverse_producer,
            args.rounds,
        )

        assert mismatches == 0
        assert ready_value == args.rounds
        assert acknowledgment_value == args.rounds
        assert states == expected_states

        print(f"rounds={args.rounds}")
        print(f"transfers_per_round={args.transfers}")
        print(
            "round_trip_us="
            f"{elapsed_ms * 1000.0 / args.rounds:.3f}"
        )
        print(
            "forward_producer_index="
            f"{states[0]}"
        )
        print(
            "reverse_producer_index="
            f"{states[2]}"
        )
        print("consumer_mismatches=0")
        print("aboleth_triton_rdma_pingpong=PASS")
    finally:
        endpoints.close()
        disable_peer(0, 1)
        disable_peer(1, 0)


def main() -> None:
    with compilation("rdma"):
        run()


if __name__ == "__main__":
    main()
