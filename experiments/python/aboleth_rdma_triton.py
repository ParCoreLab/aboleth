import struct

import torch
import triton
import triton.language as tl
from cuda.bindings import runtime as cudart

import aboleth.language as aboleth
from aboleth.compiler import compilation
from aboleth_rdma_abi_test import (
    DESCRIPTOR_WORDS,
    create_endpoints,
    load_runtime,
)


TRANSFER_COUNT = 32
BYTES_PER_TRANSFER = 256
TOTAL_BYTES = TRANSFER_COUNT * BYTES_PER_TRANSFER
SIGNAL_BYTES = 8
ALLOCATION_BYTES = TOTAL_BYTES + SIGNAL_BYTES


@triton.jit
def producer_kernel(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    source_lkey,
    destination_rkey,
    transfer_count: tl.constexpr,
    bytes_per_transfer: tl.constexpr,
):
    transfer = None
    for index in tl.static_range(0, transfer_count):
        offset = index * bytes_per_transfer
        transfer = aboleth.put(
            endpoint,
            destination + offset,
            source + offset,
            bytes_per_transfer,
            source_key=source_lkey,
            destination_key=destination_rkey,
            after=transfer,
        )
    ready = aboleth.signal(
        endpoint,
        remote_signal,
        local_signal,
        transfer,
        source_key=source_lkey,
        destination_key=destination_rkey,
    )
    aboleth.wait(ready)


@triton.jit
def consumer_kernel(
    ready_signal,
    expected_signal,
    received,
    mismatch_count,
    element_count: tl.constexpr,
    block_size: tl.constexpr,
):
    aboleth.wait_signal(ready_signal, expected_signal)
    offsets = tl.arange(0, block_size)
    mask = offsets < element_count
    values = tl.load(received + offsets, mask=mask, other=0)
    expected = ((offsets * 17 + 3) & 0xFF).to(tl.uint8)
    mismatches = tl.sum((values != expected) & mask)
    tl.store(mismatch_count, mismatches)


def cuda_check(result, operation: str) -> None:
    error = result[0]
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {error}")


def validate_wqes(
    endpoints,
    descriptor,
    source,
    destination,
    local_signal,
    remote_signal,
    source_lkey,
    destination_rkey,
) -> None:
    wqe_count = TRANSFER_COUNT + 1
    snapshot_size = wqe_count * descriptor.send_stride
    raw, dbrec_head = endpoints.send_queue_snapshot(snapshot_size)

    for index in range(wqe_count):
        offset = index * descriptor.send_stride
        opcode_and_index = struct.unpack_from(">I", raw, offset)[0]
        qpn_and_segments = struct.unpack_from(">I", raw, offset + 4)[0]
        completion_flag = raw[offset + 11]
        remote_address = struct.unpack_from(">Q", raw, offset + 16)[0]
        remote_key = struct.unpack_from(">I", raw, offset + 24)[0]

        if index < TRANSFER_COUNT:
            byte_count = struct.unpack_from(">I", raw, offset + 32)[0]
            local_key = struct.unpack_from(">I", raw, offset + 36)[0]
            local_address = struct.unpack_from(">Q", raw, offset + 40)[0]
            assert opcode_and_index == (index << 8) | 0x08
            assert qpn_and_segments == (descriptor.qpn << 8) | 3
            assert remote_key == destination_rkey
            assert local_key == source_lkey
            assert completion_flag == 0
            assert remote_address == (
                destination.data_ptr() + index * BYTES_PER_TRANSFER
            )
            assert local_address == (
                source.data_ptr() + index * BYTES_PER_TRANSFER
            )
            assert byte_count == BYTES_PER_TRANSFER
        else:
            # A payload-attached signal publishes with a responder
            # compare-and-swap so relaxed-ordering payload writes cannot
            # pass it: swap = epoch, compare = epoch - 1, result retired
            # into the endpoint-owned atomic sink.
            swap_value = struct.unpack_from(">Q", raw, offset + 32)[0]
            compare_value = struct.unpack_from(">Q", raw, offset + 40)[0]
            atomic_bytes = struct.unpack_from(">I", raw, offset + 48)[0]
            assert opcode_and_index == (index << 8) | 0x11
            assert qpn_and_segments == (descriptor.qpn << 8) | 4
            assert completion_flag == 8
            assert remote_address == remote_signal.data_ptr()
            assert swap_value == 1
            assert compare_value == 0
            assert atomic_bytes == SIGNAL_BYTES

    assert dbrec_head == wqe_count


def run() -> None:
    assert torch.cuda.device_count() >= 2
    assert torch.cuda.can_device_access_peer(0, 1)

    cuda_check(cudart.cudaSetDevice(0), "cudaSetDevice(0)")
    enable_result = cudart.cudaDeviceEnablePeerAccess(1, 0)
    if enable_result[0] not in (
        cudart.cudaError_t.cudaSuccess,
        cudart.cudaError_t.cudaErrorPeerAccessAlreadyEnabled,
    ):
        cuda_check(enable_result, "cudaDeviceEnablePeerAccess")

    descriptor = torch.empty(
        DESCRIPTOR_WORDS,
        dtype=torch.uint64,
        device="cuda:0",
    )
    source_allocation = torch.zeros(
        ALLOCATION_BYTES,
        dtype=torch.uint8,
        device="cuda:0",
    )
    destination_allocation = torch.zeros(
        ALLOCATION_BYTES,
        dtype=torch.uint8,
        device="cuda:1",
    )
    source = source_allocation[:TOTAL_BYTES]
    destination = destination_allocation[:TOTAL_BYTES]
    local_signal = source_allocation[TOTAL_BYTES:].view(torch.int64)
    remote_signal = destination_allocation[TOTAL_BYTES:].view(torch.int64)
    local_signal.fill_(1)

    pattern = (
        (torch.arange(TOTAL_BYTES, device="cuda:0") * 17 + 3) & 0xFF
    ).to(torch.uint8)
    source.copy_(pattern)
    mismatch_count = torch.full(
        (1,),
        -1,
        dtype=torch.int32,
        device="cuda:1",
    )

    library = load_runtime()
    endpoints = create_endpoints(
        library,
        descriptor,
        source_allocation,
        destination_allocation,
    )

    try:
        expected_descriptor = endpoints.descriptor
        source_lkey, destination_rkey = endpoints.keys()
        with torch.cuda.device(1):
            compiled_consumer = consumer_kernel.warmup(
                remote_signal,
                1,
                destination,
                mismatch_count,
                element_count=TOTAL_BYTES,
                block_size=TOTAL_BYTES,
                num_warps=4,
                grid=(1,),
            )
            consumer_ttir = compiled_consumer.asm["ttir"]
            consumer_ptx = compiled_consumer.asm["ptx"]
            assert "aboleth." not in consumer_ttir
            assert (
                "ld.acquire.sys.global.L1::no_allocate.u64"
                in consumer_ttir
            )
            assert (
                "ld.acquire.sys.global.L1::no_allocate.u64"
                in consumer_ptx
            )
        with torch.cuda.device(0):
            compiled_producer = producer_kernel.warmup(
                descriptor,
                destination,
                source,
                remote_signal,
                local_signal,
                source_lkey,
                destination_rkey,
                transfer_count=TRANSFER_COUNT,
                bytes_per_transfer=BYTES_PER_TRANSFER,
                num_warps=1,
                grid=(1,),
            )
            producer_ttir = compiled_producer.asm["ttir"]
            producer_ptx = compiled_producer.asm["ptx"]
            assert "aboleth." not in producer_ttir
            assert (
                producer_ttir.count("tt.atomic_rmw add, relaxed, gpu")
                == 1
            )
            # The scalar emitter keeps descriptor values' lifetimes short.
            # Queue words and every payload byte are checked below.
            assert "tt.load" in producer_ttir
            assert "tt.store" in producer_ttir
            assert (
                producer_ptx.count("atom.global.gpu.relaxed.add.u64")
                == 1
            )
            assert "st.relaxed.sys.global.L1::no_allocate.b32" in producer_ptx
            assert "st.release.sys.global.L1::no_allocate.b64" in producer_ptx
            assert "ld.volatile.global.b8" in producer_ptx
            assert "ld.volatile.global.b16" in producer_ptx
            # Validate code generation before starting the waiting receiver;
            # a failed assembly assertion must not hang endpoint teardown.
            with torch.cuda.device(1):
                consumer_kernel[(1,)](
                    remote_signal,
                    1,
                    destination,
                    mismatch_count,
                    element_count=TOTAL_BYTES,
                    block_size=TOTAL_BYTES,
                    num_warps=4,
                )
            producer_kernel[(1,)](
                descriptor,
                destination,
                source,
                remote_signal,
                local_signal,
                source_lkey,
                destination_rkey,
                transfer_count=TRANSFER_COUNT,
                bytes_per_transfer=BYTES_PER_TRANSFER,
                num_warps=1,
            )

        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        mismatches = mismatch_count.cpu().item()
        torch.testing.assert_close(
            destination.cpu(),
            pattern.cpu(),
            rtol=0,
            atol=0,
        )
        assert remote_signal.cpu().item() == 1
        assert mismatches == 0
        validate_wqes(
            endpoints,
            expected_descriptor,
            source,
            destination,
            local_signal,
            remote_signal,
            source_lkey,
            destination_rkey,
        )

        print(f"producer_wqes={TRANSFER_COUNT + 1}")
        print("producer_batch_reservations=1")
        print("producer_doorbells=1")
        print("consumer_mismatches=0")
        print("aboleth_triton_rdma=PASS")
    finally:
        endpoints.close()
        cudart.cudaSetDevice(0)
        disable_result = cudart.cudaDeviceDisablePeerAccess(1)
        if disable_result[0] not in (
            cudart.cudaError_t.cudaSuccess,
            cudart.cudaError_t.cudaErrorPeerAccessNotEnabled,
        ):
            cuda_check(disable_result, "cudaDeviceDisablePeerAccess")


def main() -> None:
    with compilation("rdma"):
        run()


if __name__ == "__main__":
    main()
