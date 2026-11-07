import torch
import triton
import triton.language as tl
from cuda.bindings import runtime as cudart

import aboleth.language as aboleth
from aboleth.compiler import compilation


TRANSFER_COUNT = 32
BYTES_PER_TRANSFER = 256
TOTAL_BYTES = TRANSFER_COUNT * BYTES_PER_TRANSFER


@triton.jit
def producer_kernel(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
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
            after=transfer,
        )
    ready = aboleth.signal(
        endpoint,
        remote_signal,
        local_signal,
        transfer,
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

    pattern = ((torch.arange(TOTAL_BYTES, device="cuda:0") * 17 + 3) & 0xFF).to(torch.uint8)
    endpoint = torch.empty(1, dtype=torch.uint8, device="cuda:0")
    local_signal = torch.ones(1, dtype=torch.int64, device="cuda:0")
    destination = torch.zeros(TOTAL_BYTES, dtype=torch.uint8, device="cuda:1")
    remote_signal = torch.zeros(1, dtype=torch.int64, device="cuda:1")
    mismatch_count = torch.full((1,), -1, dtype=torch.int32, device="cuda:1")

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
        assert "ld.acquire.sys.global" in consumer_ttir
        assert "ld.acquire.sys.global" in consumer_ptx
    with torch.cuda.device(0):
        compiled_producer = producer_kernel.warmup(
            endpoint,
            destination,
            pattern,
            remote_signal,
            local_signal,
            transfer_count=TRANSFER_COUNT,
            bytes_per_transfer=BYTES_PER_TRANSFER,
            num_warps=4,
            grid=(1,),
        )
        producer_ttir = compiled_producer.asm["ttir"]
        producer_ptx = compiled_producer.asm["ptx"]
        assert "aboleth." not in producer_ttir
        assert (
            producer_ttir.count("tt.make_range")
            == TRANSFER_COUNT
        )
        assert "bar.sync 0" in producer_ttir
        assert "tt.int_to_ptr" in producer_ttir
        assert "-> !tt.ptr<i64>" in producer_ttir
        assert "tt.atomic_rmw exch, release, sys" in producer_ttir
        assert "membar.sys" not in producer_ptx
        publication = producer_ptx.index("atom.global.sys.release.exch.b64")
        last_store = producer_ptx.rfind("st.global", 0, publication)
        copy_barrier = producer_ptx.rfind("bar.sync 0", 0, publication)
        assert 0 <= last_store < copy_barrier < publication
        # Compile and check both kernels before starting a receiver that
        # waits for the producer. An assembly assertion must not strand it.
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
            endpoint,
            destination,
            pattern,
            remote_signal,
            local_signal,
            transfer_count=TRANSFER_COUNT,
            bytes_per_transfer=BYTES_PER_TRANSFER,
            num_warps=4,
        )

    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    mismatches = mismatch_count.cpu().item()
    torch.testing.assert_close(destination.cpu(), pattern.cpu(), rtol=0, atol=0)
    assert remote_signal.cpu().item() == 1
    assert mismatches == 0

    destination[1:256].zero_()
    destination[256] = 0xA5
    remote_signal.zero_()
    torch.cuda.synchronize(1)
    with torch.cuda.device(0):
        producer_kernel[(1,)](
            endpoint,
            destination[1:256],
            pattern[1:256],
            remote_signal,
            local_signal,
            transfer_count=1,
            bytes_per_transfer=255,
            num_warps=4,
        )
    torch.cuda.synchronize(0)
    torch.testing.assert_close(
        destination[1:256].cpu(),
        pattern[1:256].cpu(),
        rtol=0,
        atol=0,
    )
    assert remote_signal.cpu().item() == 1
    assert destination[256].cpu().item() == 0xA5

    print(f"producer_transfers={TRANSFER_COUNT}")
    print(
        f"producer_copy_loop_variants="
        f"{producer_ttir.count('scf.for')}"
    )
    print(
        f"producer_cooperative_ranges="
        f"{producer_ttir.count('tt.make_range')}"
    )
    print("unaligned_fallback=PASS")
    print("aboleth_ops_after_lowering=0")
    print("consumer_mismatches=0")
    print("aboleth_triton_p2p=PASS")


def main() -> None:
    with compilation("p2p"):
        run()


if __name__ == "__main__":
    main()
