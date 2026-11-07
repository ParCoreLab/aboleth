import argparse

import torch
import triton
import triton.language as tl

import aboleth.language as aboleth
from aboleth.runtime import Runtime


BATCHES = 8
BYTES_PER_BATCH = 256
TOTAL_BYTES = BATCHES * BYTES_PER_BATCH
BLOCK_SIZE = 2048


@triton.jit
def eager_producer(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    data_source_lkey,
    data_destination_rkey,
    signal_source_lkey,
    signal_destination_rkey,
    batch_index,
    byte_count: tl.constexpr,
):
    offset = batch_index * byte_count
    transfer = aboleth.put(
        endpoint,
        destination + offset,
        source + offset,
        byte_count,
        source_key=data_source_lkey,
        destination_key=data_destination_rkey,
    )
    ready = aboleth.signal(
        endpoint,
        remote_signal + batch_index,
        local_signal + batch_index,
        transfer,
        source_key=signal_source_lkey,
        destination_key=signal_destination_rkey,
    )
    aboleth.wait(ready)


@triton.jit
def deferred_producer(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    data_source_lkey,
    data_destination_rkey,
    signal_source_lkey,
    signal_destination_rkey,
    batch_index,
    byte_count: tl.constexpr,
):
    offset = batch_index * byte_count
    group = aboleth.begin_group(endpoint)
    transfer = aboleth.put(
        endpoint,
        destination + offset,
        source + offset,
        byte_count,
        source_key=data_source_lkey,
        destination_key=data_destination_rkey,
    )
    aboleth.signal(
        endpoint,
        remote_signal + batch_index,
        local_signal + batch_index,
        transfer,
        source_key=signal_source_lkey,
        destination_key=signal_destination_rkey,
        group=group,
    )


@triton.jit
def unrelated_kernel(scratch):
    value = tl.load(scratch)
    tl.store(scratch, value + 1)


@triton.jit
def overwrite_kernel(
    source,
    value,
    byte_count: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    tl.store(source + offsets, value, mask=offsets < byte_count)


@triton.jit
def drain_and_overwrite_kernel(
    endpoint,
    source,
    value,
    byte_count: tl.constexpr,
    block_size: tl.constexpr,
):
    aboleth.drain(endpoint)
    offsets = tl.arange(0, block_size)
    tl.store(source + offsets, value, mask=offsets < byte_count)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=("eager", "deferred", "both"),
        default="both",
    )
    parser.add_argument("--debug-sync", action="store_true")
    return parser.parse_args()


def pattern(round_index: int):
    return (
        (
            torch.arange(TOTAL_BYTES, device="cuda:0")
            + round_index * 29
            + 7
        )
        & 0xFF
    ).to(torch.uint8)


def run_mode(
    *,
    deferred: bool,
    warmup_rounds: int,
    rounds: int,
    debug_sync: bool,
):
    torch.cuda.set_device(0)
    source = torch.empty(
        TOTAL_BYTES,
        dtype=torch.uint8,
        device="cuda:0",
    )
    destination = torch.zeros(
        TOTAL_BYTES,
        dtype=torch.uint8,
        device="cuda:1",
    )
    local_signals = torch.zeros(
        BATCHES,
        dtype=torch.int64,
        device="cuda:0",
    )
    remote_signals = torch.zeros(
        BATCHES,
        dtype=torch.int64,
        device="cuda:1",
    )
    scratch = torch.zeros(1, dtype=torch.int64, device="cuda:0")

    with Runtime() as runtime:
        source_registration = runtime.register(source)
        destination_registration = runtime.register(destination)
        local_signal_registration = runtime.register(local_signals)
        remote_signal_registration = runtime.register(remote_signals)
        with runtime.create_endpoint_pair(
            0,
            1,
            transport="rdma",
        ) as pair:
            if debug_sync:
                print("endpoint ready", flush=True)
            forward = pair.endpoints[0]
            source_local = forward.local_buffer(source_registration)
            destination_remote = forward.remote_buffer(
                destination_registration
            )
            signal_local = forward.local_buffer(
                local_signal_registration
            )
            signal_remote = forward.remote_buffer(
                remote_signal_registration
            )

            producer = (
                deferred_producer if deferred else eager_producer
            )
            total_rounds = warmup_rounds + rounds
            torch.cuda.set_device(0)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            with forward.compilation():
                for round_index in range(total_rounds):
                    current_pattern = pattern(round_index)
                    source.copy_(current_pattern)
                    # Each signal cell carries its own epoch, and a strong
                    # publication advances a cell by exactly one: every batch
                    # publishes round_index + 1 into its own cell.
                    signal_values = torch.full(
                        (BATCHES,),
                        round_index + 1,
                        device="cuda:0",
                        dtype=torch.int64,
                    )
                    local_signals.copy_(signal_values)

                    if round_index == warmup_rounds:
                        start.record()
                    for batch in range(BATCHES):
                        producer[(1,)](
                            forward.device_handle,
                            destination,
                            source,
                            remote_signals,
                            local_signals,
                            source_local.lkey,
                            destination_remote.rkey,
                            signal_local.lkey,
                            signal_remote.rkey,
                            batch,
                            byte_count=BYTES_PER_BATCH,
                            num_warps=1,
                        )
                        if debug_sync:
                            torch.cuda.synchronize(0)
                            print(
                                f"producer {batch} complete "
                                f"state={pair.completion_state(0)}",
                                flush=True,
                            )
                    unrelated_kernel[(1,)](scratch, num_warps=1)
                    if debug_sync:
                        torch.cuda.synchronize(0)
                        print("unrelated complete", flush=True)
                    if deferred:
                        drain_and_overwrite_kernel[(1,)](
                            forward.device_handle,
                            source,
                            0xA5,
                            byte_count=TOTAL_BYTES,
                            block_size=BLOCK_SIZE,
                            num_warps=1,
                        )
                    else:
                        overwrite_kernel[(1,)](
                            source,
                            0xA5,
                            byte_count=TOTAL_BYTES,
                            block_size=BLOCK_SIZE,
                            num_warps=1,
                        )
                    if debug_sync:
                        torch.cuda.synchronize(0)
                        print("reuse complete", flush=True)
                end.record()
                end.synchronize()

            torch.cuda.synchronize(1)
            expected = pattern(total_rounds - 1)
            torch.testing.assert_close(
                destination.cpu(),
                expected.cpu(),
                rtol=0,
                atol=0,
            )
            expected_signals = torch.full(
                (BATCHES,),
                total_rounds,
                dtype=torch.int64,
            )
            torch.testing.assert_close(
                remote_signals.cpu(),
                expected_signals,
                rtol=0,
                atol=0,
            )
            assert torch.all(source == 0xA5)
            completion = pair.completion_state(0)
            assert completion[1] == completion[2]
            assert completion[3] == 0
            expected_cqes = (
                total_rounds if deferred else total_rounds * BATCHES
            )
            assert completion[0] == expected_cqes
            return {
                "mode": "deferred" if deferred else "eager",
                "device_ms": start.elapsed_time(end),
                "cqes": completion[0],
                "completed_head": completion[1],
                "ready_head": completion[2],
            }


def main():
    args = parse_args()
    assert args.warmup_rounds >= 0
    assert args.rounds > 0
    assert torch.cuda.device_count() >= 2

    eager = None
    deferred = None
    if args.mode in ("eager", "both"):
        print("running eager", flush=True)
        eager = run_mode(
            deferred=False,
            warmup_rounds=args.warmup_rounds,
            rounds=args.rounds,
            debug_sync=args.debug_sync,
        )
        print(eager, flush=True)
    if args.mode in ("deferred", "both"):
        print("running deferred", flush=True)
        deferred = run_mode(
            deferred=True,
            warmup_rounds=args.warmup_rounds,
            rounds=args.rounds,
            debug_sync=args.debug_sync,
        )
        print(deferred, flush=True)
    if eager is not None and deferred is not None:
        print(
            "cqe_reduction="
            f"{eager['cqes'] / deferred['cqes']:.2f}x"
        )
    print("aboleth_deferred_lifecycle=PASS")


if __name__ == "__main__":
    main()
