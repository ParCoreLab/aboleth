import argparse

import torch
import triton
import triton.language as tl

import aboleth.language as aboleth
from aboleth.runtime import Runtime


@triton.jit
def producer_kernel(
    endpoint,
    destination,
    source,
    remote_signals,
    local_signals,
    data_source_key,
    data_destination_key,
    signal_source_key,
    signal_destination_key,
    transfers: tl.constexpr,
    bytes_per_transfer: tl.constexpr,
    batch_bytes: tl.constexpr,
):
    producer = tl.program_id(0)
    base = producer * batch_bytes
    transfer = None
    for index in tl.static_range(0, transfers):
        offset = base + index * bytes_per_transfer
        transfer = aboleth.put(
            endpoint,
            destination + offset,
            source + offset,
            bytes_per_transfer,
            source_key=data_source_key,
            destination_key=data_destination_key,
            after=transfer,
        )
    tl.store(local_signals + producer, 1)
    ready = aboleth.signal(
        endpoint,
        remote_signals + producer,
        local_signals + producer,
        transfer,
        source_key=signal_source_key,
        destination_key=signal_destination_key,
    )
    aboleth.wait(ready)


@triton.jit
def consumer_kernel(signals):
    producer = tl.program_id(0)
    aboleth.wait_signal(signals + producer, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--producers", type=int, default=32)
    parser.add_argument("--transfers", type=int, default=4)
    parser.add_argument("--bytes", type=int, default=256)
    parser.add_argument(
        "--transport",
        choices=("proxy", "rdma"),
        default="rdma",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert torch.cuda.device_count() >= 2
    assert args.producers > 1
    assert args.transfers > 0
    assert args.bytes > 0

    batch_bytes = args.transfers * args.bytes
    total_bytes = args.producers * batch_bytes
    source = (
        (torch.arange(total_bytes, device="cuda:0") * 17 + 3)
        & 0xFF
    ).to(torch.uint8)
    destination = torch.zeros(
        total_bytes,
        device="cuda:1",
        dtype=torch.uint8,
    )
    local_signals = torch.zeros(
        args.producers,
        device="cuda:0",
        dtype=torch.int64,
    )
    remote_signals = torch.zeros(
        args.producers,
        device="cuda:1",
        dtype=torch.int64,
    )

    with Runtime() as runtime:
        source_buffer = runtime.register(source)
        destination_buffer = runtime.register(destination)
        local_signal_buffer = runtime.register(local_signals)
        remote_signal_buffer = runtime.register(remote_signals)

        with runtime.create_endpoint_pair(
            0,
            1,
            transport=args.transport,
        ) as pair:
            forward = pair.endpoints[0]
            source_key = forward.local_buffer(source_buffer).lkey
            destination_key = forward.remote_buffer(
                destination_buffer
            ).rkey
            local_signal_key = forward.local_buffer(
                local_signal_buffer
            ).lkey
            remote_signal_key = forward.remote_buffer(
                remote_signal_buffer
            ).rkey

            with forward.compilation():
                with torch.cuda.device(1):
                    compiled_consumer = consumer_kernel.warmup(
                        remote_signals,
                        num_warps=1,
                        grid=(args.producers,),
                    )
                    compiled_consumer._init_handles()
                with torch.cuda.device(0):
                    compiled_producer = producer_kernel.warmup(
                        forward.device_handle,
                        destination,
                        source,
                        remote_signals,
                        local_signals,
                        source_key,
                        destination_key,
                        local_signal_key,
                        remote_signal_key,
                        transfers=args.transfers,
                        bytes_per_transfer=args.bytes,
                        batch_bytes=batch_bytes,
                        num_warps=1,
                        grid=(args.producers,),
                    )
                    compiled_producer._init_handles()

                with torch.cuda.device(1):
                    consumer_kernel[(args.producers,)](
                        remote_signals,
                        num_warps=1,
                    )
                with torch.cuda.device(0):
                    # Time the producer span: submission of every transfer
                    # and signal through the transport plus completion and
                    # the consumer handshake — the same span on every
                    # transport, so arms compare fairly.
                    start_event = torch.cuda.Event(
                        enable_timing=True
                    )
                    end_event = torch.cuda.Event(
                        enable_timing=True
                    )
                    start_event.record()
                    producer_kernel[(args.producers,)](
                        forward.device_handle,
                        destination,
                        source,
                        remote_signals,
                        local_signals,
                        source_key,
                        destination_key,
                        local_signal_key,
                        remote_signal_key,
                        transfers=args.transfers,
                        bytes_per_transfer=args.bytes,
                        batch_bytes=batch_bytes,
                        num_warps=1,
                    )
                    end_event.record()

                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
                producer_ms = start_event.elapsed_time(
                    end_event
                )

            torch.testing.assert_close(
                destination.cpu(),
                source.cpu(),
                rtol=0,
                atol=0,
            )
            assert torch.all(remote_signals == 1)
            if args.transport == "proxy":
                expected_state = (
                    args.producers,
                    args.producers,
                    0,
                    0,
                )
            else:
                expected_state = (
                    args.producers * (args.transfers + 1),
                    args.producers,
                    0,
                    0,
                )
            assert pair.queue_state() == expected_state
            assert pair.status() == (0, 0)

    messages = args.producers * (args.transfers + 1)
    print(
        "aboleth_concurrent_producers=PASS "
        f"producers={args.producers} "
        f"transfers={args.transfers} "
        f"transport={args.transport} "
        f"producer_ms={producer_ms:.3f} "
        f"messages={messages} "
        f"msg_per_s={messages / (producer_ms * 1e-3):.0f}"
    )


if __name__ == "__main__":
    main()
