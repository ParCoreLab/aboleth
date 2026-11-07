import argparse

import torch
import triton
import triton.language as tl

import aboleth.language as aboleth
from aboleth.runtime import Runtime, connection_info_from_bytes


TRANSFER_BYTES = 4096


@triton.jit
def producer_kernel(
    endpoint,
    destination_address,
    source,
    remote_signal_address,
    local_signal,
    data_source_lkey,
    data_destination_rkey,
    signal_source_lkey,
    signal_destination_rkey,
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
        destination,
        source,
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


@triton.jit
def consumer_kernel(
    reverse_endpoint,
    ready_signal,
    local_ack_value,
    remote_ack_address,
    source_lkey,
    destination_rkey,
):
    remote_ack = remote_ack_address.to(
        tl.pointer_type(tl.int64)
    )
    aboleth.wait_signal(ready_signal, 1)
    tl.store(local_ack_value, 1)
    acknowledgment = aboleth.signal(
        reverse_endpoint,
        remote_ack,
        local_ack_value,
        source_key=source_lkey,
        destination_key=destination_rkey,
    )
    aboleth.wait(acknowledgment)


def round_trip_connection_info(endpoint):
    return connection_info_from_bytes(
        endpoint.connection_info.to_bytes()
    )


def run(
    transport: str,
    left_device: int,
    right_device: int,
) -> None:
    control_left = torch.zeros(
        16,
        dtype=torch.uint8,
        device=f"cuda:{left_device}",
    )
    control_right = torch.zeros(
        16,
        dtype=torch.uint8,
        device=f"cuda:{right_device}",
    )
    remote_ack = control_left[:8].view(torch.int64)
    local_ready = control_left[8:16].view(torch.int64)
    remote_ready = control_right[:8].view(torch.int64)
    local_ack = control_right[8:16].view(torch.int64)

    source = torch.zeros(
        TRANSFER_BYTES,
        dtype=torch.uint8,
        device=f"cuda:{left_device}",
    )
    destination = torch.zeros(
        TRANSFER_BYTES,
        dtype=torch.uint8,
        device=f"cuda:{right_device}",
    )
    pattern = (
        (
            torch.arange(
                TRANSFER_BYTES,
                device=f"cuda:{left_device}",
            )
            * 17
            + 3
        )
        & 0xFF
    ).to(torch.uint8)
    source.copy_(pattern)
    local_ready.fill_(1)

    with (
        Runtime(device=left_device) as left_runtime,
        Runtime(device=right_device) as right_runtime,
    ):
        try:
            left_runtime.create_endpoint_pair(
                left_device,
                right_device,
                transport=transport,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "rank-local runtime accepted pair creation"
            )
        left_control_buffer = left_runtime.register(
            control_left
        )
        left_data_buffer = left_runtime.register(source)
        right_control_buffer = right_runtime.register(
            control_right
        )
        right_data_buffer = right_runtime.register(
            destination
        )

        with (
            left_runtime.create_local_endpoint(
                transport=transport
            ) as forward,
            right_runtime.create_local_endpoint(
                transport=transport
            ) as reverse,
        ):
            forward_info = round_trip_connection_info(
                forward
            )
            reverse_info = round_trip_connection_info(
                reverse
            )
            forward.connect(
                reverse_info,
                peer=right_device,
            )
            reverse.connect(
                forward_info,
                peer=left_device,
            )

            destination_remote = forward.import_remote(
                right_runtime.export_buffer(
                    right_data_buffer,
                    peer=right_device,
                )
            )
            ready_remote = forward.import_remote(
                right_runtime.export_buffer(
                    right_control_buffer,
                    peer=right_device,
                )
            )
            acknowledgment_remote = reverse.import_remote(
                left_runtime.export_buffer(
                    left_control_buffer,
                    peer=left_device,
                )
            )

            with forward.compilation():
                with torch.cuda.device(right_device):
                    consumer_kernel[(1,)](
                        reverse.device_handle,
                        remote_ready,
                        local_ack,
                        acknowledgment_remote.address,
                        reverse.local_buffer(
                            right_control_buffer
                        ).lkey,
                        acknowledgment_remote.rkey,
                        num_warps=1,
                    )

                with torch.cuda.device(left_device):
                    producer_kernel[(1,)](
                        forward.device_handle,
                        destination_remote.address,
                        source,
                        ready_remote.address,
                        local_ready,
                        forward.local_buffer(
                            left_data_buffer
                        ).lkey,
                        destination_remote.rkey,
                        forward.local_buffer(
                            left_control_buffer
                        ).lkey,
                        ready_remote.rkey,
                        byte_count=TRANSFER_BYTES,
                        num_warps=1,
                    )

                torch.cuda.synchronize(left_device)
                torch.cuda.synchronize(right_device)

            torch.testing.assert_close(
                destination.cpu(),
                pattern.cpu(),
                rtol=0,
                atol=0,
            )
            assert remote_ready.cpu().item() == 1
            assert remote_ack.cpu().item() == 1
            forward_state = forward.state()
            reverse_state = reverse.state()
            assert forward_state[2] == 0
            assert reverse_state[2] == 0
            forward_inline = forward.max_inline_bytes
            # A queue pair that grants inline at all covers the 8-byte
            # signal; zero means the adapter refused the request.
            assert forward_inline == 0 or forward_inline >= 8
            assert reverse.max_inline_bytes == forward_inline

            print(
                f"transport={transport} "
                f"left=gpu{left_device}/{left_runtime.hca} "
                f"right=gpu{right_device}/{right_runtime.hca} "
                f"left_lid={left_runtime.lid:#x} "
                f"right_lid={right_runtime.lid:#x} "
                f"forward_state={forward_state} "
                f"reverse_state={reverse_state} "
                f"capacity={forward.capacity} "
                f"max_inline_bytes={forward_inline}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=("proxy", "rdma", "efa"),
        required=True,
    )
    parser.add_argument("--left-device", type=int, default=0)
    parser.add_argument("--right-device", type=int, default=4)
    arguments = parser.parse_args()
    run(
        arguments.transport,
        arguments.left_device,
        arguments.right_device,
    )
    print("aboleth_local_endpoint=PASS")


if __name__ == "__main__":
    main()
