import torch
import triton
import triton.language as tl

import aboleth.language as aboleth
from aboleth.runtime import Runtime


TRANSFER_BYTES = 4096


@triton.jit
def producer_kernel(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    data_source_lkey,
    data_destination_rkey,
    signal_source_lkey,
    signal_destination_rkey,
    byte_count: tl.constexpr,
):
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
    remote_ack_signal,
    source_lkey,
    destination_rkey,
):
    aboleth.wait_signal(ready_signal, 1)
    tl.store(local_ack_value, 1)
    acknowledgment = aboleth.signal(
        reverse_endpoint,
        remote_ack_signal,
        local_ack_value,
        source_key=source_lkey,
        destination_key=destination_rkey,
    )
    aboleth.wait(acknowledgment)


def main() -> None:
    assert torch.cuda.device_count() >= 2

    control0 = torch.zeros(
        16,
        dtype=torch.uint8,
        device="cuda:0",
    )
    control1 = torch.zeros(
        16,
        dtype=torch.uint8,
        device="cuda:1",
    )
    remote_ack = control0[:8].view(torch.int64)
    local_ready = control0[8:16].view(torch.int64)
    remote_ready = control1[:8].view(torch.int64)
    local_ack = control1[8:16].view(torch.int64)

    source = torch.zeros(
        TRANSFER_BYTES,
        dtype=torch.uint8,
        device="cuda:0",
    )
    destination = torch.zeros(
        TRANSFER_BYTES,
        dtype=torch.uint8,
        device="cuda:1",
    )

    pattern = (
        (torch.arange(TRANSFER_BYTES, device="cuda:0") * 17 + 3)
        & 0xFF
    ).to(torch.uint8)
    source.copy_(pattern)
    local_ready.fill_(1)

    with Runtime() as runtime:
        control_buffer0 = runtime.register(control0)
        control_buffer1 = runtime.register(control1)
        assert control_buffer0.lkey > 0
        assert control_buffer1.rkey > 0
        with runtime.create_endpoint_pair(
            0,
            1,
            transport="rdma",
        ) as pair:
            forward, reverse = pair.endpoints

            # Register data after endpoint creation to prove that endpoint
            # lifetime and MR lifetime are independent.
            data_buffer0 = runtime.register(source)
            data_buffer1 = runtime.register(destination)
            forward_data_local = forward.local_buffer(data_buffer0)
            forward_data_remote = forward.remote_buffer(data_buffer1)
            forward_signal_local = forward.local_buffer(control_buffer0)
            forward_signal_remote = forward.remote_buffer(control_buffer1)
            reverse_local = reverse.local_buffer(control_buffer1)
            reverse_remote = reverse.remote_buffer(control_buffer0)

            with forward.compilation():
                with torch.cuda.device(1):
                    consumer_kernel[(1,)](
                        reverse.device_handle,
                        remote_ready,
                        local_ack,
                        remote_ack,
                        reverse_local.lkey,
                        reverse_remote.rkey,
                        num_warps=1,
                    )

                with torch.cuda.device(0):
                    producer_kernel[(1,)](
                        forward.device_handle,
                        destination,
                        source,
                        remote_ready,
                        local_ready,
                        forward_data_local.lkey,
                        forward_data_remote.rkey,
                        forward_signal_local.lkey,
                        forward_signal_remote.rkey,
                        byte_count=TRANSFER_BYTES,
                        num_warps=1,
                    )

                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)

            torch.testing.assert_close(
                destination.cpu(),
                pattern.cpu(),
                rtol=0,
                atol=0,
            )
            assert remote_ready.cpu().item() == 1
            assert remote_ack.cpu().item() == 1
            assert pair.queue_state() == (2, 1, 1, 1)
            assert pair.status() == (0, 0)

            exported = runtime.export_buffer(
                data_buffer1,
                peer=1,
            )
            imported = runtime.import_remote_buffer(
                1,
                type(exported).from_bytes(exported.to_bytes())
            )
            assert imported.address == forward_data_remote.address
            assert imported.rkey == forward_data_remote.rkey

        # Registered MRs outlive endpoints and can be reused by a new QP/CQ
        # pair without changing their registration keys.
        with runtime.create_endpoint_pair(
            0,
            1,
            transport="rdma",
        ) as replacement_pair:
            replacement_forward = replacement_pair.endpoints[0]
            assert (
                replacement_forward.local_buffer(data_buffer0).lkey
                == forward_data_local.lkey
            )
            assert (
                replacement_forward.remote_buffer(data_buffer1).rkey
                == forward_data_remote.rkey
            )

        with runtime.create_endpoint_mesh(
            (0, 1),
            transport="rdma",
        ) as mesh:
            table0 = mesh.tables[0]
            assert table0.peers == (1,)
            assert (
                table0.device_handles[1].cpu().item()
                == mesh.endpoint(0, 1).device_handle.data_ptr()
            )

    print("aboleth_runtime=PASS")


if __name__ == "__main__":
    main()
