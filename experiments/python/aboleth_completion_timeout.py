import torch
import triton

import aboleth.language as aboleth
from aboleth.runtime import Runtime


@triton.jit
def timeout_kernel(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    data_lkey,
    data_rkey,
    signal_lkey,
    signal_rkey,
):
    transfer = aboleth.put(
        endpoint,
        destination,
        source,
        256,
        source_key=data_lkey,
        destination_key=data_rkey,
    )
    ready = aboleth.signal(
        endpoint,
        remote_signal,
        local_signal,
        transfer,
        source_key=signal_lkey,
        destination_key=signal_rkey,
    )
    aboleth.wait(ready)


def main() -> None:
    source = torch.ones(256, dtype=torch.uint8, device="cuda:0")
    destination = torch.zeros(256, dtype=torch.uint8, device="cuda:1")
    local_signal = torch.ones(1, dtype=torch.int64, device="cuda:0")
    remote_signal = torch.zeros(1, dtype=torch.int64, device="cuda:1")

    with Runtime() as runtime:
        source_buffer = runtime.register(source)
        destination_buffer = runtime.register(destination)
        local_signal_buffer = runtime.register(local_signal)
        remote_signal_buffer = runtime.register(remote_signal)
        with runtime.create_endpoint_pair(
            0,
            1,
            transport="rdma",
        ) as pair:
            endpoint = pair.endpoints[0]
            fake_doorbell = torch.zeros(
                1,
                dtype=torch.int64,
                device="cuda:0",
            )
            fake_completion_queue = torch.full(
                (128,),
                0xFF,
                dtype=torch.uint8,
                device="cuda:0",
            )
            endpoint.device_handle[2] = fake_doorbell.data_ptr()
            endpoint.device_handle[3] = fake_completion_queue.data_ptr()
            endpoint.device_handle[7] = 1
            torch.cuda.synchronize(0)
            assert fake_completion_queue[63].cpu().item() == 0xFF
            assert (
                endpoint.device_handle[6].cpu().item()
                == pair.completion_state_addresses()[0]
            )
            with endpoint.compilation():
                timeout_kernel[(1,)](
                    endpoint.device_handle,
                    destination,
                    source,
                    remote_signal,
                    local_signal,
                    endpoint.local_buffer(source_buffer).lkey,
                    endpoint.remote_buffer(destination_buffer).rkey,
                    endpoint.local_buffer(local_signal_buffer).lkey,
                    endpoint.remote_buffer(remote_signal_buffer).rkey,
                    num_warps=1,
                )
                torch.cuda.synchronize(0)
            status = pair.status()
            print(
                f"completion_state={pair.completion_state(0)}",
                flush=True,
            )
            print(f"timeout_status={status}", flush=True)
            assert status[0] == 1

    print("aboleth_completion_timeout=PASS")


if __name__ == "__main__":
    main()
