import torch
import triton

import aboleth.language as aboleth
from aboleth.runtime import Runtime


@triton.jit
def invalid_proxy_put(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    source_lkey,
    invalid_destination_rkey,
    signal_lkey,
    signal_rkey,
):
    transfer = aboleth.put(
        endpoint,
        destination,
        source,
        256,
        source_key=source_lkey,
        destination_key=invalid_destination_rkey,
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
    source = torch.ones(
        256,
        dtype=torch.uint8,
        device="cuda:0",
    )
    destination = torch.zeros(
        256,
        dtype=torch.uint8,
        device="cuda:1",
    )
    local_signal = torch.ones(
        1,
        dtype=torch.int64,
        device="cuda:0",
    )
    remote_signal = torch.zeros(
        1,
        dtype=torch.int64,
        device="cuda:1",
    )

    with Runtime() as runtime:
        source_buffer = runtime.register(source)
        destination_buffer = runtime.register(destination)
        local_signal_buffer = runtime.register(local_signal)
        remote_signal_buffer = runtime.register(remote_signal)
        with runtime.create_endpoint_pair(
            0,
            1,
            transport="proxy",
        ) as pair:
            endpoint = pair.endpoints[0]
            with endpoint.compilation():
                invalid_proxy_put[(1,)](
                    endpoint.device_handle,
                    destination,
                    source,
                    remote_signal,
                    local_signal,
                    endpoint.local_buffer(
                        source_buffer
                    ).lkey,
                    0,
                    endpoint.local_buffer(
                        local_signal_buffer
                    ).lkey,
                    endpoint.remote_buffer(
                        remote_signal_buffer
                    ).rkey,
                    num_warps=1,
                )
                torch.cuda.synchronize(0)

            status = pair.status()
            assert status[0] == 2
            assert pair.queue_state() == (1, 1, 0, 0)

    print("aboleth_proxy_error=PASS")


if __name__ == "__main__":
    main()
