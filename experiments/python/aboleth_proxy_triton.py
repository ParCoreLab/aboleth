import torch
import triton
import triton.language as tl

import aboleth.language as aboleth
from aboleth.runtime import Runtime


@triton.jit(do_not_specialize=["rounds"])
def proxy_put_kernel(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    source_lkey,
    destination_rkey,
    signal_lkey,
    signal_rkey,
    elapsed_ns,
    rounds,
):
    start = tl.extra.cuda.globaltimer()
    for round_index in tl.range(0, rounds):
        transfer = None
        for index in tl.static_range(0, 4):
            offset = index * 64
            transfer = aboleth.put(
                endpoint,
                destination + offset,
                source + offset,
                64,
                source_key=source_lkey,
                destination_key=destination_rkey,
                after=transfer,
            )
        tl.store(local_signal, round_index + 1)
        ready = aboleth.signal(
            endpoint,
            remote_signal,
            local_signal,
            transfer,
            source_key=signal_lkey,
            destination_key=signal_rkey,
        )
        aboleth.wait(ready)
    end = tl.extra.cuda.globaltimer()
    tl.store(elapsed_ns, end - start)


@triton.jit(do_not_specialize=["rounds"])
def proxy_signal_kernel(
    endpoint,
    remote_signal,
    local_signal,
    signal_lkey,
    signal_rkey,
    rounds,
):
    for index in tl.range(0, rounds):
        value = index + 2
        tl.store(local_signal, value)
        ready = aboleth.signal(
            endpoint,
            remote_signal,
            local_signal,
            source_key=signal_lkey,
            destination_key=signal_rkey,
        )
        aboleth.wait(ready)


def main() -> None:
    source = (
        (torch.arange(256, device="cuda:0") * 17 + 3) & 0xFF
    ).to(torch.uint8)
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
    elapsed_ns = torch.zeros(
        1,
        dtype=torch.int64,
        device="cuda:0",
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
            arguments = (
                endpoint.device_handle,
                destination,
                source,
                remote_signal,
                local_signal,
                endpoint.local_buffer(source_buffer).lkey,
                endpoint.remote_buffer(destination_buffer).rkey,
                endpoint.local_buffer(local_signal_buffer).lkey,
                endpoint.remote_buffer(remote_signal_buffer).rkey,
                elapsed_ns,
                16,
            )
            with endpoint.compilation():
                compiled = proxy_put_kernel.warmup(
                    *arguments,
                    num_warps=1,
                    grid=(1,),
                )
                assert "aboleth." not in compiled.asm["ttir"]
                proxy_put_kernel[(1,)](
                    *arguments,
                    num_warps=1,
                )
                torch.cuda.synchronize(0)

            torch.testing.assert_close(
                destination.cpu(),
                source.cpu(),
                rtol=0,
                atol=0,
            )
            assert remote_signal.cpu().item() == 16

            reverse = pair.endpoints[1]
            reverse_arguments = (
                reverse.device_handle,
                local_signal,
                remote_signal,
                reverse.local_buffer(
                    remote_signal_buffer
                ).lkey,
                reverse.remote_buffer(
                    local_signal_buffer
                ).rkey,
                16,
            )
            with reverse.compilation():
                proxy_signal_kernel[(1,)](
                    *reverse_arguments,
                    num_warps=1,
                )
                torch.cuda.synchronize(1)

            assert local_signal.cpu().item() == 17
            assert pair.queue_state() == (16, 16, 16, 16)
            assert pair.status() == (0, 0)

    print("aboleth_proxy_triton=PASS")


if __name__ == "__main__":
    main()
