import torch
import triton
import triton.language as tl


@triton.jit
def poll_ready_kernel(
    ready,
    status,
    timeout_ns: tl.constexpr,
):
    observed = tl.atomic_poll(
        ready,
        1,
        sem="acquire",
        scope="gpu",
        timeout_ns=timeout_ns,
    )
    tl.store(status, tl.where(observed, 0, 1))


def main():
    torch.cuda.set_device(0)
    ready = torch.zeros(1, dtype=torch.int32, device="cuda")
    status = torch.zeros_like(ready)

    poll_ready_kernel[(1,)](
        ready,
        status,
        timeout_ns=0,
        num_warps=1,
    )
    torch.cuda.synchronize()
    assert status.item() == 1

    ready.fill_(1)
    status.zero_()
    poll_ready_kernel[(1,)](
        ready,
        status,
        timeout_ns=0,
        num_warps=1,
    )
    torch.cuda.synchronize()
    assert status.item() == 0
    print("Aboleth pack readiness timeout: PASS")


if __name__ == "__main__":
    main()
