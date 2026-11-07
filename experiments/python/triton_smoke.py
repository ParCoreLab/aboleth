import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x,
    y,
    output,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"triton={triton.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)}")

    element_count = 4096
    block_size = 256
    x = torch.randn(element_count, device="cuda")
    y = torch.randn(element_count, device="cuda")
    output = torch.empty_like(x)

    add_kernel[(triton.cdiv(element_count, block_size),)](
        x,
        y,
        output,
        n_elements=element_count,
        block_size=block_size,
    )
    torch.testing.assert_close(output, x + y)
    print("triton_kernel=PASS")


if __name__ == "__main__":
    main()
