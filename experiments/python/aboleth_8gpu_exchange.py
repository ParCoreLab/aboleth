import json
import statistics

import torch

from aboleth.runtime import Runtime
from aboleth_moe_exchange import (
    allocate,
    exchange_kernel,
    kernel_arguments,
    register,
)


REMOTE_TOKENS = 64
HIDDEN_SIZE = 32
WARMUP_ITERATIONS = 10
MEASURED_ITERATIONS = 100
GPU_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))


def main() -> None:
    assert torch.cuda.device_count() >= 8

    tensors = {
        device: allocate(
            device,
            REMOTE_TOKENS,
            HIDDEN_SIZE,
        )
        for pair in GPU_PAIRS
        for device in pair
    }

    with Runtime() as runtime:
        registered = {
            device: register(runtime, tensors[device])
            for device in tensors
        }
        pairs = []
        try:
            for device0, device1 in GPU_PAIRS:
                pairs.append(
                    runtime.create_endpoint_pair(
                        device0,
                        device1,
                        transport="rdma",
                    )
                )

            launches = []
            for pair, (device0, device1) in zip(
                pairs,
                GPU_PAIRS,
                strict=True,
            ):
                forward, reverse = pair.endpoints
                launches.extend(
                    (
                        (
                            device0,
                            forward,
                            kernel_arguments(
                                forward,
                                tensors[device0],
                                tensors[device1],
                                registered[device0],
                                registered[device1],
                            ),
                        ),
                        (
                            device1,
                            reverse,
                            kernel_arguments(
                                reverse,
                                tensors[device1],
                                tensors[device0],
                                registered[device1],
                                registered[device0],
                            ),
                        ),
                    )
                )

            with launches[0][1].compilation():
                for device, _, arguments in launches:
                    with torch.cuda.device(device):
                        compiled = exchange_kernel.warmup(
                            *arguments,
                            WARMUP_ITERATIONS,
                            MEASURED_ITERATIONS,
                            hidden_size=HIDDEN_SIZE,
                            num_warps=1,
                            grid=(1,),
                        )
                        compiled._init_handles()

                for device, _, arguments in launches:
                    with torch.cuda.device(device):
                        exchange_kernel[(1,)](
                            *arguments,
                            WARMUP_ITERATIONS,
                            MEASURED_ITERATIONS,
                            hidden_size=HIDDEN_SIZE,
                            num_warps=1,
                        )

                for device, _, _ in launches:
                    torch.cuda.synchronize(device)

            expected_wqes = (
                9 *
                (WARMUP_ITERATIONS + MEASURED_ITERATIONS)
            )
            expected_completions = (
                2 *
                (WARMUP_ITERATIONS + MEASURED_ITERATIONS)
            )
            for pair in pairs:
                assert pair.status() == (0, 0)
                assert pair.queue_state() == (
                    expected_wqes,
                    expected_completions,
                    expected_wqes,
                    expected_completions,
                )

            pair_times = []
            for device0, device1 in GPU_PAIRS:
                for device in (device0, device1):
                    expected = tensors[device]["send_tokens"]
                    torch.testing.assert_close(
                        tensors[device]["returned_results"],
                        expected,
                        rtol=0,
                        atol=0,
                    )
                elapsed = max(
                    tensors[device0]["elapsed_ns"].cpu().item(),
                    tensors[device1]["elapsed_ns"].cpu().item(),
                )
                pair_times.append(
                    elapsed /
                    MEASURED_ITERATIONS /
                    1_000_000
                )
        finally:
            for pair in reversed(pairs):
                pair.close()

    result = {
        "backend": "aboleth",
        "gpus": 8,
        "pairs": len(GPU_PAIRS),
        "remote_tokens": REMOTE_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "iterations": MEASURED_ITERATIONS,
        "pair_times_ms": pair_times,
        "mean_pair_ms": statistics.mean(pair_times),
        "max_pair_ms": max(pair_times),
        "status": "PASS",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
