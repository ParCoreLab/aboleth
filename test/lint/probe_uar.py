"""Probe whether this user can map the NIC's user-access region.

The RDMA transport puts the mlx5 BlueFlame doorbell page into the GPU's address
space (cudaHostRegisterMapped | cudaHostRegisterIoMemory). That mapping is what
made the hardware tests root-only, but a driver loaded with
PeerMappingOverride=1 grants it to an ordinary user, so asking "am I root?" is
the wrong question. Ask the hardware instead: create a rank-local RDMA
endpoint, which performs exactly that mapping while it builds the queue pair,
then tear it down. No connect is needed — the mapping happens at creation.

Exit 0 means the gated tests can run unprivileged. Any failure — no GPU, no
mlx5 adapter, a build without the runtime, a mapping the driver refuses — exits
non-zero with the reason on stderr, and the caller falls back to sudo.

The device is index 0 of whatever CUDA_VISIBLE_DEVICES exposes; the HCA is the
one the runtime pairs with it topologically, or ABOLETH_HCA when set.

The probe pins ABOLETH_RDMA_SUBMIT=direct: endpoint bring-up otherwise falls
back to staged submission when the driver refuses the IO mapping, and a probe
that passed via the fallback would let every direct-path test silently run
staged. Success here must keep meaning "the direct mapping works".
"""

from __future__ import annotations

import os
import sys

os.environ["ABOLETH_RDMA_SUBMIT"] = "direct"


def probe() -> str:
    import torch

    from aboleth.runtime import Runtime

    if torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA device is visible")
    with Runtime(device=0) as runtime:
        with runtime.create_local_endpoint(transport="rdma"):
            return runtime.hca


def main() -> int:
    try:
        hca = probe()
    except Exception as error:
        print(
            f"uar-probe: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(f"uar-probe=PASS hca={hca}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
