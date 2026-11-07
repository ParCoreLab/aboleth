"""dsl-moe arm: Triton-distributed's own EP dispatch and combine.

`tdist-ll`: `EPLowLatencyAllToAllLayer` (its low-latency, DeepEP-style
dispatch with online FP8 quantization, group 128, and the weighted top-k
combine at the source), as its test_ep_ll_a2a.py drives it. Runs in
Triton-distributed's environment (envs/tdist; common.ARM_PYTHON), set up as
its scripts set it up (initialize_distributed: a gloo+NCCL process group and
NVSHMEM over a UID; its launch.sh environment), with NVSHMEM's default
remote transport (IBRC). Buffers per token count: max_m = the count, the
smallest layer that serves it.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import ipaddress
import os
import socket
import subprocess
from pathlib import Path


def _plugin_path() -> str:
    spec = importlib.util.find_spec("triton_dist")
    return str(Path(list(spec.submodule_search_locations)[0]) / "lib"
               / "libtriton_dist.so")


def _own_interface() -> str | None:
    """The interface holding this host's own address (the UID bootstrap must
    listen there, not on the route to the master: lo on the master)."""
    try:
        address = socket.getaddrinfo(socket.gethostname(), None,
                                     socket.AF_INET)[0][4][0]
        listing = subprocess.run(["ip", "-o", "-4", "addr", "show"],
                                 capture_output=True, text=True,
                                 check=True).stdout
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) > 3 and str(ipaddress.ip_interface(
                fields[3]).ip) == address:
            return fields[1]
    return None


# Its scripts/launch.sh environment, before NVSHMEM or Triton load.
os.environ["TRITON_PLUGIN_PATHS"] = _plugin_path()
# NVSHMEM_HOME as its scripts/setenv.sh sets it: this environment's pip
# NVSHMEM, the host library loaded here (unset, `nvidia.nvshmem` resolves to
# the base path's 3.4.5; see tdist/tdist_nvshmem.py).
os.environ.setdefault("NVSHMEM_HOME", str(importlib.metadata.distribution(
    "nvidia-nvshmem-cu12").locate_file("nvidia/nvshmem")))
# The transport (known from the command line before anything loads): IBRC is
# NVSHMEM's default; IBGDA goes through its NVSHMEM wrapper, its documented
# route (the bitcode path has no IBGDA).
_argv = __import__("sys").argv
if "--transport" in _argv and _argv[_argv.index("--transport") + 1:][:1] == [
        "ibgda"]:
    os.environ.update({"TRITON_DIST_SHMEM_WRAPPER": "1",
                       "NVSHMEM_IBGDA_SUPPORT": "1",
                       "NVSHMEM_REMOTE_TRANSPORT": "none",
                       "NVSHMEM_IB_ENABLE_IBGDA": "1"})
for name, value in (("NVSHMEM_BOOTSTRAP", "UID"),
                    ("NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY", "AF_INET"),
                    ("NVSHMEM_DISABLE_CUDA_VMM", "1"),
                    ("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
                    ("NVSHMEM_SYMMETRIC_SIZE", "4g")):
    os.environ.setdefault(name, value)
if "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME" not in os.environ:
    interface = _own_interface()
    if interface:
        os.environ["NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"] = interface

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import moe_common  # noqa: E402


class _DistBootstrap:
    """barrier / all_gather over the process group its layers already need."""

    def __init__(self):
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()

    def barrier(self, tag: str) -> None:
        self.all_gather(tag, b"")

    def all_gather(self, tag: str, payload: bytes) -> dict:
        values = [None] * self.world_size
        dist.all_gather_object(values, payload)
        return dict(enumerate(values))


class _LowLatencyCell:
    def __init__(self, ctx, tokens: int):
        from triton_dist.layers.nvidia import EPLowLatencyAllToAllLayer
        s = ctx.spec
        self.ctx, self.tokens = ctx, tokens
        self.max_m = max(tokens, 1)
        self.layer = EPLowLatencyAllToAllLayer(
            self.max_m, s.hidden, s.topk, online_quant_fp8=True,
            rank=s.rank, num_experts=s.experts,
            local_world_size=ctx.local_world_size, world_size=s.world,
            fp8_gsize=moe_common.GROUP, dtype=torch.bfloat16)
        first = s.rank * s.local_experts
        self.factor = moe_common.expert_factor(torch.arange(
            first, first + s.local_experts, device=ctx.device))
        torch.cuda.synchronize()

    def dispatch(self, x, ids, weights):
        ids = ids.to(torch.int32)
        tokens, scales, counts, meta = self.layer.dispatch(x, None, ids)
        return tokens, scales, counts, meta, ids, weights

    def counts(self, handle):
        return handle[2]

    def expert(self, handle):
        tokens, scales = handle[0], handle[1]
        values = moe_common.dequantize(tokens, scales)
        return (values * self.factor[:, None, None]).to(torch.bfloat16)

    def combine(self, handle, y):
        _, _, _, meta, ids, weights = handle
        return self.layer.combine(y, ids, weights, meta)

    def resources(self):
        return []

    def describe(self):
        return {"max_m": self.max_m}

    def close(self):
        torch.cuda.synchronize()
        self.layer.finalize()


class _Context:
    def __init__(self, args, spec):
        if args.arm == "tdist-ll" and not spec.fp8:
            raise SystemExit("tdist-ll: its low-latency layer quantizes "
                             "online to FP8 (no BF16 dispatch)")
        if args.arm != "tdist-ll":
            raise SystemExit(f"{args.arm}: not implemented yet")
        import triton
        import triton_dist
        from triton_dist.utils import initialize_distributed
        self.spec = spec
        self.device = int(os.environ.get("LOCAL_RANK", "0"))
        self.local_world_size = int(os.environ.get(
            "LOCAL_WORLD_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE",
                                               torch.cuda.device_count())))
        os.environ["LOCAL_WORLD_SIZE"] = str(self.local_world_size)
        torch.cuda.set_device(self.device)
        free_before = torch.cuda.mem_get_info()[0]
        initialize_distributed(seed=spec.rank)
        self.bootstrap = _DistBootstrap()
        free_after = torch.cuda.mem_get_info()[0]
        # NVSHMEM's heap and NCCL's communicator (its layers need both).
        self.memory = {"free_before": free_before, "free_after": free_after,
                       "library_bytes": free_before - free_after,
                       "nvshmem_symmetric_size":
                           os.environ.get("NVSHMEM_SYMMETRIC_SIZE")}
        source = Path(triton_dist.__file__).resolve().parents[2]
        try:
            commit = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        import importlib.metadata as metadata
        self.meta = {"library": "triton-distributed", "frontend": "triton",
                     "layer": "EPLowLatencyAllToAllLayer",
                     "triton_dist_commit": commit,
                     "triton_version": triton.__version__,
                     "nvshmem": metadata.version("nvidia-nvshmem-cu12"),
                     "nvshmem_transport": ("ibgda" if os.environ.get(
                         "NVSHMEM_IB_ENABLE_IBGDA") == "1" else "ibrc"),
                     "shmem_wrapper": os.environ.get(
                         "TRITON_DIST_SHMEM_WRAPPER", "0"),
                     "nvshmem_home": os.environ.get("NVSHMEM_HOME")}

    def cell(self, tokens: int):
        return _LowLatencyCell(self, tokens)

    def close(self):
        from triton_dist.utils import finalize_distributed
        torch.cuda.synchronize()
        finalize_distributed()


def open(args, spec):  # noqa: A001 - the arm protocol's name
    return _Context(args, spec)
