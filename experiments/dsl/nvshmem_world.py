"""NVSHMEM's own world behind the surface the bench drivers use.

The DSL NVSHMEM arms (cute-nvshmem, triton-nvshmem) run ladder.py's and
alltoall.py's sweeps, checks and records over NVSHMEM instead of an Aboleth
World. Set up as bench_tier1_nvshmem.cu sets it up: the same environment
policy (configure_nvshmem), a UID handed round the bench's TCP store, the
symmetric heap sized from the arena, and the library footprint measured as
free device memory before init less after init, before the arena.

It imports no `aboleth` (the Triton-distributed arm runs it in an
environment whose Triton cannot host the Aboleth plugin): the bench's TCP
store bootstrap is loaded from its file. The arms that do use Aboleth import
it first themselves.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import nvshmem.core
import nvshmem.core.interop.torch as nvshmem_torch

SIGNAL_SET = int(nvshmem.core.SignalOp.SIGNAL_SET)
CMP_GE = int(nvshmem.core.ComparisonType.CMP_GE)
UNIQUEID_BYTES = 128  # sizeof(nvshmemx_uniqueid_t)


def configure(transport: str, heap_bytes: int, nic_only: bool,
              ctas: int) -> None:
    """bench_tier1_nvshmem.cu's configure_nvshmem: forced settings, and soft
    defaults the ambient environment overrides."""
    env = os.environ
    if nic_only:
        env["NVSHMEM_DISABLE_P2P"] = "1"
        env["NVSHMEM_ENABLE_NIC_PE_MAPPING"] = "1"
        env.setdefault("NVSHMEM_HCA_LIST", "mlx5_0:1")
    env["NVSHMEM_DISABLE_NCCL"] = "1"
    env.setdefault("NVSHMEM_IB_DISABLE_DMABUF", "1")
    env.setdefault("NVSHMEM_IB_GID_INDEX", "3")
    env.setdefault("NVSHMEM_SYMMETRIC_SIZE", str(heap_bytes))
    if transport == "ibgda":
        env["NVSHMEM_REMOTE_TRANSPORT"] = "none"
        env["NVSHMEM_IB_ENABLE_IBGDA"] = "1"
        env.setdefault("NVSHMEM_IBGDA_NUM_RC_PER_PE", str(max(ctas, 1)))
        if ctas > 1:
            env.setdefault("NVSHMEM_IBGDA_RC_MAP_BY", "cta")
    else:
        env["NVSHMEM_REMOTE_TRANSPORT"] = "ibrc"
        env["NVSHMEM_IB_ENABLE_IBGDA"] = "0"
        for name in ("NVSHMEM_IBGDA_NIC_HANDLER",
                     "NVSHMEM_IBGDA_NUM_RC_PER_PE", "NVSHMEM_IBGDA_RC_MAP_BY",
                     "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH"):
            env.pop(name, None)


def planned_arena_bytes(args, world_size: int) -> int:
    """The arena the driver will allocate (bench::plan_data_bytes)."""
    import common
    if getattr(args, "mode", None) == "latency":
        sizes = common.resolve_sweep(args, "8", "64K")
        data = max(max(sizes) * args.puts, 8)
    elif getattr(args, "mode", None) == "bandwidth":
        sizes = common.resolve_sweep(args, "4K", "256M")
        data = args.window * max(sizes)
    else:
        sizes = common.resolve_sweep(args, "1K", "64M")
        data = 2 * world_size * max(sizes)
    return (data + 7) // 8 * 8 + 8 * world_size


class _Table:
    """The region table of a symmetric heap: every peer's view of a region
    is this rank's own address, so remote(peer, name) == local(name)."""

    def __init__(self, base: int, regions: dict, world_size: int, device):
        self._regions = {
            name: SimpleNamespace(address=base + offset, lkey=0, rkey=0)
            for name, (offset, _) in regions.items()}
        self.endpoint_handles = torch.zeros(world_size, dtype=torch.int64,
                                            device=device)

    def local(self, name):
        return self._regions[name]

    def remote(self, peer, name):
        return self._regions[name]

    def close(self) -> None:
        pass


class World:
    def __init__(self, rank, world_size, device, bootstrap, memory):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.bootstrap = bootstrap
        self.bench_memory = memory
        self.endpoints_per_peer = 1
        self.hca = os.environ.get("NVSHMEM_HCA_LIST")
        # Library-initialized objects to finalize before NVSHMEM itself.
        self.kernels: list = []
        self.finalizers: list = []
        self._arena = None

    def allocate_arena(self, nbytes: int) -> torch.Tensor:
        self._arena = nvshmem_torch.bytetensor((nbytes,))
        self._arena.zero_()
        return self._arena

    def register(self, arena):
        if arena is not self._arena:
            raise RuntimeError("nvshmem world: register the symmetric arena")
        return arena

    def region(self, name, registration, *, offset, length):
        return (offset, length)

    def exchange_regions(self, regions: dict) -> _Table:
        return _Table(self._arena.data_ptr(), regions, self.world_size,
                      self._arena.device)

    def close(self) -> None:
        torch.cuda.synchronize()
        for kernel in self.kernels:
            nvshmem.core.library_finalize(kernel)
        for finalize in self.finalizers:
            finalize()
        if self._arena is not None:
            nvshmem_torch.free_tensor(self._arena)
            self._arena = None
        nvshmem.core.finalize()


WORLD: list[World] = []


def _bootstrap_class():
    """aboleth.bootstrap.TcpStoreBootstrap, without importing `aboleth` (its
    package init configures the plugin into the series' Triton)."""
    path = (Path(__file__).resolve().parents[2] / "python" / "aboleth"
            / "bootstrap.py")
    spec = importlib.util.spec_from_file_location("_bench_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TcpStoreBootstrap


def make_world(args):
    """World + bootstrap for this rank: NVSHMEM over a UID handed round the
    bench's TCP store, as the CUDA binary hands it round its rendezvous."""
    from cuda.core import Device

    TcpStoreBootstrap = _bootstrap_class()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    torch.cuda.init()
    free_before = torch.cuda.mem_get_info()[0]
    bootstrap = TcpStoreBootstrap.from_env(
        timeout=float(os.environ.get("ABOLETH_BOOTSTRAP_TIMEOUT_S", "300")))
    rank, world_size = bootstrap.rank, bootstrap.world_size
    arena = planned_arena_bytes(args, world_size)
    ladder = getattr(args, "mode", None) in ("latency", "bandwidth")
    configure(args.transport, arena * 2 + (64 << 20), ladder,
              args.ctas if getattr(args, "mode", None) == "bandwidth" else 1)
    device = Device(local)
    device.set_current()
    uid = nvshmem.core.get_unique_id(empty=rank != 0)
    raw = ctypes.string_at(uid.ptr, UNIQUEID_BYTES)
    shared = bootstrap.all_gather("nvshmem-world:uid", raw)
    ctypes.memmove(uid.ptr, shared[0], UNIQUEID_BYTES)
    nvshmem.core.init(device=device, uid=uid, rank=rank, nranks=world_size,
                      initializer_method="uid")
    free_after = torch.cuda.mem_get_info()[0]
    memory = {"free_before": free_before, "free_after": free_after,
              "library_bytes": free_before - free_after,
              "nvshmem_symmetric_size": os.environ.get(
                  "NVSHMEM_SYMMETRIC_SIZE")}
    world = World(rank, world_size, local, bootstrap, memory)
    WORLD.append(world)
    return world, bootstrap


def library_meta(frontend: str) -> dict:
    versions = {}
    for package in ("nvidia-nvshmem-cu12", "nvshmem4py-cu12",
                    "nvidia-cutlass-dsl", "triton"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    transport = os.environ.get("NVSHMEM_REMOTE_TRANSPORT")
    handler = os.environ.get("NVSHMEM_IBGDA_NIC_HANDLER", "auto")
    # The host library this process mapped: torch carries its own NVSHMEM
    # under the same SONAME, so the record names the file actually loaded.
    with open("/proc/self/maps") as maps:
        loaded = sorted({line.split()[-1] for line in maps
                         if "libnvshmem_host" in line})
    return {"library": "nvshmem", "frontend": frontend,
            "submit_mode": ("ibgda" if transport == "none" else "ibrc"),
            "nvshmem_handler": handler, "versions": versions,
            "nvshmem_host_loaded": loaded}
