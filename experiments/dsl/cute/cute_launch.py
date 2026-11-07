"""Launchers for cute_bench.py's kernels, shared by every CuTe arm.

ladder.py and alltoall.py drive these exactly as they drive tier2.py's (same
`run` protocols). Constexpr arguments go to the compilation only; the compiled
callable takes the dynamic ones. An arm module binds its policy into cute_bench and supplies
how a traced kernel is compiled (`compile_kernel(function, *arguments)`), so
an arm whose library needs per-module initialization (NVSHMEM) does it there.
Each compiled kernel's registers, spills, shared bytes and compile time land
in the driver's kernel records.
"""

from __future__ import annotations

import time

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cute_bench


def tensor(value: torch.Tensor):
    return from_dlpack(value, assumed_align=value.element_size())


def default_compile(function, *arguments):
    return cute.compile(function, *arguments)


def kernel_resources(compiled) -> dict:
    """Registers, spills and shared bytes of a compiled CuTe module's kernel."""
    from cuda.bindings import driver
    values = {}
    try:
        library = compiled.jit_module.cuda_library[0]
        _, count = driver.cuLibraryGetKernelCount(library)
        _, kernels = driver.cuLibraryEnumerateKernels(count, library)
        _, device = driver.cuCtxGetDevice()
        attribute = driver.CUfunction_attribute
        for name, key in (
                ("registers", attribute.CU_FUNC_ATTRIBUTE_NUM_REGS),
                ("local_bytes", attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
                ("shared_bytes",
                 attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)):
            error, value = driver.cuKernelGetAttribute(key, kernels[0],
                                                       device)
            if error == driver.CUresult.CUDA_SUCCESS:
                values[name] = value
    except (AttributeError, IndexError, TypeError) as error:
        values["resources_error"] = repr(error)
    values["spills"] = values.get("local_bytes", 0)
    return values


class _Kernel:
    """A compiled kernel whose record gets its resources at the first call:
    CuTe loads the module (jit_module) lazily, on that call."""

    def __init__(self, compiled, record: dict):
        self.compiled, self.record = compiled, record

    def __call__(self, *arguments):
        self.compiled(*arguments)
        if self.record is not None:
            self.record.update(kernel_resources(self.compiled))
            self.record = None


class Arm:
    """One CuTe arm: a policy bound into cute_bench and its compiler."""

    def __init__(self, policy, *, quiet_is_global: bool = False,
                 compile_kernel=default_compile, key: str = ""):
        self.policy, self.quiet_is_global = policy, quiet_is_global
        self.compile_kernel, self.key = compile_kernel, key
        self.compiled: dict[tuple, object] = {}

    def _compile(self, records: list, name: str, size: int, ctas: int,
                 function, arguments: tuple, constants: tuple,
                 threads: int = 32):
        key = (name, self.key, constants)
        compiled = self.compiled.get(key)
        if compiled is None:
            cute_bench.use(self.policy, self.quiet_is_global)
            began = time.perf_counter()
            record = {"kernel": name, "size": size, "threads": threads,
                      "ctas": ctas}
            compiled = _Kernel(
                self.compile_kernel(function, *arguments, *constants), record)
            record["compile_s"] = time.perf_counter() - began
            records.append(record)
            self.compiled[key] = compiled
        return compiled

    def pingpong(self, setup, size: int, puts: int, opcode: int):
        fuse = setup.args.fuse_signal == "on"
        endpoints = tensor(setup.table.endpoint_handles.view(torch.int64))
        samples = tensor(setup.samples)
        fixed = (endpoints, setup.peer, setup.remote_data.address,
                 setup.remote_data.rkey, setup.local_data.address,
                 setup.local_data.lkey,
                 setup.remote_flags.address + setup.rank * 8,
                 setup.remote_flags.rkey,
                 setup.local_flags.address + setup.peer * 8)
        constants = (size, puts, opcode, setup.leader, fuse)
        state = {}

        def run(warmup: int, iters: int) -> None:
            arguments = (*fixed, warmup, iters, samples)
            if "compiled" not in state:
                state["compiled"] = self._compile(
                    setup.kernels, "pingpong", size, 1, cute_bench.pingpong,
                    arguments, constants)
            state["compiled"](*arguments)
            torch.cuda.synchronize()
        return run

    def windowed(self, setup, size: int):
        endpoints = tensor(setup.table.endpoint_handles.view(torch.int64))
        samples, spans = tensor(setup.samples), tensor(setup.spans)
        barrier_buffer = torch.zeros(1, dtype=torch.int32,
                                     device=setup.device)
        barrier = tensor(barrier_buffer)
        issuers = setup.args.issuers
        stream = int(setup.args.stream == "on")
        state = {}

        def run(iters: int, record: bool, ctas: int) -> None:
            barrier_buffer.zero_()
            arguments = (endpoints, setup.peer, setup.remote_data.address,
                         setup.remote_data.rkey, setup.local_data.address,
                         setup.local_data.lkey, setup.args.window, iters,
                         setup.data_bytes // size, samples, spans,
                         int(record), barrier, ctas, stream)
            if "compiled" not in state:
                state["compiled"] = self._compile(
                    setup.kernels, "bandwidth", size, setup.args.ctas,
                    cute_bench.windowed, arguments, (size, issuers),
                    threads=max(issuers, 32))
            state["compiled"](*arguments)
        return run

    def alltoall(self, world, table, source, source_key, flags, size,
                 stride, receive, samples, barrier, records, drain=False,
                 args=None):
        if args is not None and getattr(args, "collective", "off") == "on":
            return self.alltoall_collective(world, source, size, receive,
                                            samples, records)
        rank, world_size = world.rank, world.world_size
        peers = [peer for peer in range(world_size) if peer != rank]
        device = samples.device

        def column(values, dtype=torch.int64):
            return torch.tensor(values, dtype=dtype, device=device)

        remote_data = [table.remote(peer, "data") for peer in peers]
        remote_flags = [table.remote(peer, "flags") for peer in peers]
        local_flags = table.local("flags")
        keep = {
            "dst": column([r.address + receive + rank * stride
                           for r in remote_data]),
            "dst_key": column([r.rkey for r in remote_data], torch.int32),
            "src": column([source + peer * stride for peer in peers]),
            "flag": column([r.address + rank * 8 for r in remote_flags]),
            "flag_key": column([r.rkey for r in remote_flags], torch.int32),
            "my_flag": column([local_flags.address + peer * 8
                               for peer in peers]),
        }
        views = {name: tensor(value) for name, value in keep.items()}
        endpoints = tensor(table.endpoint_handles.view(torch.int64))
        fuse = args is not None and args.fuse_signal == "on"
        warp = args is not None and args.put_issue == "warp"
        sample_view, barrier_view = tensor(samples), tensor(barrier)
        state = {}

        def run(warmup: int, iters: int) -> None:
            barrier.zero_()
            arguments = (endpoints, rank, views["dst"], views["dst_key"],
                         views["src"], source_key, views["flag"],
                         views["flag_key"], views["my_flag"], warmup, iters,
                         sample_view, barrier_view, len(peers))
            if "compiled" not in state:
                state["compiled"] = self._compile(
                    records, "alltoall", size, len(peers),
                    cute_bench.alltoall, arguments, (size, fuse, warp))
            state["compiled"](*arguments)
            torch.cuda.synchronize()
        run.keep = keep
        return run


    def alltoall_collective(self, world, source, size, receive, samples,
                            records):
        """The library's own all-to-all (the policy's alltoall_block) over
        contiguous slices at `source` and `source + receive`."""
        sample_view = tensor(samples)
        state = {}

        def run(warmup: int, iters: int) -> None:
            arguments = (source + receive, source, warmup, iters,
                         sample_view)
            if "compiled" not in state:
                state["compiled"] = self._compile(
                    records, "alltoall_collective", size, 1,
                    cute_bench.alltoall_collective, arguments,
                    (size, world.world_size),
                    threads=cute_bench.COLLECTIVE_THREADS)
            state["compiled"](*arguments)
            torch.cuda.synchronize()
        return run
