from __future__ import annotations

import ctypes
from dataclasses import dataclass

import triton
import triton.language as tl


_MEMCPY_DEVICE_TO_DEVICE = 3
_copy_api = None


def cuda_memcpy_async(
    destination: int,
    source: int,
    size: int,
    stream: int,
) -> None:
    """Submit on the caller's stream using the active PyTorch GPU runtime."""
    global _copy_api
    if _copy_api is None:
        import torch

        is_hip = torch.version.hip is not None
        library = ctypes.CDLL(
            "libamdhip64.so.7" if is_hip else "libcudart.so"
        )
        name = "hipMemcpyAsync" if is_hip else "cudaMemcpyAsync"
        function = getattr(library, name)
        function.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        function.restype = ctypes.c_int
        _copy_api = (library, function, name, "HIP" if is_hip else "CUDA")
    _, function, name, runtime = _copy_api
    status = function(
        ctypes.c_void_p(destination),
        ctypes.c_void_p(source),
        ctypes.c_size_t(size),
        _MEMCPY_DEVICE_TO_DEVICE,
        ctypes.c_void_p(stream),
    )
    if status:
        raise RuntimeError(
            f"{name} failed with {runtime} status {status}"
        )


@triton.jit
def pull_kv_runs_kernel(
    remote_layer_ptrs,
    local_layer_ptrs,
    local_run_starts,
    remote_run_starts,
    run_lengths,
    run_count,
    remote_rank,
    num_layers: tl.constexpr,
    page_words: tl.constexpr,
    copy_workers: tl.constexpr,
    block_elements: tl.constexpr,
):
    layer = tl.program_id(0)
    run = tl.program_id(1)
    worker = tl.program_id(2)
    active = run < run_count
    local_block = tl.load(
        local_run_starts + run,
        mask=active,
        other=0,
    )
    remote_block = tl.load(
        remote_run_starts + run,
        mask=active,
        other=0,
    )
    run_length = tl.load(
        run_lengths + run,
        mask=active,
        other=0,
    )
    remote_base = tl.load(
        remote_layer_ptrs
        + remote_rank * num_layers
        + layer
    ).to(tl.pointer_type(tl.uint64))
    local_base = tl.load(
        local_layer_ptrs + layer
    ).to(tl.pointer_type(tl.uint64))
    offsets = tl.arange(0, block_elements)
    run_words = run_length * page_words
    for block_start in tl.range(
        worker * block_elements,
        run_words,
        copy_workers * block_elements,
    ):
        indices = block_start + offsets
        mask = active & (indices < run_words)
        values = tl.load(
            remote_base
            + remote_block * page_words
            + indices,
            mask=mask,
            other=0,
        )
        tl.store(
            local_base
            + local_block * page_words
            + indices,
            values,
            mask=mask,
        )


@triton.jit
def pull_contiguous_kv_kernel(
    remote_layer_ptrs,
    local_layer_ptrs,
    local_block,
    remote_block,
    run_length,
    remote_rank,
    num_layers: tl.constexpr,
    page_words: tl.constexpr,
    copy_workers: tl.constexpr,
    block_elements: tl.constexpr,
):
    layer = tl.program_id(0)
    worker = tl.program_id(1)
    remote_base = tl.load(
        remote_layer_ptrs
        + remote_rank * num_layers
        + layer
    ).to(tl.pointer_type(tl.uint64))
    local_base = tl.load(
        local_layer_ptrs + layer
    ).to(tl.pointer_type(tl.uint64))
    offsets = tl.arange(0, block_elements)
    run_words = run_length * page_words
    for block_start in tl.range(
        worker * block_elements,
        run_words,
        copy_workers * block_elements,
    ):
        indices = block_start + offsets
        mask = indices < run_words
        values = tl.load(
            remote_base
            + remote_block * page_words
            + indices,
            mask=mask,
        )
        tl.store(
            local_base
            + local_block * page_words
            + indices,
            values,
            mask=mask,
        )


@dataclass(frozen=True)
class KVBlockRuns:
    local_starts: tuple[int, ...]
    remote_starts: tuple[int, ...]
    lengths: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.lengths)

    @property
    def page_count(self) -> int:
        return sum(self.lengths)


def canonicalize_block_runs(
    local_block_ids: list[int],
    remote_block_ids: list[int],
) -> KVBlockRuns:
    pairs = sorted(
        zip(local_block_ids, remote_block_ids, strict=True),
        key=lambda pair: (
            pair[0] - pair[1],
            pair[0],
            pair[1],
        ),
    )
    local_starts = []
    remote_starts = []
    lengths = []
    for local, remote in pairs:
        if (
            lengths
            and local_starts[-1] + lengths[-1] == local
            and remote_starts[-1] + lengths[-1] == remote
        ):
            lengths[-1] += 1
        else:
            local_starts.append(local)
            remote_starts.append(remote)
            lengths.append(1)
    return KVBlockRuns(
        local_starts=tuple(local_starts),
        remote_starts=tuple(remote_starts),
        lengths=tuple(lengths),
    )
