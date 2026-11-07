"""MoE dispatch/combine harness shared by every arm (dsl-moe).

One contract for every implementation, in every environment (it imports no
`aboleth`, so Triton-distributed's environment runs it too):

- Inputs: each rank's tokens (BF16, randn), top-k *distinct* experts per
  token (seeded randperm) and routing weights (multiples of 1/8, so a
  weighted sum is exact in FP32), regenerated from (rank, tokens, step) by
  any rank. Expert e lives on rank e // (E / world), as every library
  places them.
- Dispatch delivers each token to the ranks owning its experts (FP8 with
  per-128 scales, the library's own quantization, or BF16). Checked: every
  local expert's received row count equals the reference's.
- The harness's expert e computes f_e(x) = x * (1 + (e % 7) / 8) on the
  dequantized rows it received (untimed), in the layout the arm hands it.
- Combine returns, per source token, sum_k w_k * f_{e_k}(x). Checked against
  an FP32 reference of the same function on the unquantized token, within
  FP8's error (dispatch quantizes; rows are compared to their amax).

Timer, the same for every arm: per call, a one-element NCCL all-reduce
aligns the ranks' GPUs, a device sleep covers the host's enqueue of the call
(launch overhead is not counted, as CUDA-graph replays would not count it,
and arms that cannot be captured are not penalized), and CUDA events bracket
dispatch and combine separately. A host-synchronizing arm still pays its
synchronization: the GPU idles inside the bracket while the host waits.
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

GROUP = 128
FP8_MAX = 448.0

# Model shapes (hidden, experts, top-k): the dispatch side of each model's
# MoE layer.
MODELS = {"qwen3-235b": (4096, 128, 8), "qwen3-30b": (2048, 128, 8),
          "deepseek-v3": (7168, 256, 8)}


@dataclass(frozen=True)
class Spec:
    hidden: int
    experts: int
    topk: int
    world: int
    rank: int
    fp8: bool

    @property
    def local_experts(self) -> int:
        return self.experts // self.world


def bootstrap_class():
    """aboleth.bootstrap.TcpStoreBootstrap without importing `aboleth`."""
    path = (Path(__file__).resolve().parents[3] / "python" / "aboleth"
            / "bootstrap.py")
    spec = importlib.util.spec_from_file_location("_moe_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TcpStoreBootstrap


def inputs(spec: Spec, rank: int, tokens: int, step: int = 0):
    """(x [T,H] bf16, ids [T,K] int64, weights [T,K] fp32) on the CPU."""
    generator = torch.Generator().manual_seed(
        104729 * step + 7919 * rank + tokens)
    if tokens:
        ids = torch.stack([torch.randperm(spec.experts,
                                          generator=generator)[:spec.topk]
                           for _ in range(tokens)])
    else:
        ids = torch.empty((0, spec.topk), dtype=torch.int64)
    weights = torch.randint(1, 8, (tokens, spec.topk),
                            generator=generator).float() / 8
    if os.environ.get("ABOLETH_MOE_UNIT_WEIGHTS"):
        # Diagnostic: every routing weight 1 (isolates weight transport).
        weights = torch.ones_like(weights)
    x = torch.randn((tokens, spec.hidden), generator=generator).to(
        torch.bfloat16)
    return x, ids, weights


def expert_factor(experts: torch.Tensor) -> torch.Tensor:
    """f_e's scale (exact in BF16): 1 + (e % 7) / 8."""
    return 1.0 + (experts % 7).float() / 8.0


def reference_counts(spec: Spec, tokens: int, step: int = 0) -> torch.Tensor:
    """Rows each of this rank's local experts receives, over every source."""
    counts = torch.zeros(spec.local_experts, dtype=torch.int64)
    first = spec.rank * spec.local_experts
    for source in range(spec.world):
        _, ids, _ = inputs(spec, source, tokens, step)
        local = ids[(ids >= first) & (ids < first + spec.local_experts)]
        counts += torch.bincount(local - first,
                                 minlength=spec.local_experts)
    return counts


def reference_output(x, ids, weights) -> torch.Tensor:
    """sum_k w_k * f_{e_k}(x) in FP32 on the unquantized tokens."""
    coefficient = (weights * expert_factor(ids)).sum(dim=1)
    return x.float() * coefficient[:, None]


def dequantize(values: torch.Tensor, scales: torch.Tensor | None):
    """FP8 rows [..., H] with per-group scales [..., H/GROUP] -> FP32."""
    if scales is None:
        return values.float()
    shape = values.shape
    grouped = values.float().reshape(*shape[:-1], shape[-1] // GROUP, GROUP)
    return (grouped * scales.float()[..., None]).reshape(shape)


def check_output(out: torch.Tensor, x, ids, weights, fp8: bool) -> int:
    """Rows of `out` outside the contract's tolerance (0 when correct)."""
    expected = reference_output(x, ids, weights)
    got = out.float().cpu()
    if got.shape != expected.shape:
        return max(1, expected.shape[0])
    if not got.numel():
        return 0
    # FP8 E4M3 keeps 3 mantissa bits: a dequantized value is within 1/16
    # of its group's amax; the weighted sum keeps that bound per row (the
    # coefficients sum to at most top-k * 2). BF16 adds its own rounding.
    coefficient = (weights * expert_factor(ids)).sum(dim=1).abs()
    amax = x.float().abs().amax(dim=1)
    bound = coefficient * amax * (1 / 16 if fp8 else 1 / 128) + 1e-3
    error = (got - expected).abs().amax(dim=1)
    return int((error > bound).sum()) + int(torch.isnan(got).any(dim=1).sum())


def diagnose(rank, tokens, counts, expected_counts, out, x, ids,
             weights) -> None:
    """ABOLETH_MOE_DEBUG: what a failed check saw, on stderr."""
    import sys
    got = counts.cpu().to(torch.int64)
    print(f"[debug rank {rank} tokens {tokens}] counts got "
          f"{got.tolist()[:16]} expected {expected_counts.tolist()[:16]} "
          f"(sums {int(got.sum())} vs {int(expected_counts.sum())})",
          file=sys.stderr)
    expected = reference_output(x, ids, weights)
    value = out.float().cpu()
    if value.shape != expected.shape:
        print(f"[debug rank {rank}] output shape {tuple(value.shape)} vs "
              f"{tuple(expected.shape)}", file=sys.stderr)
        return
    for row in range(min(3, value.shape[0])):
        ratio = value[row] / expected[row].where(
            expected[row].abs() > 1e-3, torch.tensor(float("nan")))
        finite = ratio[torch.isfinite(ratio)]
        print(f"[debug rank {rank}] row {row}: got/expected median "
              f"{finite.median().item() if finite.numel() else 'nan':.4} "
              f"spread {finite.std().item() if finite.numel() > 1 else 0:.3} "
              f"nan {int(torch.isnan(value[row]).sum())} ids "
              f"{ids[row].tolist()} weights {weights[row].tolist()}",
              file=sys.stderr)


class Timer:
    """CUDA events around one call each, after an NCCL alignment and a
    device sleep (module doc). One per arm process."""

    def __init__(self, bootstrap, rank: int, world: int, device: int,
                 sleep_us: float = 1000.0):
        import torch.distributed as dist
        self.dist = dist
        if not dist.is_initialized():
            store = getattr(bootstrap, "_store", None)
            if store is None:
                raise RuntimeError("Timer: the bootstrap has no TCP store")
            dist.init_process_group("nccl", store=dist.PrefixStore(
                "moe-timer", store), rank=rank, world_size=world,
                device_id=torch.device("cuda", device))
        self.group = dist.group.WORLD
        self.cell = torch.zeros(1, dtype=torch.int32, device=device)
        rate = torch.cuda.get_device_properties(device).clock_rate  # kHz
        self.cycles = int(rate * 1e3 * sleep_us * 1e-6)
        self.sleep_us = sleep_us

    def align(self) -> None:
        self.dist.all_reduce(self.cell, group=self.group)
        torch.cuda._sleep(self.cycles)

    def bracket(self, fn):
        """(result, start event, end event) of fn() behind an alignment."""
        self.align()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        return result, start, end


def summarize(spans_ms: list[float]) -> dict:
    samples = [1000.0 * value for value in spans_ms]
    ordered = sorted(samples)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)
                                                 + 0.5))]
    return {"p50_us": statistics.median(samples), "p95_us": at(0.95),
            "p99_us": at(0.99), "mean_us": statistics.fmean(samples),
            "samples_us": [round(value, 3) for value in samples]}


def now() -> float:
    return time.perf_counter()


def env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
