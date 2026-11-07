#!/usr/bin/env python3
"""Markdown tables from a matrix.sh result directory, and regression checks.

  report.py DIR                 tables for every result file in DIR
  report.py --compare REF NEW   cells of NEW worse than REF by more than
                                --tolerance (default 10%); exit 1 if any

Every arm runs several times per file (matrix.sh --repeats). A latency cell
is the median over runs of each run's p50, with a 95% bootstrap interval
over runs and the run count; tails (p99) pool every run's raw samples. A
bandwidth cell is the median over runs of bytes / device span, and message
rate the same for puts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> dict:
    """A result record; committed records are gzipped."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def _records(directory: Path, prefix: str = ""):
    """(stem, path) of every JSON record, plain or gzipped."""
    for path in sorted(directory.glob(f"{prefix}*.json*")):
        if path.name.endswith((".json", ".json.gz")):
            yield path.name.split(".json")[0], path


def _size(value: int) -> str:
    for unit, scale in (("M", 1 << 20), ("K", 1 << 10)):
        if value >= scale and value % scale == 0:
            return f"{value // scale}{unit}"
    return str(value)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---:" if i else "---"
                            for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap(values: list[float], draws: int = 2000) -> tuple[float, float]:
    """95% interval of the median over runs."""
    if len(values) < 2:
        return values[0], values[0]
    rng = random.Random(0)
    medians = sorted(statistics.median(rng.choices(values, k=len(values)))
                     for _ in range(draws))
    return medians[int(0.025 * draws)], medians[int(0.975 * draws) - 1]


def _runs(record: dict):
    """(arm, inner record) for every completed run of a file."""
    for run in record.get("runs", [record]):
        inner = run.get("record") if "runs" in record else run
        if inner and not inner.get("skipped"):
            yield run.get("arm", inner.get("arm")), inner


def _metric(row: dict, metric: str) -> float:
    if metric == "p50_us" and row.get("samples_ns"):
        return _quantile(row["samples_ns"], 0.5) / 1000.0
    if metric in ("busbw_gbps", "mops") and row.get("span_ns"):
        puts = row["ctas"] * row["window"] * row["iters"]
        return (puts * row["size"] / row["span_ns"] if metric == "busbw_gbps"
                else puts / row["span_ns"] * 1e3)
    return row.get(metric, 0.0)


def _op(inner: dict, row: dict) -> str:
    """All-to-all rounds are one operation whatever the arm calls them
    (nccl-host records grouped send/recv)."""
    if inner.get("benchmark") == "alltoall":
        return "alltoall"
    return row.get("op", "put")


def cells(record: dict, metric: str) -> dict:
    """{(arm, op, size): [per-run values]}. A row that failed its
    correctness check (wrong > 0) or recorded no time is not a
    measurement: it is left out, and `failures` lists it."""
    values = defaultdict(list)
    for arm, inner in _runs(record):
        for row in inner.get("results", []):
            value = _metric(row, metric)
            if row.get("wrong") or not value:
                continue
            values[(arm, _op(inner, row), row["size"])].append(value)
    return values


def failures(record: dict) -> list[str]:
    """Every (arm, size) with a run that failed its correctness check."""
    failed = set()
    for arm, inner in _runs(record):
        for row in inner.get("results", []):
            if row.get("wrong"):
                failed.add((arm, row["size"]))
    return [f"{arm} {_size(size)}" for arm, size in sorted(failed)]


def tails(record: dict) -> dict:
    """{(arm, op, size): pooled p99 in us} over every run's raw samples."""
    pooled = defaultdict(list)
    for arm, inner in _runs(record):
        for row in inner.get("results", []):
            pooled[(arm, _op(inner, row), row["size"])] += (
                row.get("samples_ns") or [])
    return {key: _quantile(samples, 0.99) / 1000.0
            for key, samples in pooled.items() if samples}


def arm_table(record: dict, metric: str, unit: str,
              with_tail: bool = False) -> str:
    values = cells(record, metric)
    tail = tails(record) if with_tail else {}
    arms = list(dict.fromkeys(arm for arm, _, _ in values))
    keys = sorted({(op, size) for _, op, size in values},
                  key=lambda key: (key[0], key[1]))
    ops = {op for op, _ in keys}
    rows = []
    for op, size in keys:
        row = [(f"{op} " if len(ops) > 1 else "") + _size(size)]
        for arm in arms:
            runs = values.get((arm, op, size))
            if not runs:
                row.append("—")
                continue
            low, high = bootstrap(runs)
            cell = (f"{statistics.median(runs):.2f} "
                    f"[{low:.2f}, {high:.2f}] n={len(runs)}")
            if (arm, op, size) in tail:
                cell += f"; p99 {tail[(arm, op, size)]:.2f}"
            row.append(cell)
        rows.append(row)
    return _table([f"size ({unit})"] + arms, rows)


# Library of an arm token; a library's best configuration per cell is the
# one the paper quotes.
_FAMILIES = (("aboleth-triton", "Aboleth T2"),
             # dsl-moe (experiments/dsl/moe).
             ("aboleth-moe-fused", "Aboleth T2 MoE (fused pack)"),
             ("aboleth-moe", "Aboleth T2 MoE"),
             ("tdist-ll", "Triton-distributed LL"),
             ("tdist-normal", "Triton-distributed normal"),
             ("deepep", "DeepEP"),
             ("aboleth-", "Aboleth T1"),
             # The kernels in other DSLs (experiments/dsl).
             ("tdist-nvshmem", "Triton-distributed + NVSHMEM"),
             ("cute-tier1", "CuTe + Aboleth T1"),
             ("cute-nvshmem", "CuTe + NVSHMEM"),
             ("triton-tier1", "Triton + Aboleth T1"),
             ("triton-nvshmem", "Triton + NVSHMEM"),
             ("cutile-tier2", "cuTile + Aboleth T2"),
             ("cutile-nccl", "cuTile + NCCL"),
             ("cutile-p2p", "cuTile NVLink"),
             ("nvshmem38", "NVSHMEM 3.8"),
             ("nvshmem", "NVSHMEM"), ("nccl-device", "NCCL device"),
             ("nccl-host", "NCCL host"))


def family(arm: str) -> str:
    # The baked CUDA Tier-1 arm is the DSL kernels' constant-size reference,
    # not a configuration of the CUDA arm the paper quotes.
    if arm.startswith("aboleth-generic") and "bake=on" in arm:
        return "Aboleth T1 (baked)"
    # A diagnostic that skips Tier 1's per-round retirement: never Tier 1's
    # result.
    if "quiet-every=" in arm:
        return "Aboleth T1 (diagnostic: quiet every N rounds)"
    for prefix, name in _FAMILIES:
        if arm.startswith(prefix):
            return name
    return arm


def best_table(record: dict, metric: str, unit: str) -> str:
    """Each library at its best configuration per cell: the arm whose
    median over runs is lowest (latency) or highest (bandwidth, message
    rate), with its interval and how many configurations it was chosen
    from. Every configuration is in the full table."""
    values = cells(record, metric)
    higher = metric in ("busbw_gbps", "mops")
    families = list(dict.fromkeys(family(arm) for arm, _, _ in values))
    keys = sorted({(op, size) for _, op, size in values})
    ops = {op for op, _ in keys}
    rows = []
    for op, size in keys:
        row = [(f"{op} " if len(ops) > 1 else "") + _size(size)]
        for name in families:
            candidates = [(statistics.median(runs), arm, runs)
                          for (arm, cell_op, cell_size), runs
                          in values.items()
                          if family(arm) == name and cell_op == op
                          and cell_size == size and runs]
            if not candidates:
                row.append("—")
                continue
            median, arm, runs = (max if higher else min)(candidates)
            low, high = bootstrap(runs)
            label = arm.split(":", 1)[1] if ":" in arm else arm
            row.append(f"{median:.2f} [{low:.2f}, {high:.2f}] "
                       f"`{label}` of {len(candidates)}")
        rows.append(row)
    return _table([f"size ({unit})"] + families, rows)


def saturation_table(directory: Path) -> str:
    """Bandwidth by CTA count (files bandwidth_ctas<N>.json) and the fewest
    CTAs reaching 90% of the best any arm achieved at that size."""
    by_ctas = {}
    for stem, path in _records(directory, "bandwidth_ctas"):
        ctas = int(stem.removeprefix("bandwidth_ctas"))
        by_ctas[ctas] = cells(_load(path), "busbw_gbps")
    if not by_ctas:
        return ""
    keys = sorted({key for table in by_ctas.values() for key in table},
                  key=lambda key: (key[2], key[0]))
    peak = defaultdict(float)
    for table in by_ctas.values():
        for (arm, op, size), runs in table.items():
            peak[size] = max(peak[size], statistics.median(runs))
    rows = []
    for arm, op, size in keys:
        row = [f"{arm} {_size(size)}"]
        enough = "—"
        for ctas in sorted(by_ctas):
            runs = by_ctas[ctas].get((arm, op, size))
            value = statistics.median(runs) if runs else None
            row.append(f"{value:.2f}" if value is not None else "—")
            if value is not None and enough == "—" and \
                    value >= 0.9 * peak[size]:
                enough = str(ctas)
        rows.append(row + [enough])
    return _table(["arm, size"] + [f"{c} CTA" for c in sorted(by_ctas)]
                  + ["CTAs for 90% of best"], rows)


def resource_table(directory: Path) -> str:
    """Registers per kernel and library device memory, per arm."""
    registers = defaultdict(dict)
    memory = {}
    for _, path in _records(directory):
        record = _load(path)
        for arm, inner in _runs(record):
            # Python drivers list kernels; the binaries record the one
            # they launched.
            kernels = inner.get("kernels") or [
                kernel for kernel in [inner.get("kernel")] if kernel]
            for kernel in kernels:
                if kernel.get("registers") is None:
                    continue   # cuTile kernels report no register count
                name = (kernel.get("kernel") or inner.get("mode", "?")
                        ).split("[")[0]
                registers[arm][name] = max(registers[arm].get(name, 0),
                                           kernel["registers"])
            library = (inner.get("memory") or {}).get("library_bytes")
            if library is not None:
                world = inner.get("world_size", inner.get("world", "?"))
                memory[(arm, world)] = library
    names = sorted({name for kernels in registers.values()
                    for name in kernels})
    rows = [[arm] + [str(registers[arm].get(name, "—")) for name in names]
            for arm in registers]
    text = _table(["arm"] + [f"{name} registers" for name in names], rows)
    if memory:
        text += "\n\n" + _table(
            ["arm", "world", "library device memory (MiB)"],
            [[arm, str(world), f"{value / (1 << 20):.1f}"]
             for (arm, world), value in sorted(memory.items(),
                                               key=lambda item: str(item))])
    return text


def ablation_table(path: Path) -> str:
    """Rows: configurations in run order; columns: shapes."""
    values = defaultdict(list)
    kernels = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        lines = handle.read().splitlines()
    for line in lines:
        entry = json.loads(line)
        record, config = entry["record"], entry["config"]
        if "ns_per_round" in record:
            shape = f"{record['mode']} {record['chains']}, ns/round"
            values[(config, shape)].append(record["ns_per_round"])
            kernels[config] = record.get("kernel") or {}
        else:
            for row in record.get("results", []):
                shape = f"{row.get('op', 'put')} {_size(row['size'])}, p50 us"
                values[(config, shape)].append(_metric(row, "p50_us"))
            if record.get("kernels"):
                kernels[config] = record["kernels"][0]
    configs = list(dict.fromkeys(config for config, _ in values))
    shapes = list(dict.fromkeys(shape for _, shape in values))
    rows = []
    for config in configs:
        row = [config]
        for shape in shapes:
            runs = values.get((config, shape))
            row.append(f"{statistics.median(runs):.2f} n={len(runs)}"
                       if runs else "—")
        kernel = kernels.get(config, {})
        row.append(str(kernel.get("registers", "—")))
        row.append(str(kernel.get("cubin_bytes", "—")))
        rows.append(row)
    return _table(["configuration"] + shapes + ["registers", "cubin B"], rows)


_TABLES = (
    ("latency", "p50_us", "us", "Round trip, p50", True),
    ("bandwidth", "busbw_gbps", "GB/s", "One-way bandwidth", False),
    ("bandwidth", "mops", "Mops", "Message rate", False),
    ("alltoall", "p50_us", "us", "All-to-all round, p50", True),
    ("alltoall", "busbw_gbps", "GB/s", "All-to-all busbw", False),
    ("decompose", "p50_us", "us", "Decomposition round, p50", True),
    # DSL dispatch (matrix.sh dsl-*, experiments/dsl/dispatch.py).
    ("dsl_latency", "p50_us", "us", "DSLs: round trip, p50", True),
    ("dsl_bandwidth", "busbw_gbps", "GB/s", "DSLs: one-way bandwidth", False),
    ("dsl_bandwidth", "mops", "Mops", "DSLs: message rate", False),
    ("dsl_alltoall", "p50_us", "us", "DSLs: all-to-all round, p50", True),
    ("dsl_alltoall", "busbw_gbps", "GB/s", "DSLs: all-to-all busbw", False),
    ("dispatch", "p50_us", "us", "W4 FP8 dispatch round, p50", True),
    ("moe", "p50_us", "us", "MoE dispatch + combine, p50 (sum)", False),
)


def report(directory: Path) -> str:
    sections = []
    for path in sorted(directory.glob("ablation_*.jsonl*")):
        name = path.name.split(".")[0]
        sections.append(f"### {name}\n\n{ablation_table(path)}")
    for stem, path in _records(directory):
        if stem.startswith("bandwidth_ctas"):
            continue
        record = _load(path)
        failed = failures(record)
        if failed:
            sections.append(f"### Failed correctness ({path.name}): left "
                            f"out of every table\n\n- " + "\n- ".join(failed))
        for prefix, metric, unit, title, tail in _TABLES:
            if stem.startswith(prefix):
                sections.append(f"### {title}, best per library "
                                f"({path.name})\n\n"
                                f"{best_table(record, metric, unit)}")
                sections.append(f"### {title} ({path.name})\n\n"
                                f"{arm_table(record, metric, unit, tail)}")
    saturation = saturation_table(directory)
    if saturation:
        sections.append(f"### Bandwidth by CTA count\n\n{saturation}")
    sections.append(f"### Kernel resources and memory\n\n"
                    f"{resource_table(directory)}")
    return "\n\n".join(sections)


def compare(reference: Path, candidate: Path, tolerance: float) -> int:
    """Latency regresses upward, bandwidth and message rate downward."""
    regressions = 0
    candidates = dict(_records(candidate))
    for stem, path in _records(reference):
        if stem not in candidates:
            continue
        metric = "busbw_gbps" if stem.startswith("bandwidth") else "p50_us"
        before = cells(_load(path), metric)
        after = cells(_load(candidates[stem]), metric)
        for key, runs in sorted(before.items()):
            if key not in after:
                continue
            old = statistics.median(runs)
            new = statistics.median(after[key])
            worse = (new < old * (1 - tolerance) if metric == "busbw_gbps"
                     else new > old * (1 + tolerance))
            if worse:
                regressions += 1
                print(f"{stem} {key[0]} {key[1]} {_size(key[2])}: "
                      f"{old:.2f} -> {new:.2f} {metric}")
    print(f"{regressions} regression(s) beyond {tolerance:.0%}")
    return 1 if regressions else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path,
                        metavar=("REF", "NEW"))
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare, args.tolerance)
    if args.directory is None:
        parser.error("give a result directory or --compare REF NEW")
    print(report(args.directory))
    return 0


if __name__ == "__main__":
    sys.exit(main())
