#!/usr/bin/env python3
"""Per-optimization ablation of Tier-2 kernels.

Runs one workload once per optimization configuration, each from an empty
Triton cache, and appends every JSON record the workload prints to one JSONL
file, tagged with its configuration. Two sweeps:

  waterfall  everything off, then each optimization added in pipeline order,
             then deferral on top of the defaults;
  loo        the defaults, and the defaults with each optimization removed;
  emission   the RDMA lowering's descriptor-load and WQE-store options:
             both off, each alone, and both.

Configurations run --repeats times: forward, reverse, then seeded random
orders, so drift over the run shows up as spread between passes.

The workload command follows `--`; the driver sets ABOLETH_PASSES and
TRITON_CACHE_DIR around it. For example:

  ablation.py --sweep waterfall --out stream.jsonl -- \\
      python experiments/bench/stream.py --modes pipelined --chains 1 4 16 \\
      --exclusive
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from aboleth.compiler import OPTIMIZATIONS  # noqa: E402

# Pipeline order; the opt-in ones run on top of the defaults.
DEFAULTS = [name for name, on in OPTIMIZATIONS.items() if on]
OPT_IN = [name for name, on in OPTIMIZATIONS.items() if not on]


def configurations(sweep: str) -> list[tuple[str, str]]:
    """(label, ABOLETH_PASSES) pairs."""
    if sweep == "waterfall":
        configs = [("none", ",".join(f"-{name}" for name in DEFAULTS))]
        for count, name in enumerate(DEFAULTS, 1):
            configs.append((f"+{name}", ",".join(
                f"-{later}" for later in DEFAULTS[count:])))
        return configs + [(f"+{name}", f"+{name}") for name in OPT_IN]
    if sweep == "emission":
        # Absolute settings, whatever the defaults are.
        return [("scalar", "-invariant-descriptors,-vector-wqe-stores"),
                ("+invariant-descriptors",
                 "+invariant-descriptors,-vector-wqe-stores"),
                ("+vector-wqe-stores",
                 "-invariant-descriptors,+vector-wqe-stores"),
                ("+both", "+invariant-descriptors,+vector-wqe-stores")]
    return [("default", "")] + [(f"-{name}", f"-{name}")
                                for name in DEFAULTS]


def run(label: str, passes: str, command: list[str], out, index: int,
        timeout: float) -> int:
    cache = tempfile.mkdtemp(prefix="aboleth-ablation-")
    environment = {**os.environ, "ABOLETH_PASSES": passes,
                   "TRITON_CACHE_DIR": cache}
    # Under a multi-node launcher every rank runs this loop; a fresh port per
    # run keeps one run's rendezvous from meeting the last one's sockets.
    if environment.get("RANK") is not None and environment.get("MASTER_PORT"):
        environment["MASTER_PORT"] = str(
            int(environment["MASTER_PORT"]) + 1 + index)
    lead = environment.get("RANK") in (None, "0")
    if lead:
        print(f"# {label}: ABOLETH_PASSES={passes or '(defaults)'}",
              flush=True)
    child = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate()
        if lead:
            print(f"# {label}: killed after {timeout:.0f} s", flush=True)
    result = subprocess.CompletedProcess(command, child.returncode, stdout,
                                         stderr)
    shutil.rmtree(cache, ignore_errors=True)
    if not lead:
        return result.returncode
    sys.stderr.write(result.stderr[-4000:] if result.returncode else "")
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            record = json.loads(line)
            out.write(json.dumps({"config": label, "passes": passes,
                                  "record": record}, sort_keys=True) + "\n")
            out.flush()
        elif line.strip():
            print(f"  {line}", flush=True)
    if result.returncode:
        print(f"# {label}: exit {result.returncode}", flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", choices=("waterfall", "loo", "emission"),
                        default="waterfall")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1200,
                        help="seconds before a run is killed")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the workload command after --")
    configs = configurations(args.sweep)
    shuffled = random.Random(args.seed)
    order = []
    for repeat in range(args.repeats):
        order += (configs if repeat == 0 else configs[::-1] if repeat == 1
                  else shuffled.sample(configs, len(configs)))
    code = 0
    lead = os.environ.get("RANK") in (None, "0")
    if lead:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out if lead else os.devnull, "a") as out:
        for index, (label, passes) in enumerate(order):
            code = run(label, passes, command, out, index,
                       args.timeout) or code
    # Under srun --kill-on-bad-exit a failing non-lead rank would kill the
    # lead mid-sweep; the lead's records say which runs failed.
    return code if lead else 0


if __name__ == "__main__":
    sys.exit(main())
