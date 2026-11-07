# Aboleth inside other kernel DSLs

How each adoption tier reaches kernels written in other DSLs, measured against
what those DSLs' users do today. This directory holds the transcriptions: `experiments/bench`'s three kernels
(ping-pong, windowed bandwidth, all-to-all) written once per DSL over a
communication *policy*, so every arm of a DSL runs the same kernel and only
the library differs. The bench drivers (`ladder.py`, `alltoall.py`) run them
as ordinary arms: same sweeps, correctness checks before and after timing,
device-clock samples, kernel resources and records as the CUDA arms.

| Arm | DSL | Library | How it links |
|---|---|---|---|
| `cute-tier1` | CuTe DSL 4.8.1 | generated Tier-1 library | per-op bitcode through `cute.extern` (`aboleth.frontends.cute`) |
| `cute-nvshmem` | CuTe DSL 4.8.1 | NVSHMEM 3.8.0 | NVSHMEM4Py 0.4.1's CuTe device bindings |
| `triton-tier1` | Triton 3.8 (series) | generated Tier-1 library | per-op bitcode through `extern_libs` (`aboleth.frontends.triton.tier1`) |
| `triton-nvshmem` | Triton 3.8 (series) | NVSHMEM 3.8.0 | NVSHMEM's device bitcode through `extern_libs` + `nvshmemx_cumodule_init`, as Triton-distributed links it |
| `aboleth-triton` | Triton 3.8 | Tier 2 | communication in the kernel IR (`experiments/bench/tier2.py`) |
| `nvshmem38` | CUDA | NVSHMEM 3.8.0 | `bench_tier1_nvshmem` rebuilt against 3.8.0, to separate the DSL effect from the version effect (the series' `nvshmem` arm is 3.4.5); see below |
| `tdist-nvshmem` | Triton-distributed (Triton 3.7.1) | NVSHMEM 3.6.5 | its own `libshmem_device` primitives, compiled by `triton_dist.jit` |
| `cutile-*` | cuTile 1.7 | peer arrays, or a separate send kernel | `cutile/`: an NVLink all-to-all over peer-mapped arrays, and a cuTile pack followed by Aboleth's Tier-2 send or NCCL's `all_to_all_single` |

Every policy is the CUDA adapter call for call: Tier 1 is
`kernels/comm_aboleth.cuh` (generic RDMA+P2P library, kind-bit dispatch),
NVSHMEM is `kernels/comm_nvshmem.cuh` (`putmem_nbi`; fence + `signal_op(SET)`;
`putmem_signal_nbi`, fenced behind earlier puts; `uint64_p`; PE-wide quiet,
retired once per grid; `signal_wait_until(GE)`). One thread per CTA issues
(`if tid == 0`, Triton-distributed's idiom in Triton). Sizes, put counts and
the op are compile-time constants, as in the Tier-2 arm.

Triton-distributed (its own Triton 3.7.1 and plugin) runs in its own
environment (`envs/tdist`): its plugin cannot share a process with the
series' Triton and the Aboleth plugin, so it is compared by records, not
processes.

## Layout

- `cute/cute_bench.py`, `triton_dsl/triton_bench.py`: the kernels.
- `cute/cute_launch.py`, `triton_dsl/triton_launch.py`: launchers with
  `tier2.py`'s protocol.
- `cute/cute_{tier1,nvshmem}.py`, `triton_dsl/triton_{tier1,nvshmem}.py`: the
  arms (`common.KERNEL_ARMS`).
- `nvshmem_world.py`: NVSHMEM's world behind the drivers' World surface, set
  up as `bench_tier1_nvshmem.cu` does.
- `triton_dsl/nvshmem_shim.ll`: NVSHMEM at upstream Triton's extern ABI (scalar
  arguments, an i64 result), always-inlined.
- `tdist/tdist_nvshmem.py`: the Triton-distributed arm over the same kernels.
- `cutile/`: the cuTile arms (`cutile_p2p.py`, `cutile_dispatch.py`).
- `dispatch.py`, `triton_dsl/dispatch_kernels.py`: FP8 MoE dispatch with each
  library.
- `moe/`: MoE dispatch and combine per family (`moe.py`, one `arm_*.py` per
  implementation: Aboleth, Triton-distributed's EP layers, DeepEP v2 as a
  baseline, cuTile) behind one contract and timer (`moe_common.py`).
- `*/compile_check.py`: compile every kernel of every policy for sm_90a
  without a GPU.
- `setup-envs.sh`: the DSL environments.

## Running

Build the tree and its environment first (`repro/bootstrap.sh`), then create
the DSL environments, layered on `.venv`:

```bash
experiments/dsl/setup-envs.sh cute     # CuTe DSL 4.8.1, NVSHMEM4Py 0.4.1, cuTile 1.7
experiments/dsl/setup-envs.sh tdist    # Triton-distributed 7908e4e
```

They land under `ABOLETH_DSL_ROOT` (default `.dsl`). Check that every kernel
compiles, without a GPU:

```bash
CUTE_DSL_ARCH=sm_90a .dsl/envs/cute/bin/python experiments/dsl/cute/compile_check.py
.dsl/envs/cute/bin/python experiments/dsl/triton_dsl/compile_check.py
.dsl/envs/tdist/bin/python experiments/dsl/tdist/compile_check.py
```

The measurements are `experiments/bench/matrix.sh` phases, run inside an
allocation (or through `matrix.sbatch`, see
[`../bench/README.md`](../bench/README.md)), interleaved with the CUDA arms
and Tier 2:

```bash
export ABOLETH_DSL_PYTHON=$PWD/.dsl/envs/cute/bin/python
export ABOLETH_TDIST_PYTHON=$PWD/.dsl/envs/tdist/bin/python
export ABOLETH_NCCL_RUNTIME=<NCCL 2.31 lib directory>   # NCCL reference arms
ABOLETH_PLATFORM=<label> experiments/bench/matrix.sh dsl-p2p <dir>
ABOLETH_PLATFORM=<label> experiments/bench/matrix.sh dsl-ctas <dir>
ABOLETH_PLATFORM=<label> NODES=2 experiments/bench/matrix.sh dsl-a2a <dir>
ABOLETH_PLATFORM=<label> NODES=2 experiments/bench/matrix.sh dsl-moe <dir>
ABOLETH_PLATFORM=<label> NODES=4 experiments/bench/matrix.sh dsl-series <dir>
```

`DSL_REF=0` leaves the series' own arms out. The `deepep` MoE arm imports a
DeepEP v2.5.0 checkout built for the DSL environment
(`ABOLETH_DEEPEP_ROOT`). The `nvshmem38` arm expects `build/bin/bench_tier1_nvshmem38`:
`experiments/bench/bench_tier1_nvshmem.cu` built against NVSHMEM 3.8.0 with
`-DBENCH_NVSHMEM_COLLECTIVE_THREADS=256`; CMake does not build it.

`ABOLETH_MULTINODE_PYTHON` selects the interpreter `scripts/multinode-launch.sh`
runs; the DSL environment layers on the series venv (`setup-envs.sh`), so one
interleaved sweep runs the CUDA arms and the DSL arms side by side.
