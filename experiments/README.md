# Experiments

Test drivers, hardware probes and benchmarks for Aboleth. Nothing here is a
stable API or an installable library: public runtime and Python APIs live in
the top-level `include/`, `src/` and `python/` trees, and the compiler in
`compiler/`.

## Layout

- `python/` — Python drivers for the frontend, runtime, transports and
  workloads. Most are registered as CTest tests by `test/CMakeLists.txt`
  (labels `smoke`, `hardware`, `efa`); see below.
- `cuda/` — hand-written CUDA/mlx5 probes built by `experiments/CMakeLists.txt`
  when verbs and mlx5 are available, plus the EFA smoke and the Tier-1
  differential's CUDA arm.
- `bench/` — the Tier-1/Tier-2 microbenchmark suite, its comparison arms
  (NVSHMEM, NCCL GIN and host NCCL), the evaluation matrix (`matrix.sh`,
  `matrix.sbatch`) and its report generator. See
  [`bench/README.md`](bench/README.md).
- `dsl/` — the same benchmark kernels transcribed into CuTe DSL, Triton,
  Triton-distributed and cuTile, and a MoE dispatch/combine comparison. See
  [`dsl/README.md`](dsl/README.md).

The root `CMakeLists.txt` drives every build (`ABOLETH_ENABLE_EXPERIMENTS`);
binaries land in `build/bin/` and libraries in `build/lib/`.

## Registered tests

`test/CMakeLists.txt` runs the `python/` scripts and `cuda/` probes through
`scripts/run-gated.sh`, which reports a test as skipped (exit 77) when the
machine lacks a prerequisite (GPUs, an HCA, an mlx5 or EFA device, or the
privilege to map the NIC doorbell).

| Label | Needs | Examples |
|---|---|---|
| `smoke` | GPU(s); an HCA for the proxy tests; no privileges | `smoke.triton`, `smoke.frontend`, `smoke.p2p-triton`, `smoke.proxy-triton`, `smoke.kv-cache`, `smoke.moe-phase`, `smoke.multiprocess-p2p` |
| `hardware` | GPU(s) and mlx5; the direct-doorbell tests need root or an unprivileged UAR mapping, their `-staged` twins do not | `hardware.rdma-abi(-staged)`, `hardware.rdma-triton(-staged)`, `hardware.tier1-differential(-staged)`, `hardware.tier2-dispatch-p2p`, `hardware.tier2-dispatch-staged`, `hardware.world-multinode-local`, `hardware.pingpong-rdma` |
| `efa` | GPU(s) and an EFA device | `efa.staged-smoke`, `efa.host-post-smoke`, `efa.host-smoke`, `efa.proxy-smoke` |
| `bench` | GPU(s) and mlx5 | `bench.rdma-benchmark`, `bench.tier1-ladder-smoke`, `bench.tier1-alltoall-smoke` |

`ctest --test-dir build -N -L <label>` lists the registered tests.

Scripts that are not tests:

- `python/aboleth_rdma_benchmark.py` — persistent direct-mlx5 ping-pong
  (`scripts/aboleth-bench rdma-benchmark`).
- `python/aboleth_moe_exchange.py` — a MoE-shaped exchange over RDMA or the
  host proxy (`scripts/aboleth-bench moe-exchange`, `proxy-moe-exchange`; also
  registered as `hardware.moe-exchange`).
- `python/aboleth_moe_fused_pack.py` — MoE dispatch with packing fused into
  the send program against separate pack and send kernels; the Aboleth arm of
  `dsl/moe` builds on it.
- `python/aboleth_world_multinode.py` — the multi-node World smoke; run it
  across nodes with `scripts/run-multinode.sh` (single-node form:
  `hardware.world-multinode-local`).

## Benchmark presets

`scripts/aboleth-bench --list` prints the presets: `moe-exchange`,
`proxy-moe-exchange`, `rdma-benchmark`, `tier1-ladder`, `tier1-alltoall` and
`tier1-compare`. Each names a runner and its parameters; anything can be
overridden from the environment, and unrecognized arguments pass through to
the runner. Presets that map the NIC doorbell directly run under `sudo -n`
unless `--no-sudo` or `ABOLETH_BENCH_NO_SUDO=1` is given:

```bash
BENCH_ROUNDS=10000 BENCH_TRANSFERS=32 BENCH_BYTES=256 scripts/aboleth-bench rdma-benchmark
MOE_ITERATIONS=64 scripts/aboleth-bench moe-exchange
scripts/aboleth-bench tier1-ladder --mode bandwidth -b 4K
```

No measurements ship in this repository; the paper reports the results.
