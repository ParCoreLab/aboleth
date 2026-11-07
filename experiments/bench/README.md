# Tier-1 microbenchmark suite

Head-to-head, workload-free microbenchmarks in the nccl-tests style: every
arm runs the *same* templated benchmark kernels (`kernels/bench_kernels.cuh`
— latency ping-pong, windowed one-way bandwidth, symmetric all-to-all), and
only the ~40-line comm adapter differs (`kernels/comm_*.cuh`). A difference
between arms therefore measures the transport library, not the benchmark.

| `--arm` | what runs | transports (`--transport`) |
|---|---|---|
| `aboleth-generic` | Python host, compiler-generated mixed RDMA+P2P library, ordinary CUDA kernel nvcc-JIT'd per shape | `rdma` (default), `p2p`, `auto`; `proxy` uses its dedicated adapter |
| `aboleth-specialized` | the same path with the exact peer geography baked in; uniform worlds collapse to one backend and mixed worlds use peer-specialized bindings | `rdma`, `p2p`, `auto` |
| `aboleth-cpp` | `build/bin/bench_tier1_aboleth`: C++ host over the runtime C ABI (`include/aboleth/runtime_c.h`) | `rdma` (default), `proxy` |
| `nvshmem` | `build/bin/bench_tier1_nvshmem` (gated on `ABOLETH_HAS_NVSHMEM`) | `ibgda` (default), `ibrc` |
| `nccl-device` | `build/bin/bench_tier1_nccl` (gated on `ABOLETH_HAS_NCCL_GIN`); self-skips with `SKIP:` + exit 0 when the installed NCCL lacks device-API/GIN support | `gdaki` (default), `proxy`; `gdaki-lsa`, `proxy-lsa` route same-node (LSA-team) peers over NVLink load/store and the rest over GIN |

The binary arms take `--put-issue thread|warp` and `--fuse-signal off|on`,
which select the kernels' `kWarpC`/`kFuseC` instantiations exactly as the
Python Aboleth arm bakes them: NVSHMEM maps them to
`nvshmemx_putmem_nbi_warp` and `nvshmem_putmem_signal_nbi`, NCCL GIN to a
`ncclCoopWarp` put and a put carrying its `StrongSignalAdd`. `--ctas N` sizes
the windowed-bandwidth grid (aboleth-cpp clamps to 1). `--gin-contexts N`
(nccl-device) requests at most N GIN contexts instead of one per peer; CTAs
share them round-robin. Every context is fully connected, so one per peer is
peers² QPs per rank: `ncclDevCommCreate` refuses it at world 128. Both
library binaries also take `--put-issue warp --fuse-signal on`
(`nvshmemx_putmem_signal_nbi_warp`, a `ncclCoopWarp` GIN put with its
signal).

Further flags, all recordable as arm fields (`arm:transport:library:flag=value`):

| Flag | Arms | Effect |
|---|---|---|
| `--batch-window on` | Tier 1, nccl-device | windowed bandwidth posts each window as one hand-batched call (`put_many_flush`; NCCL aggregated requests): one doorbell. Off by default: each put is its own call, the plain API |
| `--collective on` | nvshmem all-to-all | NVSHMEM's own `nvshmemx_alltoallmem_block` instead of the shared kernel |
| `--graph-rounds K` | nccl-host | K rounds per CUDA graph; a sample is one graph launch over K |
| `--bake on` | Tier 1 | compile size, put count and op into the kernel (default: run time, like the library kernels) |
| `--exclusive on` | Tier 1 | the exclusive-issuer library build (no reservation atomic) |
| `--env NAME=VALUE` | all | library environment for the run (`env=NAME=VALUE` in an arm token) |
| `--completion drain` | aboleth-triton all-to-all | diagnostic: add a Tier-1-style drain after the wait (before it, the drain queues behind the unsubmitted batch and deadlocks) |
| `--stream on` | nvshmem, nccl-device bandwidth | post every window back to back and complete once at the end (libraries that reclaim their own queue; 2610.01380's message-rate method). Off: one completion per window. The Aboleth tiers refuse it: only their completions free the queue |
| `--issuers T` | Tier 1, nvshmem, nccl-device bandwidth | T threads per CTA (1, or a multiple of 32 dividing `--window`) post the CTA's window at once, each syncing its warp after every put; one quiet per window. Tier 1 needs its non-exclusive build; Tier 2's issuers are its programs (`--ctas`) |

## Running

Every driver takes the same nccl-tests-style flags: `-b/-e` size bounds
(`K/M/G` = 1024-based), `-f` multiplicative step (default 2) or `-i`
additive, `-w/--warmup` (50), `-n/--iters` (50; use ≥1000 for reported runs),
`--op put|put+signal|put_inline` (comma-separated), `--puts N`, `--window N`
(bandwidth), `--l2-flush`, `--ctas N`, `--json-out PATH`, `--smoke`
(2 sizes, 5 iters), `--local-ranks N`.
The compiler-generated mixed library is the default; `--library handwritten`
selects the hand-written RDMA parity oracle explicitly.

```bash
# Local pair on one node (self-spawns 2 ranks):
scripts/aboleth-bench tier1-ladder --arm aboleth-generic --mode latency -b 8 -e 64K -n 1000
scripts/aboleth-bench tier1-ladder --arm aboleth-generic --mode bandwidth -b 4K -e 256M
scripts/aboleth-bench tier1-alltoall --arm aboleth-cpp --local-ranks 4 -b 1K -e 64M

# Interleaved A/B..B/A comparison, one joint JSON with per-run ordering:
TIER1_ARMS="aboleth-generic,aboleth-cpp,nvshmem,nccl-device" \
  scripts/aboleth-bench tier1-compare -n 1000

# A single-point shape: 32 puts of 256 B per round:
scripts/aboleth-bench tier1-ladder --arm aboleth-generic -b 256 -e 256 --puts 32 -n 10000 -w 100
```

Direct invocation without the preset wrapper:
`PYTHONPATH=$PWD/python .venv/bin/python experiments/bench/ladder.py ...`

### Multi-node (SLURM)

The drivers read `MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK`, which
`scripts/multinode-launch.sh` exports per SLURM task. Inside an allocation,
`scripts/run-multinode.sh <nodes> <tasks-per-node> -- <args>` runs the program
named by `ABOLETH_MULTINODE_PROGRAM` on every task:

```bash
salloc -A <account> -p <partition> --nodes=2 --ntasks-per-node=1 --gpus-per-node=4
ABOLETH_MULTINODE_PROGRAM=$PWD/experiments/bench/ladder.py \
  scripts/run-multinode.sh 2 1 -- --arm aboleth-generic --mode latency -b 8 -e 64K \
    -n 1000 --json-out <out-dir>/tier1_ladder.json
```

Platform notes:

- `ABOLETH_RDMA_SUBMIT` unset → probe → **staged** where the driver refuses
  the GPU UAR mapping (the unprivileged path); the fallback notice on stderr
  is expected.
- **NVSHMEM on drivers without `nvidia_peermem`** needs
  `NVSHMEM_IB_DISABLE_DMABUF=0`: DMABUF is then the only GPUDirect
  registration path, and without it init fails with "neither nv_peer_mem, or
  nvidia_peermem detected". `matrix.sh` exports it.
- The NVSHMEM/NCCL HCA and GID defaults (`mlx5_0:1`, GID 3) are soft-set for
  RoCE hosts; on native InfiniBand export `NVSHMEM_IB_GID_INDEX=0` /
  `NCCL_IB_GID_INDEX=0` if the defaults misbehave.
- Compare arms only within one allocation; use `tier1-compare` so the
  interleaved ABBA ordering is recorded.

## Reading the output

One table row per (op, size), identical columns for every arm so results
diff cleanly, then one sorted single-line JSON record:

```
#      size      count     p50_us     p95_us     p99_us    mean_us  algbw_GBs  busbw_GBs      wrong
          8          1      19.33      19.48      19.49      19.30       0.00       0.00          0
```

- **Latency** is a full round trip (payload put(s) + strong signal out,
  signal ack back), sampled per-iteration with the device `%globaltimer`. `algbw` in latency mode is
  one-way payload over the round trip (conservative).
- **Bandwidth** is a windowed persistent kernel (`--window` puts + one
  `quiet` per iteration per CTA); the percentile columns are CTA 0's
  per-window device times. The binary arms time it on the device clock:
  bytes moved by all CTAs over max(end) − min(start) of the per-CTA
  `%globaltimer` spans (`span_ns` in each row).
- **Binary-arm records** carry raw data and resources: every row has
  `samples_ns` (latency round trips, bandwidth CTA-0 windows, alltoall
  rank-0 rounds), bandwidth rows `span_ns`/`ctas`/`window`/`iters`,
  alltoall rows `rank_mean_us`; the record has `kernel` (registers, spill
  and static shared bytes of the exact instantiation, threads, CTAs),
  `memory` (free device memory before library init vs after init and
  connection, before the benchmark arena: `library_bytes`), `versions` and
  `flags`. `wrong` also counts a pattern and inbound-flag re-check after
  the timed launch.
- **Alltoall** sizes are per peer; `algbw = n·S/t` and
  `busbw = algbw·(n−1)/n` (nccl-tests convention), with `t` the slowest
  rank's mean round; percentiles are rank 0's rounds.
- GB/s is decimal (1e9) over 1024-based sizes, as in nccl-tests.
- `wrong` comes from a pattern-checked correctness pass run before every
  timed sweep (sender-salted per-slice in alltoall, so misrouting is caught).
- Anything clamped (sizes vs free memory, window vs queue capacity, ctas)
  is printed as `# clamp: ...` and recorded in the JSON `clamped[]` — never
  silent.
- The `compile` block (aboleth-generic only) prices the per-shape nvcc JIT:
  `first_call_s` cold, `cached_call_s` warm, `variants_compiled` per
  invocation.

## Findings worth knowing before comparing

- Library arms (and Tier 1) map the kernels' operations onto the library
  calls uniformly: one ordered signal everywhere, each put its own call. What
  Aboleth's compiler derives by itself (a weaker signal or no fence where
  nothing precedes it, one doorbell per window) is never hand-applied to a
  baseline; hand-batched windows exist only as labelled `--batch-window on`
  arms for Tier 1 and NCCL. Tier 2 forms the batch from a chain of puts, and
  publishes its payload-free ping-pong echo as a plain write.
- NVSHMEM's quiet is rank-wide, so its all-to-all quiets once per rank per
  round (grid barrier, CTA 0 quiets, barrier) rather than once per peer CTA.
- The bench binaries leave library knobs at their defaults apart from
  resource parity and functional settings; `matrix.sh` runs every arm also
  as a tuned twin (`TUNED=0` skips them). Each record's `library_env` is the
  environment the library ran with.
- `put_inline` carries exactly 8 bytes on the Tier-1 facade; `nccl-device`
  refuses the op (GIN signals are counters, not flag words).
- Multi-CTA rdma needs one endpoint per CTA (`--endpoints-per-peer N`):
  the Tier-1 `quiet` is a single-consumer CQ poll per endpoint.
- `nccl-host` times the host API with a host clock (it is the host API's
  cost); every other arm uses the device `%globaltimer`.

## ctest

`ctest --test-dir build -R bench.tier1` runs `--smoke` for the ladder and
alltoall drivers (labels `bench;smoke`, needs 2 GPUs + mlx5, no sudo).

## The evaluation matrix

`matrix.sh <phase> <dir>` runs the evaluation with every library in its best
configuration, interleaved over repeated seeded orders, inside an existing
allocation and from the tree root. Phases: `ablation`, `ablation-mixed`,
`p2p`, `ctas`, `a2a`, `decompose`, `emission`, `series`, and the DSL
comparison phases `dsl-p2p`, `dsl-ctas`, `dsl-a2a`, `dsl-moe`, `dsl-series`
([`../dsl/README.md`](../dsl/README.md)); the header of `matrix.sh` describes
each one and the node count it needs. `ABOLETH_PLATFORM` labels the records
and is required.

`matrix.sbatch` is its batch form. Account and partition come from the
submitter; `ABOLETH_ENV`, if set, names a site script sourced first (modules,
`CUDA_HOME`):

```bash
ABOLETH_PLATFORM=<label> ABOLETH_COMMIT=$(git rev-parse --short HEAD) \
ABOLETH_ENV=<site-env.sh> \
  sbatch -A <account> -p <partition> --nodes=2 \
    experiments/bench/matrix.sbatch p2p <out-dir>
```

The script requests four GPUs and four tasks per node; adjust its `#SBATCH`
lines or override them on the command line for other node shapes, and set
`MATRIX_GPU_NAME` (default `H100`) and `MATRIX_GPUS_PER_NODE` (default 4) for
its GPU inventory check. It checks first that every task reaches its GPU (it names the hosts that do not and
exits 3, since one missing rank stalls every bootstrap), then measures the
fabric the job holds (`linkcheck.sh`: port rates, perftest bandwidth from GPU
and host memory on every GPU-HCA pair, 8 B latency; LINK lines in the job log,
raw output under `<dir>/linkcheck/<job>/`; `LINKCHECK_GATE=1` stops the job on
a degraded link). An interleaved arm that outlives `--arm-timeout` (1200 s) is
killed with its children and recorded with `"exit": "timeout"`.

`report.py <dir>` turns a matrix output directory into tables built from the
raw samples (medians over runs with bootstrap intervals, pooled p99, message
rate, CTAs to saturate, registers, library memory), and
`report.py --compare REF NEW` checks a run against a stored reference.

No measurements ship in this repository; the paper reports the results.
