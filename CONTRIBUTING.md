# Contributing to Aboleth

Aboleth is an optimizing compiler for GPU communication: it lowers semantic
transfer operations into program-specific transport code. Contributions are
welcome as issues and pull requests against `main`. This file covers where
things live, what to read before changing them, how to test a change, and the
invariants a change must keep.

By contributing you agree that your contribution is licensed under the
Apache License 2.0 (see [`LICENSE`](LICENSE)).

## Read first

1. [`README.md`](README.md) — pipeline shape and vocabulary (put/signal/wait,
   Queue IR, transports).
2. [`docs/PROGRAMMING_MODEL.md`](docs/PROGRAMMING_MODEL.md) — the semantic
   contract callers and backends must honour.
3. [`docs/COMPILER_SEMANTICS.md`](docs/COMPILER_SEMANTICS.md) — which IR
   transformations are legal. Reviews hold changes to it.
4. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — where in the pipeline a
   change belongs, and the ownership boundary the compiler must not cross.
5. [`docs/TIERS.md`](docs/TIERS.md) — the two tiers and framework adapters;
   it decides what the compiler may specialize for a given client.
6. [`docs/WORLD.md`](docs/WORLD.md) — for anything multi-rank.

| Doc | Read when |
| --- | --- |
| [`docs/EFFECT_ANALYSIS.md`](docs/EFFECT_ANALYSIS.md) | touching `aboleth-plan-transport` or effect schemas |
| [`docs/RDMA_ABI.md`](docs/RDMA_ABI.md) / [`docs/PROXY_ABI.md`](docs/PROXY_ABI.md) | mlx5 descriptor layout / proxy queue wire format |
| [`docs/EFA_ABI.md`](docs/EFA_ABI.md) | GPU-initiated EFA, including its platform status |
| [`docs/EFA_PROXY.md`](docs/EFA_PROXY.md) | host-proxy EFA over libfabric |
| [`docs/transport-control.md`](docs/transport-control.md) | endpoint configuration and its environment variables |
| [`docs/COMPILED_MOE.md`](docs/COMPILED_MOE.md), [`docs/VLLM_INTEGRATION.md`](docs/VLLM_INTEGRATION.md) | the MoE path and the vLLM integration |
| [`docs/KV_CACHE.md`](docs/KV_CACHE.md) | KV-cache migration (experimental) |

## Code map

| Path | What |
| --- | --- |
| `compiler/` | MLIR `aboleth` dialect and passes (C++17). Ops in `compiler/include/aboleth/IR/AbolethOps.td`, verifiers in `compiler/lib/IR/`, passes in `compiler/lib/Transforms/` (listed in `docs/COMPILER_SEMANTICS.md` § Optimization passes) |
| `compiler/lib/Target/{P2P,RDMA,Proxy,EFA,Ionic}` | backend lowerings |
| `compiler/Plugin.cpp` | the Triton seam: `tritonGetPluginInfo()` registers the dialect, the passes and the `aboleth_*` builtins. Triton dlopens `libAbolethPlugin.so`; Triton is not forked |
| `python/aboleth/` | public API: `language.py` (Triton builtins), `runtime.py` (`Runtime`, `Transport`, endpoints), `world.py` (multi-rank), `planning.py`, `pattern.py`, `phase.py`, `compiler.py` (pass switches), `tier1.py`/`device_library.py` (generated Tier-1 library), `frontends/` (Triton and CuTe DSL bindings), `moe/`, `kv/`, `integrations/` |
| `include/aboleth/` | C ABI (`runtime_c.h`), the Tier-1 device header, and the wire-format and device headers shared by compiler and runtime (`Target/`) |
| `src/runtime/` | host runtime `libaboleth_rdma_abi.so`: `core/` (mlx5 endpoints, memory, device selection, doorbell agent), `proxy/`, `efa/`, `hip/` (AMD, experimental), `dl/` (dlopen shims for every transport library) |
| `src/python/runtime_bindings.cpp` | pybind11 module behind `runtime.py` |
| `test/compiler/` | `.mlir` pass tests, including negative verifier cases, and the golden sweep (`goldens.py`) |
| `test/python/` | unit tests, one CTest per module |
| `test/lint/` | link-hygiene checks and the UAR capability probe |
| `experiments/` | hardware drivers, probes and benchmarks ([`experiments/README.md`](experiments/README.md)) |
| `integrations/vllm/` | patches against vLLM |
| `scripts/` | `run-gated.sh` (hardware gating for CTest), `aboleth-bench` (benchmark presets), `build-triton.sh`, multi-node launchers |
| `repro/` | pinned environment manifests and `bootstrap.sh` |

## Building

Build locally; see [`docs/BUILDING.md`](docs/BUILDING.md) and
[`repro/README.md`](repro/README.md). In short:

```bash
git submodule update --init --recursive
./repro/bootstrap.sh            # pinned venv, Triton, full build
cmake --build build             # afterwards
```

## Testing

Fast loop, no GPU needed — run this before anything else:

```bash
ctest --test-dir build -L python        # unit tests
ctest --test-dir build -L compiler-ir   # MLIR pass suite over test/compiler/*.mlir
ctest --test-dir build -R goldens.check # IR drift (see below)
```

`ctest --preset quick` runs the compiler, goldens, Python and lint labels
together; `ctest --preset smoke` adds the unprivileged GPU smokes.

GPU and NIC tests run through `scripts/run-gated.sh`, which reports a test as
skipped when the machine lacks what it needs (GPUs, an HCA, mlx5, EFA,
privilege). The direct-mlx5 `hardware.*` tests map the NIC doorbell into the
GPU, which needs root or a driver with `PeerMappingOverride=1`
(`test/lint/probe_uar.py` decides); their `-staged` twins run unprivileged.
The EFA tests need EFA hardware and a build with `ABOLETH_ENABLE_EFA`.

### Goldens: order matters

`.goldens/` is gitignored, so there is no committed baseline, and
`goldens.check` is only registered once a baseline exists. Record one before
your change, re-run CMake so the check is registered, then compare:

```bash
cmake --build build --target aboleth-goldens-write   # BEFORE your change
cmake build                                          # registers goldens.check
# ...make the change...
cmake --build build
ctest --test-dir build -R goldens.check              # fails on the first byte of drift
```

## Rules a change must not break

1. **No cross-backend drift.** Changing a lowering must not change what the
   other backends emit; the golden sweep runs every test module through every
   pipeline that accepts it and fails on the first byte of drift.
2. **Epoch monotonicity.** Epochs in a cell only increase and receivers wait
   for *at least* an epoch; a checked publication (CAS) replaces exactly the
   value it expects (`docs/COMPILER_SEMANTICS.md`).
3. **`submit` verifier.** Every token must belong to the same endpoint, and the
   commands form segments: a chain of posted puts, each depending on the one
   before, optionally closed by a signal that depends on the immediately
   preceding transfer (`docs/COMPILER_SEMANTICS.md`).
4. **Stay inside the compiler boundary** (`docs/ARCHITECTURE.md` § Ownership
   boundary). Aboleth does not schedule compute, assign streams, discover tile
   overlap, or replace optimized kernels.
5. **Back performance changes with evidence.** A pull request that claims a
   speedup states its correctness scope, the matched baseline it was measured
   against, the platform, and why it helps. Keep optional optimizations behind
   an `ABOLETH_PASSES` switch (`python/aboleth/compiler.py`) so they can be
   ablated.
6. **Baselines stay unmodified.** Comparison libraries (DeepEP, NVSHMEM, NCCL)
   are used as upstream ships them; client-side compatibility changes live in
   patches such as `integrations/vllm/0002-*.patch`, never in the library.
