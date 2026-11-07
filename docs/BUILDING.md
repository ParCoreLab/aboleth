# Building and testing Aboleth

This page collects the requirements, build configuration and test suite. The
[README](../README.md) has the overview and a quick start.

## Requirements

- **GPU**: NVIDIA Hopper (`sm_90`, H100/H200) or Blackwell (`sm_100`, B200).
  Device bitcode defaults to `sm_90a`; set `ABOLETH_RDMA_BITCODE_ARCHS` and
  `ABOLETH_EFA_BITCODE_ARCHS` for other deployment targets.
- **Toolchain**: Linux x86_64, CUDA toolkit (the pinned environment uses
  12.8), GCC/G++ with C++17, CMake ≥ 3.24, Ninja, Python 3.12, Git, curl.
  `repro/bootstrap.sh` installs CMake, Ninja, Python and the pinned packages
  into the project.
- **Triton**: the `third_party/triton` submodule, an unmodified upstream
  checkout built with extension support (`scripts/build-triton.sh`).
- **Optional transports**: rdma-core with mlx5 (verbs and direct RDMA),
  libfabric with the EFA provider and the `third_party/efa-dp-direct`
  submodule (EFA), ROCm/HIP (AMD, experimental). NVSHMEM and NCCL are needed
  only for the comparison benchmarks.
- **Privileges**: GPU-rung mlx5 doorbells (`ABOLETH_RDMA_SUBMIT=direct`) map
  the NIC's doorbell page into the GPU, which needs root or an NVIDIA driver
  loaded with `PeerMappingOverride=1` (`repro/configure-efa-gda-driver.sh`
  installs the module options). Without either, the runtime falls back to
  *staged* submission: the kernel still builds every work request and polls
  completions, and a host thread only forwards the doorbell. Staged
  submission, the host proxy and P2P need no privileges.

The build needs none of the GPU or transport pieces to compile the compiler
core (`--preset compiler-only`).

## Backends

| Backend | `Transport` | Status | Notes |
|---|---|---|---|
| mlx5 RDMA, GPU-rung doorbell | `RDMA` (`ABOLETH_RDMA_SUBMIT=direct`) | supported | needs root or `PeerMappingOverride=1` |
| mlx5 RDMA, staged doorbell | `RDMA` (`ABOLETH_RDMA_SUBMIT=staged`) | supported | unprivileged; the default fallback |
| Host proxy (verbs or EFA libfabric) | `PROXY` | supported | GPU enqueues commands, a host worker posts them |
| NVLink / peer memory | `P2P` | supported | same-node peers |
| AWS EFA (SRD) | `EFA` | implemented; evaluation is future work | staged submission on p5-class hosts ([`docs/EFA_ABI.md`](EFA_ABI.md)) |
| AMD MI355X + Pensando Ionic | `IONIC` | experimental, preliminary | HIP runtime and GPU-initiated Ionic backend (`ABOLETH_GPU_BACKEND=HIP`); not evaluated |

`aboleth.runtime.provider_caps()` reports which transports the loaded runtime
can use on the current machine.

## Repository layout

| Path | Contents |
|---|---|
| `compiler/` | MLIR `aboleth` dialect, passes, backend lowerings, and the Triton plugin entry (`Plugin.cpp`) |
| `python/aboleth/` | Python API: frontend builtins, `Runtime`/`World`, patterns and planning, Tier-1 library generation, MoE and KV clients |
| `include/aboleth/` | C ABI (`runtime_c.h`), Tier-1 device header, shared wire-format and device headers |
| `src/runtime/` | host runtime `libaboleth_rdma_abi.so` (mlx5, proxy, EFA, HIP) |
| `src/python/` | pybind11 bindings for the runtime |
| `integrations/vllm/` | patches that add Aboleth as a vLLM all-to-all backend |
| `test/` | MLIR pass tests, golden sweep, Python unit tests, lint checks, hardware test registration |
| `experiments/` | hardware drivers, probes, and benchmarks ([`experiments/README.md`](../experiments/README.md)) |
| `scripts/` | Triton build, CTest hardware gating, benchmark presets, multi-node launchers |
| `repro/` | pinned environment and `bootstrap.sh` |
| `docs/` | design and ABI documentation |
| `third_party/` | Triton and efa-dp-direct submodules |

## Build

`repro/bootstrap.sh` creates a project-local Python environment, builds the
Triton submodule, and configures and builds everything else:

```bash
git submodule update --init --recursive
./repro/bootstrap.sh
```

Afterwards CMake and CTest drive the tree directly. Three configure presets
exist:

| Preset | Build directory | What it builds |
|---|---|---|
| `default` | `build/` | every component whose dependencies are found |
| `full` | `build/` | Triton plugin, runtime, verbs, Python and experiments forced on; a missing dependency is a configure error (EFA stays `AUTO`) |
| `compiler-only` | `build-compiler-only/` | the compiler core from C++17 and MLIR alone: no CUDA, Triton, verbs, libfabric or tests |

```bash
cmake --preset default     # or: --preset full, --preset compiler-only
cmake --build build
ctest --test-dir build -L compiler-ir
```

The presets expect the project environment at `.venv/`.

### Configuration

Nothing is compiled in about where dependencies live. Each is a cache
variable, and the end-of-configure summary prints what was found and what was
skipped:

```bash
cmake --preset default \
  -DRdmaCore_ROOT=/opt/rdma-core \
  -DLibfabric_ROOT=/opt/amazon/efa \
  -DTriton_ROOT="$PWD/build/deps/triton"
```

| Variable | Meaning |
|---|---|
| `ABOLETH_ENABLE_{TRITON_PLUGIN,RUNTIME,VERBS,EFA,PYTHON,EXPERIMENTS}` | components; `AUTO` (default), `ON` (a missing dependency is a configure error) or `OFF` |
| `ABOLETH_ENABLE_{NVSHMEM,NCCL_GIN}_BENCHMARKS` | comparison benchmarks, same tri-state |
| `ABOLETH_ENABLE_TESTS` | register the CTest suite (`ON`) |
| `ABOLETH_GPU_BACKEND` | `auto`, `CUDA`, `HIP` or `NONE` |
| `Triton_ROOT`, `RdmaCore_ROOT`, `Libfabric_ROOT`, `NVSHMEM_ROOT`, `NCCL_ROOT` | dependency install prefixes |
| `ABOLETH_BUILD_TRITON` | build Triton from `third_party/triton` when no prefix is found (`OFF`) |
| `ABOLETH_RDMA_BITCODE_ARCHS`, `ABOLETH_EFA_BITCODE_ARCHS` | GPU architectures for the device bitcode (`sm_90a`) |
| `ABOLETH_TIER1_GENERATED` | generate the Tier-1 device library at build time (`ON`) |
| `ABOLETH_GOLDEN_DIR` | where compiler goldens live (`.goldens/`) |

The compiler core builds with every component off, on a machine with no GPU,
no Triton and no RDMA stack:

```bash
cmake --preset compiler-only && cmake --build build-compiler-only
```

The runtime never links its transport libraries. libibverbs, libmlx5, libefa,
libfabric and the CUDA driver are all dlopened on first use, so one
`libaboleth_rdma_abi.so` serves machines carrying any subset of them.
Building with `ABOLETH_ENABLE_EFA=OFF` — or on a host without an EFA-capable
libfabric — compiles the EFA control plane out entirely; the EFA entry points
then fail with "EFA support was not compiled into this runtime" instead of the
library failing to load.

CMake consumes a Triton install prefix (`scripts/build-triton.sh`, or
`-DTriton_ROOT=`). Its `lib/libtriton.so` must be a symlink to the
interpreter's `triton/_C/libtriton.so`: a separate copy gives the process two
MLIRs and breaks every TypeID comparison across the plugin boundary
(`lint.single-libtriton` checks it). The build stages the plugin and runtime
into `python/aboleth/` on every build; scripts run outside CTest load that
copy.

At run time, `ABOLETH_PASSES` switches optional compiler passes
(`-name` off, `+name` on; see
[`docs/COMPILER_SEMANTICS.md`](COMPILER_SEMANTICS.md)). The defaults are
the same on every target; on B200 (SM100), `ABOLETH_PASSES=-invariant-descriptors`
selects the RDMA lowering's scalar descriptor loads, which can be faster there.

## Testing

Test labels: `compiler-ir`, `goldens`, `python`, `lint`, `smoke`,
`hardware`, `efa`, `bench`. Test presets `quick` (compiler, goldens, Python,
lint), `smoke` (adds unprivileged GPU smokes) and `all` group them. Tests that
need hardware run through `scripts/run-gated.sh` and report as skipped when
the machine lacks a GPU, an HCA, the right NIC, or the privilege they need.

Changing a lowering must not change what the other backends emit. The golden
sweep runs every test module through every pipeline that accepts it, across
all backends and their environment-selected variants, and fails on the first
byte of drift. Goldens are not committed; record a baseline before a change
(re-run CMake once so `goldens.check` is registered):

```bash
cmake --build build --target aboleth-goldens-write     # record the current output
ctest --test-dir build -R goldens.check                # fail on any change
```

Hardware tests require a matching GPU/NIC platform; the direct-doorbell
`hardware.*` tests need the privileges above, and their `-staged` twins do
not:

```bash
ctest --test-dir build -R smoke.frontend
ctest --test-dir build -R smoke.p2p-triton
ctest --test-dir build -R hardware.rdma-triton-staged
ctest --test-dir build -R hardware.deferred-lifecycle
```

[`CONTRIBUTING.md`](../CONTRIBUTING.md) describes the development loop and the
invariants a change must keep.

## Limitations and future work

- **EFA and AMD evaluation.** The EFA backends and the AMD MI355X/Ionic
  backend are implemented but not evaluated. GPU-rung EFA doorbells are
  platform-gated on p5-class instances, where staged submission is the
  default.
- **KV-cache migration is experimental**: two ranks over P2P, one calibrated
  hardware profile.
- **Cross-kernel placement.** Every kernel completes its own transport and
  rings its own doorbell; deferring completion or doorbells across kernels,
  derived from the member kernels' IR, is future work.
- **Lifetime-scheduled synchronization**, such as receiver credits issued
  only on slot reuse and partial completion intervals, is not derived yet.
- **Tier-2 export.** Exporting a user's Tier-2 kernel as an inline fragment,
  device function or standalone kernel is not implemented.
- **No host-side Tier-1 call API** (`aboleth.put(...)` from host code).
- **API gaps**: no RDMA read (`get`) or additive signals; no device-side
  Tier-1 collectives; one queue pair per endpoint (`qps_per_peer`,
  `sm_map_policy` and DC queue pairs are refused,
  [`docs/transport-control.md`](transport-control.md)).
- A missed compare-and-swap in checked publication completes silently, and
  no backend bounds the `wait_signal` spin.
