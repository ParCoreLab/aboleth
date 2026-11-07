# Reproducible environment

This directory pins the user-space environment Aboleth is built and tested
with, and the script that recreates it.

## Included

- `bootstrap.sh`: recreates the pinned environments and builds everything
- `environment.json`: machine-readable manifest of the reference host and
  package environment
- `requirements.freeze.txt`: exact uv/Python package resolution
- `requirements.base.txt`: direct dependencies used to resolve the environment
- `gin-requirements.freeze.txt`: exact NCCL 2.30.7 GIN comparison environment
- `gin-requirements.base.txt`: resolver-consistent Torch nightly environment
  before the NCCL override
- `efa-requirements.txt`: packages for the separate EFA environment
  (`bootstrap.sh --efa`)
- `configure-efa-gda-driver.sh`: installs NVIDIA kernel-module options
  (`PeerMappingOverride=1`, `NVreg_EnableStreamMemOPs=1`) that let the GPU map
  NIC doorbell pages without root; takes effect after a reboot
- `bootstrap-gdrcopy.sh`: builds a pinned GDRCopy for the NCCL GIN baseline on
  EFA
- `SHA256SUMS`: checksums for the reproducibility files

The repository pins an unmodified upstream Triton submodule. Aboleth's dialect,
operations, passes, and Python package are built entirely out of tree.

## Tested platforms

| GPUs | Network | Doorbell path | Notes |
|---|---|---|---|
| 8× H100 80GB (NVLink, NV18) | mlx5 RoCE v2 (ConnectX VF) | direct (root), staged | Reference host in `environment.json`: driver 570.211.01, CUDA 12.8.93; the driver does not grant unprivileged UAR mapping |
| 8× H100 SXM5 (NVSwitch) | 8× mlx5 native InfiniBand NDR, one HCA per GPU | direct without root (`PeerMappingOverride=1`), staged | Same software stack |
| 4× H100 per node, multi-node | mlx5 InfiniBand | staged | No `nvidia_peermem`: GPU memory registers through dma-buf |
| 8× B200 (`sm_100`) | mlx5 RoCE v2, rail-optimized | direct, staged | Needs the Torch substitution and settings below |
| 8× H100 (AWS p5.48xlarge) | EFA gen-2 | staged EFA, EFA host proxy | GPU-rung EFA doorbells are platform-gated ([`EFA_ABI.md`](../docs/EFA_ABI.md)) |

The AMD MI355X HIP runtime and Pensando Ionic backend are experimental and
not covered by this table.

## Clean-machine setup

Install an NVIDIA driver and CUDA toolkit (the pinned environment targets the
570-series driver and CUDA 12.8), rdma-core/OFED userspace and development
headers, zlib, libxml2 and libzstd development headers, Git, curl, GCC/G++,
and Make. Then:

```bash
git clone --recurse-submodules https://github.com/ParCoreLab/aboleth.git
cd aboleth
sha256sum -c repro/SHA256SUMS
./repro/bootstrap.sh
```

The default command installs uv 0.11.28 and CPython 3.12.13 under the project,
recreates `.venv` and `.tools/venvs/gin`, restores both exact package
environments, builds stock Triton with extension support, builds the
out-of-tree Aboleth compiler plugin, and builds the Aboleth, NVSHMEM, and NCCL
GIN binaries. It requires network access for pinned Python wheels and
Triton's pinned LLVM package. Dependencies outside default locations are
passed through to CMake:

```bash
./repro/bootstrap.sh -- -DRdmaCore_ROOT=/opt/rdma-core -DLibfabric_ROOT=/opt/amazon/efa
```

The GIN environment installs the pinned CUDA 12.6 Torch nightly and then
deliberately replaces Torch's NCCL 2.29.3 dependency with NCCL 2.30.7. The
bootstrap verifies the loaded `libnccl.so.2` before compiling the benchmark.

Run the compiler, Python and unprivileged GPU suites:

```bash
./repro/bootstrap.sh --keep-env --verify
```

Also run the root-required RDMA and workload tests (needs passwordless sudo
unless the driver grants unprivileged UAR mapping) on compatible hardware:

```bash
./repro/bootstrap.sh --keep-env --hardware-tests
```

The runtime detects native InfiniBand versus RoCE, and rank-local runtimes
select the GPU-nearest mlx5 HCA from PCI topology. `ABOLETH_HCA`,
`ABOLETH_PORT`, and `ABOLETH_GID_INDEX` provide explicit overrides
([`transport-control.md`](../docs/transport-control.md)). Where the driver
refuses to map the NIC doorbell into the GPU, the runtime falls back to staged
submission, which needs no privileges ([`RDMA_ABI.md`](../docs/RDMA_ABI.md)).

## Blackwell (`sm_100`) hosts

**The pinned torch cannot run on this GPU.** `requirements.freeze.txt` pins
`torch==2.13.0+cu126`, whose `arch_list` covers `sm_50`…`sm_90`, so every GPU
test dies with `no kernel image is available for execution on the device`. The
same torch version built against CUDA 12.9 carries `sm_100` and stays on `cu12`
wheels, leaving the pinned `nvidia-nvshmem-cu12` untouched:

```bash
./.tools/bin/uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu129 "torch==2.13.0+cu129"
```

This deviates from the pinned environment; revert with
`uv pip sync repro/requirements.freeze.txt`. Build the device bitcode for the
deployment target as well, e.g.
`-DABOLETH_RDMA_BITCODE_ARCHS=sm_100a -DABOLETH_EFA_BITCODE_ARCHS=sm_100a`.

On SM100 the RDMA lowering's scalar descriptor loads can be faster than the
default; select them with `ABOLETH_PASSES=-invariant-descriptors`
([`COMPILER_SEMANTICS.md`](../docs/COMPILER_SEMANTICS.md)).

A host whose inbox `ib_core` cannot load `nvidia_peermem` registers GPU memory
through dma-buf; NVSHMEM then needs `NVSHMEM_IB_DISABLE_DMABUF=0`. On a
rail-optimized RoCE fabric, set `ABOLETH_GID_INDEX` to the routable RoCE v2
entry ([`transport-control.md`](../docs/transport-control.md)).

## Build pitfalls

**The prebuilt LLVM needs development headers.** Triton's vendored LLVM
exports `ZLIB::ZLIB` and `LibXml2::LibXml2`; without `zlib1g-dev`,
`libxml2-dev` and `libzstd-dev`, its CMake configure fails with "the target was
not found". The runtime `.so.1` files are usually already present — it is the
headers and the dev symlinks that are missing.

**Always install Triton through `scripts/build-triton.sh`.** Running
`pip install --editable third_party/triton` directly omits `TRITON_EXT_ENABLED=1`
and `TRITON_BUILD_DIR`, which silently builds a second Triton at hidden
visibility, overwrites `libtriton.so`, and disables plugin loading entirely.
Every `aboleth.*` op then fails with `'triton._C.libtriton.ir.builder' object
has no attribute 'create_aboleth_put'`. Check with
`nm -D --defined-only third_party/triton/python/triton/_C/libtriton.so | wc -l`:
a correct build exports on the order of a hundred thousand symbols, a broken
one exports 1.
