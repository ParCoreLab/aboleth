# AWS EFA proxy backend

The provider-neutral host proxy can execute through AWS EFA/libfabric. GPU
kernels enqueue semantic writes and publications through the `ProxyCommand`
ABI, while the selected host provider executes them.

## Relationship to the direct EFA backend

Aboleth reaches EFA two independent ways, and they share nothing below device
selection.

This document describes the **host-proxy** path: `Transport.PROXY` with
`ABOLETH_PROXY_PROVIDER=efa`. The queues stay on the host behind libfabric, a
NUMA-pinned worker drains a GPU-written command ring, and completion comes from
libfabric counters with `FI_DELIVERY_COMPLETE`.

[`EFA_ABI.md`](EFA_ABI.md) describes the **direct-queue** path:
`Transport.EFA`, a compiler backend that posts SRD work requests from the
kernel through efadv and the efa-dp-direct device library. On p5-class
hardware, compiler-built descriptors use staged host submission because the
platform drops GPU-originated SQ and doorbell MMIO; fully GPU-rung doorbells
are platform-gated.

The two overlap only in `src/runtime/aboleth_runtime_common.h`, which
picks the EFA device closest to a GPU on the PCI topology. Below that they
diverge completely: libfabric owns the queue pair, completion queue, doorbell,
and addressing for the proxy path, and exposes none of them, which is precisely
why the direct-queue path had to be built against efadv instead.

## Environment

The EFA environment is a separate virtual environment, `.venv-efa`, with
CUDA 13 PyTorch and NCCL packages (`repro/efa-requirements.txt`). It was
tested with EFA installer 3.1 and libfabric `2.4.0amzn5.0`. Create it without
modifying the host CUDA installation, pointing `CUDA_HOME` at a CUDA 13
toolkit:

```bash
CUDA_HOME=/usr/local/cuda-13.3 ./repro/bootstrap.sh --efa
```

The libfabric fabric is `efa` by default; `ABOLETH_EFA_FABRIC=efa-direct`
selects the thinner `efa-direct` fabric. Both use `FI_EP_RDM`.

## Runtime selection

Set:

```bash
export ABOLETH_PROXY_PROVIDER=efa
export FI_EFA_ENABLE_SHM_TRANSFER=0
```

Disabling libfabric SHM is required for same-node tests: otherwise local ranks
can bypass EFA. The runtime also requires CUDA HMEM P2P, enables device RDMA,
and rejects emulated writes.

The provider maps the existing proxy operations as follows:

- write: `fi_writemsg`
- weak/strong publication: injected 64-bit RMA write
- completion: EFA hardware counters with `FI_DELIVERY_COMPLETE`
- progress: one NUMA-pinned worker and CQ shard per active QP
- rails: four GPU-local EFA devices, selected from PCI topology

EFA does not provide mlx5 RC ordering. The correctness-first implementation
therefore retains delivery completion for every RMA operation. The fast path
does not create one CQE or GPU rendezvous per operation: libfabric posts each
command batch with bounded `FI_MORE` bursts, and the host advances the GPU
completion watermark after the EFA counter reaches the complete batch target. Strong
publication cannot issue until every earlier operation on that logical queue
has completed.

## Single-node verification

Two-GPU round trip:

```bash
env \
  ABOLETH_PROXY_PROVIDER=efa \
  FI_EFA_ENABLE_SHM_TRANSFER=0 \
  PYTHONPATH="$PWD/python:$PWD/third_party/triton/python/triton_kernels" \
  ABOLETH_PLUGIN_PATH="$PWD/build/lib/libAbolethPlugin.so" \
  TRITON_PLUGIN_PATHS="$PWD/build/lib/libAbolethPlugin.so" \
  .venv-efa/bin/python \
  experiments/python/aboleth_local_endpoint_smoke.py \
  --transport proxy --left-device 0 --right-device 1
```

Eight-GPU multiprocess round trip:

```bash
env \
  ABOLETH_PROXY_PROVIDER=efa \
  FI_EFA_ENABLE_SHM_TRANSFER=0 \
  PYTHONPATH="$PWD/python:$PWD/third_party/triton/python/triton_kernels" \
  ABOLETH_PLUGIN_PATH="$PWD/build/lib/libAbolethPlugin.so" \
  TRITON_PLUGIN_PATHS="$PWD/build/lib/libAbolethPlugin.so" \
  .venv-efa/bin/python \
  experiments/python/aboleth_multiprocess_smoke.py \
  --transport proxy --all-gpus
```

These two runs are also registered as `efa.proxy-smoke` and
`efa.proxy-multiprocess` (label `efa`) when `.venv-efa` exists or
`ABOLETH_EFA_PYTHON` names an interpreter.

## Scope

Rank host work runs on a NUMA-local CPU set separate from the explicitly
pinned proxy workers. The proxy ABI and transport planner are shared with
mlx5; provider-specific addressing, memory registration, rail selection,
posting, and completion stay below that boundary. Multi-node runs and the
performance evaluation of the EFA paths are future work.
