# Aboleth EFA backend ABI

The EFA backend compiles Queue IR into the frontend kernel, as the direct-mlx5
backend does. On hardware that accepts GPU-originated MMIO, the
kernel posts work requests and polls completions itself. On p5-class hardware,
the kernel builds descriptors into a staging ring, a host doorbell agent
submits them, and the GPU still retires completions. The backend reaches the
hardware through
[`amzn/efa-dp-direct`](https://github.com/amzn/efa-dp-direct), a git
submodule at `third_party/efa-dp-direct`.

## What SRD changes

EFA's Scalable Reliable Datagram transport is reliable but **unordered**, and
offers **no remote atomics**. Three contracts change relative to mlx5.

**Completion is counted, not indexed.** The mlx5 backend retires a batch by
polling for the completion entry whose WQE counter matches the batch end, which
is only meaningful when completions arrive in submission order. On EFA each
popped completion retires exactly one command, so a batch has retired once the
retired count reaches the batch end. `TransportCaps::completion_model` records
this as `UnorderedCount`.

**Predication cannot use an ordered placeholder.** mlx5 emits a NOP WQE for a
disabled transfer to keep the queue positions aligned. EFA has no NOP, and
needs none: a disabled transfer becomes a zero-length write, which still
produces the completion the compiler counted on, and nothing is lost by not
preserving an order SRD does not have.

**Strong publication cannot ride on same-queue ordering.** The mlx5 backend
publishes with a responder atomic compare-and-swap on the same queue, which
prior relaxed-ordering payload writes cannot bypass, and which doubles as a
checked epoch update. Neither property is available on SRD. Two legalizations
exist, both `TransportPublicationMode` values derived from capabilities rather
than chosen by the backend:

- **`FenceThenSignal`** (implemented, default). The sender flushes the batch,
  retires the payload completions ordered before the signal, and only then
  posts the epoch as an ordinary write from its registered cell — EFA has no
  inline data on RDMA writes. Observing the epoch therefore still
  implies visibility of the payload. The receiver contract is unchanged — an
  acquire load on the signal cell — which is why this is the default.
- **`ImmCounting`** (not implemented). Payload travels as write-with-immediate
  and the receiver advances the epoch once it has counted the writes belonging
  to that publication. The device library implements the receiver loop
  (`aboleth_efa_receive_advance`), but wiring it needs two things the IR does
  not yet express: `aboleth.wait_signal` carries no endpoint, so the receiver
  has no receive queue to drain, and nothing states how many writes belong to
  one epoch. Selecting it reports exactly that.

The epoch contract relaxes accordingly. Monotonicity is preserved because one
sender owns each signal cell and epochs strictly increase; what is lost is the
hardware's *check* of the predecessor value. Checked publication is an
optional property of transports with remote atomics
(`publication_is_checked`), not part of the strong-publication contract.

## Endpoint descriptor

`include/aboleth/Target/EfaAbi.h` defines `EfaEndpointDescriptor`, version 1,
72 bytes read by the kernel as nine 64-bit words. Unlike the mlx5 descriptor,
which hands the kernel raw queue memory and a doorbell register, this one
carries pointers to the efa-dp-direct queue objects, because the kernel calls
the library rather than building descriptors itself.

| Word | Contents |
| --- | --- |
| 0 | `efa_cuda_qp` device pointer |
| 1 | send `efa_cuda_cq` device pointer |
| 2 | receive `efa_cuda_cq` device pointer, zero unless counting immediates |
| 3 | completion-state block address |
| 4 | completion timeout in cycles |
| 5 | send depth \| receive depth |
| 6 | address handle \| remote queue-pair number |
| 7 | queue key \| publication scheme |
| 8 | maximum inline bytes \| ABI version |

Addressing is SRD's: an address handle plus the peer's queue-pair number and
queue key, rather than a connected queue pair. Registration keys stay per
region on the frontend binding operations, exactly as for mlx5.

The completion-state block is the same four-word layout the mlx5 backend uses:
producer index, retired head, publish index, error. Sharing it lets both
backends use the same reservation, capacity, and in-order publication helpers.

## Device primitives

`src/runtime/efa/aboleth_efa_device.cu` wraps the library in a flat
scalar ABI, because Triton extern calls carry only scalars. Each wrapper reads
what it needs from the descriptor rather than taking a long argument list.

| Symbol | Queue IR operation |
| --- | --- |
| `aboleth_efa_begin_batch` | `submit` opens a batch |
| `aboleth_efa_post_write` | `post_put` |
| `aboleth_efa_post_write_imm` | `post_put` under immediate counting |
| `aboleth_efa_post_inline` | `post_signal`, an 8-byte write from the registered epoch cell |
| `aboleth_efa_flush` | `submit` rings the doorbell |
| `aboleth_efa_completion_advance` | `poll`, `wait_group`, `drain` |
| `aboleth_efa_receive_advance` | receiver side of immediate counting |

Queue-state mutation — batch start, flush, completion pop — must be serialized
per queue. Aboleth's planned mode gives one producer CTA per endpoint, and the
caller elects a single thread for those wrappers. Work-request placement may
run from several threads at distinct indices.

## Build

The library is compiled to LLVM bitcode and linked into generated kernels as a
Triton extern library, then inlined:

```bash
cmake --build build --target aboleth_efa_sm_90a          # build/lib/aboleth_efa_sm_90a.bc
```

Two workarounds are needed and are worth knowing about when the submodule is
updated. The library declares each primitive non-static and then defines it
`static inline`, which nvcc tolerates and clang rejects, so the declaration
header is suppressed and the two enumerations it also provides are restated in
the wrapper. And the CUDA compilation's module flags conflict with Triton's own
module during linking, so they are dropped before the final bitcode is
assembled.

`ABOLETH_EFA_BITCODE` overrides the library path; `ABOLETH_EFA_ARCH` selects
the architecture suffix. The library's hash is mixed into the compilation cache
key, because a new library means a new kernel.

## Host control plane

`src/runtime/efa/aboleth_efa_runtime.cu` creates the resources the
descriptor points at. It is deliberately a separate translation unit from the
provider-neutral proxy runtime, which reaches EFA through libfabric and shares
nothing below the device-selection layer.

The creation flow, and what each step forced:

1. **Device selection.** One EFA device per GPU, chosen by deepest common PCI
   path, via `select_efa_device_name` in
   `src/runtime/aboleth_runtime_common.h` (shared with the proxy
   runtime). `ABOLETH_EFA_DEVICE` / `ABOLETH_HCA` override it. Multi-rail
   striping is future work; `select_efa_device_names` already ranks the
   candidates.
2. **Queue pair.** `efadv_create_qp_ex` with `EFADV_QP_DRIVER_TYPE_SRD` and no
   extra flags: requesting `EFADV_QP_FLAGS_INLINE_WRITE`, or more inline data
   than fits a 64-byte descriptor, makes the provider size every send slot at
   128 bytes (`efa_calc_sq_wqe_size` in rdma-core), while the device library
   hard-codes 64-byte slots — the adapter then silently ignores everything
   placed on the wrong stride. The runtime pins both and rejects any queue
   whose reported `entry_size` is not 64. The first pass over the send queue
   is phase 0, matching rdma-core's zero-initialized queue pair; a descriptor
   carrying the wrong phase bit is silently never fetched. SRD is
   connectionless, so the transitions carry no peer: INIT with the queue key,
   then RTR, then RTS. A driver queue pair may reject `IBV_QP_PKEY_INDEX`, so
   the INIT mask falls back without it.
3. **Completion queues in GPU memory.** `efadv_query_cq` on an ordinary
   `ibv_create_cq` returns host memory that `cudaHostRegister` refuses, which
   would leave the kernel unable to poll. Instead the queue is created with
   `efadv_create_cq` and `EFADV_CQ_INIT_FLAGS_EXT_MEM_DMABUF` over a
   dmabuf-exported `cudaMalloc` allocation, so completions are DMA'd straight
   into device memory the kernel already addresses. The device reports
   `EFADV_DEVICE_ATTR_CAPS_CQ_WITH_EXT_MEM_DMABUF`. dmabuf export works on
   whole allocation granules, so a queue smaller than one granule is rounded up
   rather than sub-allocated.
4. **Send queue and doorbells.** `efadv_query_qp_wqs` returns these on
   driver-mapped I/O pages, not ordinary host memory: they register only with
   `cudaHostRegisterIoMemory`, and only for a process privileged to map device
   memory — the same requirement the direct-mlx5 path carries.
5. **Memory registration.** `ibv_reg_dmabuf_mr` on the EFA protection domain.
   No relaxed-ordering flag and no atomic capability, unlike mlx5.
6. **Addressing.** `ibv_create_ah` from the peer's global identifier, then
   `efadv_query_ah` for the address-handle number the work request quotes
   alongside the peer queue-pair number and queue key.

The receive queue stays host-resident and unmapped: sender-fenced publication
posts no receive work requests. Immediate counting would have to place the
receive queue in device memory first, and `efadv` exposes no external-memory
flag for it.

### Submission modes and platform status

`ABOLETH_EFA_SUBMIT` selects who moves descriptors into the adapter:

- **`gpu`**: the kernel posts work requests and rings the doorbell itself
  through the `cudaHostRegisterIoMemory` mapping of the send queue. This needs
  a platform whose fabric delivers GPU stores to the adapter BAR, and a process
  privileged to map device memory.
- **`host`** (staged; the default on EFA gen-1/2 devices): the kernel posts
  descriptors through the unchanged device library into a host-pinned staging
  ring shaped like the send queue, and a doorbell agent
  (`src/runtime/core/aboleth_doorbell_agent.h`) copies new slots into the
  adapter's queue and rings. Completions still retire on the GPU. No queue
  memory is mapped for the GPU and no privileges are needed.

On p5-class instances (EFA gen-1/2), GPU stores through the I/O mapping do not
reach the adapter, so GPU-submitted EFA cannot complete transfers there; the
runtime selects staged submission by default and warns if `gpu` is forced.
p5en/p6-class instances are the candidates for GPU-rung doorbells.

`Transport.EFA` is wired end to end (connection-info provider code 3, endpoint
pairs, Python) on the `FI_EFA_GDA_OPS` control plane. The host proxy has two
libfabric fabrics, the default `efa` and the thinner `efa-direct`, selected by
`ABOLETH_EFA_FABRIC` ([`EFA_PROXY.md`](EFA_PROXY.md)).

The EFA backend is implemented and its tests are registered; its performance
evaluation is future work.

```bash
ctest --test-dir build -R efa.staged-smoke    # staged submission, unprivileged
ctest --test-dir build -R efa.host-post-smoke # GPU queues, descriptors posted by the host
ctest --test-dir build -R efa.host-smoke      # GPU-posted path (needs a capable platform)
```

`efa.host-post-smoke` validates the control plane, registration, and
GPU-memory completion delivery independently of whether the platform accepts
GPU MMIO. If `efa.host-smoke` fails with `status=1 retired=0` while
`efa.host-post-smoke` passes, the platform drops GPU stores to the BAR.

**Open question for strong publication on the GPU-posted path:** whether a raw
EFA TX completion proves remote delivery or only local transmission. The
sender-fenced scheme (`FenceThenSignal`) assumes the stronger reading and has
only empirical support. Resolve it against AWS SRD completion semantics before
relying on GPU-posted EFA for correctness-critical publication.

## Datapath strategy: why the library rather than generated descriptors

Aboleth generates mlx5 descriptors directly, so using a vendor library for EFA
deserves justification. Three options were considered:

1. **Call the efa-dp-direct device library** through inlined bitcode.
2. **Emit EFA descriptors from the compiler**, as `LowerRDMA` does for mlx5,
   using `efa_io_defs.h` as the hardware reference.
3. **Fork and specialize the library**, keeping the extern-call seam.

**Option 1 is implemented.** The reasoning turns on which optimizations are at
stake. Aboleth's optimizations are overwhelmingly in the scheduling layer above
Queue IR — batch formation and one doorbell per batch, deferred completion and
completion groups, capacity-derived reclamation, capacity-proven unchecked
reservation, publication-strength planning, direct-to-final materialization,
byte-aware costing, lifetime-to-completion scaling. **All of these apply
identically under any of the three options**, because they decide *what* to
submit and *when* to complete, not how a descriptor is laid out.

The datapath layer is where the options differ, and most of it still
translates. CTA-owned queues and lock-free doorbells map onto the library's own
contract, which permits concurrent placement at distinct indices with
serialized batch start and flush. Reserve, out-of-order fill, in-order publish
is the same protocol shape as `start_sq_batch`/`place_wr`/`flush_sq_wrs`.
Inline signals and zero-length predication translate directly. Compile-time
descriptor specialization is largely recovered for free, since inlined bitcode
with constant arguments constant-folds in LLVM much as hand-emission would.

What option 1 gives up is store-level control: the wrappers build a work
request on the stack and copy it into the send queue, which costs a handful of
local stores that direct emission would avoid, and any
completion-suppression trick the library's processing hints do not expose.

Against that, hand-rolling was *necessary* for mlx5 and is not for EFA. No
standalone GPU-posting library existed for mlx5 outside NVSHMEM's IBGDA, and
the mlx5 WQE format is a decade-stable documented interface. efa-dp-direct is
exactly the layer Aboleth had to hand-roll, the EFA descriptor format is
generation-specific with no published stability contract, and SRD's unordered
semantics raise the correctness cost of a from-scratch datapath. The
EFA-specific performance axis that likely matters is multi-rail — four
adapters per GPU on p5 — which maps onto the existing data-queue channel model,
not onto descriptor micro-control.

**Inlining.** The extern-bitcode path inlines completely: a kernel
calling the wrappers contains zero call instructions, with the library's
doorbell fences and completion-timeout clock reads emitted inline.

**Gates for escalating to option 3, then 2.** Escalate if post-inline SASS
shows the stack-resident work-request construction costing more than about 2%
of small-message rate against a hand-written comparison; if completion
suppression is needed for completion moderation and
`efa_cuda_wr_set_processing_hints` does not expose it; or if eager mode needs
several producer CTAs sharing one queue, which the library's serialization
contract does not allow. Option 3 comes first, since it keeps the extern-call
seam and the lowering untouched; option 2 only if fusing descriptor
construction with surrounding compute shows headroom inlining cannot reach.
