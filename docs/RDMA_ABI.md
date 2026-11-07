# Aboleth RDMA endpoint ABI

Generated Triton code receives a pointer to a versioned, GPU-resident endpoint
descriptor. The host runtime creates the QP/CQ/MRs, maps the mlx5 resources,
fills this descriptor, and copies it to GPU memory.

The canonical ABI is defined by
[`RdmaAbi.h`](../include/aboleth/Target/RdmaAbi.h).
Host code reaches the runtime that publishes it through
[`runtime_c.h`](../include/aboleth/runtime_c.h).

This is the direct-mlx5 endpoint contract. Its submission path may be GPU-rung
or host-staged depending on what the platform lets the GPU map — see
"Submission modes" below. The EFA direct-queue backend hands the kernel queue
objects rather than raw queue memory and retires work by counting completions
rather than matching a queue index; its own two submission modes are
documented in [`EFA_ABI.md`](EFA_ABI.md).

## Layout

The descriptor is 96 bytes, aligned to 8 bytes, standard-layout, and trivially
copyable.

| Offset | Type | Field | Meaning |
|---:|---|---|---|
| 0 | `u64` | `send_queue` | GPU-visible SQ base address |
| 8 | `u64` | `send_dbrec` | GPU-visible send DBREC address |
| 16 | `u64` | `blueflame` | GPU-visible doorbell target: the mlx5 UAR/BlueFlame register (direct) or a host-pinned cell a doorbell agent forwards (staged) |
| 24 | `u64` | `completion_queue` | GPU-visible CQ base address |
| 32 | `u64` | `completion_dbrec` | GPU-visible CQ consumer DBREC |
| 40 | `u64` | `producer_index_address` | GPU address of the persistent SQ producer index |
| 48 | `u64` | `completion_consumer_address` | GPU address of the persistent CQ consumer index |
| 56 | `u64` | `completion_timeout_cycles` | Device polling timeout |
| 64 | `u32` | `send_stride` | Bytes per WQEBB |
| 68 | `u32` | `send_wqe_count` | SQ capacity in WQEBBs |
| 72 | `u32` | `qpn` | mlx5 queue-pair number |
| 76 | `u32` | `completion_entry_size` | CQE size in bytes |
| 80 | `u32` | `completion_entry_count` | CQ capacity |
| 84 | `u32` | `abi_version` | Descriptor ABI version, currently `3` |
| 88 | `u32` | `atomic_result_lkey` | Local key for the endpoint-owned atomic result sink |
| 92 | `u32` | `reserved1` | Reserved, must be zero |

Compile-time offset and size assertions prevent accidental C++ layout changes.
Future ABI changes must increment `abi_version`.

Memory registration keys are deliberately not endpoint properties. Generated
RDMA operations receive the source `lkey` and destination `rkey` with their
buffer operands, allowing one endpoint to communicate with multiple registered
regions. Remote buffers are registered for 64-bit atomics as well as reads and
writes. Payload registrations may retain relaxed ordering: strong publication
uses a same-QP responder atomic, which prior relaxed writes cannot bypass. Each
registration records whether relaxed ordering was granted
(`aboleth_runtime_buffer_strictly_ordered`), and the allocation handle carries
that as bit 0 of its flags field (bytes 2-3, zero from older encoders). Only
when every exchanged registration is strictly ordered may publication planning
select a fenced write instead (`docs/COMPILER_SEMANTICS.md`).

`aboleth_runtime_local_endpoint_gpu_resident_queues` reports whether an
endpoint's send queue, doorbell records and completion queue are in GPU memory
(the DEVX staged path). Compilation narrows the doorbell's release to GPU
scope only when every endpoint a kernel posts to reports true.
`ABOLETH_REQUIRE_RELAXED_ORDERING=1` turns registration fallback into an error
for ordering litmus tests.

The completion-state allocation contains four public state words followed by
one private 64-bit atomic-result sink. Strong signal atomics write their return
value there, so the user-provided signal source remains read-only.

Zero-byte semantic puts keep their ordered SQ slot as a NOP, including
completion and dependency bookkeeping. An mlx5 data SGE with byte_count=0
must not be used to encode an empty application message. The direct and
shared-device-library lowering paths both combine the caller predicate with
`byte_count != 0`; callers need no special empty-route predicate.

## Submission modes

### Doorbell ownership

The kernel's doorbell sequence ends in one release store of a 64-bit doorbell
word to the address in `blueflame`. Everything before it — WQE construction,
producer-index advance, the DBREC store — targets ordinary memory (host
memory in the libmlx5 layout, GPU memory for the DEVX staged queues below),
and so does CQE polling. Only the UAR register itself is I/O
memory, and mapping I/O memory into the GPU
(`cudaHostRegisterIoMemory`) is what drivers without `PeerMappingOverride=1`
refuse. The two modes differ solely in who performs that final store:

- **Direct** (`submit_mode = "direct"`): `blueflame` holds the GPU-mapped UAR
  register and the kernel rings the adapter itself.
- **Staged** (`submit_mode = "staged"`): `blueflame` holds a host-pinned
  8-byte cell. The unchanged release store publishes the doorbell word there,
  and a per-(runtime, GPU) doorbell agent thread
  (`src/runtime/core/aboleth_doorbell_agent.h`) forwards it to the real UAR —
  DBREC first is not needed, the kernel already wrote it, and the release
  ordering of the cell store makes it visible before the agent can wake.
  Coalescing to the latest observed value is safe because the DBREC carries
  the authoritative count. The agent does nothing else: no WQE copies, no CQ
  polling, no submission logic. Completions retire on the GPU exactly as in
  direct mode.

Staged endpoints additionally choose where their queues live
(`ABOLETH_RDMA_STAGED_QUEUES`, default `gpu`):

- **`gpu` (default, probe-with-fallback):** the endpoint is built through
  DEVX (`src/runtime/core/aboleth_endpoint_devx.cu`) — the send queue, both
  doorbell records, and the completion queue are `cudaMalloc` memory the NIC
  reaches through dmabuf-registered umems (NVSHMEM IBGDA's placement; the
  PRM mailboxes are vendored in `src/runtime/rdma/vendor/`). No `ibv_qp`
  exists; RST->INIT->RTR->RTS run as DEVX commands. The GPU builds WQEs and
  polls CQEs at HBM latency instead of over PCIe, and the NIC fetches them
  with one peer read. The doorbell cell, the agent, and the wire bytes are
  unchanged — the agent rings a per-endpoint non-cacheable DEVX UAR.
  Any refusal (missing DEVX exports, no dmabuf export, RoCE link layer,
  firmware rejection) falls back to `host` with a one-time notice.
- **`host`:** libmlx5-allocated host buffers, cudaHostRegister-mapped into
  the GPU.

Device code, the compiler's two emitters, the descriptor layout, and the ABI
version are byte-identical across modes; the mode is a bring-up decision,
invisible to generated kernels. Selection is capability-driven: unset, the
first endpoint attempts the I/O mapping and falls back to staged if the driver
refuses it (the attempt is the capability probe — there is no sysfs shortcut);
`ABOLETH_RDMA_SUBMIT=direct` makes the refusal an error, `=staged` skips the
attempt. The variable selects who stores to the UAR and never changes a
compiled plan's premise, which is what keeps it on the right side of the
no-policy-environment rule (see `transport-control.md`). The resolved mode is
reported per endpoint by `aboleth_runtime_local_endpoint_submit_mode` /
`LocalEndpoint.submit_mode`, and `test/lint/probe_uar.py` pins `direct` so
the sudo test gate keeps proving the direct mapping rather than passing
through the fallback.

The mode was designed against the other doorbell-forwarding implementation in
the wild, NVSHMEM's CPU-assisted IBGDA (`NVSHMEM_IBGDA_NIC_HANDLER`): its
no-gdrcopy `cpu_host_memory` handler also watches one plain host-memory
producer cell, and its CPU handler writes DBREC + doorbell while the GPU
skips both behind a device-state flag. Aboleth keeps the GPU writing the
DBREC instead, which is what removes the device-code branch. NCCL GIN's
proxy backend and MSCCL++'s ProxyService are command proxies — the host
interprets descriptors and posts through verbs — which in Aboleth's taxonomy
is `Transport.PROXY`, not staged submission.

Decisions recorded as future optimization targets, none required for
correctness:

- **GPU-resident queue state for DIRECT mode.** Staged mode has it
  (`ABOLETH_RDMA_STAGED_QUEUES=gpu` above); the direct path still uses
  libmlx5-allocated host buffers and would need the same DEVX bring-up plus
  a GPU-mapped UAR.
- **CPU-side BlueFlame inlining.** The agent could copy the finished WQE into
  the BlueFlame register instead of ringing, saving the NIC's DMA fetch of
  the WQE.
- **A descriptor mode flag.** Rejected here: the flag would add a per-doorbell
  branch on the direct path, drift every rdma golden, and still need the same
  watched release-store — the pointer swap is the branch-free special case.
  Revisit only if a mode ever needs device-visible behavior differences.
- **GDRCopy-backed wakeup cell** (NVSHMEM's `cpu` handler) where the kernel
  module exists, keeping even the cell out of host memory.
- **NC-dedicated DEVX UAR** for the CPU path (NVSHMEM's preference); the
  agent currently reuses the QP's write-combining UAR with full fences.
- **Agent placement and backoff.** The agent pins to the GPU's NUMA-node
  cpuset intersected with the launcher's affinity mask (NCCL's model); an
  explicit cpuset override, a measured per-host placement sweep, and an
  adaptive backoff beyond the current single `_mm_pause` are all unmeasured.
- **Per-UAR-register lock granularity** around the agent's MMIO store,
  currently one uncontended mutex resolved at hook creation.

## Ownership

- The host owns QP/CQ/MR creation and destruction.
- The host initializes the descriptor and persistent indices.
- Generated GPU code owns SQ producer and CQ consumer advancement while the
  endpoint is active.
- The descriptor and all referenced resources must outlive every kernel using
  the endpoint.

## Verification

`ctest --test-dir build -R hardware.rdma-abi`:

1. Creates real verbs QPs, CQs, and GPU MRs.
2. Maps the actual SQ, DBREC, BlueFlame, and CQ resources.
3. Populates the descriptor in GPU memory.
4. Runs a Triton kernel that reads all twelve 64-bit descriptor words.
5. Compares every byte against the host-side structure.
6. Checks field-level invariants such as stride, QPN, CQE size, ABI version,
   and separately exported buffer registration keys.

GPU mapping of the NIC UAR needs `PeerMappingOverride=1` or root; on nodes
without either, `hardware.rdma-abi` escalates via sudo while its
`hardware.rdma-abi-staged` twin — and the rest of the `-staged` arms — run the
same scripts unprivileged over staged submission.

## Bidirectional endpoints

The persistent compiler test creates two descriptors:

- GPU 0 descriptor for data and ready signals sent to GPU 1.
- GPU 1 descriptor for acknowledgments sent to GPU 0.

Each endpoint has independent QP/CQ resources and persistent producer/consumer
indices in its owning GPU's memory. The host creates both endpoints once; the
CPU performs no work between communication rounds.

### Send ring capacity

`send_wqe_count` is the positive power-of-two physical mlx5 SQ capacity in
WQEBBs (the hardware represents it through `log_sq_size`). Synthetic bindings
must honor the same invariant. The existing CUDA `wqe_address` helper and
direct compiler lowering index it with `index & (send_wqe_count - 1)`. This
requirement applies to the SQ; CQ indexing retains its own existing contract.

See the [rdma-core mlx5 SQ indexing implementation](https://github.com/linux-rdma/rdma-core/blob/master/providers/mlx5/qp.c).
