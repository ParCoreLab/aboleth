# Aboleth provider-neutral proxy endpoint ABI

The proxy backend preserves the same compiler-visible `put`, `signal`, and
`wait` operations as the direct mlx5 backend, but delegates provider submission
and completion polling to a host progress thread. Current host providers are
mlx5 RC verbs and AWS EFA/libfabric.

The canonical layouts are defined in
`include/aboleth/Target/ProxyAbi.h`.

## Endpoint descriptor

Generated Triton code receives a 40-byte, GPU-resident descriptor:

```text
offset 0   u64 queue_state
offset 8   u64 producer_index_address
offset 16  u64 completion_timeout_cycles
offset 24  u32 slot_count
offset 28  u32 slot_stride
offset 32  u32 maximum_commands
offset 36  u32 abi_version
```

`queue_state` points to host-pinned memory mapped into the endpoint GPU.
`producer_index_address` points to a GPU-resident atomic batch sequence.
The current descriptor ABI version is 8.

## Queue layout

The current queue contains 64 batch slots. Each slot has:

```text
u64 ready_sequence
u32 command_count
u32 reserved
ProxyCommand commands[256]
```

Each 32-byte command records source and destination addresses, byte count,
opaque provider-local and remote memory keys, and a write, weak-signal,
strong-signal, or NOP opcode.
Predicated puts become NOP commands. The compiler packs each command into four
64-bit mapped-host stores. Signal commands carry their value directly. Weak
signals become inline RDMA writes. The verbs provider lowers strong signals to
64-bit compare-and-swap. The EFA provider counts delivery-complete writes in
hardware and injects the 64-bit publication value only after the preceding
batch counter reaches its target.

## Submission protocol

1. Generated code atomically reserves one sequence number for the whole batch.
2. It waits for ring capacity using the host-published completion sequence.
3. It writes every command into the selected slot.
4. One system-scope release store publishes the slot's ready sequence.
5. The host proxy acquires the slot and lowers it through the selected provider.
6. Verbs submits one prelinked WR list; EFA uses `fi_writemsg` with
   `FI_DELIVERY_COMPLETE`.
7. After provider completion, the host release-publishes the completion
   sequence.
8. The generated `poll` uses system-scope acquire loads before returning.

This supports concurrent GPU producers: reservation is atomic, publication may
occur out of order, and the host consumes batches strictly by sequence number.
Provider-specific publication must make all preceding writes visible before
the completion sequence advances.

## Status

The queue status word uses:

- `0`: success
- `1`: CQ timeout
- `2`: completion failure or unexpected completion context
- `3`: malformed command or permanent provider-posting failure

The host advances completion even on failure so sender-side polling terminates.
The endpoint should be discarded after a nonzero status.

## Scope

- mlx5 uses a shared worker/CQ. EFA uses one NUMA-pinned progress worker,
  completion counter, and CQ shard per active QP.
- EFA submits bounded `FI_MORE` bursts and coalesces completion publication.
- EFA uses four GPU-local rails, CUDA HMEM registration, `FI_EP_RDM`, and
  native device writes with SHM disabled.
- Independent registered buffers with per-command provider keys.
- No automatic direct-versus-proxy cost model.
- World connection exchange carries opaque EFA addresses.
