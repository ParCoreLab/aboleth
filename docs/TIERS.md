# The Aboleth tiers

Aboleth is one compiler with two adoption tiers. Every tier
runs compiler-generated transport code; what changes is how much of the
program the compiler sees, and so how much transport state it may specialize.

| Tier | Client writes | Compiler sees | Compiler may do |
|---|---|---|---|
| 1 Library | ordinary CUDA calling a generated device library | one operation at a time, plus topology | per-topology dispatch folding, WQE batching and one doorbell per call, QP fan-out |
| 2 Kernel | Triton kernels using `aboleth.*` operations | the whole kernel, communication and compute | everything in 1, plus batching across payload production, completion planned from declared reuse, issuer analysis, coalesced doorbells, publication ordering, epoch elision |

**Framework adapters** (`Pattern`s, the MoE and KV adapters) are not a tier.
They plan communication outside any kernel and attach it to client kernels in
one of three artifact forms (below).

## Tier 1: the device library

Arbitrary CUDA code links a compiler-generated, NVSHMEM-shaped device library
and calls one-sided operations over registered regions. Addressing comes from
the rank's peer-region table, exchanged at bootstrap; no symmetric heap.

```python
world = aboleth.World(rank, world_size, bootstrap)
world.connect_all("auto")
regions = world.exchange_regions(send=send_buffer, recv=recv_buffer)
lib = world.device_library()                 # generic RDMA+P2P, per arch
lib = world.device_library(specialize=True)  # exact peer -> transport geography
# lib.bitcode (Triton extern_libs, cute.BitCode), lib.archive (nvcc device link),
# lib.include_dir/aboleth/device.cuh
```

```cpp
#include <aboleth/device.cuh>
__global__ void exchange(aboleth::DeviceContext ctx, int peer,
                         aboleth::RemoteRegion recv, aboleth::LocalRegion send,
                         aboleth::RemoteRegion flag, std::uint64_t* local_flag) {
  aboleth::put(ctx, peer, recv, send, bytes);
  aboleth::put_inline(ctx, peer, flag, header_word);  // from registers
  aboleth::signal(ctx, peer, flag, epoch);
  aboleth::quiet(ctx, peer);                          // sender completion
  aboleth::wait_signal(ctx, local_flag, epoch);
}
```

`put`, `put_inline` and `signal` post and ring without waiting; `quiet` is the
only blocking call. The generic library dispatches on the peer handle's
transport tag; `specialize=True` bakes the peer geography into the artifact,
so a uniform world folds to one backend and `put<Peer>(...)` needs no runtime
selection. The library is built without endpoint placement, so its doorbell
keeps system scope. There is no host-side call API.

The library is generated from per-op programs (`python/aboleth/tier1.py`,
`device_library.py`, header in `emit/tier1_header.py`) lowered in *detached*
mode. Traps learned building it:

- Triton broadcasts a scalar atomic's result through shared memory, publishes
  release stores from `tid.x == 0`, and lowers every scalar `tt.store` once
  per block from lane 0. Detached mode answers all three with per-thread
  inline asm; `artifact.py`'s shared-memory-is-zero guard is the tripwire.
- Triton extern calls execute on every active lane; Triton consumers get
  per-thread semantics and must issue protocol calls from one lane.
- Two transport encodings coexist: device kind bits (rdma 0, p2p 1, proxy 2,
  efa 3; rdma 0 keeps untagged handles valid) and the bootstrap wire codes
  (p2p 1, rdma 2). Never conflate them.
- `hardware.tier1-differential(-staged)` holds generated and hand-written
  WQEs bitwise equal.

The same generated library is bound into Triton kernels through `extern_libs`
(`python/aboleth/frontends/triton/tier1.py`) and into CuTe DSL kernels through
`cute.extern` (`python/aboleth/frontends/cute.py`). In both, ops have
per-thread semantics, so a kernel issues each protocol call from one thread.

## Tier 2: Triton kernels

Communication operations appear in the client's Triton kernel and compile with
it. A fused dispatch, end to end in
`experiments/python/aboleth_tier2_dispatch.py`:

```python
ep = aboleth.endpoint(endpoints, peer)            # table + index
send = aboleth.region(send_ptr, send_key)         # address + key, .at(offset)
recv = aboleth.remote(recv_table, peer)
...
t = aboleth.put(ep, recv.at(dst), send.at(src), CHUNK, predicate=live, after=t)
t = aboleth.put_inline(ep, recv_n.at(RANK), n, after=t)   # register payload
s = aboleth.signal(ep, flag.at(RANK), epoch, after=t)     # epoch by value
aboleth.wait(s, iteration=epoch, reuse=SLOTS)             # declared reuse
```

The passes (listed in [`COMPILER_SEMANTICS.md`](COMPILER_SEMANTICS.md)
§ Optimization passes) batch the commands, choose the publication mechanism,
plan completion from the reuse distance, and the backend lowering emits mlx5
WQEs, doorbell and CQ poll (or the P2P, proxy or EFA equivalent) inline with
the compute. A statically known peer selects its backend at compile time;
dynamic mixed routing is split by the C++ planner into backend-homogeneous
launch partitions, exposed through `CompiledPhase.launches`. Each is compiled
with its exact peer set, never lowered through a representative backend or a
per-operation transport branch.

Exporting a user's Tier-2 kernel as an inline fragment, device function or
standalone kernel is not implemented.

## Framework adapters

The client keeps its kernels and declares the communication contract as a
`Pattern`; an integration author supplies a `CommunicationAttachment` naming
where the generated artifact lands, validated against the opaque kernels'
declared capabilities ([`ARCHITECTURE.md`](ARCHITECTURE.md)). Artifact forms:

- `inline_fragment`: communication materialized inside a client kernel the
  client compiles through Aboleth (Triton).
- `bindable_epilogue`: a device function (bitcode plus relocatable object)
  with a typed bindings header; callable from any CUDA compiler.
- `standalone_kernel`: a generated communication kernel plus launch metadata;
  the client launches it.

The maintained native MoE library and KV migration declare semantic programs
through this path. The C++ compiler determines communication and physical
resource plans; their client executors bind and launch the resulting artifacts.
See [`COMPILED_MOE.md`](COMPILED_MOE.md) and [`KV_CACHE.md`](KV_CACHE.md)
(experimental).

## Cross-kernel placement: derived, not declared

Every kernel completes its own transport and rings its own doorbell. Aboleth
does not accept a declared cross-kernel schedule. Cross-kernel placement is
future work and would be derived only from the member kernels' IR; the
substrate it would target (`publish_head`, `externalPublication`,
`deferCompletion` plus `drain`) is in the dialect.

## What the tiers are not

Aboleth never launches kernels, owns streams, captures CUDA graphs, or
executes the application, at any tier. Integration surfaces also hold to:

- no tier selection by application users or Python transport-policy heuristics;
- bootstrap, discovery, registration and endpoint setup stay out of
  application code;
- plans bind before CUDA graph capture and reuse stable pool addresses;
- unsupported reductions, effects or transports fail at compile time and name
  the missing contract.
