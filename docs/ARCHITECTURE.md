# Aboleth architecture

Aboleth is a transport-state subcompiler. A client supplies a communication
pattern, region effects, and attachment points; Aboleth returns transport
artifacts for that client to integrate. It is not an application runtime or a
whole-program scheduler.

## Ownership boundary

Clients own:

- The communication algorithm and routing decisions.
- Packing, tensor layouts, and optimized compute kernels.
- Kernel composition and launch order.
- Streams, events, CUDA graphs, and application execution.

Aboleth owns:

- Semantic transfer, publication, observation, and sender-completion
  obligations.
- Communication-region dataflow and lifetime analysis.
- Transport-state choices such as batching, queue use, publication strength,
  completion placement, and credits.
- Queue IR formation, target lowering, and the resource requirements needed
  for runtime binding.

Aboleth may retarget communication into client storage when effects and an
opaque-kernel capability make that rewrite legal. The client still owns the
algorithm, compute, and launch.

## Compilation pipeline

```text
Pattern + regions/effects
        ↓
semantic plan
        ↓
frontend/runtime attachment
        ↓
physicalization
        ↓
KernelTransportProgram + typed invocation manifest
        ↓
semantic transport IR → Queue IR → P2P / proxy / mlx5 / EFA
        ↓
runtime binding of regions, endpoints, and target resources
```

The stages have distinct contracts:

1. A `Pattern` describes logical regions, bounded worksets, semantic
   communication, opaque stages, and ordering. Raw pointers, registration keys,
   queue descriptors, and launches are not part of this level.
2. The semantic plan simplifies dataflow, records legal destination rewrites,
   computes region and output lifetimes, and derives transport obligations.
   It is analysis output, not executable communication.
3. A frontend/runtime attachment identifies the enclosing kernel, insertion
   point, peer partitioning, runtime buffers, and publication.
   `physicalize_pattern` derives transfer coverage, byte bounds, typed slots,
   and command structure from the compiled communication stage.
4. `KernelTransportProgram` is the frontend-neutral executable form. It
   describes one physical transport segment through typed arguments and
   semantic `put`/`signal`/`wait`/`drain` operations without selecting GPU
   thread syntax or embedding target queue descriptors.
   `invoke_transport_program` binds named transfer/publication tuples through
   the generated manifest, so frontend kernels do not maintain flat argument
   order.
5. The frontend retains dynamic address and predicate calculation because
   those values depend on application-kernel state.
6. The existing transport compiler forms Queue IR
   (`post_put`, `post_signal`, `submit`, `poll`, and `drain`) and lowers it
   through the selected backend.
7. The runtime resolves stable region identities to live allocations,
   registration keys, endpoint descriptors, queues, and peer resources.

Transport geography crosses this pipeline as a compile-time resource fact.
For a static peer the selected backend is folded directly. For dynamic mixed
routing, the C++ transport planner partitions registered peer geography into
homogeneous launch descriptions. `CompiledPhase.launches` exposes each
backend, exact peer set and grid extent. The client compiles each with
`phase.compilation_launch(launch)` and executes the launch list on its own
streams. The all-to-all benchmark uses this planner rather than grouping peers
in Python. Passing a mixed
segment to one representative backend, or inserting a per-operation transport
tag branch, is outside the compiler contract.

The artifact describes code and resource requirements. The client decides when
and where to launch it.

## Physicalization boundary

The Pattern author declares logical communication. The integration author
binds runtime buffers and frontend attachment points. The compiler derives the
physical transfer program between those two contracts.

MoE dispatch/return and KV-cache migration use this path. Their payload
operations, dispatch count transfer, and maximum byte bounds come from
compiled communication stages; readiness publication comes from the
attachment. Full-batch versus fixed-chunk dispatch is an explicit
`items_per_message` parameter, not a compiler scheduling subsystem.

The complete ownership decision is
[ADR 0001](decisions/0001-compiler-owned-physicalization.md).

## Implementation boundary

Semantic operations lower through Queue IR to P2P, host proxy, direct mlx5,
and EFA paths. Pattern analysis produces semantic dataflow and transport
plans.

`CompiledPatternPlan` carries typed `KernelTransportProgram`s produced by
`physicalize_pattern`. Triton kernels invoke generated manifests; the
materialization pass imports the fragment and inlines its semantic operations
before completion planning, batching, and target lowering. Typed
`RuntimeBindings` bind logical regions to live storage.

The generic bridge deliberately leaves packing, dynamic address expressions,
and predicates in the enclosing application kernel. `CompiledPatternPlan` also
carries a per-phase schedule selected from semantic retargeting legality,
transport topology, and payload size; the executor consumes it to choose
inline, direct-destination, or multi-CTA return templates. Schedule selection
is compiler owned; the template kernels themselves are client-owned
(`P2PRealizationProfile`), not generated artifacts.

## Opaque kernel capabilities

Opaque stages expose effects and explicit integration permissions without
exposing their instructions. The architecture groups those permissions into
three cumulative capability levels:

1. **Effect-only.** The stage declares reads, writes, outputs, and completion.
   Aboleth may schedule transport around it but cannot rewrite its
   destinations.
2. **Destination-bindable.** The stage accepts a compatible output
   destination selected by the semantic plan. This permits direct-to-final
   storage when lifetime and aliasing checks pass.
3. **Indexed-producer.** The stage also accepts an indexed scatter destination,
   allowing a producer epilogue to materialize results through a proven index
   map.

Accepted address spaces and the completion contract constrain every level.
Asynchronous stages additionally declare event ordering and maximum in-flight
work. These declarations are permissions supplied by the client adapter, not
claims that Aboleth can inspect arbitrary CUDA binaries.

## Artifact forms

Materialization must support three forms:

- **Inline fragment:** semantic transport inserted into an existing client
  kernel.
- **Bindable epilogue:** a producer or consumer epilogue attached to an opaque
  kernel through its declared capability.
- **Standalone communication kernel:** a separate kernel returned to the
  client when no legal attachment exists.

Selecting a form never transfers launch ownership to Aboleth.

## Clients

Triton is the reference frontend; the generated Tier-1 device library is also
bound into Triton and CuTe DSL kernels ([`TIERS.md`](TIERS.md)). vLLM is the
first application client: an experimental integration whose all-to-all
manager owns client lifecycle and whose prepare/finalize adapter binds vLLM
tensors to the generic MoE Pattern
([`VLLM_INTEGRATION.md`](VLLM_INTEGRATION.md)). DeepEP is used only as a
benchmark baseline. Client-specific handles or queue conventions must not
enter the core Pattern, plan, or Queue IR contracts.
