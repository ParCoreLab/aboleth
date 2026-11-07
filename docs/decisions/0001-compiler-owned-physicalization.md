# ADR 0001: Compiler-owned communication physicalization

Status: accepted

## Decision

Aboleth compiles communication around opaque application compute. It does not
own application launches, streams, routing algorithms, or expert execution.

The compiler boundary is:

1. A `Pattern` declares logical regions, worksets, transfers, ordering, and the
   effects of opaque compute.
2. A frontend/runtime attachment identifies the producer/consumer boundary,
   enclosing kernel, insertion point, runtime buffers, peer partitioning, and
   publication.
3. `physicalize_pattern` derives the physical transfers, byte bounds, typed
   argument slots, `KernelTransportProgram`, and invocation manifest.
4. The materialization pass imports that program at
   `invoke_transport_program`.
5. Queue IR selects submission and completion behavior before P2P, proxy,
   RDMA, or EFA lowering.

The Pattern author does not enumerate transport commands or maintain frontend
argument order. The attachment still computes dynamic addresses and
predicates because those values are produced inside the enclosing application
kernel.

## Consequences

- `CompiledPatternPlan` is the shared compiler/runtime artifact.
- Triton, CUDA, and framework integrations are adapters over the logical
  Pattern and physical transport program, not separate protocol definitions.
- Transfer granularity remains an explicit attachment parameter. It is not a
  public scheduling subsystem.
- Client-specific handles, queue conventions, and wire layouts remain outside
  the core Pattern and Queue IR.

## Current use

MoE dispatch/return and indexed KV-cache migration derive their transport
programs from compiled communication stages. Their Triton kernels invoke typed
manifests rather than flat, manually ordered argument tuples.
