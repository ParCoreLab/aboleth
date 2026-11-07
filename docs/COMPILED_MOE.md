# Compiled MoE dispatch and return

Distributed MoE dispatch and return are expressed as a generic Pattern and
compiled by Aboleth. The program uses `MoESpec` and `CompiledMoEPlan` from
`aboleth.moe.compiled_plan`; `aboleth.moe.native` executes it.

## Declared facts and compilation

`MoESpec` supplies the token bound, hidden size, top-k, expert count and rank
count. The client supplies expert ownership, topology, in-flight slot count
and the optimized consumer's destination/quantization capabilities. Routing,
expert placement, compute kernels, launches, streams and graphs stay with the
client, as required by [ADR 0001](decisions/0001-compiler-owned-physicalization.md).

`build_pattern` describes packing, rank-deduplicated record transfers, metadata
conversion, the opaque optimized experts and weighted gathering. The algorithm
states what is read and written. Generic compiler passes determine:

- which return fields have consumers and survive communication;
- physical storage extents, valid source-prefix bindings and output destinations;
- homogeneous P2P/RDMA launch partitions;
- endpoint lanes, issuer proofs, readiness geometry and publication;
- per-phase completion/retirement and bounded shape variants.

The plan contains the compiled Pattern, surviving dispatch/return fields,
compiled launch lists, expert destination, residual-copy fact and phase names.
It contains no application-specific QP ladder or resource-policy environment
selector. Platform endpoint profiles are measured once and passed as facts.

## Runtime binding

`NativeMoEFamily` instantiates the compiler's bounded shape family and its
independent slots. `NativeMoESlot` allocates physical roots from the storage
plan, registers the required regions, binds the expert output destination,
and executes the compiled launch list. It does not select a transport policy
or derive a second set of storage formulas.

The client agrees on the runtime bound across ranks before selecting a
variant. The thin vLLM wrapper does that once per forward when idle ranks have
dummy local counts; graph bounds already agree. Each slot/variant has private
payload, flags, epochs and endpoint reservations. Overflow and unsupported
input overlap fail rather than silently changing or truncating inputs.

A source-prefix binding reads the physical valid rows while the compiled
capacity remains the bound. Empty-source ranks still reset metadata and
participate in publication. Inactive expert choices carry a declared off-rank
placeholder at zero weight for the unchanged optimized consumer.

## Semantic transport and attribution

`records.py` uses semantic endpoints, regions, puts, inline counts, signals,
waits and drain. Native MoE contains no `external_put_group_signal`, explicit
`system_fence`, `publish_head`, magic completion interval or device QP ladder.
The compiler sees the communication and its effects before native NIC lowering.
The generic fence/publication substrate remains for KV and Tier1 clients.

`ABOLETH_PASSES` is the compiler ablation interface. Turning off storage
planning restores generic extents and copies. Turning off shape specialization
retains the upper-bound variant. The all-pass-off arm disables every switchable
optimization while preserving the model, declared bounds and expert kernels.
Dead padding can then cause very large transport/consumer work; an ablation
should report an actual OOM or deadline instead of shrinking the bound to make
that arm pass.

The vLLM bridge, supported configurations and validation scope are documented
in [VLLM_INTEGRATION.md](VLLM_INTEGRATION.md).
