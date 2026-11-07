# Aboleth programming model

Aboleth programs describe communication and the storage effects that constrain
it. They do not describe an application executor. The client owns routing,
compute, launches, streams, events, and CUDA graph integration.

## Pattern

A `Pattern` is a frontend-neutral description of communication around opaque
client stages. It declares:

- Logical regions and views.
- Bounded runtime worksets.
- Transfers, copies, reductions, publication, and observation.
- Opaque stages and their effects.
- Dataflow order and cyclic lifetime facts.
- Legal frontend attachment points.

A Pattern contains semantic facts, not raw endpoint descriptors, Python launch
callbacks, CUDA streams, or target queue commands. Pattern analysis may remove
dead materialization, retarget a transfer to final storage, or simplify a
cardinality-one reduction when the declared effects make that legal.

## Regions and effects

A logical region records the properties needed for dataflow and layout
reasoning:

- Element type, shape, strides, and index map.
- Local or peer visibility.
- Immutability, escape, registration, and donation properties.
- Reuse interval and required pool depth.
- Relationships to source regions and indexed views.

At runtime, a region effect identifies a view by allocation, allocation
generation, byte offset and length, iteration stride, count, and optional
periodic wrap. This identity is stable across endpoint replacement and is more
precise than comparing raw addresses.

Operations and opaque stages declare reads and writes over these regions. The
compiler uses the effects to determine:

- When transfer source storage may be reused.
- When destination storage may be retargeted or recycled.
- Which publication must carry payload visibility.
- When endpoint capacity forces completion.

Unknown identity, aliasing, or effects overlap conservatively. Runtime
addresses, registration keys, and endpoint descriptors are bound only after
semantic planning.

## Bounded worksets

A workset represents runtime-generated communication items without exposing a
GPU thread mapping. It declares:

- Typed fields such as source row, destination peer, and destination row.
- A compile-time maximum item count and per-field bounds.
- An optional runtime count.
- Whether destinations are unique and whether item order is significant.
- Optional byte-count and predicate fields used by transfers.

`transfer_each` applies one logical transfer to every active item. The client
still computes routing and populates the fields. Aboleth validates the bounds
and compiles transport work; it does not discover experts, pages, or peers.

`physicalize_pattern` turns the bounded workset into a fixed set of typed
transfer slots for the attached kernel. The client kernel supplies each slot's
runtime address and predicate from the populated workset fields.

The runtime resource binding may also carry a canonical peer-rank to transport
geography. It is a compile-time fact, not a dynamic endpoint property: it enters
the generated-plan/cache identity. Static peers lower directly to their
backend; dynamic mixed worksets are partitioned into backend-homogeneous
groups before lowering so no operation chooses a transport from a runtime tag.

## Opaque stages

An opaque stage represents client code that Aboleth does not inspect or
regenerate. Its contract includes:

- Regions read and written.
- Logical outputs.
- Synchronous or asynchronous completion.
- Maximum in-flight work and event ordering.
- A completion contract and accepted address spaces.
- Optional permission for destination rewrites or indexed producer output.

The capability levels and attachment rules are defined in
[`ARCHITECTURE.md`](ARCHITECTURE.md). Effects allow Aboleth to place transport
obligations around the stage; capabilities determine whether it may also bind
an output destination or epilogue. The client remains responsible for invoking
the stage and honoring its completion contract.

## Semantic transport operations

Executable frontend artifacts use a small transport-neutral contract:

- `put(endpoint, destination, source, bytes)` appends an asynchronous remote
  transfer to an endpoint's ordered stream. It does not select a queue,
  construct a descriptor, or require submission at the call site.
- `put_inline(endpoint, destination, *values)` appends the same transfer with the
  payload carried as values rather than as a region. The transferred bytes are
  the concatenation of the values' little-endian representations, so there is no
  staging buffer and no registration key to bind. A transport that cannot embed a
  payload in its command rejects the operation instead of staging it silently.
- `signal(endpoint, destination, epoch, after=...)` publishes a monotonically
  increasing epoch, passed by value. A dependency on payload transfer requires
  strong publication; a standalone acknowledgment or credit may use weak
  publication. Epochs may skip values (the compiler elides epochs it can
  prove a later one supersedes), never go backwards.
- `wait(transfer)` is a sender-side completion boundary. The represented
  transfer and preceding ordered work are complete, so their source bytes may
  be reused. `wait(transfer, iteration=e, reuse=R)` instead declares that the
  source is rewritten only R iterations later, and the compiler plans how
  often to complete.
- `wait_signal(cell, epoch)` is a receiver-side observation boundary: it
  returns once the cell holds *at least* `epoch`. Observing a strong
  publication makes the payload before it, and before every earlier epoch,
  visible; observing a weak publication establishes only the epoch.
- `drain(endpoint)` completes all outstanding sender work on the endpoint. It
  carries persistent transport state across artifact or kernel boundaries and
  is placed before deferred source storage is reused.

Backends may batch puts, delay submission, coalesce completion, and legalize
publication differently, but they must preserve these observations. Detailed
transform rules are in
[`COMPILER_SEMANTICS.md`](COMPILER_SEMANTICS.md).

## Planning, artifacts, and runtime binding

Pattern analysis produces a semantic plan containing rewritten dataflow,
lifetimes, transport obligations, region requirements, and attachment
requirements. `physicalize_pattern` combines that plan with a frontend/runtime
attachment and derives a `KernelTransportProgram`: typed arguments, ordered
transfers, publication, completion, and an invocation manifest. The frontend
supplies dynamic pointer arithmetic and predicates; Aboleth supplies the
executable communication body and argument layout.

Native MoE dispatch/return and bounded KV-cache migration use this path. Their
Triton kernels invoke program IDs at typed attachment points;
`aboleth-materialize-transport-phase` imports the compiled fragment, replaces
the invocation with semantic `put`/`signal`/`planned_wait` operations, and then
runs the existing Queue IR and backend pipeline. The other two artifact forms,
bindable epilogues and standalone kernels, are produced by
`realize_bindable_epilogue` and `realize_standalone_kernel`
(`python/aboleth/artifact.py`).

## Cross-kernel placement

Every kernel completes its own transport: submissions retire eagerly at their
completion interval, and the doorbell rings in the kernel that posted. Aboleth
does not accept a client-declared kernel schedule for cross-kernel completion
deferral, fusion, or deferred doorbells. Cross-kernel placement is future work
and is only sound where every member kernel compiles through Aboleth and their
IR itself proves the schedule; `publish_head`, external publication, and
`deferCompletion`/`drain` are the realization substrate such transforms would
target.

## Runtime binding

`Runtime` and `World` provide the control-plane binding:

- Register live allocations and validate region views.
- Create and connect transport endpoints.
- Exchange peer allocation handles and registration keys.
- Bind frontend artifact operands to device pointers and endpoint descriptors.
- Supply the connected peer transport geography used to specialize artifacts.
- Own target resources through completion and teardown.

They do not route application work, launch compute, assign streams, capture
graphs, or execute the Pattern.
