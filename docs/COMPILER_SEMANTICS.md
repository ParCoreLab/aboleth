# Aboleth compiler scheduling contract

This document defines the behavior that Aboleth may optimize without changing
the program observed by either GPU.

## Compiler boundaries

Aboleth's semantic and scheduling core is independent of Triton:

```text
frontend values
→ bind_endpoint / bind_local_region / bind_remote_region
→ begin_group / put / put_inline / signal / wait / planned_wait / wait_group
  / drain / wait_signal
→ forward staged payloads / materialize completion predicates
  / plan publication strength
→ post_put / post_inline / post_signal / post_nop / submit / poll / wait_group
  / drain
→ Triton adapter lowering
```

Core operations use `!aboleth.endpoint`, `!aboleth.local_region`,
`!aboleth.remote_region`, `!aboleth.token`, and `!aboleth.group`. Raw Triton
pointers and registration keys exist only on frontend binding operations. The
current Triton plugin creates those bindings while preserving the Python API.

The core is also independent of any one transport. Passes above the Queue IR
boundary ask what a transport can do — whether it delivers in order, offers
remote atomics or immediate data, bounds outstanding work by a queue, or can
request one completion for a whole batch — rather than testing which transport
it is. Those answers live in one registry,
[`TransportCaps.h`](../include/aboleth/Target/TransportCaps.h), shared by the
scheduling passes, the runtime, and the integration layer. A transport the
registry does not know is rejected at the boundary rather than planned as an
unknown queue.

## Optimization passes

`python/aboleth/compiler.py` runs them in this order on every Triton kernel.
`ABOLETH_PASSES` switches the optional ones by the name in the last column,
comma-separated: `-name` off, `+name` on (`ABOLETH_PASSES=-elide-epochs`).
It is part of the kernel's cache key. `form-batches` and `order-producers`
are needed to lower at all.

The RDMA lowering has two emission switches. `invariant-descriptors` (reads
endpoint-descriptor words with pure non-coherent loads that CSE shares across
a batch's WQEs) is on by default and
`vector-wqe-stores` (16-byte RDMA-write WQE stores) is off by default, on every
target. `TARGET_OPTIMIZATIONS` in `compiler.py` is a per-target override table
and is empty. On SM100 (B200) the scalar descriptor path can be faster; select
it with `ABOLETH_PASSES=-invariant-descriptors`. Both the Triton path and
detached artifacts use the compilation target, without querying a GPU during
CPU planning. The cache identity includes the target-policy revision and
explicit overrides so a forced setting cannot reuse the target default's
kernel. The same effective settings enter generated device-library identities.

| Pass | What it does | Default (switch) |
|---|---|---|
| `shape-specialization` | power-of-two capacities from declared iteration and overall bounds, with a general upper-bound fallback; nonuniform/unknown selectors and off retain one specialization; control state stays private per variant/slot | on (`plan-shapes`, planning) |
| `endpoint-fanout` | unordered record slices, endpoint lanes, measured sender geometry, count ownership and receiver readiness from exact platform and registered resource facts; unknown platforms and off use one lane and one sender warp | on (`plan-endpoints`, planning) |
| `coalesce-receives` | join a peer's independent lane signals in one receiver program when the declared kernel ABI supports it; every signal retains its acquire | on (`plan-endpoints`, planning) |
| `plan-storage` | physical row extents from all bounded reads and writes; logical shapes stay intact; off restores declared extents | on (`plan-storage`, dataflow planning) |
| `dead-communication` | removes unused logical transfers and their unobserved storage from the actual consumer graph; off retains the generic interface | on (`simplify-communication-dataflow`, planning) |
| `materialize-valid-extents` | normalizes declared valid counts and clamps transfers to the valid source prefix; off keeps generic capacity transfers with mandatory address/count safety | on (`valid-extents`) |
| `materialize-completion` | turns `wait(t, iteration=, reuse=R)` into a poll every k-th iteration, k the largest power of two within R, the transport's preference and what the queue holds; off, every iteration waits | on (`completion-planning`) |
| `forward-inline-payloads` | rewrites a `put` whose source is only staging stores into `put_inline` and deletes the stores (P2P, mlx5) | on (`forward-inline-payloads`) |
| `plan-publication` | marks standalone signals weak; picks the strong mechanism: CAS, or a fenced write when every receiver registration is strictly ordered | on (`plan-publication`) |
| `form-batches` | one `submit` per chain, across payload loads and stores | always |
| `coalesce-submissions` | merges independent same-endpoint submissions: one reservation, doorbell and CQE | on (`coalesce-submissions`) |
| `order-publications` | in a merged batch, payload first and strong signals chained, so only the first is an ordering point | on (`order-publications`) |
| `elide-epochs` | in a completion-planned loop, publishes signals only when completion is due or on the last iteration | on (`elide-epochs`) |
| `defer-announcement` | rings only when a planned poll is due | off (`defer-announcement`) |
| `order-producers` | one producer barrier before the first submission that reads a region written by several threads; a private callee's entry is as unordered as its call sites | always |
| `analyze-issuers` | `exclusiveIssuer` when the endpoint index is injective in the program id: plain-add reservation, no ordering wait, one fence | on (`analyze-issuers`) |
| `lower-rdma` `self-reclaim` | in a module with a `selfReclaim` poll whose submissions are all single-issuer and module-owned: capacity waits and polls consume CQEs themselves, and `selfReclaim` batches always request their CQE (mlx5 lowering below); any other module is unchanged | on (`self-reclaim`) |

`plan_endpoint_work` serializes a bounded unordered record domain and the
homogeneous partitions returned by `plan_transport_partitions`. The C++
`plan-endpoints` pass matches exact GPU/NIC/submission and queue-memory facts
to a measured platform profile, then emits power-of-two lanes capped by both
registered endpoints and indivisible records. It emits exact row slices, a
lane-major endpoint grid, count ownership on lane zero, and the matching
receiver grid. A receiver must acquire every lane signal before consuming the
records. Empty valid slices still publish. Unknown profiles use one lane.
The result, including measurement identity, belongs to the artifact identity.

The profile is a runtime-reported fact, like queue placement: the runtime
measures, the planner decides. `World.endpoint_profiles()` (collective) uses a
pinned, cached or seed entry when every rank names the same one; otherwise the
ranks run `endpoint_probe` once (put, strong signal and sender wait per lane
over 1-8 RDMA and 1-16 P2P lanes, ranks paired as a permutation, each trial
taken at its slowest rank), fit the samples identically
(`endpoint_profiles.fit_profile`: the fewest lanes within 95% of the widest at
the largest size where fan-out still pays 15%; bytes per endpoint = the
smallest size where that width pays 15% over one lane, divided by the
width; a held-out
size must keep the planned lanes at 90% of one lane), and every rank caches
the record in the user's cache (`$XDG_CACHE_HOME/aboleth/endpoint-profiles`,
not Triton's cache, which launchers make private per rank), keyed by
platform, queue memory and peer link. Stale (other probe protocol) or corrupt entries are never trusted;
`python -m aboleth.endpoint_profiles` lists, shows, pins, retires and
calibrates them. A disabled, impossible or failed probe leaves one lane and
says why. The resolved identity and source join the compile cache key
(`compilation(endpoint_profile=...)`) and the MoE manifest.

An opaque stage may declare `bindableOutputs`, carried from the client's
`accepts_destination_rewrite` capability. With `plan-storage` enabled, dataflow
simplification can bind a unique output/copy chain directly to its donated,
registered final destination. Direct roots, dominance, identical layouts,
non-escaping staging and ordered producer/consumer pool lifetimes are required.
The resulting opaque write binding and removed staging allocation come from
C++. This is same-value destination forwarding. It does not license reusing
unrelated live buffers or retiring a NIC read at issue time. With the storage
pass disabled, the original output and copy remain.

`plan_shape_family` sends the declared bound range, in-flight slot count and
rank-uniformity fact to C++. `plan-shapes` emits power-of-two capacities, the
first-capacity-at-least-bound selector table, and disjoint control-domain ranges
for every variant. The runtime only interprets those thresholds and realizes
the requested artifacts. It does not round capacities or pick a decode path.
Unknown uniformity uses the one largest capacity. Runtime bounds above the
declaration and slots outside their domains are rejected. Mutual exclusion
between variants is a fact for later storage analysis, not a buffer alias proof.

Rank-indexed mixed transports supply `rank_count` and one canonical participant
image in every endpoint partition. The image carries exact platform, queue
memory and allocated lane facts for every rank. C++ caps active fanout by that
common resource image; an unqualified participant selects the common one-lane
plan. A separate common `ready_stride` addresses rank-indexed signals even
when P2P and RDMA use different active lane counts. All flags fit the emitted
whole-world `signal_slots`. Inactive padding cells are neither published nor
waited on. Decode's indivisible one-record bound still gives stride one.

Shape variants and DBO slots require private publication state or an equivalent
proved sequence protocol. An application-wide epoch cannot be reused on flag
cells skipped by a smaller lane count. `records.py` is one parameterized
semantic template: payload and metadata share the planned slices, the count
uses `put_inline`, and the ordinary batching, publication and completion
passes lower its chains. The caller supplies addresses and executes the
artifact's grids. No runtime lane search or frontend Queue IR is involved.

Issuer analysis follows private Triton helper arguments to unambiguous call-site
values. A helper argument is not assumed uniform across CTAs. Endpoint
selections in distinct helpers using the same actual table are checked
together; conflicting or unknown index expressions retain shared reservations.
The degenerate `lane + peer` linearization constrains the peer grid extent to
one and keeps the lane axis. Launch checks remain mandatory.

The lowering then picks scopes from what the compilation knows: a GPU-scope
doorbell when every endpoint's queues are in GPU memory (`gpu-doorbell`), GPU-scope acquires
of words only this GPU writes. Canonicalize and CSE run after lowering.

Issuer analysis also proves a constant modulo rotation `(pid + c) % N` when
the nonnegative sum cannot overflow, and a linearized `lane * W + peer` index
over two program axes. The peer coordinate may itself be such a rotation.
The endpoint carries the selected axes and required grid extents; lowering
asserts those extents, requires unused axes to have size one, and checks that
the two-dimensional index fits nonnegative i32 before resolving the handle.
A different launch grid cannot silently reuse that proof. Arbitrary loaded
orderings remain shared: immutable permutation facts and their cache identity
must be established before that case can become exclusive.

## High-level operations

`aboleth-plan-transport` partitions a transport program's registered
`peerRanks`/`peerTransports` geography into `launchPartitions`. The verifier
requires sorted unique nonnegative peers and registered transport classes.
Each partition supplies its exact peer set, backend and grid extent to
`CompiledPatternPlan.transport_partitions` and `CompiledPhase.launches`.
These facts participate in transport IR and specialization identity.
Transport partitioning is required legalization, rather than an optional
optimization: a mixed peer set cannot lower through one backend. The client
still owns stream assignment and launch execution.

`CompiledPatternPlan.storage_plan` carries allocation facts from the C++
dataflow pass: logical and physical shapes, strides, byte counts and reader
lifetime facts. Workset index bounds constrain source and destination rows;
indexed views constrain payload access but still require their full index
metadata. Full-region uses, escaping storage and unproven layouts retain the
declared extent. Storage facts participate in the artifact cache identity.
The existing `regions` list remains a logical binding contract. Allocators
using physical bounds consume `storage_plan`; the pass currently introduces
no cross-allocation aliases or retirement operations.

At the kernel level, `region.valid(count, capacity, item_bytes)` declares a
dynamic valid prefix followed by dead padding. Its `count` is normalized by a
C++ semantic operation, so packing masks and `put_inline` metadata can use
that same value. `.at(offset)` retains the origin-relative byte displacement.
`materialize-valid-extents` bounds a put by both its requested bytes and the
remaining valid source bytes. Negative counts normalize to zero; over-bound
counts normalize to capacity. Negative or exhausted displacements yield zero
payload. Arithmetic uses i64 until the verified i32 transfer-length limit.
Ordinary batching, zero-byte NOP ordering and publication follow afterward.

With `-valid-extents`, count normalization and capacity/address safety remain
mandatory, while the transfer may include the declared dead padding. The
generic binding therefore requires the full declared source allocation. A
short physical source-prefix binding requires an explicit compiled storage
contract; this operation alone does not permit an adapter to bind fewer rows.

The dataflow contract for a short immutable input names `validCount` and
declares `deadPadding`. An opaque reader's `prefixReadCounts` states the exact
runtime mask for each read operand; an empty string means a full read. With
`plan-storage` enabled, C++ grants a `prefix_binding` only when every actual
reader uses the declaration's count, storage is dense, immutable, unregistered
and non-escaping, and all readers are ordinary ordered opaque compute. Writes,
views, unbounded uses, unknown/asynchronous readers and mismatched masks retain
full binding. The permission gives the count name, maximum rows and row bytes;
it never authorizes temporal reuse or aliases of unrelated live allocations.

The runtime checks actual count, CUDA device, dtype/layout and byte span
against this permission. Borrowed inputs cannot overlap the artifact's owned
payload or control storage. Zero rows are legal when every read is masked.
Without the storage pass, inputs retain their declared allocation and copy
path. The ordinary pack computation and its count argument stay identical.

`aboleth.put` describes an asynchronous remote-memory transfer. It does not
promise that the NIC has been notified at the call site. The compiler may delay
submission while preserving the dependencies below.

Puts may carry an explicit `after` token. Chained puts define endpoint order in
SSA rather than relying only on statement position. A put may also carry a
runtime predicate. Disabled RDMA puts become ordered mlx5 NOP WQEs; disabled
P2P puts skip their copy region.

`aboleth.put_inline` describes the same asynchronous transfer with the payload
supplied as operands instead of as a local region. The transferred bytes are the
concatenation of the values' own little-endian representations, so the byte count
is the sum of the operand widths and is a property of the operation rather than a
runtime value. There is no source region and no registration key: nothing stages
the bytes in memory, and the transport embeds them in the command it builds. A
`put_inline` orders, chains, predicates, and batches exactly as a `put` does; the
two forms are interchangeable wherever a transport can carry a payload inline,
and a transport that cannot rejects the command at the Queue IR boundary.

`aboleth-forward-inline-payloads` rewrites a `put` into a `put_inline` when the
source region is provably nothing but staging for that transfer, and removes the
staging stores. It proves, rather than assumes, every fact it needs:

- The byte count is a compile-time constant no greater than the pass's
  `inline-bytes` cap. The default cap, 28 bytes, is the largest mlx5 inline
  payload that still fits one WQEBB.
- The source address resolves to a constant byte offset from an identifiable
  base through `tt.addptr` steps and pointer bitcasts. A runtime step leaves the
  address unresolved.
- Scalar stores in the put's own block, before the put, tile the transferred
  range exactly: no gap the buffer would still have to supply, no overlap whose
  order the pass would have to guess.
- Nothing between the last such store and the put can touch that range. An
  operation is crossed only when it has no memory effects at all or when it
  provably accesses a disjoint range of the same base.
- The range is reachable from nothing else in the kernel. Because forwarding
  deletes the stores, a range any other operation can read is live storage rather
  than a staging slot. Sibling slots in one buffer are allowed, since they
  resolve to disjoint ranges of the same base.

Any fact the pass cannot prove leaves the region-sourced `put` in place; it never
guesses. Float payloads travel as their own bits through an `arith.bitcast`,
because the transport copies bytes and never interprets them.

`aboleth.signal` publishes a remote signal. When supplied an input token, the
signal cannot be submitted before that transfer. A signal without an input
token forms a one-command batch, used for receiver acknowledgments. Rotating
ring buffers use this form to publish a consumed-slot epoch on the reverse
endpoint before the sender overwrites that remote slot.

Publication strength is an internal IR property. A strong publication makes
observation of the new epoch imply visibility of payload ordered before it on
the same endpoint and peer. A weak publication makes only the epoch observable.
`aboleth-plan-publication` keeps every payload-bearing or grouped signal strong
and marks only standalone, ungrouped signal batches weak. It also selects each
strong signal's mechanism, below, for semantic `signal`s and for `post_signal`s
written directly in Queue IR. Frontends do not select this policy.

Signal epochs increase monotonically in their destination cell, and
receivers wait for *at least* an epoch (`wait_signal` spins while the cell is
below it), never for an exact one. A published epoch may therefore skip
values, which is what epoch elision does; it may never go backwards. A
checked publication (the compare-and-swap) replaces exactly the value it
expects: `epoch - 1`, or the epoch last published when elision names it.

How a backend achieves strong publication follows from what its transport can
do, not from which transport it is. `legal_strong_publication` in
[`TransportPlan.h`](../include/aboleth/Target/TransportPlan.h) derives the mode
from the capability registry:

- Peer memory reached by loads and stores publishes with a release store.
- An ordered queue with remote atomics uses compare-and-swap as both an
  ordering point and a checked counter update, since prior relaxed-ordering
  writes cannot bypass it.
- An ordered queue whose transport has a fenced write (`fenced_write`) uses
  that write instead, but only when every receiver registration is strictly
  ordered (`strict-registrations`, which `World.compilation()` sets when every
  live region exchange imported only registrations made without relaxed
  ordering). Whether a fence orders relaxed-ordering writes is unverified, so
  it is never selected for them. The fenced write does not check the
  predecessor epoch.
- An unordered queue cannot order a later command against earlier payload at
  all. The sender must instead retire the payload completions ordered before
  the signal and only then publish the epoch, or move publication to the
  receiver, which counts payload arrivals before advancing the cell.

The sequence rule is a property of the epochs, not a promise that
hardware verifies it. On a transport with remote atomics the compare-and-swap
*rejects* an out-of-sequence publication — a missed compare writes nothing and
still completes successfully — but the rejection is not currently surfaced:
nothing reads the returned predecessor, so a violated sequence is
indistinguishable from success. Surfacing it needs one atomic-result slot per
send-queue entry rather than the single per-endpoint sink that exists today.
Without remote atomics the epoch is an ordinary write, and monotonicity holds
because one sender owns each destination cell; the check is simply unavailable.
Checked publication is therefore an optional integrity property of the
transport rather than part of the strong-publication contract, which states
only that observing the epoch implies visibility of the payload published
before it.

`aboleth.wait` is a sender-side completion boundary. Every transfer represented
by its input token must be submitted and complete before execution continues.
The associated source memory may be reused afterward.

A device-loaded epoch may drive `wait(..., iteration=epoch, reuse=SLOTS)`
when that observation dominates the transfer. The compiler does not move a
load backward to manufacture this proof. A bound transport fragment carries
its own planned completion interval and queue-capacity facts through
`programId`; different fragments in one enclosing kernel keep different
intervals. All planned chains sharing an endpoint in the same block count
toward the aggregate queue bound. The actual compile-time endpoint capacity
may further shorten the interval. Disabling completion planning makes these
waits eager as well.

For a queued bound phase that may defer completion, the C++ planner emits a
final-drain obligation. `CompiledPhase.retirement_manifest` supplies a generic
semantic drain fragment in the same artifact. The client executes it for the
artifact's endpoints at slot retirement and teardown, before releasing their
source storage or registrations. The compiler supplies the protocol; the
client retains its launch and stream ownership. Legacy unbound planned waits
retain the module option as a compatibility fallback until their callers are
removed.

`aboleth.planned_wait` states the same semantic obligation at an iteration
boundary without encoding completion frequency in kernel source. Transport
planning materializes it as an unconditional or predicated `wait`.

`aboleth.wait` may carry a runtime predicate. A false predicate defers sender
completion; a later true predicate on the same single-producer endpoint
completes all intervening ordered submissions. Source regions from deferred
submissions must remain live until that later boundary. Transfer-group planning
bounds the interval by queue capacity and guarantees a final true predicate.

A predicate written by the kernel (`wait(t, predicate=p)`) carries
`selfReclaim`: no planner bounds what it defers by the queue's depth, and the
kernel must end with a true predicate (or a drain). On a backend with a bounded
queue, the submissions it defers may therefore outrun the queue. Where the
backend reclaims (mlx5 under `self-reclaim`), a submission that finds the queue
full retires the oldest completed submissions itself, so the deferral is
bounded only by the kernel's own final wait; the source-liveness rule above is
unchanged. Elsewhere the capacity wait waits for a poll as before, and a
deferral longer than the queue does not complete. Waits produced by
completion planning never carry `selfReclaim`.

`aboleth.begin_group` creates an endpoint-scoped completion group.
`aboleth.signal` may attach its submission to that group instead of being
consumed by an immediate `wait`. `aboleth.wait_group` is then the sender-side
completion boundary for every attached submission. Grouping changes completion
frequency, not endpoint order or receiver-visible signal epochs.

`aboleth.wait_signal` is a receiver-side observation boundary. After a strong
publication, its return makes the requested signal and data ordered before that
signal visible to the receiving GPU. After a weak publication, it establishes
only that the requested epoch is visible.

## IR effects and verification

High-level and scheduled communication operations carry explicit MLIR memory
effects:

- Endpoint operands read and mutate the endpoint's ordered-stream resource.
- Source regions read Aboleth local memory.
- Destination operands write the remote-memory resource.
- Completion boundaries read and mutate endpoint state.

Scheduled `submit` operations verify that every token belongs to the same
endpoint and that the commands form a sequence of segments: each segment is a
chain of posted puts that a posted signal may close, every command depends on
the one before it in its segment, and a segment's first command depends on
nothing in the batch. Batch formation makes one segment per chain;
`aboleth-coalesce-submissions` concatenates the segments of independent
same-endpoint submissions when nothing between them blocks, so the merged batch
has one reservation, one announcement and one completion while each chain keeps
its own signal. `aboleth-order-publications` then moves a merged batch's payload
ahead of its signals and chains each strong signal after the previous one: a
strong signal may depend on the strong signal immediately before it, which
orders the payload for both. Chains are unordered with respect to each other and
signals may be delayed, so only each chain's own order had to hold. A trailing
`post_nop` may stand after the last signal, in which case it is the batch's
completion carrier and the signal need not be present at all.

`submit {deferredAnnouncement}` rings only when its (predicated) poll is due;
its commands still advance the endpoint's ready head, so the next announcement
on the endpoint, at the latest a drain, covers them. `aboleth-defer-announcement`
sets it only in kernels that never wait on a peer (`wait_signal`), because a
peer's reply could depend on a command that was not announced. It is opt-in
(`ABOLETH_PASSES=+defer-announcement`); it is intended to be combined with
epoch elision.

`aboleth-elide-epochs` (on by default, `ABOLETH_PASSES=-elide-epochs` disables)
predicates the signals of a
submission in a counted loop on "completion due or last iteration": a later
epoch in the same cell supersedes a skipped one, and every payload is still
published no later than completion. A false predicate leaves an ordered no-op
in the signal's slot. A checked publication (the compare-and-swap) replaces
only the value it expects, `epoch - 1` unless `post_signal ... expect %v`
names it; elision threads the epoch last published through the loop as an
`scf.for` iteration argument (`epoch - 1` on the loop's first iteration) and
names it there. Kernels that `wait_signal` are left alone: a peer's reply could
depend on an epoch no longer published in time. A kernel that waits on a peer
by other means (a raw spin on memory) is invisible to the pass and must use
`wait_signal`, or disable elision.

`poll` must consume a token produced by
an ungrouped `submit`. Grouped signals and submissions must reference a
`begin_group` for the same endpoint, and one `wait_group` consumes the group.
These invariants remain explicit after batch formation rather than depending on
the original Python statement layout.
Every backend emits commands in the verified `submit` operand order, not by
independently walking operation kinds. That walk, the pairing of a submission
with its completion poll, and the retirement of consumed operations are shared
by all lowerings rather than repeated per backend.

## Submission scheduling

The compiler may place multiple `put` operations and their trailing `signal`
into one submission when:

- They are in the same block, with only loads, stores, pure operations,
  `scf.if`/`scf.for` holding only those, and other chains' commands between
  them. A blocking operation ends a chain.
- They target the same endpoint.
- The signal depends on the final put.
- No earlier wait requires partial completion.

Submission timing is not observable except through `wait`, `signal`, and
`wait_signal`. Performance may change because delaying submission trades
doorbell cost against communication overlap.

Batch formation follows explicit token dependencies and refuses to move
submission across unrelated side-effecting operations. A frontend store to the
local signal cell immediately before `signal` is preserved and allowed.

For grouped signals, batch formation still emits one `submit` at every signal
boundary but does not emit a `poll`. The dominating `wait_group` remains visible
to backend lowering, including when submissions execute in a nested loop.

For a predicated wait, batch formation emits a predicated `poll` and marks it
as coalesced. The submit remains at every signal boundary; only completion is
deferred.

## Lower-level operations

`aboleth.post_put` constructs one transfer command without notifying the
backend.

`aboleth.post_inline` constructs one transfer command whose payload is its own
operands. It is `post_put` without a source region or local key, and it follows
the same batch rules: it may chain through `after`, it may be predicated, and it
may never be a submission's final command.

`aboleth.post_signal` constructs one signal command without notifying the
backend. Its optional `weakPublication` attribute records that no payload
visibility is attached; absence of the attribute means strong publication. Its
optional `fencedSignal` attribute selects which of the transport's ordering
points carries a strong publication: the epoch is published by a fenced write
rather than by a remote atomic. It is legal only on a strong publication and
does not change what publication strength means.

`aboleth.post_nop` constructs one ordered no-op command whose only purpose is to
carry its submission's completion. It must be the final command of an ungrouped,
non-deferred submission with an unpredicated poll, and when it is present no
payload or signal command in that submission requests a completion.

`aboleth.submit` groups posted commands into one backend notification. Its
`batchSize` attribute records the number of commands in the group. It is a
command count and not a queue-slot count: how many slots a command occupies is a
property of the transport, and every backend but mlx5 consumes exactly one slot
per command. The mlx5 lowering derives its own slot arithmetic from the posted
commands, as described under its execution lowering below.

`aboleth.submit {headPublication}` publishes the queue at whatever the endpoint's
producer index has reached rather than at the end of its own reservation, and
does not wait for earlier reservations to publish first. The submission
therefore notifies the backend of commands it did not itself build, which is
legal only where the caller holds a barrier proving every reserved command is
complete. Without the attribute a submission publishes its own range in
reservation order.

`aboleth.submit {reservationScope = "cta"}` states that one CTA owns the
endpoint's queue for the submission and that queue capacity is already proven
for the whole round. The reservation is then a block-scoped increment that
reloads no completion state. The default, `"gpu"`, is a device-scoped increment
that waits for capacity.

`aboleth.poll` waits for sender-side completion of the submitted group.
Its optional predicate implements compiler-selected completion moderation.

`aboleth.wait_group` waits for all endpoint submissions associated with one
`aboleth.begin_group`.

`aboleth.submit {deferCompletion}` publishes a batch without requesting or
polling sender completion in that kernel. `aboleth.drain` later appends the
backend completion boundary and retires every outstanding ordered submission
on the endpoint before source storage is reused. Persistent endpoint state,
not an SSA token, connects the two kernels.

The current `aboleth-form-batches` pass transforms:

```text
put × N → signal → wait
```

into:

```text
post_put × N → post_signal → submit(batchSize=N+1) → poll
```

`submit` is placed at the signal boundary, while `poll` replaces the later
`wait`. Pure computation between signal and wait can therefore overlap the
submitted communication without changing the source API.

The grouped form transforms:

```text
begin_group
  (put × N → signal(group)) × K
wait_group
```

into:

```text
begin_group
  (post_put × N → post_signal(group) → submit(group)) × K
wait_group
```

## Direct P2P execution lowering

When the compilation context selects `p2p`, the Aboleth plugin runs
`aboleth-lower-p2p` after batch formation:

- `post_put` checks source, destination, and byte-count alignment. Aligned
  transfers become an `scf.for` loop of 64-bit `tt.load`/`tt.store`
  operations; the fallback remains a byte-copy loop.
- `post_signal` becomes a system-scope release atomic exchange. P2P does not
  need a distinct responder-ordering primitive, but weak credits still retain
  release ordering for local consumer reads.
- `submit`, `poll`, `begin_group`, and `wait_group` become compile-time
  token/scheduling constructs and are removed.
- `wait_signal` remains a system-scope acquire observation loop.

The pass eliminates every Aboleth operation and token before Triton's TTIR to
TTGIR conversion. This produces an ordinary Triton kernel; it does not call a
handwritten transfer kernel or an opaque communication function.

## Host verbs-proxy execution lowering

When the compilation context selects `proxy`, the Aboleth plugin runs
`aboleth-lower-proxy` after batch formation:

- One GPU atomic reserves a ring slot for the complete submission.
- Every posted put and signal becomes a compact command in that slot.
- Predicated puts become NOP commands.
- Commands are published as four packed 64-bit stores; signal commands carry
  their value inline.
- The host posts strong signals as RC atomic compare-and-swap operations and
  weak signals as ordinary inline RDMA writes.
- `submit` publishes the batch with one system-scope release store.
- A host progress thread patches one prelinked RC verbs list and calls
  `ibv_post_send` once.
- `poll` waits on the host-published completion sequence with system-scope
  acquire loads.
- A false predicated `poll` emits no GPU wait. A true one waits for the final
  sequence and therefore covers prior ordered submissions.
- `wait_group` loads the final producer sequence and waits once for that
  sequence; intermediate grouped submissions do not execute GPU polls.
- `wait_signal` retains the same receiver-side acquire semantics.

The queue protocol supports concurrent GPU producers and bounded ring reuse.
The proxy ABI and status codes are documented in [`PROXY_ABI.md`](PROXY_ABI.md).

## mlx5 RDMA execution lowering

When the compilation context selects `rdma`, the Aboleth plugin runs
`aboleth-lower-rdma` after batch formation:

- `submit` reserves one contiguous SQ range for the whole batch; each
  `post_put` generates its mlx5 control, remote-address, and data segments in
  that range.
- `post_inline` generates a control segment, a remote-address segment, an
  inline-data header of `0x80000000 | bytes`, and the payload bytes themselves.
  Only segment headers are big-endian: inline data is not byte-swapped, so each
  value contributes the bytes of its own little-endian representation and the
  emitted 32-bit stores land them at the right addresses. The whole inline
  segment is written, including the padding the adapter ignores, so a queue slot
  never carries bytes from a previous generation. There is no data segment, no
  lkey, and no adapter read of host or device memory for the payload.
- A predicated `post_inline` becomes an ordered NOP that keeps the write's
  segment count rather than shrinking to one segment, so the command occupies the
  same slots whether or not it transfers and the next command still begins where
  the reservation says it does. That NOP requests no completion: a predicated
  placeholder must not compete with the batch's completion carrier.
- Strong `post_signal` generates a 64-bit atomic compare-and-swap WQE. Atomic
  response ordering makes prior relaxed-ordering payload writes visible before
  the epoch can advance. It compares against `expect` when present, else
  `epoch - 1`; a predicated one becomes a no-op of the same size. The returned predecessor is written to an
  endpoint-owned result sink, not the signal source. Weak publication generates
  an inline 8-byte RDMA write of the epoch.
- With a predicated wait, the signal requests a CQE only when the predicate is
  true. The corresponding coalesced poll advances completion from the
  endpoint's current completed head through that signal.
- Grouped signal WQEs do not request CQ updates. `wait_group` appends one
  signaled NOP WQE, polls its CQE, and retires the complete grouped SQ range.
- Deferred submissions likewise omit CQ updates. A later `drain` appends one
  signaled NOP, polls once, and advances the persistent completed SQ head
  through all prior endpoint work.
- Self-reclaim (`self-reclaim`, a module with a `selfReclaim` poll in which
  every submission is single-issuer and reserved, published and polled by the
  module, with no groups, drains, deferred announcements or multi-WQEBB final
  command; any other module is lowered as without it): a `selfReclaim` batch
  requests its CQE whether or not its poll is due. Every capacity wait loads
  the CQ consumer index and the completed head once and, while the batch does
  not fit, consumes the next CQE (owner bit, opcode, and the endpoint's poll
  timeout as the poll checks them) and lifts its 16-bit WQE counter onto the
  completed head: in-order RC completion retires that WQE and everything
  before it. Every poll consumes CQEs the same way until the completed head
  reaches its batch end, since earlier batches' CQEs may precede its own. The
  state is carried in registers through the loop, which is sound because one
  program issues on the endpoint, and is published once afterwards (consumer
  index, CQ DBREC relaxed, completed head released at GPU scope) if anything
  was consumed. A CQE that does not arrive in time or reports an error ends
  the loop and records the error in endpoint state, as the poll does. The CQ
  holds at least as many entries as the send queue, and a batch carries at
  most one CQE, so the CQ cannot overflow.
- `submit` updates the send DBREC with a relaxed store and rings BlueFlame (or
  the staged cell) with one release store: system scope by default, GPU scope
  under `gpu-scope-doorbell`, which the compilation sets only when every
  endpoint's queues are in GPU memory (the NIC then reads the commands from
  GPU memory, where a GPU-scope release has already made them visible).
- `submit {headPublication}` reads the producer index with an acquire load and
  rings the doorbell there instead of at its own batch end, and emits no wait on
  the published-ready word. The post-doorbell release store still advances that
  word, so a later publisher can still read how far the queue was published.
- `submit {reservationScope = "cta"}` reserves with a block-scoped atomic and
  emits no SQ-capacity wait.
- `submit {exclusiveIssuer}` reserves with a plain add and does not wait for
  earlier reservations; it stores its ready head relaxed before the doorbell,
  whose release covers it. Shared issuers release the ready head after the
  ring. A batch requests one CQE, on its last command.
- `submit {producerBarrier}` emits one `ttg.barrier` before the batch.
- A strong `post_signal` whose `after` is a strong `post_signal` becomes a
  plain inline write: the queue executes it after that signal, whose ordering
  point already made the payload before both visible.
- A strong `post_signal {fencedSignal}` becomes an inline write of the epoch
  whose control segment carries fm = 3 (`0x60` in `fm_ce_se`, the kernel
  driver's strong-ordering mode) rather than an atomic compare-and-swap.
- `post_nop` becomes one signaled NOP WQE inside the batch's reserved range. The
  batch's payload and signal WQEs then request no CQ update and the poll retires
  the range through that NOP.
- `poll` checks CQ ownership, opcode, and WQE counter, then advances the CQ
  consumer DBREC.
- `wait_signal` uses a system-scope acquire load loop, avoiding a GPU atomic
  RMW racing a NIC atomic update.

### Queue accounting in WQEBBs

An mlx5 send queue is an array of 64-byte WQEBBs, and its producer index, its
doorbell, its capacity, and its CQE WQE counter are all expressed in WQEBBs. Every
command this lowering builds except an inline write occupies exactly one, which is
why command counts and slot counts have so far been the same number.

An inline write occupies `ceil((36 + bytes) / 64)` WQEBBs: 16 bytes of control, 16
of remote address, 4 of inline-data header, and the payload, rounded up to a whole
WQEBB. Its `ds` field is `2 + ceil((4 + bytes) / 16)` segments. The 28-byte
default forwarding cap is exactly the largest payload for which that is one WQEBB.

The lowering therefore derives its own slot arithmetic rather than reading a slot
count out of Queue IR: every command's size is a compile-time constant, and a
WQEBB is an mlx5 fact that no other backend shares. `submit` keeps `batchSize` as
the command count it has always been, and the lowering computes

- the reservation amount as the batch's total WQEBBs,
- each command's WQE index as the batch base plus the WQEBBs its predecessors
  consumed,
- the batch end — which drives the capacity wait, the DBREC and BlueFlame head,
  and the poll's expected WQE counter — as the batch base plus that total.

The poll's expected counter stays `batchEnd - 1` because a batch's completion is
always carried by its last command, and the last command is always a signal or a
no-op, each one WQEBB. When every command in a batch is one WQEBB these
expressions fold to the numbers the lowering emitted before, so a batch without an
inline write reserves, indexes, publishes, and polls byte-identical IR.

The send queue is circular in WQEBBs, so a command longer than one WQEBB wraps at
the end of the buffer instead of running past it, exactly as mlx5's own providers
continue a long WQE at offset zero. Each WQEBB of an inline write therefore
resolves its own slot rather than being addressed as a linear run from the first.

In device-library mode the library's inline primitive is typed on a 64-bit value
and always builds an unconditional write, so `post_inline` is legal there only for
an 8-byte payload with a predicate the lowering can prove is always active.
Anything else is an error naming the direct emitter rather than a command silently
reinterpreted; extending the device library is not part of this contract.

The lowering reads every resource from the versioned endpoint descriptor in
[`RDMA_ABI.md`](RDMA_ABI.md). It emits ordinary Triton, arith, SCF, and inline
PTX operations and eliminates all Aboleth tokens before TTGIR conversion.

Persistent Triton loops retain queue state through the descriptor's producer
and CQ-consumer addresses. The same generated body can therefore cross SQ, CQ,
and 16-bit WQE index wraparound without host intervention.

The current lowering reserves a whole batch with one atomic operation. Batches
wait for SQ capacity before reusing wrapped slots, publish and ring doorbells in
reservation order, and serialize CQ consumption by the corresponding WQE
range. Completion polling records timeout, CQ-opcode, and WQE-counter errors in
endpoint state. The attributes above select the planned alternative to each of
those defaults; absent the attributes the emission is unchanged.

The pass also takes the mlx5 device library's path, and then emits a call into
that library for each command instead of the segment stores. Every such call is
predicated on the block's first lane. The inline emission carries that predicate
on its own publishing stores, but a call site has none of its own and would
otherwise build one command and ring one doorbell per lane.

## AWS EFA execution lowering

When the compilation context selects `efa`, the Aboleth plugin runs
`aboleth-lower-efa` after batch formation. The pass emits calls into the
efa-dp-direct device library, which is linked as Triton extern bitcode and
inlined, so the generated kernel contains the library's instruction sequence
rather than a call.

SRD is reliable but unordered and offers no remote atomics, which changes three
things relative to the mlx5 lowering:

- `submit` reserves a contiguous range on the endpoint's producer index, waits
  for the send queue to retire enough of it, and opens a batch. Posted commands
  fill their reserved slots; the batch is published by ringing the doorbell in
  reservation order, as for mlx5.
- `post_put` becomes an RDMA write. A disabled put becomes a zero-length write
  rather than an ordered NOP: the batch must still produce the completion count
  the compiler planned, and there is no delivery order for a placeholder to
  preserve.
- `post_signal` becomes an 8-byte RDMA write from a registered source region;
  EFA does not support inline data on RDMA writes. A weak signal posts
  directly. A strong signal first flushes the batch and retires the payload
  completions ordered before it, then opens a new batch for the epoch, because
  nothing else prevents the signal from overtaking the payload.
- `poll`, `wait_group`, and `drain` retire by count. Each consumed completion
  retires exactly one command, so a batch has completed once the retired count
  reaches its end; a predicated poll defers that to the later true predicate.
  No completion placeholder is needed, since the producer index already counts
  outstanding commands.
- `wait_signal` is unchanged: a system-scope acquire load loop on the local
  cell, exactly as for the other backends.

The alternative publication scheme, in which payload carries immediate data and
the receiver counts arrivals, is described in [`EFA_ABI.md`](EFA_ABI.md). It
requires `aboleth.wait_signal` to carry a receive endpoint and a per-epoch write
count, which the current contract does not express.

## Current pass scope

A chain must stay in one block, without a blocking operation inside it;
other shapes are rejected rather than split. Completion groups may
dominate nested loops and complete after those loops. Overlapping completion groups on one endpoint, alias analysis, and
latency-aware early submission are not yet implemented. Unsupported shapes
produce a compiler error instead of being silently transformed.
