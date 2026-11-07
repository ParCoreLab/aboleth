# Transport effect analysis

Aboleth consumes an existing execution order. It does not choose streams,
reorder compute, fuse kernels, or capture a CUDA graph.

Each communication-bearing kernel contributes semantic transfers and a
summary of the local regions it reads. Opaque kernels contribute a reusable
read/write effect schema. A region is identified by allocation, generation,
byte offset/length, per-iteration stride, and optional periodic wrap. Unknown
aliases conservatively overlap.

For every transfer, the compiler finds the first later write or free that may
overlap its source region. That operation is the local-completion deadline.
Transfers on the same endpoint may remain outstanding until the earliest
deadline or queue-capacity boundary.

Example:

```text
pack             writes send.slot.0
dispatch         reads send.slot.0; posts endpoint E
opaque experts   reads recv.slot.0
next pack        writes send.slot.0
```

The generated phase plan computes the number of iterations for which dispatch
WQEs may remain outstanding. `aboleth-materialize-completion` turns that
interval into the predicate on the phase's sender-completion operation. Direct
mlx5 lowering requests and polls a CQE only on the due iteration.

Aboleth does not insert work into a later client kernel on the strength of a
user-declared schedule; the declared-schedule session that once did was
retired. An integration that needs cross-kernel lifetime control supplies an
explicit phase entry/exit hook, and the compiler optimizes the protocol inside
that contract. Cross-kernel placement remains open to return as a
compiler-derived transform once every member kernel compiles through Aboleth
and their IR itself proves the schedule; `publish_head` and external
publication preserve the substrate it would target.

The same effects are the intended input to remote consumer-credit planning. A
reused destination slot may be released only after every opaque consumer read
and forwarding transfer that touches it has completed. Inferring and
materializing those reverse credits remains open.

Publication is legalized separately from sender completion. A signal attached
to a transfer token carries payload effects and remains strong. A standalone,
ungrouped reverse credit has no attached payload effect and is marked weak.
Unknown or grouped cases remain strong. Under relaxed-ordering memory
registration, direct mlx5 and proxy strong publication use a responder atomic;
weak credits use the cheaper ordinary write.

Frontends provide:

- Semantic transfer and publication operations.
- Stable region identity and conservative effects.
- A fixed happens-before relation supplied by the existing caller.
- Endpoint capabilities and persistent queue ownership.

The compiler derives:

- Submission groups and doorbell boundaries.
- Strong versus weak notification.
- CQE and poll placement.
- Queue-capacity completion boundaries.

Reverse consumer credits and partial completion intervals are not yet derived
by this analysis.

An opaque effect schema is the only required manual contract. Model users do
not provide an effect graph, completion interval, or queue plan.
