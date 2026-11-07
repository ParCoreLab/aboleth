# Compiled KV-cache migration (experimental)

> **Experimental.** KV-cache page migration ships with this release but is
> not part of the evaluated surface. Its interfaces may change, and its
> profiles are not calibrated beyond the one listed below.

`CompiledKVCacheMigrationPlan`
(`python/aboleth/patterns/kv_cache_transport.py`) compiles KV-cache page
migration through the same generic communication compiler as MoE dispatch.
The vLLM runtime (`aboleth.integrations.vllm_kv`) supports two ranks connected
over P2P; there is no inter-node path.

The frontend supplies:

- maximum pages and physical page bytes;
- rank and consumer domains;
- destination injectivity;
- asynchronous consumer depth;
- stable-address and contiguous-run facts.
- an explicit consumer order when layer pipelining is legal.

The generic compiler returns `CompiledPhase` artifacts and selects:

- direct final-page placement when the transfer-to-materialization chain can
  be legally retargeted;
- a page batch size bounded by target bytes and consumer partitions;
- a profile-defined transfer-engine crossover: generated P2P for small contiguous
  runs and one-stream direct copy-engine placement for large runs;
- generated guarded runs for fragmented layouts;
- asynchronous completion when consumer scheduling can overlap transfer
  progress;
- ordered consumer-readiness batches that admit a request before whole-cache
  completion while preserving per-layer first-use dependencies;
- fused completion and consumer readiness publication;
- a one-page fallback phase for failed runtime layout guards;
- destination slot depth from the consumer lifetime.

The runtime packs arbitrary page descriptors into contiguous runs and asks the
compiled plan to select the legal engine from transfer bytes and layout facts.
If the number of runs exceeds the compiled batch capacity, it selects the
fallback phase rather than truncating or assuming contiguity.

Consumer readiness is indexed by `(slot, consumer, source_rank)`. The last copy
worker for a consumer performs the system fence and publishes the epoch. This
allows independent consumers to observe only their own page dependency.

For vLLM, the compiled plan groups four ordered layers per readiness event.
Layer 0 waits for copies of layers 0–3, then computation overlaps the next copy
group. A completed whole-transfer event removes all layer guards. The compiler
rejects this schedule unless consumer order is proven and the copy engine uses
one ordered stream.

## Runtime ownership

The implementation is split by responsibility:

- `aboleth.kv.profile` defines typed hardware profiles and forced ablation
  overrides;
- `aboleth.kv.copy` owns generated Triton copies, CUDA copy-engine submission,
  and block-run canonicalization;
- `aboleth.kv.lifecycle` owns the typed per-request transfer/admission/
  notification state machine;
- `aboleth.kv.manager` combines compiled transfer tickets with that lifecycle;
- `aboleth.integrations.vllm_kv` owns world construction, semantic layer
  regions, plan compilation, streams, and transfer execution;
- the vLLM KV connector added by
  `integrations/vllm/0001-aboleth-all2all-backend.patch` translates `ReadSpec`
  and invokes the generic manager.

`consumer_ordered` defaults to false. The vLLM adapter must provide every KV
layer exactly once before the compiler can select consumer-layer pipelining.
Regions are exchanged by semantic layer name, so ranks need not share Python
dictionary insertion order.

## Target profiles

`aboleth.kv.profile` selects a typed hardware profile from the device name.
The 64 MiB crossover, one copy stream, sixteen generated workers, and
four-layer readiness batch belong to the `h200-nvlink` profile, selected on
H200. They are not universal defaults. Every other device selects the
conservative `generic-generated-p2p` profile, which never uses the copy
engine and publishes readiness per consumer.

`experiments/python/kv_cache_api_smoke.py` (registered as `smoke.kv-cache`)
checks the API on two local GPUs.
