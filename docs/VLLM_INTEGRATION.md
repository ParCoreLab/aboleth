# vLLM expert-parallel integration

Aboleth integrates with vLLM expert parallelism as an all-to-all backend. The
integration is developed against vLLM
`38a466e7b6e087d67c35e7f924c04c245423c99f`; apply the patches and select the
backend as described in [the integration README](../integrations/vllm/README.md).
The compiled MoE path it drives is described in
[`COMPILED_MOE.md`](COMPILED_MOE.md).

## Ownership and execution

vLLM owns model execution, routing, expert placement, scheduling, streams,
CUDA graphs and the unchanged optimized expert kernels. The adapter supplies
bounded shapes, tensor contracts, placement and in-flight slot identities.
These are client facts; the compiler does not choose a model layout.

`moe/compiled_plan.py` expresses rank-deduplicated records as a generic Pattern.
The compiler determines which communication fields survive, physical storage
extents and destinations, transport partitions, endpoint geometry, publication,
completion and bounded shape variants. `moe/native.py` allocates the planned
resources and executes those plans. `moe/kernels.py` handles ordinary packing,
metadata conversion and gathering. `records.py` supplies semantic transport
operations that are compiled into native NIC queue code.

The prepare/finalize flow is:

```text
hidden states + top-k IDs/weights
  -> rank-deduplicated dispatch records
  -> unchanged vLLM optimized experts at the compiler-selected destination
  -> native return and weighted gathering
  -> original token order
```

Inactive choices carry a valid off-rank expert at zero weight so the reused
expert workspace is initialized safely. Independent slots have separate
payloads, flags, epochs and endpoint reservations. A once-per-forward maximum
keeps dummy and active DP ranks in the same shape variant; captured graph
bounds already agree. The engine retains stream and handoff ownership.

## Files

- `python/aboleth/moe/compiled_plan.py`: semantic consumer graph.
- `python/aboleth/moe/native.py`: allocation and compiled-plan execution.
- `python/aboleth/moe/kernels.py`: compute-only packing and gathering.
- `python/aboleth/records.py`: generic semantic transport template.
- `python/aboleth/integrations/compiled_vllm.py`: thin framework bridge.
- `integrations/vllm/0001-aboleth-all2all-backend.patch`: native factory,
  prepare/finalize, destination capability, configuration and an experimental
  Aboleth KV connector ([`KV_CACHE.md`](KV_CACHE.md)).
- `integrations/vllm/0002-current-deepep-client-and-nccl-abi.patch`: optional
  DeepEP baseline-client compatibility, inactive-ID/count handling and stream
  safety.

The native patch does not modify DeepEP. The optional baseline patch does not
modify DeepEP's library or tune its resource policy. The native patch declares
NCCL's actual property structure version before a newer library writes it,
and profiles unsplit microbatches before locking expert workspace sizes.

## Scope and checks

The path supports BF16 and block-scaled E4M3 FP8, balanced linear or
round-robin experts, TP1/PP1, DP=EP>=2, CUDA graphs and engine-owned overlap
slots. Other parallel layouts are not supported. The FP8 factory passes its
optimized consumer's rounding contract, the compiler sizes byte payloads and
scale metadata, and the return path remains BF16. Quantization is fused into
packing; expert kernels are unchanged.

`experiments/python/compiled_moe_smoke.py` exercises the path on two local
GPUs, or under a two-rank `torchrun` job with `--internode`. It checks
changing routes, invalid choices, empty/partial/full counts and concurrent graph
slots against an independent scalar-expert oracle. It is registered as
`smoke.moe-phase`. CPU checks use one process per module, matching
`test/python/CMakeLists.txt`.

End-to-end serving measurements are reported in the paper; none ship in this
repository.
