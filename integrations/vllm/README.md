# vLLM integration

Patches that add Aboleth as a vLLM expert-parallel all-to-all backend. The
integration contract is described in
[`docs/VLLM_INTEGRATION.md`](../../docs/VLLM_INTEGRATION.md).

## Applying the patches

Apply the native patch to vLLM `38a466e7b6e087d67c35e7f924c04c245423c99f`:

```bash
git checkout 38a466e7b6e087d67c35e7f924c04c245423c99f
git apply /path/to/aboleth/integrations/vllm/0001-aboleth-all2all-backend.patch
```

Select `--enable-expert-parallel --all2all-backend aboleth --moe-backend triton`,
set `VLLM_ENABLE_ABOLETH=1` before importing vLLM, and put the built Aboleth
`python/` directory on `PYTHONPATH`. The compiler plugin and Triton must share
one extension-enabled build. No MoE tier or resource-policy environment
variables are required.

vLLM retains scheduling, streams, routing, placement, expert computation and
graph ownership. The semantic record program is lowered through the C++
communication compiler. The native patch also registers an experimental
Aboleth KV connector, exposes an optional expert destination, profiles valid
unsplit microbatches, and fixes the versioned NCCL property structure
declaration.

To run DeepEP as a baseline through the same vLLM checkout, apply
`0002-current-deepep-client-and-nccl-abi.patch` after 0001. Despite its
filename, the NCCL ABI fix is in 0001; 0002 contains only DeepEP API
compatibility and consumer/stream-lifetime corrections.

## Scope

The path supports BF16 and block-scaled E4M3 FP8, balanced linear/round-robin
expert placement, TP1/PP1, two or more expert/data-parallel ranks, CUDA graphs
and engine-owned microbatch slots. Same-node links use P2P and remote links use
native RDMA. FP8 carries per-token K-block FP32 scales and the optimized
consumer's rounding contract; packing and quantization are fused before the
semantic transfers. Optimized expert kernels remain framework-owned.

For FP8 models with a floating-scale precision policy on B200, set
`VLLM_USE_DEEP_GEMM_E8M0=0` and `VLLM_USE_DEEP_GEMM=0`: the pinned DeepGEMM
dense kernel rejects floating scales. These are framework precision/backend
settings; the native transport uses the consumer fact passed by vLLM.

## Checking the install

```bash
python experiments/python/compiled_moe_smoke.py
```

It uses two GPUs, changing and invalid routes, empty/partial/full inputs, two
concurrent graph slots, and an independent scalar-expert oracle.

## The DeepEP baseline patch

`0002` changes only vLLM's DeepEP client: it accepts DeepEP 2.5's `EPBuffer`,
maps inactive top-k slots to a zero-weight off-rank expert (the same consumer
contract the Aboleth path states), passes no per-expert counts in decode as
the class documents, records stream lifetimes, and adds the dual-batch-overlap
yields the Aboleth path also has. None of Aboleth's compiler-derived
optimizations are applied to DeepEP, and DeepEP's library source and SM/QP
policy are untouched.
