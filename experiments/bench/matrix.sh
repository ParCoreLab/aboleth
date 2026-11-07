#!/usr/bin/env bash
# The paper's evaluation matrix. Run inside an allocation (matrix.sbatch),
# from the tree root, with the environment set up (repro/README.md):
#
#   ABOLETH_PLATFORM=h100-staged experiments/bench/matrix.sh <phase> <dir>
#
#   ablation  1 node:   Tier-2 waterfall and leave-one-out (stream,
#                       multi-chain ping-pong, world-4 all-to-all)
#   ablation-mixed  2 nodes: the same across nodes (ping-pong over the
#                       fabric, world-8 mixed all-to-all)
#   p2p       2 nodes:  one rank per node; latency, bandwidth and message
#                       rate
#   ctas      2 nodes:  bandwidth by CTA count (SMs to saturate the link)
#   a2a       N nodes:  mixed NVLink+RDMA all-to-all at world 4N; run once
#                       per node count (NODES=2, 4, 8, ...)
#   decompose 2 nodes:  where Tier 2's all-to-all lead comes from
#   emission  2 nodes:  the RDMA lowering's descriptor-load and WQE-store
#                       options on both tiers (ablation.py --sweep emission)
#   series    N nodes:  p2p, ctas, decompose, a2a at every node count up to
#                       N, and both ablations, in one allocation
#   dsl-p2p   2 nodes:  DSL comparison (experiments/dsl): the series' p2p, ctas and
#   dsl-ctas  2 nodes:  a2a phases with the kernels transcribed into CuTe DSL,
#   dsl-a2a   N nodes:  Triton and Triton-distributed, each over the
#                       generated Tier-1 library or NVSHMEM, interleaved with
#                       the series' own arms (DSL_REF=0 leaves those out);
#                       run from the DSL environment (ABOLETH_DSL_PYTHON,
#                       ABOLETH_TDIST_PYTHON: setup-envs.sh cute, tdist)
#   dsl-moe   N nodes:  DSL dispatch/combine per family (experiments/dsl/moe)
#   dsl-series N nodes: dsl-p2p, dsl-ctas, then dsl-a2a at every node count
#                       in DSL_A2A_NODES (2 4 by default) up to N
#
# Every library runs its best configuration, interleaved over REPEATS
# passes (forward, reverse, seeded random orders) in one allocation; every
# record carries raw device-clock samples and provenance.
# experiments/bench/report.py <dir> turns <dir> into tables.
set -euo pipefail
phase=${1:?phase}
out=${2:?output directory}
mkdir -p "$out"
: "${ABOLETH_PLATFORM:?label the platform, e.g. h100-staged}"
export ABOLETH_PLATFORM
# GPUDirect registration on drivers without nvidia_peermem (bench/README.md).
export NVSHMEM_IB_DISABLE_DMABUF=0
iters=${ITERS:-1000}
warmup=${WARMUP:-200}
repeats=${REPEATS:-5}
bench=experiments/bench
interleave=(--interleave --repeats "$repeats" -n "$iters" -w "$warmup")
# Lists from the environment: spaces or + (sbatch --export cannot carry a
# space or a comma safely). Set but empty skips that part of dsl-ctas (the
# CTA sweep, or the rate and streaming runs), so it can run as several jobs.
dsl_ctas=${DSL_CTAS-1 2 4 8 16}
dsl_rate_ctas=${DSL_RATE_CTAS-1 16 64}
dsl_a2a_nodes=${DSL_A2A_NODES:-2 4}

multi() {
  local nodes=$1 tasks=$2 program=$3
  shift 3
  ABOLETH_MULTINODE_PROGRAM=$PWD/$bench/$program \
    scripts/run-multinode.sh "$nodes" "$tasks" -- "$@"
}

# Best configuration of every library. Tier 1 and the baselines fuse the
# epoch into the payload put where their API can (warp-issued too, where the
# library has that call: large NVLink copies), and Tier 1 runs under the
# exclusive-issuer contract the kernels honour (one issuer per endpoint, as
# NCCL's exclusive contexts); Tier 2 leaves all of it to the compiler. The
# NCCL device API runs GIN for remote peers and LSA for same-node ones, and
# NCCL host also runs CUDA-graph-captured (its launch overhead removed).
t1=aboleth-generic
p2p_arms=$t1:rdma::fuse-signal=on:exclusive=on,aboleth-triton:rdma,nvshmem:ibgda::fuse-signal=on,nvshmem:ibrc::fuse-signal=on,nccl-device:gdaki::fuse-signal=on,nccl-device:proxy::fuse-signal=on
# Windows: every library issues each put on its own (the plain API); Tier
# 2 batches a window's doorbells by itself. The hand-batched windows Tier 1
# (put_many_flush) and NCCL (aggregated requests) can express are labelled
# arms of their own, an upper bound that takes hand work.
bw_arms=$t1:rdma::exclusive=on,$t1:rdma::exclusive=on:batch-window=on,aboleth-triton:rdma,nvshmem:ibgda,nvshmem:ibrc,nccl-device:gdaki,nccl-device:gdaki::batch-window=on,nccl-device:proxy
# Message rate at each library's best issue width: a CTA's window posted by
# 1, 32 or 128 of its threads at once (--issuers; each library coalesces a
# converged warp's postings as it does by itself, e.g. NVSHMEM's and DOCA's
# warp-aggregated doorbells), against a 128-put window (128 issuers post one
# put each; it fits Tier 1's default 257-entry queue, which only quiet
# reclaims). Tier 1 shares the CTA's endpoint through its non-exclusive
# build; Tier 2's issuers are its programs (the CTA sweep). The dissecting
# study (arXiv 2610.01380) reached 250 M msg/s with NVSHMEM at 16 QPs x 256
# threads, where one thread per CTA reaches a few.
# Streaming peak rate (2610.01380's method): libraries that reclaim their
# own queue post every window back to back and complete once at the end, as
# Tier 2 does through the compiler's self-reclaim (its windowed arm stays
# beside it); Tier 1 cannot skip completions (only its quiet frees the
# queue), so it runs its windowed arm. A separate record from the windowed
# rate.
stream_arms=$t1:rdma::exclusive=on,aboleth-triton:rdma,aboleth-triton:rdma::stream=on,nvshmem:ibgda::issuers=128:stream=on,nvshmem:ibgda::issuers=256:window=256:stream=on,nvshmem:ibrc::issuers=128:stream=on,nccl-device:gdaki::stream=on,nccl-device:gdaki::issuers=128:stream=on,nccl-device:proxy::issuers=128:stream=on
rate_arms=$t1:rdma::exclusive=on,aboleth-triton:rdma
for lib in $t1:rdma nvshmem:ibgda nvshmem:ibrc nccl-device:gdaki nccl-device:proxy; do
  [[ $lib == "$t1:rdma" ]] || rate_arms+=",$lib"
  rate_arms+=",$lib::issuers=32,$lib::issuers=128"
done
# Thread-issued NVLink copies collapse above 4 KiB in every library (world
# 8), so large sizes run only the warp-issued and
# compiler-issued arms.
# NCCL GIN contexts: one fully connected context per peer is NCCL's fastest
# at world 32, but it exceeds the NIC's QP budget (8k-16k per rank) from
# world 128, where few shared contexts win at small sizes and more at large.
# Large worlds interleave the caps in NCCL_GIN_CONTEXTS
# (0: one per peer) and report the best.
nccl_arms="" nccl_large_arms=""
for gin in ${NCCL_GIN_CONTEXTS:-0}; do
  cap=""
  ((gin)) && cap=":gin-contexts=$gin"
  nccl_arms+=",nccl-device:gdaki-lsa::fuse-signal=on$cap"
  nccl_arms+=",nccl-device:gdaki-lsa::put-issue=warp:fuse-signal=on$cap"
  nccl_large_arms+=",nccl-device:gdaki-lsa::put-issue=warp:fuse-signal=on$cap"
done
# NVSHMEM's own all-to-all collective is a reference arm next to the
# shared kernel (which quiets once per rank per round: nvshmem_quiet is
# rank-wide).
nvshmem_arms=nvshmem:ibgda::fuse-signal=on,nvshmem:ibrc::fuse-signal=on,nvshmem:ibgda::put-issue=warp:fuse-signal=on,nvshmem:ibrc::put-issue=warp:fuse-signal=on,nvshmem:ibgda::collective=on,nvshmem:ibrc::collective=on
nvshmem_large_arms=nvshmem:ibgda::put-issue=warp:fuse-signal=on,nvshmem:ibrc::put-issue=warp:fuse-signal=on,nvshmem:ibgda::collective=on,nvshmem:ibrc::collective=on
host_arms=nccl-host,nccl-host:host::graph-rounds=20

# Tuned twins. Every arm also runs with its library's tuning knobs set
# (TUNED=0 skips them), Aboleth's included, so the paper can show either
# the defaults or every library tuned. The bundles follow published
# guidance (DeepEP's NVSHMEM settings, nccl-tests' NCCL ones) and are
# recorded per run (library_env / provenance.environment).
tune_aboleth=:env=ABOLETH_QP_DEPTH=1024:env=ABOLETH_RD_ATOMIC_DEPTH=16
tune_nvshmem=:env=NVSHMEM_QP_DEPTH=1024
# Submission capability is platform policy, not a workload tuning parameter.
# Leave auto at the library default when the platform tag does not specify it.
case ${ABOLETH_RDMA_SUBMIT:-auto} in
  direct) tune_nvshmem+=:env=NVSHMEM_IBGDA_NIC_HANDLER=gpu ;;
  staged) tune_nvshmem+=:env=NVSHMEM_IBGDA_NIC_HANDLER=cpu ;;
  auto)
    case $ABOLETH_PLATFORM in
      *-direct) tune_nvshmem+=:env=NVSHMEM_IBGDA_NIC_HANDLER=gpu ;;
      *-staged) tune_nvshmem+=:env=NVSHMEM_IBGDA_NIC_HANDLER=cpu ;;
    esac ;;
esac
# NCCL_GIN_PROXY_NTHREADS (NCCL >= 2.31) is the dissecting study's GIN
# proxy setting (8 threads); GDAKI ignores it.
tune_nccl_device=:env=NCCL_GIN_GDAKI_QP_DEPTH=1024:env=NCCL_GIN_PROXY_NTHREADS=8
tune_nccl_host=:env=NCCL_IB_QPS_PER_CONNECTION=4:env=NCCL_P2P_NET_CHUNKSIZE=524288
with_tuned() {
  local arms=$1 arm tuned=""
  ((${TUNED:-1})) || { printf '%s' "$arms"; return; }
  for arm in ${arms//,/ }; do
    # Flags go after the library slot: arm:transport:library:flag...
    while (($(tr -dc : <<<"$arm" | wc -c) < 2)); do arm+=:; done
    case $arm in
      aboleth-*|*-tier1:*) tuned+=",$arm$tune_aboleth" ;;
      nvshmem:*|nvshmem38:*|*-nvshmem:*) tuned+=",$arm$tune_nvshmem" ;;
      nccl-device:*) tuned+=",$arm$tune_nccl_device" ;;
      nccl-host*) tuned+=",$arm$tune_nccl_host" ;;
    esac
  done
  printf '%s%s' "$arms" "$tuned"
}
# A library arm token's fields need a library slot before flags.
host_arms_tunable=nccl-host:host:,nccl-host:host::graph-rounds=20
a2a_large_arms_base="$t1:auto::put-issue=warp:exclusive=on,aboleth-triton:auto,$nvshmem_large_arms$nccl_large_arms,$host_arms_tunable"
a2a_arms_base="$t1:auto::fuse-signal=on:exclusive=on,$t1:auto::put-issue=warp:exclusive=on,aboleth-triton:auto,$nvshmem_arms$nccl_arms,$host_arms_tunable"
a2a_large_arms=${A2A_LARGE_ARMS:-$(with_tuned "$a2a_large_arms_base")}
a2a_arms=${A2A_ARMS:-$(with_tuned "$a2a_arms_base")}
p2p_arms_base=$p2p_arms bw_arms_base=$bw_arms rate_arms_base=$rate_arms
stream_arms_base=$stream_arms
p2p_arms=$(with_tuned "$p2p_arms")
bw_arms=$(with_tuned "$bw_arms")
rate_arms=$(with_tuned "$rate_arms")
stream_arms=$(with_tuned "$stream_arms")

case $phase in
  ablation)
    for sweep in waterfall loo; do
      python $bench/ablation.py --sweep $sweep --repeats "$repeats" \
        --out "$out/ablation_stream_$sweep.jsonl" -- \
        python $bench/stream.py --modes pipelined --chains 1 4 16 --exclusive
      # Eight put(+signal) chains per round: what coalescing, ordering and
      # publication planning act on.
      python $bench/ablation.py --sweep $sweep --repeats "$repeats" \
        --out "$out/ablation_pingpong_$sweep.jsonl" -- \
        python $bench/ladder.py --arm aboleth-triton --transport rdma \
        --mode latency --op put,put+signal --puts 8 -b 64 -e 4K -f 64 \
        -n "$iters" -w "$warmup"
      python $bench/ablation.py --sweep $sweep --repeats "$repeats" \
        --out "$out/ablation_alltoall_$sweep.jsonl" -- \
        python $bench/alltoall.py --arm aboleth-triton --transport auto \
        --local-ranks 4 -b 256 -e 16K -f 64 -n "$iters" -w "$warmup"
    done
    ;;
  ablation-mixed)
    # The same ablation across nodes: ping-pong over the fabric (one rank
    # per node) and the world-8 mixed NVLink+RDMA all-to-all, where the
    # passes have submissions to act on (on one node every peer is NVLink).
    for sweep in waterfall loo; do
      multi 2 1 ablation.py --sweep $sweep --repeats "$repeats" \
        --out "$out/ablation_pingpong2n_$sweep.jsonl" -- \
        python "$PWD/$bench/ladder.py" --arm aboleth-triton --transport rdma \
        --mode latency --op put,put+signal --puts 8 -b 64 -e 4K -f 64 \
        -n "$iters" -w "$warmup"
      multi 2 4 ablation.py --sweep $sweep --repeats "$repeats" \
        --out "$out/ablation_alltoall_w8_$sweep.jsonl" -- \
        python "$PWD/$bench/alltoall.py" --arm aboleth-triton --transport auto \
        -b 256 -e 64K -f 16 -n "$iters" -w "$warmup"
    done
    ;;
  p2p)
    # A failed arm fails its sweep but is in the record; run every sweep.
    status=0
    multi 2 1 ladder.py --arms "$p2p_arms" "${interleave[@]}" \
      --mode latency -b 8 -e 1M --json-out "$out/latency.json" || status=$?
    multi 2 1 ladder.py --arms "$bw_arms" "${interleave[@]}" \
      --mode bandwidth -b 8 -e 1M --json-out "$out/bandwidth.json" ||
      status=$?
    # Large windows move gigabytes each: 50 iterations are seconds of
    # transfer per size.
    multi 2 1 ladder.py --arms "$bw_arms" --interleave --repeats "$repeats" \
      -n 50 -w 5 --mode bandwidth -b 2M -e 64M \
      --json-out "$out/bandwidth_large.json" || status=$?
    exit "$status"
    ;;
  ctas)
    # Message rate and SMs to saturate the link: CTA count sweep, one
    # endpoint per CTA where the library has endpoints. A failed arm fails
    # its sweep but is in the record; run every sweep.
    status=0
    for ctas in 1 2 4 8 16; do
      multi 2 1 ladder.py --arms "$bw_arms" "${interleave[@]}" \
        --mode bandwidth -b 8 -e 1M -f 8 --ctas "$ctas" \
        --endpoints-per-peer "$ctas" \
        --json-out "$out/bandwidth_ctas$ctas.json" || status=$?
    done
    # Message rate at one, sixteen and sixty-four queue pairs (endpoint, RC
    # QP, GIN context; one per CTA), two ways: windowed (every window
    # completes before the next, as an application round does) and
    # streaming (windows back to back, one completion at the end;
    # 2610.01380's method, for the libraries whose queue reclaims itself).
    for ctas in 1 16 64; do
      multi 2 1 ladder.py --arms "$rate_arms" "${interleave[@]}" \
        --mode bandwidth -b 8 -e 32K -f 8 --window 128 --ctas "$ctas" \
        --endpoints-per-peer "$ctas" \
        --json-out "$out/bandwidth_rate_ctas$ctas.json" || status=$?
      multi 2 1 ladder.py --arms "$stream_arms" "${interleave[@]}" \
        --mode bandwidth -b 8 -e 32K -f 8 --window 128 --ctas "$ctas" \
        --endpoints-per-peer "$ctas" \
        --json-out "$out/bandwidth_stream_ctas$ctas.json" || status=$?
    done
    exit "$status"
    ;;
  a2a)
    nodes=${NODES:?NODES=2|4|8|...}
    # A failed arm fails its sweep but is in the record; run both sweeps.
    status=0
    multi "$nodes" 4 alltoall.py --arms "$a2a_arms" "${interleave[@]}" \
      -b 256 -e 16K -f 4 --json-out "$out/alltoall_w$((nodes * 4)).json" ||
      status=$?
    multi "$nodes" 4 alltoall.py --arms "$a2a_large_arms" "${interleave[@]}" \
      -b 64K -e 4M -f 4 --json-out "$out/alltoall_large_w$((nodes * 4)).json" ||
      status=$?
    exit "$status"
    ;;
  dsl-p2p|dsl-ctas|dsl-a2a)
    # The series' phases, shape for shape, with each DSL's arms:
    # every DSL arm calls the kernel transcribed once per DSL
    # (experiments/dsl/{cute,triton_dsl}: CuTe's, and Triton's for
    # triton-tier1 and Triton-distributed alike); only the library differs.
    # Each library runs the series' configurations of its CUDA arm (fused
    # put+signal, warp-issued large all-to-all, 1/32/128 issuers, streaming
    # where the library reclaims its own queue). The DSL kernels take sizes
    # as compile-time constants; a baked CUDA Tier-1 arm (bake=on) is their
    # like-for-like CUDA reference. The series' own arms
    # (DSL_REF=1) run interleaved in the same records, so the DSL lines sit
    # on the paper's scale without crossing allocations. NVSHMEM in the DSLs
    # is the version their bindings need (CuTe 3.8.0, Triton-distributed
    # 3.6.5); nvshmem38 is the CUDA arm on 3.8.0.
    export ABOLETH_MULTINODE_PYTHON=${ABOLETH_DSL_PYTHON:?the DSL environment python}
    export ABOLETH_TDIST_PYTHON=${ABOLETH_TDIST_PYTHON:?Triton-distributed environment python}
    # The NCCL (2.31.2) library directory for the NCCL reference arms.
    export ABOLETH_NCCL_RUNTIME=${ABOLETH_NCCL_RUNTIME:?the NCCL 2.31 lib directory}
    # Compiled DSL kernels per job (CuTe's default cache is under $HOME).
    export CUTE_DSL_CACHE_DIR=${CUTE_DSL_CACHE_DIR:-${ABOLETH_BENCH_CACHE:-$out/.cache}/cute}
    ref=${DSL_REF:-1}
    # NVSHMEM's transports in the DSL arms: IBGDA (its CPU proxy on staged
    # platforms) and IBRC; the best per cell is the library's number.
    # Triton-distributed links NVSHMEM's bitcode by default (IBRC); its IBGDA
    # goes through its NVSHMEM wrapper (TRITON_DIST_SHMEM_WRAPPER, set by the
    # arm for IBGDA: the plain bitcode path segfaults in IBGDA init).
    # IBGDA is NVSHMEM's message-rate path even on staged platforms.
    nv=${DSL_NVSHMEM:-ibgda ibrc}
    nv_tdist=${DSL_TDIST_NVSHMEM:-ibgda ibrc}
    dsl_nv() {  # flags (may be empty): every NVSHMEM DSL arm x transport
      local t
      for t in $nv; do printf ',cute-nvshmem:%s%s' "$t" "${1:+::$1}"; done
      for t in $nv_tdist; do printf ',tdist-nvshmem:%s%s' "$t" "${1:+::$1}"; done
    }
    dsl_t1() {  # transport flags: both Tier-1 DSL arms
      printf ',cute-tier1:%s%s,triton-tier1:%s%s' "$1" "${2:+::$2}" "$1" "${2:+::$2}"
    }
    pick() {  # series-arms dsl-arms: the joint list
      if ((ref)); then printf '%s%s' "$1" "$2"; else printf '%s' "${2#,}"; fi
    }
    status=0
    case $phase in
      dsl-p2p)
        # latency.json, bandwidth.json, bandwidth_large.json. The ping-pong
        # also runs nvshmem38 (the version control) and triton-nvshmem (a
        # labelled diagnostic: NVSHMEM's bitcode on the series' Triton
        # through our shim, against Triton-distributed's own linking).
        lat=$(with_tuned "$(pick "$p2p_arms_base,$t1:rdma::fuse-signal=on:exclusive=on:bake=on,nvshmem38:ibgda::fuse-signal=on,nvshmem38:ibrc::fuse-signal=on" \
          "$(dsl_t1 rdma fuse-signal=on:exclusive=on)$(dsl_nv fuse-signal=on),triton-nvshmem:ibrc::fuse-signal=on")")
        bw=$(with_tuned "$(pick "$bw_arms_base" "$(dsl_t1 rdma exclusive=on)$(dsl_nv)")")
        multi 2 1 ladder.py --arms "$lat" "${interleave[@]}" \
          --mode latency -b 8 -e 1M --json-out "$out/dsl_latency.json" ||
          status=$?
        multi 2 1 ladder.py --arms "$bw" "${interleave[@]}" \
          --mode bandwidth -b 8 -e 1M --json-out "$out/dsl_bandwidth.json" ||
          status=$?
        multi 2 1 ladder.py --arms "$bw" --interleave --repeats "$repeats" \
          -n 50 -w 5 --mode bandwidth -b 2M -e 64M \
          --json-out "$out/dsl_bandwidth_large.json" || status=$?
        ;;
      dsl-ctas)
        # bandwidth_ctas*.json, bandwidth_rate_ctas*.json,
        # bandwidth_stream_ctas*.json: one endpoint, RC QP or context per CTA.
        bw=$(with_tuned "$(pick "$bw_arms_base" "$(dsl_t1 rdma exclusive=on)$(dsl_nv)")")
        # Message rate at 1, 32 and 128 issuers per CTA (Tier 1 shares the
        # CTA's endpoint through its non-exclusive build beyond one).
        rate_dsl="$(dsl_t1 rdma exclusive=on)$(dsl_t1 rdma issuers=32)$(dsl_t1 rdma issuers=128)$(dsl_nv)$(dsl_nv issuers=32)$(dsl_nv issuers=128)"
        # nvshmem38: the CUDA NVSHMEM arm on the DSL arms' NVSHMEM version.
        rate=$(with_tuned "$(pick "$rate_arms_base,nvshmem38:ibgda::issuers=128,nvshmem38:ibrc::issuers=128" "$rate_dsl")")
        # DSL_RATE_ARMS (+ or comma separated) replaces the rate arms, tuned
        # twins added; DSL_STREAM=0 skips the streaming runs (a rerun of
        # some libraries' rate cells).
        [[ -z ${DSL_RATE_ARMS:-} ]] || rate=$(with_tuned "${DSL_RATE_ARMS//+/,}")
        # Streaming: NVSHMEM reclaims its own queue; the Tier-1 arms cannot
        # skip completions and run their windowed form (as the series).
        stream_dsl="$(dsl_t1 rdma exclusive=on)$(dsl_nv issuers=128:stream=on)"
        # NVSHMEM's 256-issuer stream over IBGDA, as the series ran it.
        stream_dsl+="$(nv=ibgda nv_tdist=ibgda dsl_nv issuers=256:window=256:stream=on)"
        stream=$(with_tuned "$(pick "$stream_arms_base,nvshmem38:ibgda::issuers=256:window=256:stream=on" "$stream_dsl")")
        for ctas in ${dsl_ctas//+/ }; do
          multi 2 1 ladder.py --arms "$bw" "${interleave[@]}" \
            --mode bandwidth -b 8 -e 1M -f 8 --ctas "$ctas" \
            --endpoints-per-peer "$ctas" \
            --json-out "$out/dsl_bandwidth_ctas$ctas.json" || status=$?
        done
        for ctas in ${dsl_rate_ctas//+/ }; do
          multi 2 1 ladder.py --arms "$rate" "${interleave[@]}" \
            --mode bandwidth -b 8 -e 32K -f 8 --window 128 --ctas "$ctas" \
            --endpoints-per-peer "$ctas" \
            --json-out "$out/dsl_bandwidth_rate_ctas$ctas.json" || status=$?
          ((${DSL_STREAM:-1})) || continue
          multi 2 1 ladder.py --arms "$stream" "${interleave[@]}" \
            --mode bandwidth -b 8 -e 32K -f 8 --window 128 --ctas "$ctas" \
            --endpoints-per-peer "$ctas" \
            --json-out "$out/dsl_bandwidth_stream_ctas$ctas.json" || status=$?
        done
        ;;
      dsl-a2a)
        # alltoall_w*.json and alltoall_large_w*.json at world 4N: small
        # sizes thread-issued with the fused signal, and warp-issued (put_warp,
        # fused where the library has a warp put-with-signal); large sizes
        # warp-issued only, as the series' arms.
        nodes=${NODES:?NODES=2|4|8|16}
        warp_dsl="$(dsl_t1 auto put-issue=warp:exclusive=on)$(dsl_nv put-issue=warp:fuse-signal=on)"
        # NVSHMEM's own all-to-all where its DSL binding ships it (NVSHMEM4Py
        # in CuTe: alltoall_block; Triton-distributed has no in-kernel
        # all-to-all collective), as the series' CUDA collective arm.
        warp_dsl+="$(nv_tdist= dsl_nv collective=on)"
        v38="nvshmem38:ibrc::collective=on,nvshmem38:ibrc::put-issue=warp:fuse-signal=on"
        small=$(with_tuned "$(pick "$a2a_arms_base,$t1:auto::fuse-signal=on:exclusive=on:bake=on,$v38" "$(dsl_t1 auto fuse-signal=on:exclusive=on)$(dsl_nv fuse-signal=on)$warp_dsl")")
        large=$(with_tuned "$(pick "$a2a_large_arms_base,$v38" "$warp_dsl")")
        multi "$nodes" 4 alltoall.py --arms "$small" "${interleave[@]}" \
          -b 256 -e 16K -f 4 --json-out "$out/dsl_alltoall_w$((nodes * 4)).json" ||
          status=$?
        multi "$nodes" 4 alltoall.py --arms "$large" "${interleave[@]}" \
          -b 64K -e 4M -f 4 \
          --json-out "$out/dsl_alltoall_large_w$((nodes * 4)).json" ||
          status=$?
        ;;
    esac
    exit "$status"
    ;;
  dsl-moe)
    # DSL dispatch/combine (experiments/dsl/moe): each family's arms and
    # the implementations its developers ship, one contract, one timer
    # (moe_common.py), interleaved. FP8 decode: tokens per rank in
    # DSL_MOE_TOKENS, per model in DSL_MOE_MODELS, world 4N.
    nodes=${NODES:?NODES=1|2|4}
    export ABOLETH_MULTINODE_PYTHON=${ABOLETH_DSL_PYTHON:?the DSL environment python}
    export ABOLETH_TDIST_PYTHON=${ABOLETH_TDIST_PYTHON:?Triton-distributed environment python}
    # Triton-distributed's LL layer: IBRC (its default) by default; its IBGDA
    # (wrapper) runs once NVSHMEM_HOME names its own NVSHMEM (otherwise it
    # compiles against the base environment's NVSHMEM and faults beyond one
    # CTA).
    # DSL_MOE_ARMS: arm tokens separated by + (sbatch --export splits on
    # commas) or commas.
    moe_arms=${DSL_MOE_ARMS:-aboleth-moe:auto,aboleth-moe-fused:auto,tdist-ll:ibrc,deepep:gin,cutile-tier2-moe:auto,cutile-nccl-moe:auto}
    moe_arms=${moe_arms//+/,}
    status=0
    for model in ${DSL_MOE_MODELS:-qwen3-235b qwen3-30b}; do
      ABOLETH_MULTINODE_PROGRAM=$PWD/experiments/dsl/moe/moe.py \
        scripts/run-multinode.sh "$nodes" 4 -- --arms "$moe_arms" \
        --interleave --repeats "$repeats" -n "${MOE_ITERS:-200}" \
        -w "${MOE_WARMUP:-20}" --model "$model" \
        --tokens "${DSL_MOE_TOKENS:-1,8,32,64,128}" \
        --json-out "$out/dsl_moe_${model}_w$((nodes * 4)).json" || status=$?
    done
    exit "$status"
    ;;
  dsl-series)
    # The DSL phases in one allocation, on one commit and one set of nodes.
    status=0
    for name in dsl-p2p dsl-ctas; do
      echo "dsl-series: $name start $(date -u +%FT%TZ)"
      bash "${BASH_SOURCE[0]}" "$name" "$out" || status=1
    done
    for nodes in ${dsl_a2a_nodes//+/ }; do
      ((nodes <= NODES)) || continue
      echo "dsl-series: dsl-a2a@$nodes start $(date -u +%FT%TZ)"
      NODES=$nodes bash "${BASH_SOURCE[0]}" dsl-a2a "$out" || status=1
    done
    exit "$status"
    ;;
  series)
    # Every paper phase in one allocation, on one commit and one set of
    # nodes: the two-node phases on the allocation's first nodes, then the
    # all-to-all at each node count in SERIES_A2A_NODES that fits (NCCL's
    # GIN-context caps from SERIES_GIN_FROM_NODES nodes up, where one
    # context per peer stops being NCCL's best), then the ablations. A
    # failed phase is in the record; the rest still run.
    status=0
    phase_run() {
      local name=$1 start=$SECONDS code=0
      shift
      echo "series: $name start $(date -u +%FT%TZ)"
      env "$@" bash "${BASH_SOURCE[0]}" "${name%%@*}" "$out" || code=$?
      echo "series: $name exit $code after $(((SECONDS - start) / 60)) min"
      ((code == 0)) || status=1
    }
    for name in ${SERIES_PHASES:-p2p ctas decompose}; do
      phase_run "$name" NODES=2
    done
    for nodes in ${SERIES_A2A_NODES:-2 4 8 16 32 64}; do
      ((nodes <= NODES)) || { echo "series: a2a@$nodes skipped ($NODES nodes)"; continue; }
      caps=0
      ((nodes >= ${SERIES_GIN_FROM_NODES:-16})) && caps=${SERIES_GIN_CAPS:-16 31 63}
      phase_run "a2a@$nodes" NODES="$nodes" NCCL_GIN_CONTEXTS="$caps"
    done
    for name in ${SERIES_LATE_PHASES:-ablation-mixed ablation}; do
      phase_run "$name" NODES=2
    done
    exit "$status"
    ;;
  decompose)
    # Where Tier 2's all-to-all lead comes from (2 nodes): Tier 1 with its
    # issuer contract (exclusive=on: no reservation atomic) and the world's
    # geography (specialized: no per-peer transport switch) supplied by
    # hand, and Tier 2 retiring each round through a Tier-1-style drain
    # (after its wait: a drain ahead of it queues behind the unsubmitted
    # batch); with every switchable pass on and off, NCCL interleaved with
    # the passes-off runs as their control. World 8 mixed, world 2 over the
    # NIC only (one rank per node), world 4 over NVLink only.
    arms=aboleth-generic:auto::fuse-signal=on,aboleth-generic:auto::fuse-signal=on:exclusive=on,aboleth-specialized:auto::fuse-signal=on:exclusive=on,aboleth-triton:auto,aboleth-triton:auto::completion=drain,nccl-device:gdaki-lsa::fuse-signal=on
    # The drain diagnostic runs with the passes on only: with every pass off
    # it hangs.
    tier2=aboleth-triton:auto,nccl-device:gdaki-lsa::fuse-signal=on
    none=-completion-planning,-forward-inline-payloads,-plan-publication,-coalesce-submissions,-order-publications,-elide-epochs,-analyze-issuers,-gpu-doorbell
    sizes=(-b 256 -e 4K -f 16)
    status=0
    for shape in w8:2:4 w2:2:1; do
      IFS=: read -r name nodes tasks <<<"$shape"
      multi "$nodes" "$tasks" alltoall.py --arms "$arms" "${interleave[@]}" \
        "${sizes[@]}" --json-out "$out/decompose_$name.json" || status=$?
      ABOLETH_PASSES=$none multi "$nodes" "$tasks" alltoall.py \
        --arms "$tier2" "${interleave[@]}" "${sizes[@]}" \
        --json-out "$out/decompose_${name}_none.json" || status=$?
    done
    python $bench/alltoall.py --arms "$arms" "${interleave[@]}" \
      --local-ranks 4 "${sizes[@]}" --json-out "$out/decompose_w4.json" ||
      status=$?
    ABOLETH_PASSES=$none python $bench/alltoall.py --arms "$tier2" \
      "${interleave[@]}" --local-ranks 4 "${sizes[@]}" \
      --json-out "$out/decompose_w4_none.json" || status=$?
    exit "$status"
    ;;
  emission)
    # The RDMA lowering's descriptor-load and WQE-store options
    # (ablation.py --sweep emission) on both tiers (2 nodes): the round trip
    # over the fabric, the 16-CTA window, the world-8 mixed all-to-all, and
    # the one-node stream. Records carry each kernel's registers.
    status=0
    multi 2 1 ablation.py --sweep emission --repeats "$repeats" \
      --out "$out/ablation_emission_latency.jsonl" -- \
      python "$PWD/$bench/ladder.py" \
      --arms "$t1:rdma::fuse-signal=on:exclusive=on,aboleth-triton:rdma" \
      --mode latency -b 8 -e 64K -f 8 -n "$iters" -w "$warmup" || status=$?
    multi 2 1 ablation.py --sweep emission --repeats "$repeats" \
      --out "$out/ablation_emission_window_ctas16.jsonl" -- \
      python "$PWD/$bench/ladder.py" \
      --arms "$t1:rdma::exclusive=on,aboleth-triton:rdma" \
      --mode bandwidth -b 8 -e 64K -f 8 --ctas 16 --endpoints-per-peer 16 \
      -n "$iters" -w "$warmup" || status=$?
    multi 2 4 ablation.py --sweep emission --repeats "$repeats" \
      --out "$out/ablation_emission_alltoall_w8.jsonl" -- \
      python "$PWD/$bench/alltoall.py" \
      --arms "$t1:auto::fuse-signal=on:exclusive=on,aboleth-triton:auto" \
      -b 256 -e 64K -f 4 -n "$iters" -w "$warmup" || status=$?
    python $bench/ablation.py --sweep emission --repeats "$repeats" \
      --out "$out/ablation_emission_stream.jsonl" -- \
      python $bench/stream.py --modes pipelined --chains 1 4 16 --exclusive ||
      status=$?
    exit "$status"
    ;;
  *)
    echo "matrix: unknown phase $phase" >&2
    exit 2
    ;;
esac
