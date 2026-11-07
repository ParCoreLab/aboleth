#!/usr/bin/env bash
# Environment for one rank of the Aboleth multi-node smoke, then exec it.
#
# srun starts one copy of this per task, so everything below is derived from
# that task's own SLURM variables and the same file serves every rank. The
# environment is set here rather than sourced: a compute node needs this file
# and the tree, nothing else. scripts/
# run-multinode.sh is the usual caller; running it by hand needs the same
# SLURM variables in the environment.
set -euo pipefail

die() { printf 'multinode-launch: %s\n' "$*" >&2; exit 1; }

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname -- "$SCRIPT_DIRECTORY")"

[[ -n "${SLURM_JOB_NODELIST:-}" ]] ||
  die "SLURM_JOB_NODELIST is unset: run this through srun"
[[ -n "${SLURM_PROCID:-}" && -n "${SLURM_NTASKS:-}" ]] ||
  die "SLURM_PROCID/SLURM_NTASKS are unset: run this through srun"

# Rank 0 hosts the bootstrap store, and rank 0 is the first task, which SLURM
# places on the first node of the *step*. That is not always the first node of
# the job: a step narrower than the allocation (srun --nodes=1 inside an
# 8-node salloc) is placed wherever SLURM likes, and deriving the address from
# SLURM_JOB_NODELIST then points every rank at a node the step never used —
# they all fail with a TCPStore connect timeout.
STEP_NODELIST="${SLURM_STEP_NODELIST:-$SLURM_JOB_NODELIST}"
MASTER_ADDR="$(scontrol show hostnames "$STEP_NODELIST" | head -n1)"
[[ -n "$MASTER_ADDR" ]] || die "no hostname in $STEP_NODELIST"
export MASTER_ADDR
# One port per job, so concurrent jobs on the same nodes do not collide,
# below Linux's ephemeral range (32768 up): a port there can already be the
# source port of an outgoing connection, and the store then fails to listen
# (EADDRINUSE). Drivers that run one store per arm add a small offset.
export MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"
export RANK="$SLURM_PROCID"
export WORLD_SIZE="$SLURM_NTASKS"
# The rank's GPU. With one task per node this is 0, and the task still sees
# every GPU on the node; it simply uses the first.
export LOCAL_RANK="${SLURM_LOCALID:-0}"

export PYTHONPATH="$ROOT/python:$ROOT/third_party/triton/python:$ROOT/third_party/triton/python/triton_kernels:${PYTHONPATH:-}"
export ABOLETH_BUILD_DIR="$ROOT/build"
export ABOLETH_PLUGIN_PATH="$ROOT/build/lib/libAbolethPlugin.so"
export TRITON_PLUGIN_PATHS="$ROOT/build/lib/libAbolethPlugin.so"
MULTINODE_CACHE_ID="${SLURM_JOB_ID:-manual}"
CACHE_ROOT="${ABOLETH_MULTINODE_CACHE_ROOT:-$ROOT/.triton/multinode-cache/$MULTINODE_CACHE_ID}"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton-rank$RANK"
export TRITON_HOME="$CACHE_ROOT/triton-home-rank$RANK"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch-rank$RANK"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$ROOT/.venv/bin:${PATH:-}"
# ABOLETH_NCCL_RUNTIME: the lib directory of an NCCL built outside the venv
# (the bench binaries' NCCL_ROOT); it must precede the venv's copy, which
# LD_LIBRARY_PATH would otherwise load ahead of the binaries' RUNPATH.
NCCL_RUNTIME="${ABOLETH_NCCL_RUNTIME:-$ROOT/.venv/lib/python3.12/site-packages/nvidia/nccl/lib}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NCCL_RUNTIME:${LD_LIBRARY_PATH:-}"
# Clusters whose login profile pins an older libstdc++ than the toolchain
# needs point ABOLETH_GCC_RUNTIME at the newer runtime's lib64.
if [[ -n "${ABOLETH_GCC_RUNTIME:-}" ]]; then
  export LD_LIBRARY_PATH="$ABOLETH_GCC_RUNTIME:$LD_LIBRARY_PATH"
fi

# ABOLETH_MULTINODE_PYTHON: another interpreter for the program, e.g. a DSL
# environment layered on the tree's venv (experiments/dsl/README.md).
PYTHON="${ABOLETH_MULTINODE_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || die "$PYTHON is missing"

PROGRAM="${ABOLETH_MULTINODE_PROGRAM:-$ROOT/experiments/python/aboleth_world_multinode.py}"
[[ -f "$PROGRAM" ]] || die "multi-node program is missing: $PROGRAM"

exec "$PYTHON" \
  "$PROGRAM" "$@"
