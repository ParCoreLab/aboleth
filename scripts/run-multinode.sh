#!/usr/bin/env bash
# Run the Aboleth multi-node smoke across a SLURM allocation.
#
# One task per rank; scripts/multinode-launch.sh turns each task's SLURM
# variables into the RANK/WORLD_SIZE/MASTER_* the smoke reads. --overlap is
# required: inside an existing allocation srun otherwise waits for resources
# the outer job already holds.
set -euo pipefail

die() { printf 'run-multinode: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: scripts/run-multinode.sh <nodes> [tasks-per-node] [-- <smoke args>...]

Examples:
  scripts/run-multinode.sh 2                     # 2 ranks, one per node
  scripts/run-multinode.sh 4 4                   # 16 ranks, 4 per node
  scripts/run-multinode.sh 2 -- --stage a        # stage A only
  scripts/run-multinode.sh 2 1 -- --rounds 32

With one task per node the task still sees all four GPUs and uses the first
(LOCAL_RANK=0). Smoke arguments are passed through to
experiments/python/aboleth_world_multinode.py.
EOF
}

(($#)) || { usage >&2; exit 2; }
case "$1" in
  -h|--help) usage; exit 0 ;;
esac

NODES="$1"; shift
[[ "$NODES" =~ ^[1-9][0-9]*$ ]] || { usage >&2; die "nodes must be a positive integer"; }

TASKS_PER_NODE=1
if (($#)) && [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
  TASKS_PER_NODE="$1"; shift
fi
if (($#)) && [[ "$1" == "--" ]]; then
  shift
fi

command -v srun >/dev/null 2>&1 || die "srun is not on PATH"

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

printf 'run-multinode: %s node(s) x %s task(s) = %s ranks\n' \
  "$NODES" "$TASKS_PER_NODE" "$((NODES * TASKS_PER_NODE))"

# Inside an allocation larger than the step, srun places each step on
# whichever nodes it picks: a 64-node series put its two-node files on
# about twenty different pairs, whose staged small-message rates differ by
# up to 15%. Pin every step to the same nodes, starting at relative node
# MULTINODE_RELATIVE (0 by default).
placement=()
[[ -n ${SLURM_JOB_ID:-} ]] && placement=(--relative="${MULTINODE_RELATIVE:-0}")

exec srun --nodes="$NODES" --ntasks-per-node="$TASKS_PER_NODE" \
  "${placement[@]}" \
  --gpus-per-node=4 --cpus-per-task=16 --overlap --export=ALL \
  --kill-on-bad-exit=1 \
  bash "$SCRIPT_DIRECTORY/multinode-launch.sh" "$@"
