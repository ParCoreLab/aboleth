#!/bin/bash
# Batch form of matrix.sh, for runs larger than an interactive allocation:
#
#   sbatch -A <account> -p <partition> --nodes=N \
#       experiments/bench/matrix.sbatch <phase> <dir>
#
# Submit from the tree root with ABOLETH_PLATFORM and ABOLETH_COMMIT set (the
# job inherits them); NODES defaults to the allocation's node count.
# ABOLETH_ENV, if set, is a site script sourced first (modules, CUDA_HOME).
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --job-name=aboleth-matrix
#SBATCH --output=aboleth-matrix-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -n ${ABOLETH_ENV:-} ]] && source "$ABOLETH_ENV"
source .venv/bin/activate
# This tree's package and build, whatever the submitting shell exported:
# single-node runs (decompose's world 4) otherwise import from PYTHONPATH
# (multinode-launch.sh already puts its own tree first).
export PYTHONPATH=$PWD/python ABOLETH_BUILD_DIR=$PWD/build
export NODES=${NODES:-$SLURM_JOB_NUM_NODES}
# The Tier-1 shape cache keys on the runtime objects' paths, not their
# contents, so a rebuilt runtime would reuse shapes linked against the old
# one: each job compiles its own.
export ABOLETH_BENCH_CACHE=${ABOLETH_BENCH_CACHE:-$PWD/.bench-cache/job-$SLURM_JOB_ID}
# Preflight: every task must start Python and reach its GPU. One stuck rank
# stalls every arm's bootstrap, so name the hosts that did not answer and
# stop before the matrix spends the allocation (resubmit with --exclude);
# the failing tasks' errors stay in the job log.
expected=$((NODES * SLURM_NTASKS_PER_NODE))
answered=$(srun --nodes="$NODES" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
  --gpus-per-node=4 --cpus-per-task=16 --overlap --kill-on-bad-exit=0 \
  timeout 300 python -c '
import os, socket, torch
torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
torch.ones(1, device="cuda").sum().item()
print("ready", socket.gethostname(), os.environ["SLURM_PROCID"])' \
  | grep '^ready' || true)
count=$(grep -c '^ready' <<<"$answered" || true)
echo "matrix preflight: $count/$expected tasks ready"
if ((count != expected)); then
  scontrol show hostnames "$SLURM_JOB_NODELIST" | while read -r host; do
    ready=$(grep -c " $host " <<<"$answered" || true)
    ((ready == SLURM_NTASKS_PER_NODE)) ||
      echo "matrix preflight: $host has $ready/$SLURM_NTASKS_PER_NODE tasks"
  done
  exit 3
fi
# GPU inventory: every node shows MATRIX_GPUS_PER_NODE GPUs named
# MATRIX_GPU_NAME to nvidia-smi (the preflight above only proves that each
# task's own device answers).
inventory=$(srun --nodes="$NODES" --ntasks-per-node=1 --gpus-per-node=4 \
  --cpus-per-task=16 --overlap --kill-on-bad-exit=0 timeout 120 bash -c \
  'echo "gpus $(hostname -s) $(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "${MATRIX_GPU_NAME:-H100}")"' \
  | grep '^gpus' || true)
short=$(awk -v want="${MATRIX_GPUS_PER_NODE:-4}" '$3 != want {print $2 ": " $3}' <<<"$inventory")
listed=$(grep -c '^gpus' <<<"$inventory" || true)
echo "matrix inventory: $listed/$NODES nodes listed their GPUs"
if [[ -n $short ]] || ((listed != NODES)); then
  echo "matrix inventory: nodes without ${MATRIX_GPUS_PER_NODE:-4} GPUs: ${short:-(a node did not answer)}"
  exit 3
fi
# The fabric this job holds, measured before the matrix (linkcheck.sh): a
# link that trained down would otherwise read as a library's result. With
# LINKCHECK_GATE=1 a GPU-memory link under the threshold, or any test that
# could not run, stops the job once a second check confirms it (a single
# failed connection has been transient); host-memory-only LOW readings are
# a known second-socket PCIe effect and only reported.
gate_lines() {
  grep -E '^LINK .* gpu bidir_avg_Gbps=.*(LOW|FAILED)|^LINK .*FAILED' "$1" || true
}
links=${2:?output directory}/linkcheck/$SLURM_JOB_ID
experiments/bench/linkcheck.sh "$links" || true
if ((${LINKCHECK_GATE:-0})); then
  bad=$(gate_lines "$links/links.txt")
  if [[ -n $bad ]]; then
    echo "matrix linkcheck: $(wc -l <<<"$bad") bad readings; checking again"
    experiments/bench/linkcheck.sh "$links/retry" || true
    bad=$(gate_lines "$links/retry/links.txt")
    if [[ -n $bad ]]; then
      echo "matrix linkcheck: still bad, stopping (resubmit with --exclude):"
      echo "$bad"
      exit 4
    fi
  fi
  echo "matrix linkcheck: gate passed"
fi
exec experiments/bench/matrix.sh "$@"
