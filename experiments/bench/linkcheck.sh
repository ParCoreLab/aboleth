#!/usr/bin/env bash
# Fabric check of the nodes a batch job holds, run before the benchmark
# (matrix.sbatch): a link that trained down or a slow PCIe path would
# otherwise read as a library's result. Inside an allocation:
#
#   experiments/bench/linkcheck.sh <dir>
#
# Every node reports its mlx5 ports (state, rate, link layer) and which HCA
# sits nearest each GPU (the runtime's rule: deepest shared PCI path). The
# nodes then pair up over a ring in two rounds (one pair at 2 nodes), so
# each node is measured against two neighbours; each pair runs perftest on
# all four GPU<->HCA pairs at once: bidirectional RDMA-write bandwidth
# (1 MiB, 3 s) from GPU memory (DMA-BUF, as the benchmarks register it)
# and from host memory (separating the link from the PCIe path), plus 8 B
# write latency. One LINK line per measurement goes to stdout; bidirectional
# bandwidth under LINKCHECK_LOW_GBPS (default 350, ~90% of 2 x NDR200) is
# flagged LOW. Raw perftest output lands in <dir>/<host>_r<round>.*.
# Never fails the job: the record says what the fabric did.
set -uo pipefail
out=${1:?output directory}
mkdir -p "$out"
export LC_OUT=$out LC_LOW=${LINKCHECK_LOW_GBPS:-350}
LC_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
export LC_HOSTS
nodes=$(wc -w <<<"$LC_HOSTS")
echo "linkcheck: $nodes nodes: $LC_HOSTS"

read -r -d '' node_script <<'NODE'
round=$1
read -ra H <<<"$LC_HOSTS"
N=${#H[@]}
me=$SLURM_NODEID
host=$(hostname -s)
log=$LC_OUT/${host}_r$round
if ((round == 0)); then
  if ((me % 2 == 0)); then role=server; peer=$((me + 1)); else role=client; peer=$((me - 1)); fi
else
  if ((me % 2 == 1)); then role=server; peer=$(((me + 1) % N)); else role=client; peer=$(((me - 1 + N) % N)); fi
fi
peerhost=${H[$peer]}
if ((round == 0)); then
  for d in /sys/class/infiniband/mlx5_*; do
    echo "PORT $host $(basename "$d") $(cat "$d/ports/1/state") $(cat "$d/ports/1/rate") $(cat "$d/ports/1/link_layer")"
  done
fi
depth() {
  local IFS=/
  local -a a b
  read -ra a <<<"$1"
  read -ra b <<<"$2"
  local i=0
  while ((i < ${#a[@]} && i < ${#b[@]})) && [[ ${a[i]} == "${b[i]}" ]]; do ((i++)); done
  echo "$i"
}
declare -a HCA
for k in 0 1 2 3; do
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$k" | tr 'A-F' 'a-f')
  gpath=$(readlink -f "/sys/bus/pci/devices/${bus:4}")
  best="" bestd=-1
  for d in /sys/class/infiniband/mlx5_*; do
    [[ $(cat "$d/ports/1/link_layer") == InfiniBand ]] || continue
    [[ $(cat "$d/ports/1/state") == *ACTIVE* ]] || continue
    dd=$(depth "$gpath" "$(readlink -f "$d/device")")
    ((dd > bestd)) && { bestd=$dd; best=$(basename "$d"); }
  done
  HCA[k]=$best
done
((round == 0)) && echo "MAP $host gpu0..3 -> ${HCA[*]}"
bw="-F --report_gbits -s 1048576 -D 3 -b"
# tag port command...: the server listens once; the client retries until
# the server answers (perftest does not), so start-up skew cannot fail it.
test_one() {
  local tag=$1 port=$2
  shift 2
  if [[ $role == server ]]; then
    timeout 150 "$@" -p "$port" >"$log.$tag.server" 2>&1
  else
    local attempt
    for attempt in $(seq 1 40); do
      timeout 40 "$@" -p "$port" "$peerhost" >"$log.$tag" 2>&1 && return
      grep -q -i "couldn't connect\|unable to connect\|connection refused" "$log.$tag" || return
      sleep 1
    done
  fi
}
if [[ $role == server ]]; then
  # Every listener at once; each waits for its client.
  for k in 0 1 2 3; do
    test_one gpu$k $((18515 + k)) ib_write_bw -d "${HCA[k]}" --use_cuda="$k" --use_cuda_dmabuf $bw &
    test_one host$k $((18525 + k)) ib_write_bw -d "${HCA[k]}" $bw &
  done
  test_one lat 18535 ib_write_lat -d "${HCA[0]}" -F -s 8 -n 20000 &
  wait
  exit 0
fi
for k in 0 1 2 3; do
  test_one gpu$k $((18515 + k)) ib_write_bw -d "${HCA[k]}" --use_cuda="$k" --use_cuda_dmabuf $bw &
done
wait
for k in 0 1 2 3; do
  test_one host$k $((18525 + k)) ib_write_bw -d "${HCA[k]}" $bw &
done
wait
test_one lat 18535 ib_write_lat -d "${HCA[0]}" -F -s 8 -n 20000
for k in 0 1 2 3; do
  for mem in gpu host; do
    f=$log.$mem$k
    value=$(awk '$1 == 1048576 {print $4}' "$f" | tail -1)
    flag=""
    if [[ -z $value ]]; then
      flag=" FAILED($(grep -m1 -i -E 'error|fail|unable|couldn' "$f" | tr -s ' ' | cut -c1-80))"
    elif awk -v b="$value" -v low="$LC_LOW" 'BEGIN{exit !(b < low)}'; then
      flag=" LOW"
    fi
    echo "LINK r$round $peerhost<->$host gpu$k ${HCA[k]} $mem bidir_avg_Gbps=${value:-NA}$flag"
  done
done
lat=$(awk '$1 == 8 {print $5, $6}' "$log.lat" | tail -1)
echo "LINK r$round $peerhost<->$host ${HCA[0]} write_lat_8B_us(typical,avg)=${lat:-NA}"
NODE

rounds=1
((nodes > 2)) && rounds=2
for ((round = 0; round < rounds; round++)); do
  srun --overlap --nodes="$nodes" --ntasks-per-node=1 --cpus-per-task=16 \
    --gpus-per-node=4 --kill-on-bad-exit=0 timeout 300 \
    bash -c "$node_script" linkcheck "$round" | sort
done | tee "$out/links.txt"
links=$(grep -c '^LINK .*_Gbps=' "$out/links.txt")
bad=$(grep -c '^LINK .*\(LOW\|FAILED\)' "$out/links.txt")
echo "linkcheck: done: $bad of $links bandwidth measurements LOW or FAILED"
exit 0
