#!/usr/bin/env bash
# Run a test only when this machine can actually run it.
#
# CTest reads exit code 77 as "skipped" (SKIP_RETURN_CODE), so a laptop with no
# HCA reports the RDMA tests as skipped rather than failed, and the suite stays
# meaningful on every machine. Anything else — a real failure, a timeout — is
# passed through unchanged.
set -uo pipefail

TIMEOUT=""
SUDO=0
NEEDS=()
NEEDS_MLX5=0

usage() {
  cat <<'EOF'
Usage: scripts/run-gated.sh [--needs <gpu|multi-gpu|hca|mlx5|efa|sudo>]...
                            [--needs gpus=<n>] [--sudo] [--timeout <seconds>]
                            -- <command>...

Exits 77 (ctest "skipped") when a prerequisite is absent.
EOF
}

skip() {
  printf 'SKIP: %s\n' "$*"
  exit 77
}

while (($#)); do
  case "$1" in
    --needs) NEEDS+=("$2"); shift 2 ;;
    --sudo) SUDO=1; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

(($#)) || { usage >&2; exit 2; }

gpu_count() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
  nvidia-smi --list-gpus 2>/dev/null | wc -l
}

has_verbs_device() {
  local pattern="$1"
  local directory
  [[ -d /sys/class/infiniband ]] || return 1
  for directory in /sys/class/infiniband/*; do
    [[ -e "$directory" ]] || return 1
    [[ "$(basename -- "$directory")" == $pattern ]] && return 0
    # Predictable IB names may be ibp*, while the provider is still mlx5.
    [[ "$pattern" == 'mlx5_*' && "${directory}/device/driver" -ef /sys/bus/pci/drivers/mlx5_core ]] && return 0
  done
  return 1
}

for need in "${NEEDS[@]}"; do
  case "$need" in
    gpu)
      (($(gpu_count) >= 1)) || skip "no GPU visible to nvidia-smi" ;;
    multi-gpu)
      (($(gpu_count) >= 2)) || skip "fewer than two GPUs" ;;
    gpus=*)
      # Several workloads pair fixed device indices (0-4, 1-5, ...) and need a
      # specific number of GPUs, not merely more than one.
      required="${need#gpus=}"
      (($(gpu_count) >= required)) || skip "fewer than ${required} GPUs" ;;
    hca)
      has_verbs_device '*' || skip "no InfiniBand/RoCE device in /sys/class/infiniband" ;;
    mlx5)
      NEEDS_MLX5=1
      has_verbs_device 'mlx5_*' || skip "no mlx5 device" ;;
    efa)
      has_verbs_device 'efa_*' || skip "no EFA adapter" ;;
    sudo)
      SUDO=1 ;;
    *)
      printf 'run-gated: unknown prerequisite: %s\n' "$need" >&2; exit 2 ;;
  esac
done

COMMAND=("$@")

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(dirname -- "$SCRIPT_DIRECTORY")"
UAR_PROBE="${REPOSITORY_ROOT}/test/lint/probe_uar.py"

# The interpreter the gated command itself uses, so the probe imports the same
# aboleth package and the same _runtime extension the test will.
probe_python() {
  local candidate
  if [[ -n "${ABOLETH_PROBE_PYTHON:-}" ]]; then
    printf '%s\n' "${ABOLETH_PROBE_PYTHON}"
    return 0
  fi
  if [[ "$(basename -- "${COMMAND[0]}")" == python* ]]; then
    printf '%s\n' "${COMMAND[0]}"
    return 0
  fi
  # A compiled probe (gpu_wqe_loopback, aboleth_backend_test) carries no
  # interpreter of its own; the venv CMake configured with is the next best.
  for candidate in "${REPOSITORY_ROOT}/.venv/bin/python" "$(command -v python3 2>/dev/null)"; do
    [[ -n "$candidate" && -x "$candidate" ]] && { printf '%s\n' "$candidate"; return 0; }
  done
  return 1
}

uar_cache_path() {
  local user boot
  user="${USER:-$(id -un 2>/dev/null || echo unknown)}"
  boot=unknown
  [[ -r /proc/sys/kernel/random/boot_id ]] &&
    IFS= read -r boot < /proc/sys/kernel/random/boot_id
  printf '%s/aboleth-uar-probe-%s-%s\n' "${TMPDIR:-/tmp}" "$user" "$boot"
}

uar_cache_write() {
  local path="$1" verdict="$2" temporary
  temporary="$(mktemp "${path}.XXXXXX" 2>/dev/null)" || return 0
  if printf '%s\n' "$verdict" >"$temporary" 2>/dev/null; then
    mv -f "$temporary" "$path" 2>/dev/null || rm -f "$temporary"
  else
    rm -f "$temporary"
  fi
  return 0
}

# Whether this user can map the NIC's user-access region without root. Only the
# mapping itself answers that — a driver with PeerMappingOverride=1 hands it to
# any user — so run the real thing once and remember the verdict, which cannot
# change without a driver reload, for the rest of the boot. Delete the cache
# file to force a re-probe.
UAR_REASON=""
uar_probe_ok() {
  local cache verdict python output status
  cache="$(uar_cache_path)"
  if [[ -r "$cache" ]]; then
    IFS= read -r verdict <"$cache" || verdict=""
    case "$verdict" in
      ok) return 0 ;;
      fail:*) UAR_REASON="${verdict#fail: }"; return 1 ;;
    esac
  fi

  if [[ ! -r "$UAR_PROBE" ]]; then
    UAR_REASON="probe script is missing: ${UAR_PROBE}"
    return 1
  fi
  python="$(probe_python)" || {
    UAR_REASON="no python interpreter to run the probe with"
    return 1
  }

  output="$("$python" "$UAR_PROBE" 2>&1 >/dev/null)"
  status=$?
  if ((status == 0)); then
    uar_cache_write "$cache" "ok"
    return 0
  fi
  UAR_REASON="${output##*$'\n'}"
  [[ -n "$UAR_REASON" ]] || UAR_REASON="probe exited ${status}"
  uar_cache_write "$cache" "fail: ${UAR_REASON}"
  return 1
}

if ((SUDO)); then
  # The probe exercises the mlx5 UAR map specifically, so its verdict only
  # transfers to tests that need that adapter; anything else keeps the sudo
  # requirement it always had.
  if ((NEEDS_MLX5)) && uar_probe_ok; then
    : # The mapping works unprivileged; run the command as this user.
  elif sudo -n true 2>/dev/null; then
    # sudo scrubs the environment, and everything here — PYTHONPATH, the plugin
    # path, the Triton cache — arrives that way. Re-export it explicitly.
    ENV_ARGS=()
    for name in PYTHONPATH PATH LD_LIBRARY_PATH \
                ABOLETH_PLUGIN_PATH TRITON_PLUGIN_PATHS ABOLETH_BUILD_DIR \
                ABOLETH_EFA_BITCODE TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
                TRITON_ALWAYS_COMPILE TRITON_HOME PYTHONUNBUFFERED CUDA_VISIBLE_DEVICES; do
      [[ -n "${!name:-}" ]] && ENV_ARGS+=("$name=${!name}")
    done
    # Test-specific variables set by CTest's ENVIRONMENT property.
    while IFS='=' read -r name _; do
      case "$name" in
        ABOLETH_*|FI_*|NCCL_*|NVSHMEM_*|EP8_*) ENV_ARGS+=("$name=${!name}") ;;
      esac
    done < <(env)
    COMMAND=(sudo -n env "${ENV_ARGS[@]}" "${COMMAND[@]}")
  elif ((NEEDS_MLX5)); then
    skip "NIC UAR mapping needs root here: unprivileged probe failed (${UAR_REASON}) and passwordless sudo is unavailable"
  else
    skip "passwordless sudo is required (NIC UAR mapping)"
  fi
fi

if [[ -n "$TIMEOUT" ]]; then
  COMMAND=(timeout "${TIMEOUT}s" "${COMMAND[@]}")
fi

"${COMMAND[@]}"
status=$?

# 77 from the command itself would be misread as a skip we did not decide.
if ((status == 77)); then
  printf 'run-gated: command exited 77, which ctest reserves for skips\n' >&2
  exit 1
fi
exit $status
