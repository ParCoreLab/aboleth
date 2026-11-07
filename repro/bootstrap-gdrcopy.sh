#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GDRCOPY_ROOT="${GDRCOPY_ROOT:-"$ROOT/../gdrcopy"}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
GDRCOPY_REVISION="fcec3ce0bb40a97a6cc45dd4afeec4bccb509712"
JOBS="${MAX_JOBS:-16}"

if [[ ! -d "$GDRCOPY_ROOT/.git" ]]; then
  git clone https://github.com/NVIDIA/gdrcopy.git \
    "$GDRCOPY_ROOT"
fi

git -C "$GDRCOPY_ROOT" checkout "$GDRCOPY_REVISION"
make -C "$GDRCOPY_ROOT" -j "$JOBS" \
  prefix="$GDRCOPY_ROOT/install" \
  CUDA="$CUDA_HOME" \
  all install

if [[ ! -e /dev/gdrdrv ]]; then
  (
    cd "$GDRCOPY_ROOT"
    sudo ./insmod.sh
  )
fi

printf 'GDRCopy ready: %s\n' "$GDRCOPY_ROOT/install"
