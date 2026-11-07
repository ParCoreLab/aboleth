#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV_VERSION="0.11.28"
PYTHON_VERSION="3.12.13"
TRITON_REVISION="8d749c1660d399b71d8a115bd0aacfa7f955d65b"
JOBS="${MAX_JOBS:-8}"
BUILD_DIR="${ABOLETH_BUILD_DIR:-$ROOT/build}"
VERIFY=0
HARDWARE_TESTS=0
RECREATE=1
EFA=0
CMAKE_ARGS=()

usage() {
  cat <<'EOF'
Usage: repro/bootstrap.sh [options] [-- <extra cmake -D flags>]

Recreate the pinned Aboleth and NCCL GIN environments, build stock Triton, and
configure and build everything with CMake.

Options:
  --verify          Run the compiler, Python and non-root GPU suites.
  --hardware-tests  Also run the root-required RDMA and workload tests.
                    Implies --verify and requires passwordless sudo.
  --efa             Additionally create .venv-efa and register the EFA tests.
  --keep-env        Reuse .venv instead of recreating it.
  -h, --help        Show this help.

Dependency locations are CMake's job now. Anything the machine does not keep in
a default location is passed through, for example:

  repro/bootstrap.sh -- -DRdmaCore_ROOT=/opt/rdma-core -DLibfabric_ROOT=/opt/amazon/efa

Set MAX_JOBS to control build parallelism (default: 8).
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --verify) VERIFY=1 ;;
    --hardware-tests) VERIFY=1; HARDWARE_TESTS=1 ;;
    --efa) EFA=1 ;;
    --keep-env) RECREATE=0 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; CMAKE_ARGS=("$@"); break ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

cd "$ROOT"

# CMake reports what it cannot find, with the flag that fixes it, so this only
# checks the tools needed to get as far as running CMake.
for tool in curl git gcc g++; do
  command -v "$tool" >/dev/null 2>&1 || die "required command not found: $tool"
done

git -C "$ROOT" submodule update --init --recursive third_party/triton
[[ "$(git -C "$ROOT/third_party/triton" rev-parse HEAD)" == "$TRITON_REVISION" ]] ||
  die "third_party/triton is not at the pinned stock revision"
for file in requirements.freeze.txt gin-requirements.base.txt gin-requirements.freeze.txt; do
  [[ -f "$ROOT/repro/$file" ]] || die "repro/$file is missing"
done

mkdir -p "$ROOT/.tools/bin" "$ROOT/.uv/python" "$ROOT/.triton"
UV="$ROOT/.tools/bin/uv"

if [[ ! -x "$UV" ]] || [[ "$("$UV" --version 2>/dev/null || true)" != "uv $UV_VERSION "* ]]; then
  printf 'Installing uv %s...\n' "$UV_VERSION"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" |
    env UV_INSTALL_DIR="$ROOT/.tools/bin" sh
fi

export UV_PYTHON_INSTALL_DIR="$ROOT/.uv/python"
export TRITON_HOME="$ROOT/.triton"

"$UV" python install "$PYTHON_VERSION"

if ((RECREATE)); then
  "$UV" venv --clear --python "$PYTHON_VERSION" "$ROOT/.venv"
  "$UV" venv --clear --python "$PYTHON_VERSION" "$ROOT/.tools/venvs/gin"
elif [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$UV" venv --python "$PYTHON_VERSION" "$ROOT/.venv"
fi
if [[ ! -x "$ROOT/.tools/venvs/gin/bin/python" ]]; then
  "$UV" venv --python "$PYTHON_VERSION" "$ROOT/.tools/venvs/gin"
fi

PYTHON="$ROOT/.venv/bin/python"
GIN_PYTHON="$ROOT/.tools/venvs/gin/bin/python"

printf 'Synchronizing pinned Python packages...\n'
"$UV" pip sync \
  --python "$PYTHON" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  "$ROOT/repro/requirements.freeze.txt"

# Triton's setup.py locates ninja through PATH, so the pinned build tools have
# to come ahead of whatever the system provides.
export PATH="$ROOT/.venv/bin:$PATH"

printf 'Synchronizing pinned CUDA 12.6 Torch nightly GIN environment...\n'
"$UV" pip sync \
  --python "$GIN_PYTHON" \
  --index-strategy unsafe-best-match \
  "$ROOT/repro/gin-requirements.base.txt"
"$UV" pip install --python "$GIN_PYTHON" --no-deps "nvidia-nccl-cu12==2.30.7"

"$GIN_PYTHON" - <<'PY'
import ctypes
import importlib.metadata
from pathlib import Path

distribution = importlib.metadata.distribution("nvidia-nccl-cu12")
root = Path(distribution.locate_file("nvidia/nccl"))
library_path = root / "lib" / "libnccl.so.2"
library = ctypes.CDLL(str(library_path))
version = ctypes.c_int()
status = library.ncclGetVersion(ctypes.byref(version))

print(f"gin_torch={importlib.metadata.version('torch')}")
print(f"gin_nccl_package={distribution.version}")
print(f"gin_nccl_library={library_path}")
print(f"gin_nccl_runtime={version.value}")

if status != 0 or version.value != 23007:
    raise SystemExit("the isolated GIN environment did not load NCCL 2.30.7")
PY

printf 'Building stock Triton with extension support...\n'
MAX_JOBS="$JOBS" "$ROOT/scripts/build-triton.sh" \
  --source "$ROOT/third_party/triton" \
  --build-dir "$BUILD_DIR/deps/triton-build" \
  --prefix "$BUILD_DIR/deps/triton" \
  --python "$PYTHON" \
  --jobs "$JOBS"

"$PYTHON" - <<'PY'
import pathlib
import sys
import torch
import triton

root = pathlib.Path.cwd().resolve()
triton_source = pathlib.Path(triton.__file__).resolve()
expected = root / "third_party" / "triton"

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"triton={triton.__version__}, source={triton_source}")

if expected not in triton_source.parents:
    raise SystemExit(f"Triton does not resolve to the stock submodule: {triton_source}")
if not triton.__version__.startswith("3.8.0"):
    raise SystemExit(f"expected Triton 3.8.0, found {triton.__version__}")
PY

if ((EFA)); then
  # The EFA workloads run under their own interpreter: a different CUDA and a
  # different NCCL from the main environment.
  if ((RECREATE)) || [[ ! -x "$ROOT/.venv-efa/bin/python" ]]; then
    "$UV" venv --clear --python "$PYTHON_VERSION" "$ROOT/.venv-efa"
  fi
  "$UV" pip install --python "$ROOT/.venv-efa/bin/python" \
    -r "$ROOT/repro/efa-requirements.txt"
  "$UV" pip install --python "$ROOT/.venv-efa/bin/python" --no-deps \
    "nvidia-nccl-cu13==2.30.7"
  TRITON_EXT_ENABLED=1 MAX_JOBS="$JOBS" "$UV" pip install \
    --python "$ROOT/.venv-efa/bin/python" \
    --no-build-isolation --no-deps --editable "$ROOT/third_party/triton"
  CMAKE_ARGS+=(
    "-DABOLETH_ENABLE_EFA=ON"
    "-DABOLETH_EFA_PYTHON=$ROOT/.venv-efa/bin/python"
  )
fi

printf 'Configuring and building Aboleth...\n'
CMAKE="$ROOT/.venv/bin/cmake"
"$CMAKE" -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_MAKE_PROGRAM="$ROOT/.venv/bin/ninja" \
  -DPython_EXECUTABLE="$PYTHON" \
  -DTriton_ROOT="$BUILD_DIR/deps/triton" \
  "${CMAKE_ARGS[@]}"
"$CMAKE" --build "$BUILD_DIR" --parallel "$JOBS"

"$UV" pip install --python "$PYTHON" --no-build-isolation --no-deps --editable "$ROOT"
if ((EFA)); then
  "$UV" pip install --python "$ROOT/.venv-efa/bin/python" \
    --no-build-isolation --no-deps --editable "$ROOT"
fi

CTEST="$ROOT/.venv/bin/ctest"

if ((VERIFY)); then
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for --verify"
  nvidia-smi >/dev/null
  "$CTEST" --test-dir "$BUILD_DIR" --output-on-failure \
    -L 'compiler-ir|goldens|python|lint|smoke'
fi

if ((HARDWARE_TESTS)); then
  sudo -n true ||
    die "--hardware-tests requires passwordless sudo for NIC UAR mapping"
  "$CTEST" --test-dir "$BUILD_DIR" --output-on-failure -L 'hardware|bench'
fi

printf 'Aboleth rebuild complete: %s\n' "$BUILD_DIR"
