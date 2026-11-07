#!/bin/bash
# Python environments for the DSL comparison arms, layered on the tree's own
# environment (.venv: torch, Triton + the Aboleth plugin, NVSHMEM, NCCL), so
# every DSL arm runs the same torch and CUDA runtime as the CUDA arms.
#
#   experiments/dsl/setup-envs.sh cute     # CuTe DSL 4.8.1, NVSHMEM4Py 0.4.1, cuTile 1.7
#   experiments/dsl/setup-envs.sh tdist    # Triton-distributed 7908e4e
#
# cute:  site.addsitedir(base venv) behind a local site-packages, so the base
#        torch/Triton/aboleth import and the local CuTe DSL, NVSHMEM 3.8.0 and
#        NVSHMEM4Py shadow theirs. --no-deps keeps numpy and the CUDA Python
#        bindings at the base versions.
# tdist: its own Triton (upstream 3.7.1 + Triton-distributed's patches,
#        TRITON_EXT_ENABLED). The base venv is a plain path entry (no .pth
#        processing), so its editable Triton finder never loads and torch
#        still imports. NVSHMEM 3.6.5 + NVSHMEM4Py 0.3.0, as
#        Triton-distributed's setup.py pins.
#
# Environment:
#   ABOLETH_DSL_ROOT   where envs/, src/ and cache/ go (default: .dsl)
#   ABOLETH_BASE_VENV  the tree's environment (default: .venv)
#   UV                 the uv binary (default: uv on PATH)
#
# matrix.sh's dsl-* phases then take ABOLETH_DSL_PYTHON=$ABOLETH_DSL_ROOT/
# envs/cute/bin/python and ABOLETH_TDIST_PYTHON=$ABOLETH_DSL_ROOT/envs/tdist/
# bin/python.
set -euo pipefail
which=${1:?cute or tdist}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
D=${ABOLETH_DSL_ROOT:-$ROOT/.dsl}
N=${ABOLETH_BASE_VENV:-$ROOT/.venv}
PYTHON=$(readlink -f "$N/bin/python")
PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
BASE=$N/lib/python$PYVER/site-packages
UV=${UV:-uv}
export UV_CACHE_DIR=$D/cache/uv PIP_CACHE_DIR=$D/cache/pip
mkdir -p "$D"

case $which in
cute)
  E=$D/envs/cute
  [ -d "$E" ] || "$UV" venv --python "$PYTHON" "$E"
  echo "import site; site.addsitedir(\"$BASE\")" \
    > "$E/lib/python$PYVER/site-packages/zz_aboleth_base.pth"
  "$UV" pip install --no-deps --python "$E/bin/python" \
    nvidia-cutlass-dsl==4.8.1 nvidia-cutlass-dsl-libs-base==4.8.1 \
    nvidia-cutlass-dsl-libs-core==4.8.1 nvidia-cutlass-dsl-libs-cu12==4.8.1 \
    nvidia-nvshmem-cu12==3.8.0 nvshmem4py-cu12==0.4.1 \
    cuda-core==1.2.2 cuda-pathfinder==1.8.4 nvidia-cuda-cccl-cu12==12.9.27
  # cuTile 1.7 and its TileIR compiler (CTK 13.3 tileiras, under nvidia/cu13):
  # Hopper needs tileiras >= 13.3.
  "$UV" pip install --python "$E/bin/python" cuda-tile==1.7.0 \
    "cuda-toolkit[tileiras,nvvm,nvcc]==13.3.*" "nvidia-nvjitlink==13.3.*"
  ;;
tdist)
  E=$D/envs/tdist
  T=$D/src/triton-distributed
  COMMIT=7908e4e
  export TRITON_HOME=$D/cache/tdist-home   # LLVM download, not in $HOME
  if [ ! -d "$T/.git" ]; then
    git clone https://github.com/ByteDance-Seed/Triton-distributed.git "$T"
  fi
  git -C "$T" checkout --quiet $COMMIT
  git -C "$T" submodule update --init --depth 1 3rdparty/triton
  [ -d "$E" ] || "$UV" venv --python "$PYTHON" "$E"
  echo "$BASE" > "$E/lib/python$PYVER/site-packages/zz_aboleth_base_path.pth"
  "$UV" pip install --python "$E/bin/python" pip "setuptools<80" wheel pybind11 \
    cmake ninja lit nvidia-ml-py
  "$UV" pip install --no-deps --python "$E/bin/python" \
    nvidia-nvshmem-cu12==3.6.5 nvshmem4py-cu12==0.3.0 \
    cuda-core==1.2.2 cuda-pathfinder==1.8.4
  source "$E/bin/activate"
  cd "$T"
  MAX_JOBS=${MAX_JOBS:-16} bash scripts/build_triton.sh
  export USE_TRITON_DISTRIBUTED_AOT=0
  MAX_JOBS=${MAX_JOBS:-16} pip install -e python --no-deps \
    --no-build-isolation --use-pep517 --verbose
  cd /
  python -c "import triton, triton_dist, torch; \
print('triton', triton.__version__); print('torch', torch.__version__)"
  ;;
*) echo "unknown env $which" >&2; exit 2 ;;
esac
echo "setup-envs: $which ready in $E"
