#!/usr/bin/env bash
# Build stock Triton and turn its build tree into an install prefix Aboleth can
# consume.
#
# Triton is built through pip rather than plain CMake because its CMakeLists
# expects the wheel layout (TRITON_WHEEL_DIR, the vendored LLVM download, the
# nanobind extension). TRITON_EXT_ENABLED=1 is what compiles the plugin API that
# libAbolethPlugin.so links against.
set -euo pipefail

SOURCE=""
BUILD_DIR=""
PREFIX=""
PYTHON=""
JOBS="${MAX_JOBS:-8}"

usage() {
  cat <<'EOF'
Usage: scripts/build-triton.sh --source <dir> --build-dir <dir> --prefix <dir> --python <exe>

Builds Triton as an editable install and produces an Aboleth-consumable prefix
(<prefix>/include, <prefix>/lib/libtriton.so).
EOF
}

die() { printf 'build-triton: %s\n' "$*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

[[ -n "$SOURCE" && -n "$BUILD_DIR" && -n "$PREFIX" && -n "$PYTHON" ]] ||
  { usage >&2; die "missing required option"; }
[[ -f "$SOURCE/CMakeLists.txt" ]] || die "$SOURCE is not a Triton source tree"
[[ -x "$PYTHON" ]] || die "$PYTHON is not executable"

mkdir -p "$BUILD_DIR" "$PREFIX"

printf 'Building Triton (%s jobs) into %s...\n' "$JOBS" "$BUILD_DIR"
TRITON_EXT_ENABLED=1 \
TRITON_BUILD_DIR="$BUILD_DIR" \
MAX_JOBS="$JOBS" \
  "$PYTHON" -m pip install --no-build-isolation --no-deps --editable "$SOURCE"

# `pip install --editable` leaves the CMake build tree behind; install from it.
# Triton's get_cmake_dir() configures straight into TRITON_BUILD_DIR when that
# is set, and only falls back to a `cmake.<platform>` subdirectory when it is
# not — so look for the tree itself before looking for one underneath it.
# `find` on a directory that does not exist fails, which under `set -e` and
# `pipefail` would end this script without a word, so guard each search.
find_cmake_dir() {
  local root="$1"
  [[ -d "$root" ]] || return 0
  [[ -f "$root/CMakeCache.txt" ]] && { printf '%s\n' "$root"; return 0; }
  find "$root" -maxdepth 1 -name 'cmake.*' -type d | head -n 1
}

TRITON_CMAKE_DIR="$(find_cmake_dir "$BUILD_DIR")"
[[ -n "$TRITON_CMAKE_DIR" ]] ||
  TRITON_CMAKE_DIR="$(find_cmake_dir "$SOURCE/build")"
[[ -n "$TRITON_CMAKE_DIR" ]] || die "no Triton CMake build tree found under $BUILD_DIR"

CMAKE="$(dirname -- "$PYTHON")/cmake"
[[ -x "$CMAKE" ]] || CMAKE="$(command -v cmake)" || die "cmake not found"

printf 'Installing Triton headers and library into %s...\n' "$PREFIX"
"$CMAKE" --install "$TRITON_CMAKE_DIR" --prefix "$PREFIX" >/dev/null

# glibc keys shared objects by inode. The interpreter loads
# triton/_C/libtriton.so; if the plugin resolved the installed *copy* instead,
# the process would hold two MLIRs and every TypeID comparison across the
# plugin boundary would fail. Point the prefix at the live file.
LIVE_LIBTRITON="$("$PYTHON" -c \
  "import triton, pathlib; print(pathlib.Path(triton.__file__).parent / '_C' / 'libtriton.so')")"
[[ -f "$LIVE_LIBTRITON" ]] || die "the interpreter has no triton/_C/libtriton.so"
ln -sfn "$LIVE_LIBTRITON" "$PREFIX/lib/libtriton.so"

printf 'Triton prefix ready: %s\n' "$PREFIX"
