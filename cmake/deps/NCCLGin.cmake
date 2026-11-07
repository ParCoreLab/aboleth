include_guard(GLOBAL)

# NCCL with GPU-initiated networking, used only by the ep8 comparison benchmark.
# It needs a newer NCCL than torch ships, so bootstrap.sh installs it into a
# side venv (.tools/venvs/gin) that this module probes by default.

if(ABOLETH_ENABLE_NCCL_GIN_BENCHMARKS STREQUAL "OFF" OR NOT ABOLETH_HAS_CUDA)
  return()
endif()

if(NOT NCCL_ROOT)
  set(_gin_site "${PROJECT_SOURCE_DIR}/.tools/venvs/gin/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/nvidia/nccl")
  if(EXISTS "${_gin_site}")
    set(NCCL_ROOT "${_gin_site}")
  endif()
endif()

find_path(ABOLETH_NCCL_INCLUDE_DIR NAMES nccl.h
  HINTS ${NCCL_ROOT} PATH_SUFFIXES include)
# NAMES_PER_DIR: the wheel ships libnccl.so.2 with no .so symlink, and without
# this a system libnccl.so would outrank the pinned one in NCCL_ROOT.
find_library(ABOLETH_NCCL_LIBRARY NAMES nccl libnccl.so.2 NAMES_PER_DIR
  HINTS ${NCCL_ROOT} PATH_SUFFIXES lib lib64)

if(NOT ABOLETH_NCCL_INCLUDE_DIR OR NOT ABOLETH_NCCL_LIBRARY)
  if(ABOLETH_ENABLE_NCCL_GIN_BENCHMARKS STREQUAL "ON")
    message(FATAL_ERROR "ABOLETH_ENABLE_NCCL_GIN_BENCHMARKS=ON but NCCL was "
      "not found (pass -DNCCL_ROOT=<prefix>)")
  endif()
  message(STATUS "[Aboleth] NCCL (gin): not found (pass -DNCCL_ROOT=<prefix>)")
  return()
endif()

add_library(nccl::nccl SHARED IMPORTED GLOBAL)
set_target_properties(nccl::nccl PROPERTIES
  IMPORTED_LOCATION "${ABOLETH_NCCL_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ABOLETH_NCCL_INCLUDE_DIR}")

aboleth_enable_feature(NCCL_GIN)
message(STATUS "[Aboleth] NCCL (gin): ${ABOLETH_NCCL_LIBRARY}")
