include_guard(GLOBAL)

# NVSHMEM, used only by the comparison benchmarks in experiments/cuda. The
# default prefix is the nvidia-nvshmem wheel installed in the build venv.

if(ABOLETH_ENABLE_NVSHMEM_BENCHMARKS STREQUAL "OFF" OR NOT ABOLETH_HAS_CUDA)
  return()
endif()

# NVSHMEM's device-inlined IBGDA path (ibgda_device.cuh) includes
# infiniband/mlx5dv.h, so the benchmarks need rdma-core headers even though
# NVSHMEM itself carries the transport libraries.
if(NOT TARGET rdma_headers OR NOT EXISTS "${ABOLETH_RDMA_INCLUDE_DIR}/infiniband/mlx5dv.h")
  if(ABOLETH_ENABLE_NVSHMEM_BENCHMARKS STREQUAL "ON")
    message(FATAL_ERROR "ABOLETH_ENABLE_NVSHMEM_BENCHMARKS=ON but the rdma-core "
      "headers (infiniband/mlx5dv.h) NVSHMEM's IBGDA device path compiles against are missing")
  endif()
  message(STATUS "[Aboleth] NVSHMEM benchmarks: auto-disabled (rdma-core headers missing)")
  return()
endif()

if(NOT NVSHMEM_ROOT)
  aboleth_python_query(_nvshmem_dir
    "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])")
  if(_nvshmem_dir)
    set(NVSHMEM_ROOT "${_nvshmem_dir}")
  endif()
endif()

find_path(ABOLETH_NVSHMEM_INCLUDE_DIR NAMES nvshmem.h
  HINTS ${NVSHMEM_ROOT} PATH_SUFFIXES include)
find_library(ABOLETH_NVSHMEM_HOST_LIBRARY NAMES nvshmem_host libnvshmem_host.so.3
  NAMES_PER_DIR HINTS ${NVSHMEM_ROOT} PATH_SUFFIXES lib lib64)
find_library(ABOLETH_NVSHMEM_DEVICE_LIBRARY NAMES nvshmem_device
  HINTS ${NVSHMEM_ROOT} PATH_SUFFIXES lib lib64)

if(NOT ABOLETH_NVSHMEM_INCLUDE_DIR
    OR NOT ABOLETH_NVSHMEM_HOST_LIBRARY
    OR NOT ABOLETH_NVSHMEM_DEVICE_LIBRARY)
  if(ABOLETH_ENABLE_NVSHMEM_BENCHMARKS STREQUAL "ON")
    message(FATAL_ERROR "ABOLETH_ENABLE_NVSHMEM_BENCHMARKS=ON but NVSHMEM was "
      "not found (pass -DNVSHMEM_ROOT=<prefix>)")
  endif()
  message(STATUS "[Aboleth] NVSHMEM: not found (pass -DNVSHMEM_ROOT=<prefix>)")
  return()
endif()

get_filename_component(ABOLETH_NVSHMEM_LIB_DIR "${ABOLETH_NVSHMEM_HOST_LIBRARY}" DIRECTORY)

add_library(nvshmem::host SHARED IMPORTED GLOBAL)
set_target_properties(nvshmem::host PROPERTIES
  IMPORTED_LOCATION "${ABOLETH_NVSHMEM_HOST_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ABOLETH_NVSHMEM_INCLUDE_DIR}")

add_library(nvshmem::device STATIC IMPORTED GLOBAL)
set_target_properties(nvshmem::device PROPERTIES
  IMPORTED_LOCATION "${ABOLETH_NVSHMEM_DEVICE_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ABOLETH_NVSHMEM_INCLUDE_DIR}")

aboleth_enable_feature(NVSHMEM)
message(STATUS "[Aboleth] NVSHMEM: ${ABOLETH_NVSHMEM_LIB_DIR}")
