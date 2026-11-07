# libaboleth_rdma_abi.so — the host-side transport runtime.
#
# One shared object carries every provider. Which providers are compiled in is a
# build-time decision (the ABOLETH_HAS_* feature macros); whether a compiled-in
# provider is usable is a runtime decision, answered by the dlopen shims in dl/
# and reported through aboleth_runtime_provider_caps().

if(ABOLETH_HAS_HIP)
  add_subdirectory(hip)
  return()
endif()

set(ABOLETH_RUNTIME_SOURCES
  core/aboleth_connection.cu
  core/aboleth_device_select.cu
  core/aboleth_endpoint_devx.cu
  core/aboleth_endpoint_rc.cu
  core/aboleth_memory.cu
  core/aboleth_runtime.cu
  provider_caps.cpp
  proxy/aboleth_proxy.cu
)

if(ABOLETH_HAS_EFA)
  # The EFA-only halves of the runtime — the libfabric fabric, the endpoints
  # built on it, the raw-efadv GPU-initiated control plane — and the vendored
  # device-primitive host half. The fabric and endpoint files share the types
  # in efa/aboleth_efa_internal.h with the mlx5/libfabric translation unit;
  # aboleth_efa_runtime.cu shares only aboleth_runtime_common.h.
  list(APPEND ABOLETH_RUNTIME_SOURCES
    efa/aboleth_efa_control.cu
    efa/aboleth_efa_endpoints.cu
    efa/aboleth_efa_runtime.cu
    "${ABOLETH_EFA_DP_DIR}/host/efa_cuda_dp.cpp"
  )
endif()

aboleth_add_library(aboleth_rdma_abi SHARED ${ABOLETH_RUNTIME_SOURCES})
set_target_properties(aboleth_rdma_abi PROPERTIES
  OUTPUT_NAME aboleth_rdma_abi
  CUDA_VISIBILITY_PRESET default
  CXX_VISIBILITY_PRESET default)

target_include_directories(aboleth_rdma_abi PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(ABOLETH_HAS_EFA)
  target_include_directories(aboleth_rdma_abi SYSTEM PRIVATE
    "${ABOLETH_EFA_DP_DIR}/common"
    "${ABOLETH_EFA_DP_DIR}/device"
    "${ABOLETH_EFA_DP_DIR}/host")
endif()

find_package(Threads REQUIRED)
# cudart_static, not the shared one: this library is copied into the Python
# package, and a DT_NEEDED on libcudart.so.N would drag the build host's CUDA
# rpath along with it. It also matches what nvcc links by default.
target_link_libraries(aboleth_rdma_abi PRIVATE
  CUDA::cudart_static
  Threads::Threads)

# Transport headers are consumed for their types and static inlines only; the
# entry points are resolved at runtime by dl/, so none of these libraries may
# become a DT_NEEDED (see the lint.no-dt-needed test).
target_link_libraries(aboleth_rdma_abi PRIVATE rdma_headers)
# libfabric appears in this translation unit only inside the EFA-OFI proxy's
# #if ABOLETH_HAS_EFA regions; dl/ carries it for the shims themselves.
if(ABOLETH_HAS_EFA)
  target_link_libraries(aboleth_rdma_abi PRIVATE libfabric_headers)
endif()

add_subdirectory(dl)
target_link_libraries(aboleth_rdma_abi PRIVATE aboleth::dl)

# EFA device primitives are linked into generated Triton kernels as bitcode
# rather than compiled into this library.
if(ABOLETH_HAS_EFA)
  foreach(arch IN LISTS ABOLETH_EFA_BITCODE_ARCHS)
    aboleth_add_device_bitcode(aboleth_efa_${arch}
      SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/efa/aboleth_efa_device.cu"
      ARCH ${arch}
      INCLUDES
        "${PROJECT_SOURCE_DIR}/include"
        "${ABOLETH_EFA_DP_DIR}/device"
        "${ABOLETH_EFA_DP_DIR}/common"
      DEPENDS "${PROJECT_SOURCE_DIR}/include/aboleth/Target/EfaAbi.h")
  endforeach()
endif()

# Shared mlx5 device primitives, consumed by the RDMA lowering's optional
# bitcode-path mode and by external clients that link the device library. Tier 1
# clients take either form: the bitcode (Triton extern_libs, clang
# -mlink-builtin-bitcode) or the archive (nvcc -rdc=true).
foreach(arch IN LISTS ABOLETH_RDMA_BITCODE_ARCHS)
  aboleth_add_device_bitcode(aboleth_rdma_${arch}
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rdma/aboleth_rdma_device.cu"
    ARCH ${arch}
    INCLUDES "${PROJECT_SOURCE_DIR}/include"
    DEPENDS
      "${PROJECT_SOURCE_DIR}/include/aboleth/Target/RdmaDevice.cuh"
      "${PROJECT_SOURCE_DIR}/include/aboleth/Target/RdmaAbi.h")

  aboleth_add_device_archive(aboleth_device_${arch}
    BITCODE "${ABOLETH_LIB_DIR}/aboleth_rdma_${arch}.bc"
    ARCH ${arch}
    DEPENDS aboleth_rdma_${arch})
endforeach()
