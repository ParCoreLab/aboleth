add_subdirectory(../dl ${CMAKE_CURRENT_BINARY_DIR}/dl)
find_package(Threads REQUIRED)
aboleth_add_library(aboleth_rdma_abi SHARED runtime.cpp)
set_target_properties(aboleth_rdma_abi PROPERTIES
  CXX_VISIBILITY_PRESET default)
target_include_directories(aboleth_rdma_abi PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(aboleth_rdma_abi PRIVATE
  hip::host aboleth::dl Threads::Threads)

set(ABOLETH_HIP_ARCH "gfx950" CACHE STRING "HIP device primitive architecture")
add_custom_command(OUTPUT "${ABOLETH_LIB_DIR}/aboleth_hip.bc"
  COMMAND "${ABOLETH_HIPCC}" --offload-arch=${ABOLETH_HIP_ARCH}
    --offload-device-only -fgpu-rdc -mcode-object-version=5 -O3 -emit-llvm -c
    "${CMAKE_CURRENT_SOURCE_DIR}/device.cpp"
    -o "${ABOLETH_LIB_DIR}/aboleth_hip.bc"
  DEPENDS device.cpp VERBATIM)
add_custom_target(aboleth_hip_bitcode ALL
  DEPENDS "${ABOLETH_LIB_DIR}/aboleth_hip.bc")
add_custom_command(TARGET aboleth_hip_bitcode POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${ABOLETH_LIB_DIR}/aboleth_hip.bc"
    "${PROJECT_SOURCE_DIR}/python/aboleth/aboleth_hip.bc")

add_custom_command(OUTPUT "${ABOLETH_LIB_DIR}/aboleth_ionic.bc"
  COMMAND "${ABOLETH_HIPCC}" --offload-arch=${ABOLETH_HIP_ARCH}
    --offload-device-only -fgpu-rdc -mcode-object-version=5 -O3 -emit-llvm -c
    -I "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/ionic_device.cpp"
    -o "${ABOLETH_LIB_DIR}/aboleth_ionic.bc"
  DEPENDS ionic_device.cpp "${PROJECT_SOURCE_DIR}/include/aboleth/Target/IonicAbi.h"
  VERBATIM)
add_custom_target(aboleth_ionic_bitcode ALL
  DEPENDS "${ABOLETH_LIB_DIR}/aboleth_ionic.bc")
add_custom_command(TARGET aboleth_ionic_bitcode POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${ABOLETH_LIB_DIR}/aboleth_ionic.bc"
    "${PROJECT_SOURCE_DIR}/python/aboleth/aboleth_ionic.bc")
