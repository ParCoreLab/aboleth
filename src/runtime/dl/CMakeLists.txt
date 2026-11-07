# Provider resolution. Kept in its own static library so the symbol tables are
# compiled by the host compiler rather than nvcc, and so the linker sees exactly
# one dlopen dependency for the whole runtime.

set(ABOLETH_DL_SOURCES aboleth_dl.cpp)
if(ABOLETH_HAS_VERBS)
  list(APPEND ABOLETH_DL_SOURCES verbs_shim.cpp)
endif()
if(ABOLETH_HAS_MLX5)
  list(APPEND ABOLETH_DL_SOURCES mlx5dv_shim.cpp)
endif()
if(ABOLETH_HAS_EFA)
  list(APPEND ABOLETH_DL_SOURCES efadv_shim.cpp)
endif()
if(ABOLETH_HAS_LIBFABRIC)
  list(APPEND ABOLETH_DL_SOURCES fabric_shim.cpp)
endif()
if(ABOLETH_HAS_CUDA)
  list(APPEND ABOLETH_DL_SOURCES cuda_shim.cpp)
endif()

aboleth_add_library(aboleth_dl STATIC ${ABOLETH_DL_SOURCES})
add_library(aboleth::dl ALIAS aboleth_dl)

target_include_directories(aboleth_dl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(aboleth_dl PUBLIC rdma_headers)
if(TARGET libfabric_headers)
  target_link_libraries(aboleth_dl PUBLIC libfabric_headers)
endif()
if(ABOLETH_HAS_CUDA)
  # cuda.h only; libcuda itself is what these shims exist to avoid linking.
  target_include_directories(aboleth_dl SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()
target_link_libraries(aboleth_dl PRIVATE ${CMAKE_DL_LIBS})
