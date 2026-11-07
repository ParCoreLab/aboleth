include_guard(GLOBAL)

# rdma-core: verbs, mlx5 direct-verbs, and the EFA direct-verbs extension.
#
# The runtime library only ever needs the *headers*: every entry point it calls
# goes through the dlopen shims in src/runtime/dl, and the hot path
# (ibv_post_send, ibv_poll_cq, mlx5dv_*) is static inlines over the context ops
# vtable, which need no symbol at all. The imported link targets below exist for
# the experiments/ binaries, which are developer tools and link directly.

# Only the plain booleans are seeded here: seeding the find_* result variables
# would shadow the cache entries those calls write.
set(ABOLETH_HAVE_EFADV OFF)

if(ABOLETH_ENABLE_VERBS STREQUAL "OFF" AND ABOLETH_ENABLE_EFA STREQUAL "OFF")
  message(STATUS "[Aboleth] rdma-core: not probed (verbs and EFA both OFF)")
  return()
endif()

set(_hints)
if(RdmaCore_ROOT)
  list(APPEND _hints "${RdmaCore_ROOT}")
endif()
if(DEFINED ENV{RdmaCore_ROOT})
  list(APPEND _hints "$ENV{RdmaCore_ROOT}")
endif()

if(NOT _hints)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(ABOLETH_RDMA_PKG QUIET libibverbs)
  endif()
endif()

if(ABOLETH_RDMA_PKG_FOUND)
  list(GET ABOLETH_RDMA_PKG_INCLUDE_DIRS 0 ABOLETH_RDMA_INCLUDE_DIR)
  set(_lib_hints ${ABOLETH_RDMA_PKG_LIBRARY_DIRS})
else()
  find_path(ABOLETH_RDMA_INCLUDE_DIR
    NAMES infiniband/verbs.h
    HINTS ${_hints}
    PATH_SUFFIXES include)
  set(_lib_hints)
  foreach(hint IN LISTS _hints)
    list(APPEND _lib_hints "${hint}/lib" "${hint}/lib64")
  endforeach()
endif()

if(ABOLETH_RDMA_INCLUDE_DIR)
  find_library(ABOLETH_IBVERBS_LIBRARY NAMES ibverbs HINTS ${_lib_hints})
  find_library(ABOLETH_MLX5_LIBRARY NAMES mlx5 HINTS ${_lib_hints})
  find_library(ABOLETH_EFA_LIBRARY NAMES efa HINTS ${_lib_hints})
  if(EXISTS "${ABOLETH_RDMA_INCLUDE_DIR}/infiniband/efadv.h")
    set(ABOLETH_HAVE_EFADV ON)
  endif()
endif()

if(NOT ABOLETH_RDMA_INCLUDE_DIR)
  message(STATUS "[Aboleth] rdma-core: not found (pass -DRdmaCore_ROOT=<prefix>)")
  return()
endif()

message(STATUS "[Aboleth] rdma-core: ${ABOLETH_RDMA_INCLUDE_DIR}")

# Header-only view, used by the shimmed runtime.
add_library(rdma_headers INTERFACE)
add_library(rdma::headers ALIAS rdma_headers)
target_include_directories(rdma_headers SYSTEM INTERFACE "${ABOLETH_RDMA_INCLUDE_DIR}")

function(_aboleth_rdma_imported name path)
  if(NOT path)
    return()
  endif()
  add_library(rdma::${name} SHARED IMPORTED GLOBAL)
  set_target_properties(rdma::${name} PROPERTIES
    IMPORTED_LOCATION "${path}"
    INTERFACE_LINK_LIBRARIES rdma_headers)
endfunction()

_aboleth_rdma_imported(ibverbs "${ABOLETH_IBVERBS_LIBRARY}")
_aboleth_rdma_imported(mlx5 "${ABOLETH_MLX5_LIBRARY}")
_aboleth_rdma_imported(efa "${ABOLETH_EFA_LIBRARY}")
