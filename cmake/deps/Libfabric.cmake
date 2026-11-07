include_guard(GLOBAL)

# libfabric, needed for the EFA provider's control plane (fi_getinfo/fi_fabric).
# Libfabric_ROOT replaces the AWS efa-installer's hardcoded /opt/amazon/efa: a
# host that builds libfabric itself points at that prefix instead.

set(ABOLETH_LIBFABRIC_HAS_EFA OFF)

# Nothing outside the EFA provider links or includes libfabric, so an explicit
# ABOLETH_ENABLE_EFA=OFF makes the probe pointless.
if(ABOLETH_ENABLE_EFA STREQUAL "OFF")
  message(STATUS "[Aboleth] libfabric: not probed (EFA OFF)")
  return()
endif()

set(_hints ${Libfabric_ROOT} $ENV{Libfabric_ROOT})

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  foreach(hint IN LISTS _hints)
    set(ENV{PKG_CONFIG_PATH} "${hint}/lib/pkgconfig:${hint}/lib64/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  endforeach()
  pkg_check_modules(ABOLETH_LIBFABRIC_PKG QUIET libfabric)
endif()

if(ABOLETH_LIBFABRIC_PKG_FOUND)
  list(GET ABOLETH_LIBFABRIC_PKG_INCLUDE_DIRS 0 ABOLETH_LIBFABRIC_INCLUDE_DIR)
  find_library(ABOLETH_LIBFABRIC_LIBRARY NAMES fabric
    HINTS ${ABOLETH_LIBFABRIC_PKG_LIBRARY_DIRS})
else()
  set(_lib_hints)
  foreach(hint IN LISTS _hints)
    list(APPEND _lib_hints "${hint}/lib" "${hint}/lib64")
  endforeach()
  find_path(ABOLETH_LIBFABRIC_INCLUDE_DIR NAMES rdma/fabric.h
    HINTS ${_hints} PATH_SUFFIXES include)
  find_library(ABOLETH_LIBFABRIC_LIBRARY NAMES fabric HINTS ${_lib_hints})
endif()

# fi_ext_efa.h only ships when libfabric was built with the EFA provider; without
# it the runtime's EFA control plane cannot be compiled at all.
if(ABOLETH_LIBFABRIC_INCLUDE_DIR AND EXISTS "${ABOLETH_LIBFABRIC_INCLUDE_DIR}/rdma/fi_ext_efa.h")
  set(ABOLETH_LIBFABRIC_HAS_EFA ON)
endif()

if(NOT ABOLETH_LIBFABRIC_INCLUDE_DIR OR NOT ABOLETH_LIBFABRIC_LIBRARY)
  message(STATUS "[Aboleth] libfabric: not found (pass -DLibfabric_ROOT=<prefix>)")
  return()
endif()

message(STATUS "[Aboleth] libfabric: ${ABOLETH_LIBFABRIC_LIBRARY} (EFA provider: ${ABOLETH_LIBFABRIC_HAS_EFA})")

add_library(libfabric_headers INTERFACE)
add_library(libfabric::headers ALIAS libfabric_headers)
target_include_directories(libfabric_headers SYSTEM INTERFACE "${ABOLETH_LIBFABRIC_INCLUDE_DIR}")

add_library(libfabric::libfabric SHARED IMPORTED GLOBAL)
set_target_properties(libfabric::libfabric PROPERTIES
  IMPORTED_LOCATION "${ABOLETH_LIBFABRIC_LIBRARY}"
  INTERFACE_LINK_LIBRARIES libfabric_headers)
