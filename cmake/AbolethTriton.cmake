include_guard(GLOBAL)

# Triton acquisition.
#
# Aboleth consumes Triton as an *install prefix*: `include/triton/...` plus
# `lib/libtriton.so`. That is the only shape the plugin needs, it is what
# `cmake --install` on a Triton build tree produces, and it keeps Triton's
# private build layout (build/cmake.<platform>/, nanobind, third_party/nvidia)
# out of Aboleth's include path. Adding Triton via add_subdirectory or
# FetchContent is not an option: its CMakeLists mutates global flags and
# downloads ~2GB of LLVM at configure time.

if(ABOLETH_ENABLE_TRITON_PLUGIN STREQUAL "OFF")
  message(STATUS "[Aboleth] Triton: disabled by user")
  set(ABOLETH_TRITON_FOUND OFF)
  return()
endif()

set(ABOLETH_TRITON_BUILD_PREFIX "${ABOLETH_DEPS_DIR}/triton")

# aboleth_probe_triton_prefix(<prefix>) -> sets Triton_INCLUDE_DIR/Triton_LIBRARY
function(_aboleth_probe_triton_prefix prefix out_include out_library)
  set(${out_include} "" PARENT_SCOPE)
  set(${out_library} "" PARENT_SCOPE)
  if(NOT prefix)
    return()
  endif()
  if(EXISTS "${prefix}/include/triton/Tools/PluginUtils.h"
      AND EXISTS "${prefix}/lib/libtriton.so")
    set(${out_include} "${prefix}/include" PARENT_SCOPE)
    set(${out_library} "${prefix}/lib/libtriton.so" PARENT_SCOPE)
  endif()
endfunction()

set(_triton_prefixes ${Triton_ROOT} $ENV{Triton_ROOT} ${CMAKE_PREFIX_PATH} "${ABOLETH_TRITON_BUILD_PREFIX}")
set(Triton_INCLUDE_DIR "")
set(Triton_LIBRARY "")
foreach(prefix IN LISTS _triton_prefixes)
  _aboleth_probe_triton_prefix("${prefix}" _include _library)
  if(_include)
    set(Triton_INCLUDE_DIR "${_include}")
    set(Triton_LIBRARY "${_library}")
    set(Triton_PREFIX "${prefix}")
    break()
  endif()
endforeach()

set(_triton_external OFF)
if(NOT Triton_LIBRARY AND ABOLETH_BUILD_TRITON)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/triton/CMakeLists.txt")
    message(FATAL_ERROR
      "ABOLETH_BUILD_TRITON=ON but third_party/triton is empty; run `git submodule update --init`")
  endif()
  set(_triton_external ON)
  set(Triton_PREFIX "${ABOLETH_TRITON_BUILD_PREFIX}")
  set(Triton_INCLUDE_DIR "${Triton_PREFIX}/include")
  set(Triton_LIBRARY "${Triton_PREFIX}/lib/libtriton.so")
  file(MAKE_DIRECTORY "${Triton_INCLUDE_DIR}")

  include(ExternalProject)
  ExternalProject_Add(triton_external
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/triton"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
      "${PROJECT_SOURCE_DIR}/scripts/build-triton.sh"
        --source "${PROJECT_SOURCE_DIR}/third_party/triton"
        --build-dir "${ABOLETH_DEPS_DIR}/triton-build"
        --prefix "${Triton_PREFIX}"
        --python "${Python_EXECUTABLE}"
    BUILD_IN_SOURCE OFF
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${Triton_LIBRARY}"
    USES_TERMINAL_BUILD ON
    BUILD_ALWAYS OFF
  )
endif()

aboleth_resolve_tristate(ABOLETH_ENABLE_TRITON_PLUGIN ABOLETH_TRITON_FOUND
  Triton_LIBRARY
  "no Triton install prefix found; pass -DTriton_ROOT=<prefix> or -DABOLETH_BUILD_TRITON=ON")

if(NOT ABOLETH_TRITON_FOUND)
  return()
endif()

# glibc identifies a shared object by inode, not by soname: if the plugin
# resolves a different libtriton.so file than the one Python already loaded as
# triton._C.libtriton, the process ends up with two copies of MLIR and every
# TypeID comparison across the boundary fails. Prefer the interpreter's live
# copy; scripts/build-triton.sh symlinks the install prefix at it so the two
# agree by construction.
if(NOT _triton_external)
  aboleth_python_query(_venv_libtriton
    "import triton, pathlib; print(pathlib.Path(triton.__file__).parent / '_C' / 'libtriton.so')")
  if(_venv_libtriton AND EXISTS "${_venv_libtriton}")
    file(REAL_PATH "${Triton_LIBRARY}" _prefix_real)
    file(REAL_PATH "${_venv_libtriton}" _venv_real)
    if(NOT _prefix_real STREQUAL _venv_real)
      message(WARNING
        "[Aboleth] Triton: ${Triton_LIBRARY} and the interpreter's "
        "${_venv_libtriton} are different files. Loading the plugin into that "
        "interpreter would map MLIR twice. Using the interpreter's copy.")
    endif()
    set(Triton_LIBRARY "${_venv_libtriton}")
  endif()
endif()

add_library(Triton::triton SHARED IMPORTED GLOBAL)
set_target_properties(Triton::triton PROPERTIES
  IMPORTED_LOCATION "${Triton_LIBRARY}"
  IMPORTED_NO_SONAME ON)
target_include_directories(Triton::triton SYSTEM INTERFACE "${Triton_INCLUDE_DIR}")
if(_triton_external)
  add_dependencies(Triton::triton triton_external)
endif()

aboleth_enable_feature(TRITON_PLUGIN)
message(STATUS "[Aboleth] Triton: ${Triton_LIBRARY}")
