# The _runtime extension module.
#
# It is copied next to the pure-Python package together with everything it
# depends on, so `python/aboleth/` is a self-contained, relocatable package: the
# Makefile-era build baked an absolute rpath into the extension instead, which
# broke the moment the tree moved.

pybind11_add_module(_runtime MODULE runtime_bindings.cpp)
aboleth_apply_compile_options(_runtime)

target_link_libraries(_runtime PRIVATE aboleth_rdma_abi)
if(ABOLETH_HAS_HIP)
  aboleth_python_query(_torch_include
    "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('torch').origin).parent / 'include')")
  target_include_directories(_runtime SYSTEM PRIVATE "${_torch_include}")
  target_link_libraries(_runtime PRIVATE hip::host)
endif()
set_target_properties(_runtime PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

set(ABOLETH_PACKAGE_DIR "${PROJECT_SOURCE_DIR}/python/aboleth")

set(_packaged $<TARGET_FILE:_runtime> $<TARGET_FILE:aboleth_rdma_abi>)
if(ABOLETH_HAS_TRITON_PLUGIN)
  list(APPEND _packaged $<TARGET_FILE:AbolethPlugin>)
endif()

# Staged on every build, not as a post-build step of _runtime: the plugin and
# the bitcode change without the extension relinking, and `aboleth._plugin`
# prefers the staged plugin, so a copy staged only when _runtime relinks goes
# stale silently.
if(ABOLETH_HAS_EFA)
  foreach(arch IN LISTS ABOLETH_EFA_BITCODE_ARCHS)
    list(APPEND _packaged "${ABOLETH_LIB_DIR}/aboleth_efa_${arch}.bc")
  endforeach()
endif()
add_custom_target(aboleth_stage_package ALL
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${_packaged} "${ABOLETH_PACKAGE_DIR}/"
  COMMENT "Staging the native artifacts into python/aboleth/"
  VERBATIM)
add_dependencies(aboleth_stage_package _runtime aboleth_rdma_abi)
if(ABOLETH_HAS_EFA)
  foreach(arch IN LISTS ABOLETH_EFA_BITCODE_ARCHS)
    add_dependencies(aboleth_stage_package aboleth_efa_${arch})
  endforeach()
endif()
if(ABOLETH_HAS_TRITON_PLUGIN)
  add_dependencies(aboleth_stage_package AbolethPlugin)
endif()
