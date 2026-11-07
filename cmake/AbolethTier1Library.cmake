include_guard(GLOBAL)

# The compiler-generated mixed RDMA+P2P Tier-1 device library
# (ABOLETH_TIER1_GENERATED).
#
# Runs `python -m aboleth.device_library` per RDMA bitcode arch: every Tier-1
# op compiled as a one-op transport program with detached inline emission,
# packaged as per-op bitcode + an nvcc-linkable archive + the generated
# aboleth/device.cuh under ${ABOLETH_LIB_DIR}/tier1/<arch>/. Needs the Triton
# plugin (the generator drives the real lowering) and nvcc for the archive;
# no GPU and no connected World.

function(aboleth_add_tier1_library arch)
  if(NOT ABOLETH_HAS_TRITON_PLUGIN)
    message(STATUS
      "[Aboleth] tier1 generated library (${arch}): skipped (needs the "
      "Triton plugin)")
    return()
  endif()

  set(_output "${ABOLETH_LIB_DIR}/tier1/${arch}")
  set(_manifest "${_output}/manifest.json")
  file(GLOB _generator_sources
    "${PROJECT_SOURCE_DIR}/python/aboleth/tier1.py"
    "${PROJECT_SOURCE_DIR}/python/aboleth/device_library.py"
    "${PROJECT_SOURCE_DIR}/python/aboleth/artifact.py"
    "${PROJECT_SOURCE_DIR}/python/aboleth/emit/*.py")

  add_custom_command(
    OUTPUT "${_manifest}"
    COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
            "ABOLETH_BUILD_DIR=${CMAKE_BINARY_DIR}"
            "ABOLETH_PLUGIN_PATH=$<TARGET_FILE:AbolethPlugin>"
            "TRITON_CACHE_DIR=${PROJECT_SOURCE_DIR}/.triton/cache"
            "${Python_EXECUTABLE}" -m aboleth.device_library
            --arch "${arch}" --output-dir "${_output}"
    DEPENDS AbolethPlugin ${_generator_sources}
    COMMENT "Generating Aboleth Tier-1 device library (${arch})"
    VERBATIM)

  add_custom_target(aboleth_tier1_${arch} ALL DEPENDS "${_manifest}")
endfunction()

if(ABOLETH_TIER1_GENERATED AND NOT ABOLETH_HAS_HIP)
  foreach(_arch IN LISTS ABOLETH_RDMA_BITCODE_ARCHS)
    aboleth_add_tier1_library(${_arch})
  endforeach()
endif()
