include_guard(GLOBAL)

# EFA device primitives are linked into generated Triton kernels as LLVM
# bitcode, not as a CUDA object, so they bypass the CMake CUDA toolchain
# entirely and are compiled with the clang++ from the same LLVM distribution
# that produced libtriton. Module flags emitted by the CUDA compilation conflict
# with Triton's own module during linking and are stripped before `opt` runs.

# clang's CUDA header wrapper unconditionally includes curand_mtgp32_kernel.h.
# CUDA 13 unbundled cuRAND from the toolkit, so on those installs the header has
# to be found wherever cuRAND now lives — usually the nvidia-curand wheel in the
# build interpreter's site-packages.
set(ABOLETH_BITCODE_EXTRA_INCLUDE_DIRS "" CACHE STRING
  "Extra include directories for the device bitcode compilation")
if(ABOLETH_HAS_CUDA AND NOT ABOLETH_BITCODE_EXTRA_INCLUDE_DIRS
    AND NOT EXISTS "${CUDAToolkit_INCLUDE_DIR}/curand_mtgp32_kernel.h")
  aboleth_python_query(_curand_dir "import nvidia.curand; print(nvidia.curand.__path__[0])")
  if(_curand_dir AND EXISTS "${_curand_dir}/include/curand_mtgp32_kernel.h")
    set(ABOLETH_BITCODE_EXTRA_INCLUDE_DIRS "${_curand_dir}/include" CACHE STRING "" FORCE)
    message(STATUS "[Aboleth] bitcode: cuRAND headers from ${_curand_dir}/include")
  endif()
endif()

# aboleth_add_device_bitcode(<target>
#   SOURCE <file> ARCH <sm_xx> [INCLUDES <dir>...] [DEPENDS <file>...])
function(aboleth_add_device_bitcode target)
  cmake_parse_arguments(arg "" "SOURCE;ARCH" "INCLUDES;DEPENDS" ${ARGN})

  if(NOT ABOLETH_CLANGXX OR NOT ABOLETH_LLVM_OPT)
    message(FATAL_ERROR
      "device bitcode needs clang++ and opt; set -DABOLETH_CLANGXX=/-DABOLETH_LLVM_OPT=")
  endif()

  set(_stem "${ABOLETH_LIB_DIR}/${target}")
  set(_includes)
  foreach(dir IN LISTS arg_INCLUDES ABOLETH_BITCODE_EXTRA_INCLUDE_DIRS)
    list(APPEND _includes "-I${dir}")
  endforeach()

  add_custom_command(
    OUTPUT "${_stem}.bc"
    # --no-cuda-version-check: the tarball clang is usually older than the
    # installed CUDA, and only the device intrinsics matter here.
    COMMAND "${ABOLETH_CLANGXX}" -x cuda --cuda-device-only
            --cuda-gpu-arch=${arg_ARCH} --no-cuda-version-check
            --cuda-path=${ABOLETH_CUDA_ROOT}
            -O3 -std=gnu++17 -fno-strict-aliasing
            -S -emit-llvm "${arg_SOURCE}" -o "${_stem}.ll" ${_includes}
    COMMAND "${CMAKE_COMMAND}" -DINPUT=${_stem}.ll -DOUTPUT=${_stem}.clean.ll
            -P "${PROJECT_SOURCE_DIR}/cmake/StripModuleFlags.cmake"
    COMMAND "${ABOLETH_LLVM_OPT}" -o "${_stem}.bc" "${_stem}.clean.ll"
    DEPENDS "${arg_SOURCE}" ${arg_DEPENDS}
    BYPRODUCTS "${_stem}.ll" "${_stem}.clean.ll"
    COMMENT "Emitting EFA device bitcode ${target}.bc (${arg_ARCH})"
    VERBATIM)

  add_custom_target(${target} ALL DEPENDS "${_stem}.bc")
  set_target_properties(${target} PROPERTIES ABOLETH_BITCODE_FILE "${_stem}.bc")
endfunction()

# _aboleth_llc_needs_ptx84(<llc> <sm_xx> <out-var>)
#
# +ptx84 is the floor the mlx5 inline asm needs; llc emits the per-target
# minimum ISA when no -mattr pins one, and that minimum only needs raising to
# the floor on targets whose own minimum falls short of it (pinning sm_100+
# down to 8.4 is exactly the bug this avoids — its minimum is already 8.6).
# The probe result is cached per arch so re-configures don't re-run llc.
function(_aboleth_llc_needs_ptx84 llc arch out_var)
  string(MAKE_C_IDENTIFIER "_ABOLETH_LLC_NEEDS_PTX84_${arch}" _cache_var)
  if(DEFINED CACHE{${_cache_var}})
    set(${out_var} "${${_cache_var}}" PARENT_SCOPE)
    return()
  endif()
  set(_probe_ll "${CMAKE_BINARY_DIR}/CMakeFiles/aboleth-ptx-probe.ll")
  file(WRITE "${_probe_ll}" "target triple = \"nvptx64-nvidia-cuda\"\n")
  execute_process(
    COMMAND "${llc}" -march=nvptx64 -mcpu=${arch} "${_probe_ll}" -o -
    OUTPUT_VARIABLE _ptx
    ERROR_VARIABLE _ptx_error
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR
      "[Aboleth] probing llc's default PTX ISA for ${arch} failed: ${_ptx_error}")
  endif()
  if(NOT _ptx MATCHES "\\.version ([0-9]+)\\.([0-9]+)")
    message(FATAL_ERROR
      "[Aboleth] llc's PTX output for ${arch} has no .version directive")
  endif()
  math(EXPR _ptx_isa "${CMAKE_MATCH_1} * 100 + ${CMAKE_MATCH_2}")
  if(_ptx_isa LESS 804)
    set(_needed TRUE)
  else()
    set(_needed FALSE)
  endif()
  set(${_cache_var} "${_needed}" CACHE INTERNAL "")
  set(${out_var} "${_needed}" PARENT_SCOPE)
endfunction()

# aboleth_add_device_archive(<target> BITCODE <file> ARCH <sm_xx> [DEPENDS ...])
#
# The same device library as a static archive, for clients that separate-compile
# (`nvcc -rdc=true client.cu lib<target>.a`) instead of linking bitcode into
# their own module. nvlink cannot search an archive of bare cubins — the members
# have to be host objects carrying relocatable device code, exactly like
# libcudadevrt.a — so nvcc, not ptxas, assembles the PTX here.
function(aboleth_add_device_archive target)
  cmake_parse_arguments(arg "" "BITCODE;ARCH" "DEPENDS" ${ARGN})

  if(NOT ABOLETH_LLVM_LLC OR NOT CMAKE_CUDA_COMPILER)
    message(STATUS
      "[Aboleth] ${target}: skipped (needs llc and nvcc; set -DABOLETH_LLVM_LLC=)")
    return()
  endif()

  set(_stem "${ABOLETH_LIB_DIR}/${target}")
  set(_archive "${ABOLETH_LIB_DIR}/lib${target}.a")
  string(REPLACE "sm_" "compute_" _virtual "${arg_ARCH}")

  _aboleth_llc_needs_ptx84("${ABOLETH_LLVM_LLC}" "${arg_ARCH}" _needs_ptx84)
  set(_llc_mattr)
  if(_needs_ptx84)
    set(_llc_mattr -mattr=+ptx84)
  endif()

  add_custom_command(
    OUTPUT "${_archive}"
    COMMAND "${ABOLETH_LLVM_LLC}" -march=nvptx64 -mcpu=${arg_ARCH}
            ${_llc_mattr} "${arg_BITCODE}" -o "${_stem}.ptx"
    COMMAND "${CMAKE_CUDA_COMPILER}"
            -gencode=arch=${_virtual},code=${arg_ARCH} -rdc=true
            -dc "${_stem}.ptx" -o "${_stem}.o"
    # ar appends, so a stale archive would keep an older member.
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${_archive}"
    COMMAND "${CMAKE_AR}" qcs "${_archive}" "${_stem}.o"
    DEPENDS "${arg_BITCODE}" ${arg_DEPENDS}
    BYPRODUCTS "${_stem}.ptx" "${_stem}.o"
    COMMENT "Archiving Aboleth device library lib${target}.a (${arg_ARCH})"
    VERBATIM)

  add_custom_target(${target} ALL DEPENDS "${_archive}")
  set_target_properties(${target} PROPERTIES
    ABOLETH_DEVICE_ARCHIVE "${_archive}")
endfunction()
