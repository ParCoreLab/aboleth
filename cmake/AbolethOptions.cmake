include_guard(GLOBAL)

# All user-facing build knobs live here so `cmake -LH` and the configure summary
# have a single source of truth.
#
# Component options are tri-state:
#   AUTO  enable when the dependencies are found, disable quietly otherwise
#   ON    enable, and fail configuration when a dependency is missing
#   OFF   never enable, and skip the dependency probes entirely

set(ABOLETH_TRISTATE_VALUES AUTO ON OFF)

# aboleth_tristate_option(<NAME> <DOCSTRING> <DEFAULT>)
function(aboleth_tristate_option NAME DOCSTRING DEFAULT)
  set(${NAME} "${DEFAULT}" CACHE STRING "${DOCSTRING} (AUTO/ON/OFF)")
  set_property(CACHE ${NAME} PROPERTY STRINGS ${ABOLETH_TRISTATE_VALUES})

  # Accept the usual CMake boolean spellings so -DABOLETH_ENABLE_EFA=0 works.
  string(TOUPPER "${${NAME}}" _value)
  if(_value MATCHES "^(ON|YES|TRUE|1)$")
    set(_value ON)
  elseif(_value MATCHES "^(OFF|NO|FALSE|0|IGNORE|NOTFOUND)$")
    set(_value OFF)
  elseif(NOT _value STREQUAL "AUTO")
    message(FATAL_ERROR "${NAME} must be one of AUTO/ON/OFF (got '${${NAME}}')")
  endif()
  set(${NAME} "${_value}" CACHE STRING "${DOCSTRING} (AUTO/ON/OFF)" FORCE)
endfunction()

# aboleth_resolve_tristate(<OPTION_NAME> <RESULT_VAR> <AVAILABLE> <REASON>)
#
# Turns a tri-state option plus a probe result into a plain boolean in the
# caller's scope, and turns `ON` + missing dependency into a hard error naming
# what was missing.
macro(aboleth_resolve_tristate OPTION_NAME RESULT_VAR AVAILABLE REASON)
  if(NOT ${OPTION_NAME})
    set(${RESULT_VAR} OFF)
  elseif(${OPTION_NAME} STREQUAL "OFF")
    set(${RESULT_VAR} OFF)
  elseif(${AVAILABLE})
    set(${RESULT_VAR} ON)
  elseif(${OPTION_NAME} STREQUAL "ON")
    message(FATAL_ERROR "${OPTION_NAME}=ON but ${REASON}")
  else()
    set(${RESULT_VAR} OFF)
    message(STATUS "[Aboleth] ${OPTION_NAME}: auto-disabled (${REASON})")
  endif()
endmacro()

# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

aboleth_tristate_option(ABOLETH_ENABLE_TRITON_PLUGIN
  "Build the Triton codegen adapter and the libAbolethPlugin loadable module" AUTO)
aboleth_tristate_option(ABOLETH_ENABLE_RUNTIME
  "Build libaboleth_rdma_abi, the host-side transport runtime" AUTO)
aboleth_tristate_option(ABOLETH_ENABLE_VERBS
  "Build the mlx5/verbs transport provider" AUTO)
aboleth_tristate_option(ABOLETH_ENABLE_EFA
  "Build the AWS EFA transport provider and its device bitcode" AUTO)
aboleth_tristate_option(ABOLETH_ENABLE_PYTHON
  "Build the _runtime pybind11 extension" AUTO)
aboleth_tristate_option(ABOLETH_ENABLE_EXPERIMENTS
  "Build the CUDA benchmark and probe executables" AUTO)
aboleth_tristate_option(ABOLETH_ENABLE_NVSHMEM_BENCHMARKS
  "Build the NVSHMEM comparison benchmarks" AUTO)
aboleth_tristate_option(ABOLETH_ENABLE_NCCL_GIN_BENCHMARKS
  "Build the NCCL GPU-initiated-networking comparison benchmark" AUTO)

option(ABOLETH_ENABLE_TESTS "Register the CTest suite" ON)
option(ABOLETH_BUILD_TRITON
  "Build Triton from third_party/triton when no install prefix is found" OFF)

# ---------------------------------------------------------------------------
# Discovery hints
# ---------------------------------------------------------------------------

set(Triton_ROOT "" CACHE PATH "Triton install prefix (include/triton + lib/libtriton.so)")
set(RdmaCore_ROOT "" CACHE PATH "rdma-core install prefix (verbs.h, mlx5dv.h, efadv.h)")
set(Libfabric_ROOT "" CACHE PATH "libfabric install prefix carrying the EFA provider")
set(NVSHMEM_ROOT "" CACHE PATH "NVSHMEM install prefix; defaults to the venv's nvidia.nvshmem")
set(NCCL_ROOT "" CACHE PATH "NCCL install prefix; defaults to .tools/venvs/gin")

set(ABOLETH_EFA_BITCODE_ARCHS "sm_90a" CACHE STRING
  "GPU architectures to emit EFA device bitcode for; these name the deployment \
target, not the build host")
set(ABOLETH_RDMA_BITCODE_ARCHS "sm_90a" CACHE STRING
  "GPU architectures to emit shared mlx5 device bitcode for; these name the \
deployment target, not the build host")
set(ABOLETH_GOLDEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.goldens" CACHE PATH
  "Directory holding the compiler golden outputs")
option(ABOLETH_TIER1_GENERATED
  "Generate the compiler-emitted Tier-1 device library at build time \
(needs the Triton plugin and the Python package)" ON)
set(ABOLETH_EFA_PYTHON "" CACHE FILEPATH
  "Python interpreter for the EFA-only test venv; defaults to Python_EXECUTABLE")

# LLVM_SYSPATH is how the Makefile-era build and Triton itself name an LLVM
# install root. Honour it as an MLIR_DIR seed so existing invocations keep working.
set(LLVM_SYSPATH "" CACHE PATH "LLVM/MLIR install root (compatibility alias for MLIR_DIR)")
if(LLVM_SYSPATH AND NOT MLIR_DIR)
  set(MLIR_DIR "${LLVM_SYSPATH}/lib/cmake/mlir" CACHE PATH "" FORCE)
endif()
