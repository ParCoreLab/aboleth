include_guard(GLOBAL)

# MLIR discovery.
#
# Two consumption modes are exposed:
#
#   aboleth::mlir_headers  include paths and definitions only, no libraries
#   aboleth::mlir_libs     the static MLIR/LLVM archives
#
# Everything that ends up in the same link closure as libtriton.so uses
# mlir_headers: libtriton already contains a full copy of MLIR, and linking the
# archives too would give the process two sets of MLIR TypeIDs, so a Type
# created by Triton would not compare equal to the "same" Type created by a
# plugin pass. aboleth::mlir_libs exists for future standalone tools that link
# MLIR and never see libtriton.

# When neither MLIR_DIR nor LLVM_SYSPATH is given, fall back to the LLVM
# distribution Triton downloads for its own build.
if(NOT MLIR_DIR)
  set(_llvm_candidates)
  if(DEFINED ENV{TRITON_HOME})
    file(GLOB _globbed "$ENV{TRITON_HOME}/.triton/llvm/llvm-*")
    list(APPEND _llvm_candidates ${_globbed})
  endif()
  file(GLOB _globbed "${PROJECT_SOURCE_DIR}/.triton/.triton/llvm/llvm-*")
  list(APPEND _llvm_candidates ${_globbed})
  file(GLOB _globbed "$ENV{HOME}/.triton/llvm/llvm-*")
  list(APPEND _llvm_candidates ${_globbed})

  foreach(candidate IN LISTS _llvm_candidates)
    if(EXISTS "${candidate}/lib/cmake/mlir/MLIRConfig.cmake")
      set(MLIR_DIR "${candidate}/lib/cmake/mlir" CACHE PATH "" FORCE)
      message(STATUS "[Aboleth] MLIR: using Triton's LLVM distribution at ${candidate}")
      break()
    endif()
  endforeach()
  unset(_globbed)
  unset(_llvm_candidates)
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG HINTS "${LLVM_DIR}" "${MLIR_DIR}/../llvm")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# The LLVM install root backs the IR-test and bitcode toolchain defaults below.
get_filename_component(ABOLETH_LLVM_ROOT "${LLVM_INSTALL_PREFIX}" ABSOLUTE)
if(NOT ABOLETH_LLVM_ROOT)
  get_filename_component(ABOLETH_LLVM_ROOT "${LLVM_TOOLS_BINARY_DIR}/.." ABSOLUTE)
endif()

add_library(aboleth_mlir_headers INTERFACE)
add_library(aboleth::mlir_headers ALIAS aboleth_mlir_headers)
target_include_directories(aboleth_mlir_headers SYSTEM INTERFACE
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(aboleth_mlir_headers INTERFACE ${LLVM_DEFINITIONS})

add_library(aboleth_mlir_libs INTERFACE)
add_library(aboleth::mlir_libs ALIAS aboleth_mlir_libs)
target_link_libraries(aboleth_mlir_libs INTERFACE
  aboleth_mlir_headers
  MLIRAnalysis
  MLIRArithDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  LLVMSupport
)

message(STATUS "[Aboleth] MLIR: ${MLIR_DIR} (LLVM ${LLVM_PACKAGE_VERSION})")

# Tools taken from the same LLVM distribution unless overridden.
find_program(ABOLETH_FILECHECK NAMES FileCheck
  HINTS "${LLVM_TOOLS_BINARY_DIR}" "${ABOLETH_LLVM_ROOT}/bin")
find_program(ABOLETH_CLANGXX NAMES clang++
  HINTS "${LLVM_TOOLS_BINARY_DIR}" "${ABOLETH_LLVM_ROOT}/bin")
find_program(ABOLETH_LLVM_OPT NAMES opt
  HINTS "${LLVM_TOOLS_BINARY_DIR}" "${ABOLETH_LLVM_ROOT}/bin")
find_program(ABOLETH_LLVM_LLC NAMES llc
  HINTS "${LLVM_TOOLS_BINARY_DIR}" "${ABOLETH_LLVM_ROOT}/bin")
