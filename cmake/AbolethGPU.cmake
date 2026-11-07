include_guard(GLOBAL)

# GPU backend selection.
#
# CUDA is deliberately absent from project(LANGUAGES ...): the compiler half of
# Aboleth is pure C++17 + MLIR and must configure on machines with no GPU
# toolchain at all. check_language() probes without failing, and the language is
# only enabled once we know we want it.

set(ABOLETH_GPU_BACKEND "auto" CACHE STRING "GPU backend: auto, CUDA, HIP, or NONE")
set_property(CACHE ABOLETH_GPU_BACKEND PROPERTY STRINGS auto CUDA HIP NONE)

include(CheckLanguage)

if(ABOLETH_GPU_BACKEND STREQUAL "auto")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(ABOLETH_GPU_BACKEND "CUDA")
  else()
    set(ABOLETH_GPU_BACKEND "NONE")
  endif()
endif()

if(NOT ABOLETH_GPU_BACKEND MATCHES "^(CUDA|HIP|NONE)$")
  message(FATAL_ERROR "ABOLETH_GPU_BACKEND must be auto, CUDA, HIP or NONE (got '${ABOLETH_GPU_BACKEND}')")
endif()

if(ABOLETH_GPU_BACKEND STREQUAL "CUDA")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # "native" builds for the GPUs in this machine. The EFA device bitcode is a
    # deployment artifact and picks its architectures separately.
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  aboleth_enable_feature(CUDA)
  # The toolkit root (the directory owning bin/nvcc), which is what clang's
  # --cuda-path expects; CUDAToolkit_TARGET_DIR points one level deeper.
  set(ABOLETH_CUDA_ROOT "${CUDAToolkit_LIBRARY_ROOT}")
  if(NOT ABOLETH_CUDA_ROOT)
    set(ABOLETH_CUDA_ROOT "${CUDAToolkit_TARGET_DIR}")
  endif()
  message(STATUS "[Aboleth] GPU backend: CUDA ${CUDAToolkit_VERSION} (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
elseif(ABOLETH_GPU_BACKEND STREQUAL "HIP")
  # The control plane uses the HIP host API. Device primitives are LLVM
  # bitcode linked by Triton, so they use a separate device-only compilation.
  find_package(hip REQUIRED CONFIG)
  find_program(ABOLETH_HIPCC NAMES hipcc REQUIRED)
  aboleth_enable_feature(HIP)
  message(STATUS "[Aboleth] GPU backend: HIP ${hip_VERSION}")
else()
  message(STATUS "[Aboleth] GPU backend: NONE (compiler-only build)")
endif()
