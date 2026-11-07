cmake_minimum_required(VERSION 3.24)
project(Aboleth VERSION 0.1.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(AbolethOptions)
include(AbolethFeatures)
include(AbolethGPU)
include(AbolethCompilerConfig)
include(AbolethPython)
include(AbolethMLIR)
include(AbolethTriton)
include(deps/RdmaCore)
include(deps/Libfabric)
include(deps/NVSHMEM)
include(deps/NCCLGin)
include(deps/Pybind11)
include(AbolethProviders)
include(AbolethBitcode)

# Every ABOLETH_HAS_* is settled by this point; bake them into the flags target
# the archetypes link.
aboleth_load_feature_flags()

add_subdirectory(compiler)

include(AbolethTier1Library)

if(ABOLETH_BUILD_RUNTIME)
  add_subdirectory(src/runtime)
endif()
if(ABOLETH_HAS_PYTHON_BINDINGS)
  add_subdirectory(src/python)
endif()
if(ABOLETH_BUILD_EXPERIMENTS)
  add_subdirectory(experiments)
endif()

if(ABOLETH_ENABLE_TESTS)
  include(AbolethTesting)
  enable_testing()
  add_subdirectory(test)
endif()

include(AbolethInstall)
include(AbolethSummary)
