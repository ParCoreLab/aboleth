# The compiler is two halves. AbolethIR + AbolethScheduling are C++17 + MLIR and
# nothing else: they build on a machine with no GPU, no Triton and no RDMA
# stack. AbolethTritonAdapter and AbolethPlugin are the Triton-facing half.

# LLVM's tablegen() builds its -I list from the directory's INCLUDE_DIRECTORIES
# property, so the .td files can only resolve `include "mlir/..."` if MLIR's
# headers are on the *directory* include path. This is the one place the modern
# target-scoped style does not apply.
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

add_subdirectory(include/aboleth/IR)
add_subdirectory(include/aboleth/Transforms)

set(ABOLETH_COMPILER_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/include"
)
# The tablegen output is machine-written and carries warnings we cannot fix.
set(ABOLETH_COMPILER_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")

aboleth_add_library(AbolethIR STATIC
  lib/IR/Dialect.cpp
  lib/IR/LogicalRegions.cpp
)
add_dependencies(AbolethIR AbolethTableGen)
target_include_directories(AbolethIR PUBLIC ${ABOLETH_COMPILER_INCLUDE_DIRS})
target_include_directories(AbolethIR SYSTEM PUBLIC "${ABOLETH_COMPILER_GENERATED_INCLUDE_DIR}")
target_link_libraries(AbolethIR PUBLIC aboleth::mlir_headers)
aboleth_use_mlir_abi(AbolethIR)

aboleth_add_library(AbolethScheduling STATIC
  lib/Transforms/CoalesceSubmissions.cpp
  lib/Transforms/DeferAnnouncement.cpp
  lib/Transforms/ElideEpochs.cpp
  lib/Transforms/FormBatches.cpp
  lib/Transforms/MaterializeCompletion.cpp
  lib/Transforms/MaterializeTransportPhase.cpp
  lib/Transforms/MaterializeValidExtents.cpp
  lib/Transforms/OrderPublications.cpp
  lib/Transforms/OrderProducers.cpp
  lib/Transforms/PlanPublication.cpp
  lib/Transforms/PlanStorage.cpp
  lib/Transforms/PlanEndpoints.cpp
  lib/Transforms/PlanShapes.cpp
  lib/Transforms/PlanTransport.cpp
  lib/Transforms/SimplifyCommunicationDataflow.cpp
)
add_dependencies(AbolethScheduling AbolethTableGen AbolethTransformsIncGen)
target_link_libraries(AbolethScheduling PUBLIC AbolethIR)
aboleth_use_mlir_abi(AbolethScheduling)

if(NOT ABOLETH_HAS_TRITON_PLUGIN)
  return()
endif()

add_subdirectory(include/aboleth/Target)

# aboleth-forward-inline-payloads is a scheduling transform by name and by
# option, but it is the one transform that reads frontend stores: proving that a
# staging buffer is written entirely by scalar `tt.store` operations needs the
# Triton dialect. It therefore builds here rather than in AbolethScheduling,
# which must keep building on a machine with no Triton.
aboleth_add_library(AbolethTritonAdapter STATIC
  lib/Target/Common/QueueLowering.cpp
  lib/Target/EFA/LowerEFA.cpp
  lib/Target/Ionic/LowerIonic.cpp
  lib/Target/P2P/LowerP2P.cpp
  lib/Target/Proxy/LowerProxy.cpp
  lib/Target/RDMA/LowerRDMA.cpp
  lib/Transforms/AnalyzeIssuers.cpp
  lib/Transforms/ForwardInlinePayloads.cpp
)
add_dependencies(AbolethTritonAdapter
  AbolethTableGen AbolethTargetPassesIncGen AbolethTransformsIncGen)
target_link_libraries(AbolethTritonAdapter PUBLIC AbolethIR Triton::triton)
aboleth_use_mlir_abi(AbolethTritonAdapter)

# MODULE, not SHARED: the plugin is dlopened by Triton, never linked against.
aboleth_add_library(AbolethPlugin MODULE
  Plugin.cpp
)
add_dependencies(AbolethPlugin AbolethTableGen AbolethTransformsIncGen AbolethTargetPassesIncGen)
set_target_properties(AbolethPlugin PROPERTIES PREFIX "lib")
target_link_libraries(AbolethPlugin PRIVATE
  AbolethIR
  AbolethScheduling
  AbolethTritonAdapter
  Triton::triton
)
aboleth_use_mlir_abi(AbolethPlugin)

# No MLIR archive may enter this link closure. libtriton.so already contains a
# complete MLIR; linking the archives as well would give the loaded plugin its
# own MLIR statics, and a Type or Attribute built by a plugin pass would then
# carry a different TypeID than the identical one Triton built. Upstream's
# examples/plugins links `triton` alone for the same reason.
get_target_property(_plugin_libs AbolethPlugin LINK_LIBRARIES)
foreach(lib IN LISTS _plugin_libs)
  if(lib MATCHES "^(MLIR|LLVM)")
    message(FATAL_ERROR
      "AbolethPlugin links ${lib}; MLIR/LLVM archives must resolve from libtriton.so only")
  endif()
endforeach()
