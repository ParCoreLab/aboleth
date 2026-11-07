#pragma once

#include "aboleth/IR/Dialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aboleth {

struct MaterializedRegion {
  Value pointer;
  Value key;
};

inline FailureOr<Value> materializeEndpoint(Value endpoint,
                                            Operation *consumer) {
  auto binding = endpoint.getDefiningOp<BindEndpointOp>();
  if (!binding) {
    consumer->emitError(
        "endpoint must be produced by aboleth.bind_endpoint");
    return failure();
  }
  return binding.getValue();
}

inline FailureOr<MaterializedRegion>
materializeLocalRegion(Value region, Operation *consumer) {
  auto binding = region.getDefiningOp<BindLocalRegionOp>();
  if (!binding) {
    consumer->emitError(
        "local region must be produced by aboleth.bind_local_region");
    return failure();
  }
  return MaterializedRegion{binding.getPointer(), binding.getKey()};
}

inline FailureOr<MaterializedRegion>
materializeRemoteRegion(Value region, Operation *consumer) {
  auto binding = region.getDefiningOp<BindRemoteRegionOp>();
  if (!binding) {
    consumer->emitError(
        "remote region must be produced by aboleth.bind_remote_region");
    return failure();
  }
  return MaterializedRegion{binding.getPointer(), binding.getKey()};
}

// The epoch a signal publishes: its value operand, or the i64 its local
// region holds.
inline FailureOr<Value> materializeSignalValue(OpBuilder &builder, Value source,
                                               Operation *consumer) {
  if (source.getType().isInteger(64))
    return source;
  auto region = materializeLocalRegion(source, consumer);
  if (failed(region))
    return failure();
  auto type = dyn_cast<triton::PointerType>(region->pointer.getType());
  if (!type || !type.getPointeeType().isInteger(64)) {
    consumer->emitError("a signal's source region must hold an i64 epoch");
    return failure();
  }
  return triton::LoadOp::create(builder, consumer->getLoc(), region->pointer,
                                triton::CacheModifier::NONE,
                                triton::EvictionPolicy::NORMAL,
                                /*isVolatile=*/false)
      .getResult();
}

inline LogicalResult eraseTritonBindings(ModuleOp module) {
  llvm::SmallVector<Operation *> bindings;
  module.walk([&](Operation *operation) {
    if (isa<BindEndpointOp, BindLocalRegionOp, BindRemoteRegionOp>(
            operation))
      bindings.push_back(operation);
  });
  for (Operation *binding : llvm::reverse(bindings)) {
    if (!binding->use_empty())
      return binding->emitError(
          "frontend binding still has uses after Triton lowering");
    binding->erase();
  }
  return success();
}

} // namespace mlir::aboleth
