#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "aboleth/IR/Dialect.h.inc"

namespace mlir::aboleth {

struct EndpointState
    : public SideEffects::Resource::Base<EndpointState> {
  StringRef getName() const final { return "<AbolethEndpointState>"; }
};

bool areEquivalentEndpoints(Value first, Value second);

} // namespace mlir::aboleth

#define GET_TYPEDEF_CLASSES
#include "aboleth/IR/Types.h.inc"

#define GET_OP_CLASSES
#include "aboleth/IR/Ops.h.inc"
