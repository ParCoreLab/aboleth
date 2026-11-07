#pragma once

// The target lowerings are the only part of the compiler that depends on
// Triton: their generated getDependentDialects() names TritonDialect, so
// including this header pulls libtriton into the link closure. Passes that must
// stay Triton-free live in aboleth/Transforms/Passes.h instead.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "aboleth/IR/Dialect.h"

namespace mlir::aboleth {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "aboleth/Target/Passes.h.inc"

} // namespace mlir::aboleth
