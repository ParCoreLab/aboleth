#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "aboleth/IR/Dialect.h"

namespace mlir::aboleth {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "aboleth/Transforms/Passes.h.inc"

// Whether a submission may move past `operation`: true for anything that
// cannot block, synchronize with other threads, or wait for completion.
bool isCrossableBySubmission(Operation *operation);

} // namespace mlir::aboleth
