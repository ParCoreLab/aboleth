#pragma once

#include "aboleth/IR/Dialect.h"

#include <optional>

namespace mlir::aboleth {

struct LogicalRegionMetadata {
  Type elementType;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  AffineMap indexMap;
  StringRef visibility;
  bool immutable;
  bool escapes;
  bool registered;
  bool donated;
  int64_t reuseInterval;
  int64_t poolDepth;
  bool indexed;
  bool injective;
};

// Follows destination-passing SSA states without confusing the value with a
// different storage allocation. Views retain their own logical metadata.
std::optional<LogicalRegionMetadata> getLogicalRegionMetadata(Value value);

// The nearest declared region/view behind a destination-passing SSA state.
Value getLogicalStorageView(Value value);

// The original declaration behind destination-passing states and views.
Value getLogicalStorageRoot(Value value);

} // namespace mlir::aboleth
