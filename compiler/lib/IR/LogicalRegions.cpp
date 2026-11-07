#include "aboleth/IR/LogicalRegions.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::aboleth {

Value getLogicalStorageView(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return {};
  if (isa<RegionOp, ViewOp, IndexedViewOp>(op))
    return value;
  if (isa<TransferOp, ReductionOp, CopyOp, TransformOp>(op))
    return getLogicalStorageView(op->getOperand(1));
  if (auto transfer = dyn_cast<TransferEachOp>(op))
    return getLogicalStorageView(transfer.getDestination());
  if (auto copy = dyn_cast<CopyEachOp>(op))
    return getLogicalStorageView(copy.getDestination());
  if (auto stage = dyn_cast<OpaqueStageOp>(op)) {
    auto result = dyn_cast<OpResult>(value);
    if (result && result.getResultNumber() < stage.getWrites().size())
      return getLogicalStorageView(stage.getWrites()[result.getResultNumber()]);
  }
  return {};
}

std::optional<LogicalRegionMetadata> getLogicalRegionMetadata(Value value) {
  Value declaration = getLogicalStorageView(value);
  Operation *definingOp = declaration ? declaration.getDefiningOp() : nullptr;
  if (!definingOp)
    return std::nullopt;

  auto elementType = definingOp->getAttrOfType<TypeAttr>("elementType");
  auto shape = definingOp->getAttrOfType<DenseI64ArrayAttr>("shape");
  auto strides = definingOp->getAttrOfType<DenseI64ArrayAttr>("strides");
  auto indexMap = definingOp->getAttrOfType<AffineMapAttr>("indexMap");
  auto visibility = definingOp->getAttrOfType<StringAttr>("visibility");
  auto immutable = definingOp->getAttrOfType<BoolAttr>("immutable");
  auto escapes = definingOp->getAttrOfType<BoolAttr>("escapes");
  if (!elementType || !shape || !strides || !indexMap || !visibility ||
      !immutable || !escapes)
    return std::nullopt;
  auto registered = definingOp->getAttrOfType<BoolAttr>("registered");
  auto donated = definingOp->getAttrOfType<BoolAttr>("donated");
  auto reuseInterval =
      definingOp->getAttrOfType<IntegerAttr>("reuseInterval");
  auto poolDepth = definingOp->getAttrOfType<IntegerAttr>("poolDepth");
  auto injective = definingOp->getAttrOfType<BoolAttr>("injective");

  return LogicalRegionMetadata{
      elementType.getValue(),
      llvm::to_vector(shape.asArrayRef()),
      llvm::to_vector(strides.asArrayRef()),
      indexMap.getValue(),
      visibility.getValue(),
      immutable.getValue(),
      escapes.getValue(),
      registered && registered.getValue(),
      donated && donated.getValue(),
      reuseInterval ? reuseInterval.getInt() : 0,
      poolDepth ? poolDepth.getInt() : 0,
      isa<IndexedViewOp>(definingOp),
      injective && injective.getValue(),
  };
}

Value getLogicalStorageRoot(Value value) {
  Value declaration = getLogicalStorageView(value);
  if (!declaration || declaration.getDefiningOp<RegionOp>())
    return declaration;
  return getLogicalStorageRoot(declaration.getDefiningOp()->getOperand(0));
}

} // namespace mlir::aboleth
