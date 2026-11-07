#include "aboleth/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHORDERPRODUCERSPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct OrderProducers
    : public impl::AbolethOrderProducersPassBase<OrderProducers> {
  void runOnOperation() override;
};

// The block barrier the frontend or an earlier submission already placed. This
// library stays free of Triton and the GPU dialect, so it is recognized by name.
bool isBarrier(Operation *operation) {
  if (auto submission = dyn_cast<SubmitOp>(operation))
    return static_cast<bool>(submission.getProducerBarrier());
  return operation->getName().getStringRef() == "ttg.barrier";
}

// A write another thread of the block may have performed. A scalar store is
// issued once per block by the thread that also announces submissions, and
// Aboleth's own operations publish their state themselves.
bool writesForOtherThreads(Operation *operation) {
  if (operation->getDialect() &&
      operation->getDialect()->getNamespace() ==
          AbolethDialect::getDialectNamespace())
    return false;
  if (operation->getName().getStringRef() == "tt.store")
    return isa<RankedTensorType>(operation->getOperand(1).getType());
  if (operation->getNumRegions() != 0)
    return operation
        ->walk([](Operation *nested) {
          return nested->getNumRegions() == 0 &&
                         writesForOtherThreads(nested)
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        })
        .wasInterrupted();
  auto effects = dyn_cast<MemoryEffectOpInterface>(operation);
  return !effects || effects.hasEffect<MemoryEffects::Write>();
}

bool readsRegion(SubmitOp submission) {
  return llvm::any_of(submission.getTransfers(), [](Value transfer) {
    Operation *command = transfer.getDefiningOp();
    if (auto signal = dyn_cast_or_null<PostSignalOp>(command))
      return isa<LocalRegionType>(signal.getSource().getType());
    return isa_and_nonnull<PostPutOp>(command);
  });
}

bool producedSinceBarrier(Operation *operation);

// Only a kernel entry orders every earlier write. A private function's body
// follows whatever its callers wrote before the call (the fused MoE dispatch
// packs, then calls the shared send chain), so its entry is as unordered as
// any of its call sites. Triton functions do not recurse.
bool producedBeforeEntry(Operation *function) {
  auto symbol = dyn_cast<SymbolOpInterface>(function);
  if (!symbol || symbol.isPublic())
    return false;
  auto uses = SymbolTable::getSymbolUses(
      symbol, function->getParentOfType<ModuleOp>());
  if (!uses)
    return true;
  return llvm::any_of(*uses, [](const SymbolTable::SymbolUse &use) {
    return producedSinceBarrier(use.getUser());
  });
}

// Whether a write may have happened since the last barrier, looking backwards
// from `operation` through its block and then through every enclosing one. An
// enclosing loop can carry a write from the end of one iteration to a
// submission early in the next, so a loop body that writes anywhere counts.
bool producedSinceBarrier(Operation *operation) {
  for (Operation *cursor = operation; cursor;
       cursor = cursor->getParentOp()) {
    for (Operation *previous = cursor->getPrevNode(); previous;
         previous = previous->getPrevNode()) {
      if (isBarrier(previous))
        return false;
      if (writesForOtherThreads(previous))
        return true;
    }
    Operation *parent = cursor->getParentOp();
    if (!parent || isa<ModuleOp>(parent))
      return false;
    if (isa_and_nonnull<ModuleOp>(parent->getParentOp()))
      return producedBeforeEntry(parent);
    StringRef name = parent->getName().getStringRef();
    if (name != "scf.if" && writesForOtherThreads(parent))
      return true;
  }
  return false;
}

} // namespace

void OrderProducers::runOnOperation() {
  getOperation().walk([](SubmitOp submission) {
    if (submission.getExternalPublication() || !readsRegion(submission) ||
        !producedSinceBarrier(submission))
      return;
    submission.setProducerBarrierAttr(UnitAttr::get(submission.getContext()));
  });
}

} // namespace mlir::aboleth
