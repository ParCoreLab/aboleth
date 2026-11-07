#include "aboleth/Transforms/Passes.h"

#include "aboleth/Target/TransportCaps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHORDERPUBLICATIONSPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct OrderPublications
    : public impl::AbolethOrderPublicationsPassBase<OrderPublications> {
  using Base::Base;
  void runOnOperation() override;
};

void setAfter(Value command, Value after) {
  Operation *op = command.getDefiningOp();
  MutableOperandRange range =
      TypeSwitch<Operation *, MutableOperandRange>(op)
          .Case<PostPutOp, PostInlineOp, PostSignalOp>(
              [](auto post) { return post.getAfterMutable(); });
  if (after)
    range.assign(after);
  else
    range.clear();
}

// Put every payload command of the batch ahead of its signals and chain the
// strong signals one after another. The first then orders all of the batch's
// payload, and each later one executes after it at the responder, so it needs
// no ordering point of its own. Segments are independent chains and signals
// may be delayed, so only the order within each chain had to hold, and it
// still does.
void orderPublications(SubmitOp submission) {
  SmallVector<Value> payloads, strong, weak;
  for (Value transfer : submission.getTransfers()) {
    Operation *command = transfer.getDefiningOp();
    if (isa<PostPutOp, PostInlineOp>(command))
      payloads.push_back(transfer);
    else if (auto signal = dyn_cast<PostSignalOp>(command))
      (signal.getWeakPublication() ? weak : strong).push_back(transfer);
    else
      return; // A completion carrier keeps the batch as it is.
  }
  if (strong.size() < 2)
    return;

  Value previous;
  for (Value command : llvm::concat<Value>(payloads, strong)) {
    setAfter(command, previous);
    previous = command;
  }
  SmallVector<Value> order(payloads);
  llvm::append_range(order, strong);
  llvm::append_range(order, weak);
  // Commands are lowered at their submission, so constructing them right
  // before it changes nothing but keeps every new `after` defined first.
  for (Value command : order)
    command.getDefiningOp()->moveBefore(submission);
  submission.getTransfersMutable().assign(order);
}

} // namespace

void OrderPublications::runOnOperation() {
  const std::string name = transport;
  const auto *caps =
      ::aboleth::abi::lookup_transport_caps(name.data(), name.size());
  if (!caps || !caps->gpu_initiated || !caps->ordered_delivery ||
      caps->load_store_addressable)
    return;
  getOperation().walk([](SubmitOp submission) {
    if (!submission.getGroup())
      orderPublications(submission);
  });
}

} // namespace mlir::aboleth
