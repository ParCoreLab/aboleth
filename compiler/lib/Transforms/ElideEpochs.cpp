#include "aboleth/Transforms/Passes.h"

#include "aboleth/Target/TransportCaps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHELIDEEPOCHSPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct ElideEpochs
    : public impl::AbolethElideEpochsPassBase<ElideEpochs> {
  using Base::Base;
  void runOnOperation() override;
};

PollOp plannedPoll(SubmitOp submission) {
  for (Operation *user : submission.getToken().getUsers())
    if (auto poll = dyn_cast<PollOp>(user))
      return poll.getPredicate() ? poll : PollOp{};
  return {};
}

// The counted loop whose body holds the submission directly, or null. Loops
// are matched by name, like the rest of this library.
Operation *countedLoop(SubmitOp submission) {
  Operation *loop = submission->getParentOp();
  if (!loop || loop->getName().getStringRef() != "scf.for" ||
      loop->getNumOperands() < 3 ||
      submission->getBlock()->getNumArguments() < 1)
    return nullptr;
  return loop;
}

// The predicate "this is the loop's last iteration".
Value lastIteration(Operation *loop, SubmitOp submission) {
  Value iv = submission->getBlock()->getArgument(0);
  OpBuilder builder(submission);
  Value next = arith::AddIOp::create(builder, submission.getLoc(), iv,
                                     loop->getOperand(2));
  return arith::CmpIOp::create(builder, submission.getLoc(),
                               arith::CmpIPredicate::sge, next,
                               loop->getOperand(1));
}

// Rebuilds the loop with one more carried value starting at `init` and returns
// its block argument. The caller appends the value the body yields for it.
BlockArgument addCarried(Operation *&loop, Value init) {
  OperationState state(loop->getLoc(), loop->getName());
  state.addOperands(loop->getOperands());
  state.addOperands(init);
  state.addAttributes(loop->getAttrs());
  state.propertiesAttr = loop->getPropertiesAsAttribute();
  state.addTypes(loop->getResultTypes());
  state.addTypes(init.getType());
  state.addRegion()->takeBody(loop->getRegion(0));
  OpBuilder builder(loop);
  Operation *rebuilt = builder.create(state);
  loop->replaceAllUsesWith(
      rebuilt->getResults().take_front(loop->getNumResults()));
  loop->erase();
  loop = rebuilt;
  return rebuilt->getRegion(0).front().addArgument(init.getType(),
                                                   init.getLoc());
}

// A checked signal replaces only the value it expects. Without elision that is
// epoch - 1; with it, the epoch last published, which the loop carries: the
// cell holds epoch - 1 when the loop starts, and each iteration hands on the
// epoch it published or the value it found.
void carryExpected(Operation *&loop, PostSignalOp signal, Value predicate) {
  OpBuilder outside(loop);
  Value init = arith::ConstantOp::create(outside, signal.getLoc(),
                                         outside.getI64IntegerAttr(0));
  BlockArgument carried = addCarried(loop, init);
  Block *body = carried.getOwner();
  OpBuilder builder(signal);
  Location loc = signal.getLoc();
  Value epoch = signal.getSource();
  Value first = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                      body->getArgument(0),
                                      loop->getOperand(0));
  Value previous = arith::SubIOp::create(
      builder, loc, epoch,
      arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(1)));
  Value expected =
      arith::SelectOp::create(builder, loc, first, previous, carried);
  Value next = arith::SelectOp::create(builder, loc, predicate, epoch, expected);
  signal.getExpectedMutable().assign(expected);
  Operation *yield = body->getTerminator();
  yield->insertOperands(yield->getNumOperands(), next);
}

} // namespace

// A signal's epoch tells the receiver that its payload, and everything before
// it, has landed; a later epoch in the same cell tells it the same about more.
// In a loop that completes only every k-th iteration, publishing only on those
// iterations, and on the loop's last, delivers every payload no later than
// completion already does and drops the other iterations' ordering points.
// Receivers that wait for "at least epoch e" see the same epochs, only later;
// a checked publication compares against the epoch last published instead;
// a kernel that itself waits on a peer is left alone, since the peer's reply
// could depend on an epoch this kernel no longer publishes in time.
void ElideEpochs::runOnOperation() {
  const std::string name = transport;
  const auto *caps =
      ::aboleth::abi::lookup_transport_caps(name.data(), name.size());
  if (!caps || !caps->gpu_initiated || !caps->ordered_delivery ||
      caps->load_store_addressable)
    return;
  for (Operation &function : getOperation().getBody()->getOperations()) {
    if (function.walk([](WaitSignalOp) { return WalkResult::interrupt(); })
            .wasInterrupted())
      continue;
    SmallVector<SubmitOp> submissions;
    function.walk([&](SubmitOp submission) {
      submissions.push_back(submission);
    });
    for (SubmitOp submission : submissions) {
      PollOp poll = plannedPoll(submission);
      Operation *loop = countedLoop(submission);
      // The signal is published where the submission stands, so the poll's
      // predicate must already be known there.
      if (!poll || !loop || submission.getGroup() ||
          !DominanceInfo(&function).properlyDominates(poll.getPredicate(),
                                                      submission))
        continue;
      // Publication planning may be disabled. If the batch's first strong
      // signal still needs the default checked mechanism on a transport
      // without atomics, leave the entire batch unpredicated. Predicating
      // only its ordered followers breaks their common publication chain.
      if (!caps->native_atomics &&
          llvm::any_of(submission.getTransfers(), [](Value transfer) {
            auto signal = transfer.getDefiningOp<PostSignalOp>();
            return signal && signal.isChecked();
          }))
        continue;
      // Signals chained after a strong signal (publication ordering) share
      // its predicate, so a chain is published or skipped as a whole and each
      // published follower still finds its ordering point published.
      Value predicate;
      for (Value transfer : submission.getTransfers()) {
        auto signal = transfer.getDefiningOp<PostSignalOp>();
        if (!signal || signal.getPredicate() || signal.getExpected())
          continue;
        if (!predicate) {
          OpBuilder builder(submission);
          predicate = arith::OrIOp::create(builder, submission.getLoc(),
                                           poll.getPredicate(),
                                           lastIteration(loop, submission));
        }
        // Commands are lowered at their submission; constructing this one
        // there instead lets it use the predicate built in front of it.
        signal->moveBefore(submission);
        signal.getPredicateMutable().assign(predicate);
        // Whether it is checked depends on the predicate: a follower sharing
        // its predecessor's is ordered by it. A checked signal reading its
        // epoch from a cell has no value to carry and publishes every epoch.
        if (!signal.isChecked())
          continue;
        if (signal.getSource().getType().isInteger(64))
          carryExpected(loop, signal, predicate);
        else
          signal.getPredicateMutable().clear();
      }
    }
  }
}

} // namespace mlir::aboleth
