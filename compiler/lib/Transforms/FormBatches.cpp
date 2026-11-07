#include "aboleth/Transforms/Passes.h"

#include <algorithm>

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHFORMBATCHESPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct FormBatches
    : public impl::AbolethFormBatchesPassBase<FormBatches> {
  void runOnOperation() override;
};

// The endpoint of a semantic payload command, or null when the operation is not
// one. Region-sourced and inline transfers batch identically, so the chain walk
// asks this rather than naming one op.
Value payloadEndpoint(Operation *operation) {
  if (auto put = dyn_cast_or_null<PutOp>(operation))
    return put.getEndpoint();
  if (auto inlinePut = dyn_cast_or_null<PutInlineOp>(operation))
    return inlinePut.getEndpoint();
  return {};
}

Value payloadAfter(Operation *operation) {
  if (auto put = dyn_cast<PutOp>(operation))
    return put.getAfter();
  return cast<PutInlineOp>(operation).getAfter();
}

// Walk the `after` chain back from `cursor` and return its payload commands
// in program order. Every member must share `endpoint` and `block`.
FailureOr<SmallVector<Operation *>>
collectChain(Value cursor, Value endpoint, Block *block, Operation *anchor) {
  SmallVector<Operation *> puts;
  llvm::DenseSet<Value> seen;
  while (cursor) {
    if (!seen.insert(cursor).second)
      return anchor->emitError("put dependency chain contains a cycle");
    Operation *put = cursor.getDefiningOp();
    Value putEndpoint = payloadEndpoint(put);
    if (!putEndpoint)
      return anchor->emitError(
          "put dependency chain must contain only aboleth.put or "
          "aboleth.put_inline tokens");
    if (put->getBlock() != block ||
        !areEquivalentEndpoints(putEndpoint, endpoint))
      return put->emitError(
          "put dependency chain must stay in one block and endpoint");
    puts.push_back(put);
    cursor = payloadAfter(put);
  }
  std::reverse(puts.begin(), puts.end());
  return puts;
}

// The submission replaces the chain at its tail, so every operation between
// the first command and the tail must be one the submission may cross.
LogicalResult checkCrossings(ArrayRef<Operation *> puts, Operation *tail) {
  if (puts.empty())
    return success();
  for (Operation *operation = puts.front();
       operation && operation != tail; operation = operation->getNextNode()) {
    if (isCrossableBySubmission(operation))
      continue;
    return operation->emitError(
        "operation cannot be crossed by an Aboleth submission batch");
  }
  return success();
}

// Replace a semantic payload command with its Queue IR form.
Value postPayload(Operation *operation) {
  OpBuilder builder(operation);
  Value token = operation->getResult(0);
  Value posted;
  if (auto put = dyn_cast<PutOp>(operation)) {
    posted = PostPutOp::create(
        builder, put.getLoc(), token.getType(), put.getEndpoint(),
        put.getDestination(), put.getSource(), put.getBytes(),
        put.getPredicate(), put.getAfter()).getToken();
  } else {
    auto inlinePut = cast<PutInlineOp>(operation);
    posted = PostInlineOp::create(
        builder, inlinePut.getLoc(), token.getType(),
        inlinePut.getEndpoint(), inlinePut.getDestination(),
        inlinePut.getValues(), inlinePut.getPredicate(),
        inlinePut.getAfter()).getToken();
  }
  token.replaceAllUsesWith(posted);
  return posted;
}

// Form one submission ending at `tail`, which is either a signal or the last
// payload command of a chain a `wait` consumes directly.
LogicalResult formBatch(Operation *tail, WaitOp wait) {
  if (wait && tail->getBlock() != wait->getBlock())
    return wait.emitError(
        "cross-block communication batches are not supported");

  auto signal = dyn_cast<SignalOp>(tail);
  Value endpoint = signal ? signal.getEndpoint() : payloadEndpoint(tail);
  Value cursor = signal ? Value(signal.getAfter()) : tail->getResult(0);
  auto puts = collectChain(cursor, endpoint, tail->getBlock(), tail);
  if (failed(puts) || failed(checkCrossings(*puts, tail)))
    return failure();

  SmallVector<Value> postedTokens;
  postedTokens.reserve(puts->size() + 1);
  for (Operation *put : *puts)
    postedTokens.push_back(postPayload(put));

  Value group;
  if (signal) {
    OpBuilder signalBuilder(signal);
    auto postedSignal = PostSignalOp::create(
        signalBuilder, signal.getLoc(), signal.getToken().getType(),
        signal.getEndpoint(), signal.getDestination(), signal.getSource(),
        signal.getAfter(), signal.getGroup(), /*predicate=*/Value{},
        /*expected=*/Value{},
        signal.getWeakPublicationAttr(), signal.getFencedSignalAttr(),
        signal.getCompletedPrefixAttr());
    signal.getToken().replaceAllUsesWith(postedSignal.getToken());
    postedTokens.push_back(postedSignal.getToken());
    group = signal.getGroup();
  }

  OpBuilder submitBuilder(tail);
  auto submission = SubmitOp::create(
      submitBuilder, tail->getLoc(), postedTokens.back().getType(), endpoint,
      ValueRange(postedTokens), group,
      submitBuilder.getI32IntegerAttr(
          static_cast<int32_t>(postedTokens.size())),
      /*deferCompletion=*/UnitAttr{}, /*headPublication=*/UnitAttr{},
      /*plannedReservation=*/UnitAttr{},
      /*reservationScope=*/StringAttr{}, /*issuerScope=*/StringAttr{},
      /*externalPublication=*/UnitAttr{}, /*producerBarrier=*/UnitAttr{},
      /*exclusiveIssuer=*/UnitAttr{}, /*deferredAnnouncement=*/UnitAttr{});
  if (wait) {
    OpBuilder waitBuilder(wait);
    UnitAttr coalesced =
        wait.getPredicate() ? waitBuilder.getUnitAttr() : UnitAttr{};
    PollOp::create(waitBuilder, wait.getLoc(), submission.getToken(),
                   wait.getPredicate(), coalesced, wait.getSelfReclaimAttr());
    wait.erase();
  }

  if (signal)
    signal.erase();
  for (Operation *put : llvm::reverse(*puts))
    put->erase();
  return success();
}

} // namespace

// The commands are built and announced where the submission stands, so
// ordinary loads and stores before it may be crossed: a store that produces a
// later command's payload still precedes the announcement, and a command's own
// source may not change before its completion in any case. Commands of other
// chains may be crossed as well, since chains are unordered with respect to
// each other. Anything that can block, synchronize with other threads, or wait
// for completion ends the region a submission may span. This library stays
// free of Triton, so its memory operations are recognized by name.
bool isCrossableBySubmission(Operation *operation) {
  if (isMemoryEffectFree(operation) ||
      isa<PutOp, PutInlineOp, SignalOp, PostPutOp, PostInlineOp,
          PostSignalOp>(operation))
    return true;
  StringRef name = operation->getName().getStringRef();
  if (name == "tt.load" || name == "tt.store")
    return true;
  if (name != "scf.if" && name != "scf.for")
    return false;
  return !operation
              ->walk([](Operation *nested) {
                return nested->getNumRegions() == 0 &&
                               !isCrossableBySubmission(nested)
                           ? WalkResult::interrupt()
                           : WalkResult::advance();
              })
              .wasInterrupted();
}

void FormBatches::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<SignalOp> signals;
  module.walk([&](SignalOp signal) { signals.push_back(signal); });
  SmallVector<WaitOp> payloadWaits;
  module.walk([&](WaitOp wait) {
    if (payloadEndpoint(
            wait.getTransfer().getDefiningOp()
        ))
      payloadWaits.push_back(wait);
  });

  for (SignalOp signal : signals) {
    WaitOp wait;
    for (Operation *user : signal.getToken().getUsers()) {
      if (auto candidate = dyn_cast<WaitOp>(user)) {
        wait = candidate;
        break;
      }
    }
    if (signal.getGroup() && wait) {
      signal.emitError(
          "grouped signals must be completed by aboleth.wait_group");
      return signalPassFailure();
    }
    if (!signal.getGroup() && !wait) {
      signal.emitError(
          "ungrouped signal must be consumed by aboleth.wait");
      return signalPassFailure();
    }
    if (failed(formBatch(signal, wait)))
      return signalPassFailure();
  }
  for (WaitOp wait : payloadWaits) {
    Operation *tail =
        wait.getTransfer().getDefiningOp();
    if (failed(formBatch(tail, wait)))
      return signalPassFailure();
  }
}

} // namespace mlir::aboleth
