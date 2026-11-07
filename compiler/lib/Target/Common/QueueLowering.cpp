#include "aboleth/Target/QueueLowering.h"

#include "aboleth/Target/TritonBridge.h"

#include "aboleth/Target/TritonEmit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>

namespace mlir::aboleth {

LogicalResult forEachTransfer(
    SubmitOp submission,
    llvm::function_ref<LogicalResult(PostPutOp, std::size_t)> onPut,
    llvm::function_ref<LogicalResult(PostSignalOp, std::size_t)> onSignal,
    llvm::function_ref<LogicalResult(PostNopOp, std::size_t)> onNop,
    llvm::function_ref<LogicalResult(PostInlineOp, std::size_t)> onInline) {
  for (auto [index, transfer] :
       llvm::enumerate(submission.getTransfers())) {
    if (auto put = transfer.getDefiningOp<PostPutOp>()) {
      if (failed(onPut(put, index)))
        return failure();
      continue;
    }
    if (auto inlinePut = transfer.getDefiningOp<PostInlineOp>()) {
      if (!onInline)
        return inlinePut.emitError(
            "backend does not support an inline-payload command");
      if (failed(onInline(inlinePut, index)))
        return failure();
      continue;
    }
    if (auto nop = transfer.getDefiningOp<PostNopOp>()) {
      if (!onNop)
        return submission.emitError(
            "does not support a completion no-op command");
      if (failed(onNop(nop, index)))
        return failure();
      continue;
    }
    auto signal = transfer.getDefiningOp<PostSignalOp>();
    if (!signal)
      return submission.emitError(
          "contains an unsupported transfer token");
    if (failed(onSignal(signal, index)))
      return failure();
  }
  return success();
}

namespace {

// Replace each table-selected endpoint with the descriptor pointer it names,
// so every backend sees the binding it already lowers. A selection the issuer
// analysis proved exclusive also checks the launch domain of that proof.
// Checking only unused axes is insufficient for modulo rotations and a
// linearized peer/lane index: a wrong peer extent can alias endpoints.
void materializeEndpointSelections(ModuleOp module, std::uint64_t kindMask) {
  SmallVector<EndpointOp> selections;
  module.walk([&](EndpointOp op) { selections.push_back(op); });
  for (EndpointOp op : selections) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    SmallVector<int32_t, 2> axes;
    if (auto axis = op.getExclusiveAxis())
      axes.push_back(*axis);
    else if (auto attr = op.getExclusiveAxesAttr())
      llvm::append_range(axes, attr.asArrayRef());
    if (!axes.empty()) {
      std::array<bool, 3> selected = {false, false, false};
      std::array<Value, 3> extents;
      for (int axis : axes)
        selected[axis] = true;
      auto extent = [&](int axis) -> Value {
        if (!extents[axis])
          extents[axis] = triton::GetNumProgramsOp::create(builder, loc, axis);
        return extents[axis];
      };
      Value single;
      auto require = [&](Value condition) {
        single = single ? arith::AndIOp::create(builder, loc, single, condition)
                        : condition;
      };
      auto grid = op.getExclusiveGridShapeAttr();
      for (int axis = 0; axis < 3; ++axis) {
        int64_t expected = selected[axis] ? (grid ? grid[axis] : 0) : 1;
        if (!expected)
          continue;
        require(arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, extent(axis),
            emit::i32Constant(builder, loc, expected)));
      }
      if (axes.size() == 2) {
        Value first = arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                             extent(axes[0]));
        Value second = arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                              extent(axes[1]));
        Value programs = arith::MulIOp::create(builder, loc, first, second);
        // The index is i32 and pointer arithmetic sign-extends it. All
        // linearized indices must remain nonnegative and representable.
        require(arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::ule, programs,
            emit::i64Constant(builder, loc, uint64_t{1} << 31)));
      }
      triton::AssertOp::create(
          builder, loc, single,
          axes.size() == 1 && !grid
              ? "aboleth: an exclusive endpoint index needs a one-dimensional grid"
              : "aboleth: an exclusive endpoint index violates its proven launch grid");
    }
    Value slot = triton::AddPtrOp::create(builder, loc, op.getTable().getType(),
                                          op.getTable(), op.getIndex());
    Value handle = emit::loadValue(builder, loc, slot);
    Value address = arith::AndIOp::create(
        builder, loc, handle, emit::i64Constant(builder, loc, ~kindMask));
    Value descriptor = triton::IntToPtrOp::create(
        builder, loc,
        triton::PointerType::get(builder.getI64Type(), /*addressSpace=*/1),
        address);
    op.replaceAllUsesWith(
        BindEndpointOp::create(builder, loc, op.getType(), descriptor)
            .getEndpoint());
    op.erase();
  }
}

} // namespace

LogicalResult runQueueLowering(ModuleOp module,
                               const QueueLoweringHooks &hooks) {
  materializeEndpointSelections(module, hooks.endpointKindMask);
  // Both payload forms chain through the same `after` tokens, so they retire as
  // one sequence in reverse program order rather than as two lists.
  SmallVector<Operation *> payloads;
  SmallVector<PostSignalOp> signals;
  SmallVector<PostNopOp> nops;
  SmallVector<SubmitOp> submissions;
  SmallVector<PollOp> polls;
  SmallVector<BeginGroupOp> groups;
  SmallVector<WaitGroupOp> groupWaits;
  SmallVector<DrainOp> drains;
  SmallVector<SystemFenceOp> fences;
  SmallVector<PublishHeadOp> publications;
  SmallVector<WaitSignalOp> signalWaits;
  module.walk([&](Operation *op) {
    if (isa<PostPutOp, PostInlineOp>(op))
      payloads.push_back(op);
  });
  module.walk([&](PostSignalOp op) { signals.push_back(op); });
  module.walk([&](PostNopOp op) { nops.push_back(op); });
  module.walk([&](SubmitOp op) { submissions.push_back(op); });
  module.walk([&](PollOp op) { polls.push_back(op); });
  module.walk([&](BeginGroupOp op) { groups.push_back(op); });
  module.walk([&](WaitGroupOp op) { groupWaits.push_back(op); });
  module.walk([&](DrainOp op) { drains.push_back(op); });
  module.walk([&](SystemFenceOp op) { fences.push_back(op); });
  module.walk([&](PublishHeadOp op) { publications.push_back(op); });
  module.walk([&](WaitSignalOp op) { signalWaits.push_back(op); });

  if (!hooks.supportsCompletedPrefix)
    for (PostSignalOp signal : signals)
      if (signal.getCompletedPrefix())
        return signal.emitError(hooks.backendName)
               << " cannot realize completed-prefix publication";

  llvm::DenseMap<Operation *, PollOp> submissionPolls;
  for (PollOp poll : polls) {
    if (auto submission = poll.getSubmission().getDefiningOp<SubmitOp>())
      submissionPolls[submission] = poll;
  }

  for (SubmitOp submission : submissions) {
    // Payload other threads of the block produced is complete before the
    // thread that builds and announces the submission reads it.
    if (submission.getProducerBarrier()) {
      OpBuilder builder(submission);
      triton::gpu::BarrierOp::create(builder, submission.getLoc(),
                                     triton::gpu::AddrSpace::All);
    }
    PollOp poll;
    if (auto found = submissionPolls.find(submission);
        found != submissionPolls.end())
      poll = found->second;
    if (failed(hooks.lowerSubmission(submission, poll)))
      return failure();
  }

  if (hooks.lowerPoll) {
    for (PollOp poll : polls) {
      if (!poll.getSubmission().getDefiningOp<SubmitOp>())
        return poll.emitError("poll has no matching submission");
      if (failed(hooks.lowerPoll(poll)))
        return failure();
    }
  }

  if (hooks.lowerBeginGroup) {
    for (BeginGroupOp group : groups) {
      if (failed(hooks.lowerBeginGroup(group)))
        return failure();
    }
  }

  if (hooks.lowerWaitGroup) {
    for (WaitGroupOp wait : groupWaits) {
      if (failed(hooks.lowerWaitGroup(wait)))
        return failure();
    }
  }

  if (!drains.empty() && !hooks.lowerDrain)
    return drains.front().emitError(hooks.backendName)
           << " cannot drain an endpoint";
  if (hooks.lowerDrain) {
    for (DrainOp drain : drains) {
      if (failed(hooks.lowerDrain(drain)))
        return failure();
    }
  }

  if (!fences.empty() && !hooks.lowerSystemFence)
    return fences.front().emitError(hooks.backendName)
           << " cannot realize a phase system fence";
  if (hooks.lowerSystemFence) {
    for (SystemFenceOp fence : fences) {
      if (failed(hooks.lowerSystemFence(fence)))
        return failure();
    }
  }

  if (!publications.empty() && !hooks.lowerPublishHead)
    return publications.front().emitError(hooks.backendName)
           << " cannot publish an externally constructed mlx5 queue head";
  if (hooks.lowerPublishHead) {
    for (PublishHeadOp publication : publications) {
      if (failed(hooks.lowerPublishHead(publication)))
        return failure();
    }
  }

  if (hooks.lowerWaitSignal) {
    for (WaitSignalOp wait : signalWaits) {
      if (failed(hooks.lowerWaitSignal(wait)))
        return failure();
    }
  }

  // Retire in dependency order: consumers before the tokens they consume, and
  // posted puts in reverse so a chained `after` operand dies before its
  // producer.
  for (PollOp poll : polls)
    poll.erase();
  for (WaitGroupOp wait : groupWaits)
    wait.erase();
  for (DrainOp drain : drains)
    drain.erase();
  for (SystemFenceOp fence : fences)
    fence.erase();
  for (PublishHeadOp publication : publications)
    publication.erase();
  for (SubmitOp submission : submissions) {
    if (hooks.requireUnusedTokens && !submission.getToken().use_empty())
      return submission.emitError(hooks.backendName)
             << " submission token has unsupported uses";
    submission.erase();
  }
  for (PostNopOp nop : nops) {
    if (hooks.requireUnusedTokens && !nop.getToken().use_empty())
      return nop.emitError(hooks.backendName)
             << " completion no-op token has unsupported uses";
    nop.erase();
  }
  // A signal may follow another signal, so they retire in reverse too.
  for (PostSignalOp signal : llvm::reverse(signals)) {
    if (hooks.requireUnusedTokens && !signal.getToken().use_empty())
      return signal.emitError(hooks.backendName)
             << " signal token has unsupported uses";
    signal.erase();
  }
  for (Operation *payload : llvm::reverse(payloads)) {
    if (hooks.requireUnusedTokens && !payload->getResult(0).use_empty())
      return payload->emitError(hooks.backendName)
             << " put token has unsupported uses";
    payload->erase();
  }
  for (WaitSignalOp wait : signalWaits)
    wait.erase();
  for (BeginGroupOp group : groups) {
    if (!group.getGroup().use_empty())
      return group.emitError(hooks.backendName)
             << " completion group has unsupported uses";
    group.erase();
  }
  if (failed(eraseTritonBindings(module)))
    return failure();

  bool hasAbolethOps = false;
  module.walk([&](Operation *op) {
    if (op->getDialect() &&
        op->getDialect()->getNamespace() ==
            AbolethDialect::getDialectNamespace())
      hasAbolethOps = true;
  });
  if (hasAbolethOps)
    return module.emitError(hooks.backendName)
           << " lowering left unsupported Aboleth operations";
  return success();
}

} // namespace mlir::aboleth
