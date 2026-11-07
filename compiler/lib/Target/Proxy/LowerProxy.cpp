#include "aboleth/Target/Passes.h"
#include "aboleth/Target/QueueLowering.h"
#include "aboleth/Target/TritonBridge.h"
#include "aboleth/Target/TritonEmit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "aboleth/Target/ProxyAbi.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHLOWERPROXYPASS
#include "aboleth/Target/Passes.h.inc"

namespace {

struct LowerProxy : public impl::AbolethLowerProxyPassBase<LowerProxy> {
  using Base::Base;
  void runOnOperation() override;
};

using emit::acquireLoad64;
using emit::addressOffset;
using emit::endpointWord;
using emit::high32;
using emit::i32Constant;
using emit::i64Constant;
using emit::loadValue;
using emit::low32;
using emit::pointerAddress;
using emit::pointerFromAddress;
using emit::releaseStore64;
using emit::store32;
using emit::store64;
using emit::zeroExtend64;

Value pack32(OpBuilder &builder, Location loc, Value low, Value high) {
  Value low64 = zeroExtend64(builder, loc, low);
  Value high64 = arith::ShLIOp::create(
      builder, loc, zeroExtend64(builder, loc, high),
      i64Constant(builder, loc, 32));
  return arith::OrIOp::create(builder, loc, low64, high64);
}

void waitUntilAtLeast(OpBuilder &builder, Location loc, Value pointer,
                      Value target) {
  auto loop =
      scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
  Block *before = builder.createBlock(&loop.getBefore());
  builder.setInsertionPointToStart(before);
  Value observed = acquireLoad64(builder, loc, pointer);
  Value keepWaiting = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, observed, target);
  scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});
  Block *after = builder.createBlock(&loop.getAfter());
  builder.setInsertionPointToStart(after);
  scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(loop);
}

LogicalResult validateEndpoint(Value endpoint, Operation *operation) {
  auto pointerType = dyn_cast<triton::PointerType>(endpoint.getType());
  if (!pointerType || !pointerType.getPointeeType().isInteger(64))
    return operation->emitError(
        "proxy endpoint must be a pointer to the 40-byte i64 descriptor");
  return success();
}

Value reserveBatch(OpBuilder &builder, Location loc, Value endpoint) {
  Value stateAddress = endpointWord(builder, loc, endpoint, 1);
  Value statePointer = pointerFromAddress(
      builder, loc, stateAddress, builder.getI64Type());
  if (emit::hipDetached(builder))
    return emit::hipCall(builder, loc, builder.getI64Type(),
                         "aboleth_hip_reserve64", {stateAddress});
  return triton::AtomicRMWOp::create(
      builder, loc, builder.getI64Type(), triton::RMWOp::ADD,
      statePointer, i64Constant(builder, loc, 1), /*mask=*/nullptr,
      triton::MemSemantic::RELAXED, triton::MemSyncScope::GPU);
}

void waitForQueueCapacity(OpBuilder &builder, Location loc, Value queueState,
                          Value sequence, Value slotCount) {
  Value completedPointer = pointerFromAddress(
      builder, loc,
      addressOffset(
          builder, loc, queueState,
          offsetof(::aboleth::abi::ProxyQueueState, completed_sequence)),
      builder.getI64Type());
  Value capacity = zeroExtend64(builder, loc, slotCount);

  auto loop =
      scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
  Block *before = builder.createBlock(&loop.getBefore());
  builder.setInsertionPointToStart(before);
  Value completed = acquireLoad64(builder, loc, completedPointer);
  Value availableEnd =
      arith::AddIOp::create(builder, loc, completed, capacity);
  Value keepWaiting = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::uge, sequence, availableEnd);
  scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});
  Block *after = builder.createBlock(&loop.getAfter());
  builder.setInsertionPointToStart(after);
  scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(loop);
}

void writeCommand(OpBuilder &builder, Location loc, Value commandAddress,
                  Value destination, Value source, Value bytes,
                  Value sourceKey, Value destinationKey, Value predicate) {
  Value opcode = arith::SelectOp::create(
      builder, loc, predicate,
      i32Constant(builder, loc, ::aboleth::abi::kProxyCommandWrite),
      i32Constant(builder, loc, ::aboleth::abi::kProxyCommandNop));
  store64(
      builder, loc, commandAddress,
      offsetof(::aboleth::abi::ProxyCommand, source_address),
      pointerAddress(builder, loc, source));
  store64(
      builder, loc, commandAddress,
      offsetof(::aboleth::abi::ProxyCommand, destination_address),
      pointerAddress(builder, loc, destination));
  store64(
      builder, loc, commandAddress,
      offsetof(::aboleth::abi::ProxyCommand, bytes),
      pack32(builder, loc, bytes, sourceKey));
  store64(
      builder, loc, commandAddress,
      offsetof(::aboleth::abi::ProxyCommand, destination_key),
      pack32(builder, loc, destinationKey, opcode));
}

// The command carries the epoch itself where a write carries its source.
void writeSignalCommand(
    OpBuilder &builder, Location loc, Value commandAddress,
    Value destination, Value signalValue, Value destinationKey,
    bool weakPublication) {
  store64(
      builder, loc, commandAddress,
      offsetof(::aboleth::abi::ProxyCommand, source_address),
      signalValue);
  store64(
      builder, loc, commandAddress,
      offsetof(::aboleth::abi::ProxyCommand, destination_address),
      pointerAddress(builder, loc, destination));
  store64(
      builder, loc, commandAddress,
      offsetof(::aboleth::abi::ProxyCommand, bytes),
      pack32(
          builder, loc,
          i32Constant(builder, loc, sizeof(std::uint64_t)),
          i32Constant(builder, loc, 0)));
  store64(
      builder, loc, commandAddress,
      offsetof(::aboleth::abi::ProxyCommand, destination_key),
      pack32(
          builder, loc, destinationKey,
          i32Constant(
              builder, loc,
              weakPublication
                  ? ::aboleth::abi::kProxyCommandWeakSignal
                  : ::aboleth::abi::kProxyCommandStrongSignal)));
}

LogicalResult lowerPoll(PollOp op, Value queueState, Value target) {
  OpBuilder builder(op);
  if (op.getPredicate()) {
    auto conditional = scf::IfOp::create(
        builder, op.getLoc(), TypeRange{},
        op.getPredicate(), /*withElseRegion=*/false);
    builder.setInsertionPoint(
        conditional.thenBlock()->getTerminator());
  }
  Value completedPointer = pointerFromAddress(
      builder, op.getLoc(),
      addressOffset(
          builder, op.getLoc(), queueState,
          offsetof(::aboleth::abi::ProxyQueueState, completed_sequence)),
      builder.getI64Type());
  waitUntilAtLeast(builder, op.getLoc(), completedPointer, target);
  return success();
}

LogicalResult lowerGroupWait(WaitGroupOp op) {
  auto begin = op.getGroup().getDefiningOp<BeginGroupOp>();
  if (!begin)
    return op.emitError("completion group has no matching begin_group");
  auto materializedEndpoint = materializeEndpoint(begin.getEndpoint(), op);
  if (failed(materializedEndpoint))
    return failure();
  Value endpoint = *materializedEndpoint;
  if (failed(validateEndpoint(endpoint, op)))
    return failure();

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value queueState = endpointWord(builder, loc, endpoint, 0);
  Value producerAddress = endpointWord(builder, loc, endpoint, 1);
  Value producerPointer = pointerFromAddress(
      builder, loc, producerAddress, builder.getI64Type());
  Value target = loadValue(builder, loc, producerPointer, /*isVolatile=*/true);
  Value completedPointer = pointerFromAddress(
      builder, loc,
      addressOffset(
          builder, loc, queueState,
          offsetof(::aboleth::abi::ProxyQueueState, completed_sequence)),
      builder.getI64Type());
  waitUntilAtLeast(builder, loc, completedPointer, target);
  return success();
}

LogicalResult lowerDrain(DrainOp op) {
  auto materializedEndpoint = materializeEndpoint(op.getEndpoint(), op);
  if (failed(materializedEndpoint))
    return failure();
  Value endpoint = *materializedEndpoint;
  if (failed(validateEndpoint(endpoint, op)))
    return failure();

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value queueState = endpointWord(builder, loc, endpoint, 0);
  Value producerAddress = endpointWord(builder, loc, endpoint, 1);
  Value producerPointer = pointerFromAddress(
      builder, loc, producerAddress, builder.getI64Type());
  Value target = loadValue(builder, loc, producerPointer, /*isVolatile=*/true);
  Value completedPointer = pointerFromAddress(
      builder, loc,
      addressOffset(
          builder, loc, queueState,
          offsetof(::aboleth::abi::ProxyQueueState, completed_sequence)),
      builder.getI64Type());
  waitUntilAtLeast(builder, loc, completedPointer, target);
  return success();
}

LogicalResult lowerWaitSignal(WaitSignalOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  auto signal = materializeLocalRegion(op.getSignal(), op);
  if (failed(signal))
    return failure();
  auto signalType =
      dyn_cast<triton::PointerType>(signal->pointer.getType());
  if (!signalType || !signalType.getPointeeType().isInteger(64))
    return op.emitError("proxy wait_signal requires an i64 signal");

  auto loop =
      scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
  Block *before = builder.createBlock(&loop.getBefore());
  builder.setInsertionPointToStart(before);
  Value observed = acquireLoad64(builder, loc, signal->pointer);
  Value keepWaiting = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, observed, op.getExpected());
  scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});
  Block *after = builder.createBlock(&loop.getAfter());
  builder.setInsertionPointToStart(after);
  scf::YieldOp::create(builder, loc);
  return success();
}

} // namespace

void LowerProxy::runOnOperation() {
  if (detached) {
    if (hipDeviceLibrary.empty()) {
      getOperation().emitError("detached proxy currently requires HIP helpers");
      return signalPassFailure();
    }
    getOperation()->setAttr("aboleth.hip-detached", UnitAttr::get(&getContext()));
  }
  if (!hipDeviceLibrary.empty())
    getOperation()->setAttr("aboleth.hip-device-library",
                            StringAttr::get(&getContext(), hipDeviceLibrary));
  // Queue state and the sequence this batch will reach, per submission.
  llvm::DenseMap<Operation *, std::pair<Value, Value>> submissionStates;

  auto lowerSubmission = [&](SubmitOp submission, PollOp) -> LogicalResult {
    auto materializedEndpoint =
        materializeEndpoint(submission.getEndpoint(), submission);
    if (failed(materializedEndpoint))
      return failure();
    Value endpoint = *materializedEndpoint;
    if (failed(validateEndpoint(endpoint, submission)))
      return failure();
    if (submission.getBatchSize() >
        ::aboleth::abi::kProxyMaximumCommands)
      return submission.emitError(
          "proxy batch exceeds maximum command count");
    if (submission.getExternalPublication())
      return submission.emitError(
          "proxy cannot delegate publication to an mlx5 phase adapter");

    OpBuilder builder(submission);
    Location loc = submission.getLoc();
    Value queueState = endpointWord(builder, loc, endpoint, 0);
    Value sequence = reserveBatch(builder, loc, endpoint);
    Value layout = endpointWord(builder, loc, endpoint, 3);
    Value slotCount = low32(builder, loc, layout);
    Value slotStride = high32(builder, loc, layout);
    waitForQueueCapacity(
        builder, loc, queueState, sequence, slotCount);

    Value slot = arith::RemUIOp::create(
        builder, loc, sequence,
        zeroExtend64(builder, loc, slotCount));
    Value slotOffset = arith::MulIOp::create(
        builder, loc, slot,
        zeroExtend64(builder, loc, slotStride));
    Value slotAddress = arith::AddIOp::create(
        builder, loc,
        addressOffset(
            builder, loc, queueState,
            offsetof(::aboleth::abi::ProxyQueueState, slots)),
        slotOffset);
    Value commandsAddress = addressOffset(
        builder, loc, slotAddress,
        offsetof(::aboleth::abi::ProxyBatchSlot, commands));

    auto commandAt = [&](std::size_t index) {
      return addressOffset(
          builder, loc, commandsAddress,
          index * sizeof(::aboleth::abi::ProxyCommand));
    };
    auto onPut = [&](PostPutOp put, std::size_t index) -> LogicalResult {
      auto source = materializeLocalRegion(put.getSource(), put);
      auto destination =
          materializeRemoteRegion(put.getDestination(), put);
      if (failed(source) || failed(destination))
        return failure();
      writeCommand(
          builder, put.getLoc(), commandAt(index), destination->pointer,
          source->pointer, put.getBytes(), source->key,
          destination->key, put.getPredicate());
      return success();
    };
    auto onSignal = [&](PostSignalOp signal,
                        std::size_t index) -> LogicalResult {
      if (!hipDeviceLibrary.empty() && signal.getExpected())
        return signal.emitError(
            "HIP proxy cannot realize an explicitly checked CAS publication");
      auto signalValue =
          materializeSignalValue(builder, signal.getSource(), signal);
      auto destination =
          materializeRemoteRegion(signal.getDestination(), signal);
      if (failed(signalValue) || failed(destination))
        return failure();
      writeSignalCommand(builder, signal.getLoc(), commandAt(index),
                         destination->pointer, *signalValue, destination->key,
                         static_cast<bool>(signal.getWeakPublication()));
      return success();
    };
    auto onInline = [&](PostInlineOp put, std::size_t index) -> LogicalResult {
      if (hipDeviceLibrary.empty() || put.getPayloadBytes() != 8)
        return put.emitError("HIP proxy inline writes require exactly 8 bytes");
      auto destination = materializeRemoteRegion(put.getDestination(), put);
      if (failed(destination)) return failure();
      auto words = emit::packInlinePayload(builder, put.getLoc(), put.getValues(), 2);
      auto value = pack32(builder, put.getLoc(), words[0], words[1]);
      writeSignalCommand(builder, put.getLoc(), commandAt(index),
                         destination->pointer, value, destination->key, true);
      auto opcode = arith::SelectOp::create(
          builder, put.getLoc(), put.getPredicate(),
          i32Constant(builder, put.getLoc(), ::aboleth::abi::kProxyCommandWeakSignal),
          i32Constant(builder, put.getLoc(), ::aboleth::abi::kProxyCommandNop));
      store32(builder, put.getLoc(), commandAt(index),
              offsetof(::aboleth::abi::ProxyCommand, opcode), opcode);
      return success();
    };
    auto onNop = [&](PostNopOp nop, std::size_t index) -> LogicalResult {
      store32(builder, nop.getLoc(), commandAt(index),
              offsetof(::aboleth::abi::ProxyCommand, opcode),
              i32Constant(builder, nop.getLoc(), ::aboleth::abi::kProxyCommandNop));
      return success();
    };
    auto transferred = hipDeviceLibrary.empty()
        ? forEachTransfer(submission, onPut, onSignal)
        : forEachTransfer(submission, onPut, onSignal, onNop, onInline);
    if (failed(transferred)) return failure();

    store32(
        builder, loc, slotAddress,
        offsetof(::aboleth::abi::ProxyBatchSlot, command_count),
        i32Constant(
            builder, loc,
            static_cast<std::uint32_t>(submission.getBatchSize())));
    Value target = arith::AddIOp::create(
        builder, loc, sequence, i64Constant(builder, loc, 1));
    Value readyPointer = pointerFromAddress(
        builder, loc,
        addressOffset(
            builder, loc, slotAddress,
            offsetof(::aboleth::abi::ProxyBatchSlot, ready_sequence)),
        builder.getI64Type());
    releaseStore64(builder, loc, readyPointer, target);
    submissionStates[submission] = {queueState, target};
    return success();
  };

  auto lowerPollOp = [&](PollOp poll) -> LogicalResult {
    auto submission = poll.getSubmission().getDefiningOp<SubmitOp>();
    auto state = submissionStates.find(submission);
    if (state == submissionStates.end())
      return poll.emitError("submission has no proxy queue state");
    return lowerPoll(poll, state->second.first, state->second.second);
  };

  QueueLoweringHooks hooks;
  hooks.backendName = "proxy";
  hooks.lowerSubmission = lowerSubmission;
  hooks.lowerPoll = lowerPollOp;
  hooks.lowerWaitGroup = lowerGroupWait;
  hooks.lowerDrain = lowerDrain;
  hooks.lowerWaitSignal = lowerWaitSignal;
  if (failed(runQueueLowering(getOperation(), hooks)))
    signalPassFailure();
}

} // namespace mlir::aboleth
