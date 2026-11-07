#include "aboleth/Target/Passes.h"
#include "aboleth/Target/QueueLowering.h"
#include "aboleth/Target/TritonBridge.h"
#include "aboleth/Target/TritonEmit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "aboleth/Target/EfaAbi.h"
#include "aboleth/Target/TransportCaps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHLOWEREFAPASS
#include "aboleth/Target/Passes.h.inc"

namespace {

namespace efa = ::aboleth::abi;

using emit::addressOffset;
using emit::acquireLoad64;
using emit::endpointWord;
using emit::high32;
using emit::i32Constant;
using emit::i64Constant;
using emit::loadValue;
using emit::low32;
using emit::pointerAddress;
using emit::pointerFromAddress;
using emit::releaseStore64;
using emit::waitUntilEqual;
using emit::zeroExtend64;

struct LowerEFA : public impl::AbolethLowerEFAPassBase<LowerEFA> {
  using Base = impl::AbolethLowerEFAPassBase<LowerEFA>;
  using Base::Base;
  void runOnOperation() override;
};

std::string environmentOr(const char *name, StringRef configured) {
  if (!configured.empty())
    return configured.str();
  const char *value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

// Immediate-counting publication needs receive-side machinery on both ends and
// is selected explicitly; sender-fenced signalling is the default because it
// leaves the receiver contract unchanged.
bool immediateCountingSelected(StringRef configured) {
  const std::string scheme =
      environmentOr("ABOLETH_EFA_PUBLICATION", configured);
  return StringRef(scheme).equals_insensitive("imm") ||
         StringRef(scheme).equals_insensitive("immediate");
}

std::string bitcodeLibraryPath(StringRef configured) {
  return environmentOr("ABOLETH_EFA_BITCODE", configured);
}

LogicalResult validateEndpoint(Value endpoint, Operation *operation) {
  auto pointerType = dyn_cast<triton::PointerType>(endpoint.getType());
  if (!pointerType || !pointerType.getPointeeType().isInteger(64))
    return operation->emitError(
        "EFA endpoint must be a pointer to the 72-byte i64 descriptor");
  return success();
}

// Every device primitive is an extern call into the linked EFA bitcode. The
// linker inlines them, so this emits the library's instruction sequence rather
// than a call.
struct DeviceLibrary {
  std::string path;

  Value call(OpBuilder &builder, Location loc, StringRef symbol,
             ValueRange arguments) const {
    auto op = triton::ExternElementwiseOp::create(
        builder, loc, builder.getI64Type(), arguments,
        efa::kEfaLibraryName, path, symbol, /*pure=*/false);
    return op.getResult();
  }
};

Value endpointAddress(OpBuilder &builder, Location loc, Value endpoint) {
  return pointerAddress(builder, loc, endpoint);
}

Value completionStatePointer(OpBuilder &builder, Location loc, Value endpoint,
                             std::uint32_t word) {
  Value stateAddress =
      endpointWord(builder, loc, endpoint, efa::kEfaWordCompletionState);
  return emit::stateWordPointer(builder, loc, stateAddress, word);
}

// The send queue bounds how many commands may be outstanding, and SRD makes
// tracking that the application's responsibility.
Value reserveBatch(OpBuilder &builder, Location loc, Value endpoint,
                   std::uint32_t batchSize) {
  Value stateAddress =
      endpointWord(builder, loc, endpoint, efa::kEfaWordCompletionState);
  Value producer = pointerFromAddress(
      builder, loc,
      addressOffset(builder, loc, stateAddress,
                    efa::kEfaStateProducer * sizeof(std::uint64_t)),
      builder.getI64Type());
  return triton::AtomicRMWOp::create(
      builder, loc, builder.getI64Type(), triton::RMWOp::ADD, producer,
      i64Constant(builder, loc, batchSize), /*mask=*/nullptr,
      triton::MemSemantic::RELAXED, triton::MemSyncScope::GPU);
}

// A predicated transfer keeps its slot as a zero-length write so the batch
// still produces the number of completions the compiler planned. SRD is
// unordered, so nothing is lost by not emitting an ordered placeholder.
Value predicatedBytes(OpBuilder &builder, Location loc, Value bytes,
                      Value predicate) {
  if (!predicate)
    return bytes;
  return arith::SelectOp::create(
      builder, loc, predicate, bytes, i32Constant(builder, loc, 0));
}

} // namespace

void LowerEFA::runOnOperation() {
  const DeviceLibrary library{bitcodeLibraryPath(bitcodePath)};
  if (library.path.empty()) {
    getOperation().emitError(
        "EFA lowering requires the device bitcode path; set "
        "ABOLETH_EFA_BITCODE or the bitcode-path pass option");
    return signalPassFailure();
  }
  // Immediate counting moves publication to the receiver, which needs two
  // things this lowering cannot yet express: `aboleth.wait_signal` carries no
  // endpoint, so the receiver has no receive queue to drain, and nothing
  // states how many writes belong to one epoch. The device library already
  // implements the receiver loop; wiring it needs those IR operands and the
  // host-side receive-queue plumbing.
  if (immediateCountingSelected(publicationScheme)) {
    getOperation().emitError(
        "immediate-counting publication is not implemented; it requires "
        "aboleth.wait_signal to carry a receive endpoint and a per-epoch "
        "write count");
    return signalPassFailure();
  }

  llvm::DenseMap<Operation *, std::pair<Value, Value>> submissionRanges;
  llvm::DenseMap<Operation *, std::pair<Value, Value>> groupStates;

  auto lowerSubmission = [&](SubmitOp submission, PollOp) -> LogicalResult {
    if (submission.getExternalPublication())
      return submission.emitError(
          "EFA cannot delegate publication to an mlx5 phase adapter");
    OpBuilder builder(submission);
    Location loc = submission.getLoc();
    auto materializedEndpoint =
        materializeEndpoint(submission.getEndpoint(), submission);
    if (failed(materializedEndpoint))
      return failure();
    Value endpoint = *materializedEndpoint;
    if (failed(validateEndpoint(endpoint, submission)))
      return failure();

    const auto batchSize =
        static_cast<std::uint32_t>(submission.getBatchSize());
    Value batchBase = reserveBatch(builder, loc, endpoint, batchSize);
    Value batchEnd = arith::AddIOp::create(
        builder, loc, batchBase, i64Constant(builder, loc, batchSize));
    Value depths = endpointWord(builder, loc, endpoint, efa::kEfaWordDepths);
    Value sendDepth = low32(builder, loc, depths);
    Value retired =
        completionStatePointer(builder, loc, endpoint, efa::kEfaStateRetired);
    emit::waitForQueueCapacity(builder, loc, retired, batchEnd, sendDepth);

    Value address = endpointAddress(builder, loc, endpoint);

    // Sender-fenced publication splits the hardware batch at every strong
    // signal: the payload prefix is flushed and retired before the signal may
    // travel, so each segment opens its own batch and slot indices restart at
    // the segment base. Flushing the whole reserved batch instead would hand
    // the adapter slots nothing was written into, and it stalls silently on
    // the first garbage descriptor.
    const auto transfers = submission.getTransfers();
    const std::size_t transferCount = transfers.size();
    SmallVector<std::size_t> segmentOf(transferCount, 0);
    llvm::DenseMap<std::size_t, std::uint32_t> segmentSizes;
    {
      std::size_t base = 0;
      for (std::size_t index = 0; index < transferCount; ++index) {
        segmentOf[index] = base;
        auto signal = transfers[index].getDefiningOp<PostSignalOp>();
        if (signal && !signal.getWeakPublication())
          base = index + 1;
        else
          ++segmentSizes[base];
      }
    }
    if (segmentSizes.lookup(0) > 0)
      library.call(
          builder, loc, efa::kEfaSymbolBeginBatch,
          ValueRange{address,
                     i32Constant(builder, loc, segmentSizes.lookup(0))});

    // The first operation of a segment that follows a strong signal opens
    // the segment's batch.
    auto maybeBeginSegment = [&](std::size_t index, Location opLoc) {
      if (index != 0 && segmentOf[index] == index)
        library.call(
            builder, opLoc, efa::kEfaSymbolBeginBatch,
            ValueRange{address,
                       i32Constant(builder, opLoc,
                                   segmentSizes.lookup(index))});
    };
    auto slotConstant = [&](std::size_t index) {
      return i32Constant(
          builder, loc,
          static_cast<std::uint32_t>(index - segmentOf[index]));
    };
    auto requestId = [&](std::size_t index) {
      return i32Constant(builder, loc, static_cast<std::uint32_t>(index));
    };

    auto onPut = [&](PostPutOp put, std::size_t index) -> LogicalResult {
      maybeBeginSegment(index, put.getLoc());
      auto source = materializeLocalRegion(put.getSource(), put);
      auto destination = materializeRemoteRegion(put.getDestination(), put);
      if (failed(source) || failed(destination))
        return failure();
      Value bytes =
          predicatedBytes(builder, put.getLoc(), put.getBytes(),
                          put.getPredicate());
      SmallVector<Value> arguments{
          address,
          slotConstant(index),
          pointerAddress(builder, put.getLoc(), source->pointer),
          source->key,
          pointerAddress(builder, put.getLoc(), destination->pointer),
          destination->key,
          bytes,
          requestId(index),
      };
      library.call(builder, put.getLoc(), efa::kEfaSymbolPostWrite, arguments);
      return success();
    };

    auto onSignal = [&](PostSignalOp signal,
                        std::size_t index) -> LogicalResult {
      // TODO: signals by value. SRD RDMA writes carry no inline data,
      // so the epoch must travel from registered memory; the endpoint needs a
      // registered scratch cell the lowering can stage it into.
      if (signal.getSource().getType().isInteger(64))
        return signal.emitError(
            "EFA signals publish an epoch from a registered local region");
      auto source = materializeLocalRegion(signal.getSource(), signal);
      auto destination =
          materializeRemoteRegion(signal.getDestination(), signal);
      if (failed(source) || failed(destination))
        return failure();
      Location signalLoc = signal.getLoc();
      if (signal.getWeakPublication())
        maybeBeginSegment(index, signalLoc);

      // Sender-fenced publication: retire this batch's payload before the
      // epoch is allowed on the wire, because SRD would otherwise let the
      // signal overtake it.
      if (!signal.getWeakPublication()) {
        Value payloadEnd = arith::AddIOp::create(
            builder, signalLoc, batchBase,
            i64Constant(builder, signalLoc, index));
        library.call(builder, signalLoc, efa::kEfaSymbolFlush,
                     ValueRange{address});
        library.call(builder, signalLoc, efa::kEfaSymbolCompletionAdvance,
                     ValueRange{address, payloadEnd});
        library.call(builder, signalLoc, efa::kEfaSymbolBeginBatch,
                     ValueRange{address, i32Constant(builder, signalLoc, 1)});
      }

      // The epoch travels from its registered cell, not by value: EFA has no
      // inline data on RDMA writes, so the wrapper posts an ordinary 8-byte
      // write from the source region.
      // A strong signal opens its own single-entry batch, so it posts at
      // slot zero; a weak one rides in its segment's batch.
      Value signalSlot = signal.getWeakPublication()
                             ? slotConstant(index)
                             : i32Constant(builder, signalLoc, 0);
      library.call(
          builder, signalLoc, efa::kEfaSymbolPostInline,
          ValueRange{
              address,
              signalSlot,
              pointerAddress(builder, signalLoc, source->pointer),
              source->key,
              pointerAddress(builder, signalLoc, destination->pointer),
              destination->key,
              requestId(index),
          });
      return success();
    };
    // TODO: put_inline on EFA. Lower post_inline from a per-endpoint
    // registered cell, as signal epochs already travel, and investigate
    // write-with-immediate for values of at most 4 bytes.
    if (failed(forEachTransfer(submission, onPut, onSignal)))
      return failure();

    // Batches ring the doorbell in reservation order so concurrent producers
    // publish a contiguous run.
    Value publish =
        completionStatePointer(builder, loc, endpoint, efa::kEfaStatePublish);
    waitUntilEqual(builder, loc, publish, batchBase);
    library.call(builder, loc, efa::kEfaSymbolFlush, ValueRange{address});
    releaseStore64(builder, loc, publish, batchEnd);
    submissionRanges[submission] = {batchBase, batchEnd};
    return success();
  };

  auto lowerPollOp = [&](PollOp poll) -> LogicalResult {
    auto submission = poll.getSubmission().getDefiningOp<SubmitOp>();
    auto range = submissionRanges.find(submission);
    if (range == submissionRanges.end())
      return poll.emitError("submission has no reserved queue range");
    auto endpoint = materializeEndpoint(submission.getEndpoint(), poll);
    if (failed(endpoint))
      return failure();
    OpBuilder builder(poll);
    Location loc = poll.getLoc();
    Value address = endpointAddress(builder, loc, *endpoint);
    Value target = range->second.second;

    // A predicated poll defers completion; the later true predicate retires
    // everything ordered before it.
    if (Value predicate = poll.getPredicate()) {
      auto guard = scf::IfOp::create(builder, loc, predicate, /*else=*/false);
      OpBuilder guarded(guard.thenBlock(), guard.thenBlock()->begin());
      library.call(guarded, loc, efa::kEfaSymbolCompletionAdvance,
                   ValueRange{address, target});
      return success();
    }
    library.call(builder, loc, efa::kEfaSymbolCompletionAdvance,
                 ValueRange{address, target});
    return success();
  };

  auto lowerBeginGroupOp = [&](BeginGroupOp group) -> LogicalResult {
    auto endpoint = materializeEndpoint(group.getEndpoint(), group);
    if (failed(endpoint))
      return failure();
    OpBuilder builder(group);
    Value base = reserveBatch(builder, group.getLoc(), *endpoint, 0);
    groupStates[group] = {*endpoint, base};
    return success();
  };

  // A completion group retires every command submitted since it opened. EFA
  // needs no completion placeholder: the producer index already counts them.
  auto retireThroughProducer = [&](Location loc, Value endpoint,
                                   OpBuilder &builder) {
    Value address = endpointAddress(builder, loc, endpoint);
    Value producer =
        completionStatePointer(builder, loc, endpoint, efa::kEfaStateProducer);
    Value target = loadValue(builder, loc, producer, /*isVolatile=*/true);
    library.call(builder, loc, efa::kEfaSymbolCompletionAdvance,
                 ValueRange{address, target});
  };

  auto lowerWaitGroupOp = [&](WaitGroupOp wait) -> LogicalResult {
    auto begin = wait.getGroup().getDefiningOp<BeginGroupOp>();
    if (!begin)
      return wait.emitError("completion group has no matching begin_group");
    auto state = groupStates.find(begin);
    if (state == groupStates.end())
      return wait.emitError("completion group has no direct EFA state");
    OpBuilder builder(wait);
    retireThroughProducer(wait.getLoc(), state->second.first, builder);
    return success();
  };

  auto lowerDrainOp = [&](DrainOp drain) -> LogicalResult {
    auto endpoint = materializeEndpoint(drain.getEndpoint(), drain);
    if (failed(endpoint))
      return failure();
    if (failed(validateEndpoint(*endpoint, drain)))
      return failure();
    OpBuilder builder(drain);
    retireThroughProducer(drain.getLoc(), *endpoint, builder);
    return success();
  };

  auto lowerWaitSignalOp = [&](WaitSignalOp wait) -> LogicalResult {
    OpBuilder builder(wait);
    Location loc = wait.getLoc();
    auto signal = materializeLocalRegion(wait.getSignal(), wait);
    if (failed(signal))
      return failure();
    auto signalType =
        dyn_cast<triton::PointerType>(signal->pointer.getType());
    if (!signalType || !signalType.getPointeeType().isInteger(64))
      return wait.emitError("EFA wait_signal requires an i64 signal");

    // Sender-fenced publication leaves the receiver contract unchanged: the
    // epoch cell is written by the peer and observed with an acquire load.
    auto loop = scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
    Block *before = builder.createBlock(&loop.getBefore());
    builder.setInsertionPointToStart(before);
    Value observed = acquireLoad64(builder, loc, signal->pointer);
    Value keepWaiting = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ult, observed,
        wait.getExpected());
    scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});
    Block *after = builder.createBlock(&loop.getAfter());
    builder.setInsertionPointToStart(after);
    scf::YieldOp::create(builder, loc);
    return success();
  };

  QueueLoweringHooks hooks;
  hooks.backendName = "EFA";
  hooks.lowerSubmission = lowerSubmission;
  hooks.lowerPoll = lowerPollOp;
  hooks.lowerBeginGroup = lowerBeginGroupOp;
  hooks.lowerWaitGroup = lowerWaitGroupOp;
  hooks.lowerDrain = lowerDrainOp;
  hooks.lowerWaitSignal = lowerWaitSignalOp;
  if (failed(runQueueLowering(getOperation(), hooks)))
    signalPassFailure();
}

} // namespace mlir::aboleth
