#include "aboleth/Target/Passes.h"
#include "aboleth/Target/QueueLowering.h"
#include "aboleth/Target/TritonBridge.h"
#include "aboleth/Target/TritonEmit.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::aboleth {
#define GEN_PASS_DEF_ABOLETHLOWERIONICPASS
#include "aboleth/Target/Passes.h.inc"
namespace {
struct LowerIonic : impl::AbolethLowerIonicPassBase<LowerIonic> {
  using Base = impl::AbolethLowerIonicPassBase<LowerIonic>;
  using Base::Base;
  void runOnOperation() override;
};
using emit::i32Constant;
using emit::i64Constant;
using emit::pointerAddress;

// The pinned upstream AMD load lowering loses tt.load's volatile flag. Keep
// explicit volatile consumer reads in HIP LLVM bitcode, where volatility and
// masks survive. This is limited to Ionic modules with an observation boundary;
// ordinary producer modules and other backends retain their existing lowering.
LogicalResult preserveVolatileObserverLoads(Operation *module,
                                            StringRef bitcodePath) {
  SmallVector<triton::LoadOp> loads;
  module->walk([&](triton::LoadOp op) {
    if (op.getIsVolatile()) loads.push_back(op);
  });
  for (triton::LoadOp op : loads) {
    OpBuilder b(op); Location loc = op.getLoc();
    Type resultType = op.getType();
    Type elementType = getElementTypeOrSelf(resultType);
    bool pointer = isa<triton::PointerType>(elementType);
    unsigned width = pointer ? 64 : elementType.getIntOrFloatBitWidth();
    if (width != 1 && width != 8 && width != 16 && width != 32 && width != 64)
      return op.emitError("Ionic volatile observer loads require 1/8/16/32/64-bit elements");
    unsigned storageWidth = std::max(8u, width);
    auto shapedInt = [&](unsigned bits) -> Type {
      Type integer = b.getIntegerType(bits);
      if (auto tensor = dyn_cast<RankedTensorType>(resultType))
        return RankedTensorType::get(tensor.getShape(), integer, tensor.getEncoding());
      return integer;
    };
    auto constant = [&](unsigned bits, int64_t value) -> Value {
      Value scalar = arith::ConstantIntOp::create(b, loc, value, bits);
      if (auto tensor = dyn_cast<RankedTensorType>(resultType))
        return triton::SplatOp::create(b, loc, shapedInt(bits), scalar);
      return scalar;
    };
    Value address = triton::PtrToIntOp::create(b, loc, shapedInt(64), op.getPtr());
    Value mask = op.getMask()
        ? Value(arith::ExtUIOp::create(b, loc, shapedInt(32), op.getMask()))
        : constant(32, 1);
    Value other = op.getOther();
    if (!other) other = constant(storageWidth, 0);
    else if (pointer)
      other = triton::PtrToIntOp::create(b, loc, shapedInt(64), other);
    else if (isa<FloatType>(elementType))
      other = triton::BitcastOp::create(b, loc, shapedInt(width), other);
    else if (width == 1)
      other = arith::ExtUIOp::create(b, loc, shapedInt(8), other);
    std::string symbol = "aboleth_ionic_load_u" + std::to_string(storageWidth);
    Value bits = triton::ExternElementwiseOp::create(b, loc,
        shapedInt(storageWidth), ValueRange{address, mask, other},
        "aboleth_ionic", bitcodePath, symbol, false).getResult();
    Value result = bits;
    if (pointer)
      result = triton::IntToPtrOp::create(b, loc, resultType, bits);
    else if (isa<FloatType>(elementType))
      result = triton::BitcastOp::create(b, loc, resultType, bits);
    else if (width == 1)
      result = arith::TruncIOp::create(b, loc, resultType, bits);
    op.getResult().replaceAllUsesWith(result);
    op.erase();
  }
  return success();
}

void LowerIonic::runOnOperation() {
  bool observes = false;
  getOperation().walk([&](WaitSignalOp) { observes = true; });
  if (bitcodePath.empty()) {
    getOperation().emitError("Ionic lowering requires its HIP device bitcode path");
    return signalPassFailure();
  }
  auto call = [&](OpBuilder &b, Location loc, StringRef symbol, ValueRange args) -> Value {
    std::string entry = symbol.str();
    if (!detached && symbol != "aboleth_ionic_wait_signal" && symbol != "aboleth_ionic_fence")
      entry += "_cta";
    return triton::ExternElementwiseOp::create(b, loc, b.getI64Type(), args,
        "aboleth_ionic", bitcodePath, entry, false).getResult();
  };
  auto address = [&](Value binding, Operation *op, OpBuilder &b) -> FailureOr<Value> {
    auto ep = materializeEndpoint(binding, op);
    if (failed(ep)) return failure();
    return pointerAddress(b, op->getLoc(), *ep);
  };
  llvm::DenseMap<Operation *, std::pair<Value, Value>> targets;
  llvm::DenseSet<Operation *> exclusiveSubmissions;
  llvm::DenseSet<Operation *> sharedFenceSubmissions;
  auto submit = [&](SubmitOp op, PollOp) -> LogicalResult {
    if (op.getExternalPublication())
      return op.emitError("Ionic requires GPU-owned doorbell publication");
    OpBuilder b(op); auto loc = op.getLoc();
    auto ep = address(op.getEndpoint(), op, b);
    if (failed(ep)) return failure();
    uint32_t count = 0;
    for (Value transfer : op.getTransfers())
      if (!transfer.getDefiningOp<PostNopOp>()) ++count;
    if (!count) {
      Value end = call(b, loc, "aboleth_ionic_drain", ValueRange{*ep});
      targets[op] = {*ep, end}; return success();
    }
    // The proof is per Triton program. Detached per-thread callers do not
    // inherit CTA exclusivity. Retain the shared ABI without that proof.
    const bool exclusive = !detached && static_cast<bool>(op.getExclusiveIssuer());
    if (exclusive) exclusiveSubmissions.insert(op);
    // All frontend producers precede this SubmitOp. Reserve keeps the
    // all-lane producer fence; later commands only build WQEs on its issuer.
    const bool shareFence = exclusive && shareProducerFence;
    if (shareFence) sharedFenceSubmissions.insert(op);
    Value base = call(b, loc, exclusive ? "aboleth_ionic_reserve_exclusive"
                                      : "aboleth_ionic_reserve",
                      ValueRange{*ep, i32Constant(b, loc, count)});
    uint32_t next = 0;
    auto slot = [&]() -> Value {
      return arith::AddIOp::create(b, loc, base, i64Constant(b, loc, next++));
    };
    auto put = [&](PostPutOp put, std::size_t) -> LogicalResult {
      auto src = materializeLocalRegion(put.getSource(), put);
      auto dst = materializeRemoteRegion(put.getDestination(), put);
      if (failed(src) || failed(dst)) return failure();
      call(b, put.getLoc(), shareFence ? "aboleth_ionic_put_posted"
                                     : "aboleth_ionic_put", ValueRange{*ep, slot(),
          pointerAddress(b, loc, src->pointer), src->key,
          pointerAddress(b, loc, dst->pointer), dst->key, put.getBytes(),
          arith::ExtUIOp::create(b, loc, b.getI32Type(), put.getPredicate())});
      return success();
    };
    auto signal = [&](PostSignalOp signal, std::size_t) -> LogicalResult {
      if (signal.getExpected())
        return signal.emitError("Ionic explicitly checked CAS publication is not implemented");
      auto dst = materializeRemoteRegion(signal.getDestination(), signal);
      auto epoch = materializeSignalValue(b, signal.getSource(), signal);
      if (failed(dst) || failed(epoch)) return failure();
      // The planner proves same-queue visibility only for strict receiver
      // registrations. The fenced WQE replaces the intermediate prefix poll;
      // the submission still commits and its final Poll protects source reuse.
      const bool fenced = fencedPublication && signal.getFencedSignal() &&
                          !signal.getWeakPublication();
      if (!signal.getWeakPublication() && !fenced) {
        Value prefix = arith::AddIOp::create(b, loc, base, i64Constant(b, loc, next));
        if (Value predicate = signal.getPredicate()) {
          auto guard = scf::IfOp::create(b, loc, predicate, false);
          OpBuilder::InsertionGuard insertion(b);
          b.setInsertionPointToStart(guard.thenBlock());
          call(b, loc, shareFence ? "aboleth_ionic_publish_prefix_posted"
                                 : "aboleth_ionic_publish_prefix", ValueRange{*ep, prefix});
        } else {
          call(b, loc, shareFence ? "aboleth_ionic_publish_prefix_posted"
                                 : "aboleth_ionic_publish_prefix", ValueRange{*ep, prefix});
        }
      }
      Value predicate = signal.getPredicate()
          ? Value(arith::ExtUIOp::create(b, loc, b.getI32Type(), signal.getPredicate()))
          : i32Constant(b, loc, 1);
      call(b, loc, fenced
          ? (shareFence ? "aboleth_ionic_inline_fenced_posted"
                        : "aboleth_ionic_inline_fenced")
          : (shareFence ? "aboleth_ionic_inline_posted"
                        : "aboleth_ionic_inline"), ValueRange{*ep, slot(),
          pointerAddress(b, loc, dst->pointer), dst->key, *epoch,
          i64Constant(b, loc, 0), i32Constant(b, loc, 8), predicate});
      return success();
    };
    auto nop = [&](PostNopOp, std::size_t) { return success(); };
    auto inlined = [&](PostInlineOp put, std::size_t) -> LogicalResult {
      if (put.getPayloadBytes() > 16)
        return put.emitError("Ionic inline writes require a payload of at most 16 bytes");
      auto dst = materializeRemoteRegion(put.getDestination(), put);
      if (failed(dst)) return failure();
      auto words = emit::packInlinePayload(b, loc, put.getValues(), 4);
      auto pack = [&](Value low, Value high) -> Value {
        return arith::OrIOp::create(b, loc, emit::zeroExtend64(b, loc, low),
          arith::ShLIOp::create(b, loc, emit::zeroExtend64(b, loc, high), i64Constant(b, loc, 32)));
      };
      call(b, loc, shareFence ? "aboleth_ionic_inline_posted"
                             : "aboleth_ionic_inline", ValueRange{*ep, slot(),
          pointerAddress(b, loc, dst->pointer), dst->key,
          pack(words[0], words[1]), pack(words[2], words[3]),
          i32Constant(b, loc, put.getPayloadBytes()),
          arith::ExtUIOp::create(b, loc, b.getI32Type(), put.getPredicate())});
      return success();
    };
    if (failed(forEachTransfer(op, put, signal, nop, inlined))) return failure();
    Value end = arith::AddIOp::create(b, loc, base, i64Constant(b, loc, count));
    call(b, loc, shareFence ? "aboleth_ionic_commit_posted"
                           : exclusive ? "aboleth_ionic_commit_exclusive"
                           : "aboleth_ionic_commit", ValueRange{*ep, end});
    targets[op] = {*ep, end}; return success();
  };
  auto poll = [&](PollOp op) -> LogicalResult {
    auto it = targets.find(op.getSubmission().getDefiningOp());
    if (it == targets.end()) return op.emitError("Ionic submission has no queue range");
    OpBuilder b(op); auto loc = op.getLoc();
    if (Value predicate = op.getPredicate()) {
      auto guard = scf::IfOp::create(b, loc, predicate, false);
      b.setInsertionPointToStart(guard.thenBlock());
    }
    const bool exclusive = exclusiveSubmissions.contains(op.getSubmission().getDefiningOp());
    const bool shareFence = sharedFenceSubmissions.contains(op.getSubmission().getDefiningOp());
    call(b, loc, shareFence ? "aboleth_ionic_poll_posted"
                           : exclusive ? "aboleth_ionic_poll_exclusive"
                           : "aboleth_ionic_poll", ValueRange{it->second.first, it->second.second});
    return success();
  };
  auto drain = [&](DrainOp op) -> LogicalResult {
    OpBuilder b(op); auto ep = address(op.getEndpoint(), op, b);
    if (failed(ep)) return failure();
    call(b, op.getLoc(), "aboleth_ionic_drain", ValueRange{*ep}); return success();
  };
  llvm::DenseMap<Operation *, Value> groups;
  auto begin = [&](BeginGroupOp op) -> LogicalResult {
    OpBuilder b(op); auto ep = address(op.getEndpoint(), op, b);
    if (failed(ep)) return failure(); groups[op] = *ep; return success();
  };
  auto waitGroup = [&](WaitGroupOp op) -> LogicalResult {
    auto it = groups.find(op.getGroup().getDefiningOp());
    if (it == groups.end()) return op.emitError("Ionic completion group has no endpoint");
    OpBuilder b(op); call(b, op.getLoc(), "aboleth_ionic_drain", ValueRange{it->second});
    return success();
  };
  auto waitSignal = [&](WaitSignalOp op) -> LogicalResult {
    OpBuilder b(op); auto region = materializeLocalRegion(op.getSignal(), op);
    if (failed(region)) return failure();
    call(b, op.getLoc(), "aboleth_ionic_wait_signal",
      ValueRange{pointerAddress(b, op.getLoc(), region->pointer), op.getExpected()});
    return success();
  };
  auto fence = [&](SystemFenceOp op) -> LogicalResult {
    OpBuilder b(op); call(b, op.getLoc(), "aboleth_ionic_fence", ValueRange{});
    return success();
  };
  QueueLoweringHooks hooks;
  hooks.endpointKindMask = 7;
  hooks.supportsCompletedPrefix = true;
  hooks.backendName = "Ionic"; hooks.lowerSubmission = submit;
  hooks.lowerPoll = poll; hooks.lowerBeginGroup = begin;
  hooks.lowerWaitGroup = waitGroup; hooks.lowerDrain = drain;
  hooks.lowerWaitSignal = waitSignal; hooks.lowerSystemFence = fence;
  if (failed(runQueueLowering(getOperation(), hooks))) {
    signalPassFailure(); return;
  }
  if (observes && failed(preserveVolatileObserverLoads(getOperation(), bitcodePath.getValue())))
    signalPassFailure();
}
} // namespace
} // namespace mlir::aboleth
