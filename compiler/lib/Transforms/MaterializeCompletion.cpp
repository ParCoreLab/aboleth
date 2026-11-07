#include "aboleth/Transforms/Passes.h"

#include "aboleth/Target/RdmaAbi.h"
#include "aboleth/Target/TransportCaps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"

#include <algorithm>
#include <limits>

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHMATERIALIZECOMPLETIONPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

// Queue slots the signal chain a planned wait completes occupies in one
// iteration. An inline write may span several slots on mlx5; counting it that
// way elsewhere only underestimates how many iterations fit.
std::int64_t chainSlots(SignalOp signal) {
  std::int64_t slots = 1;
  for (Value cursor = signal.getAfter(); cursor;) {
    Operation *command = cursor.getDefiningOp();
    if (auto put = dyn_cast<PutOp>(command)) {
      ++slots;
      cursor = put.getAfter();
    } else if (auto put = dyn_cast<PutInlineOp>(command)) {
      slots += ::aboleth::abi::inline_wqe_blocks(put.getPayloadBytes());
      cursor = put.getAfter();
    } else {
      break;
    }
  }
  return slots;
}

// The completion predicate also controls the CQE requested by the producer.
// Frontends may form its integer bounds after the semantic transfer. Move only
// speculatable, effect-free dependencies in the same block before that transfer.
// An observation such as a load cannot be moved; its safe fallback is eager
// completion, rather than invalid SSA or a guessed completion predicate.
bool collectCompletionBounds(Value value, Operation *producer,
                             DominanceInfo &dominance,
                             llvm::DenseSet<Operation *> &seen,
                             SmallVectorImpl<Operation *> &operations) {
  if (dominance.dominates(value, producer))
    return true;
  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getBlock() != producer->getBlock() ||
      operation->getNumRegions() || !isPure(operation))
    return false;
  if (!seen.insert(operation).second)
    return true;
  for (Value operand : operation->getOperands())
    if (!collectCompletionBounds(operand, producer, dominance, seen,
                                 operations))
      return false;
  operations.push_back(operation);
  return true;
}

class MaterializeCompletion
    : public impl::AbolethMaterializeCompletionPassBase<
          MaterializeCompletion> {
public:
  using Base::Base;

  FailureOr<int64_t> queueWindow(PlannedWaitOp wait, ArrayRef<PlannedWaitOp> waits) {
    const std::string name = transport;
    const auto *caps = ::aboleth::abi::lookup_transport_caps(name.data(), name.size());
    if (!caps)
      return wait.emitOpError("planned completion needs the kernel's transport");
    if (!caps->queue_capacity_limited)
      return std::numeric_limits<int64_t>::max();
    int64_t capacity = queueCapacity > 0 ? int64_t(queueCapacity)
                                      : std::numeric_limits<int64_t>::max();
    int64_t slots = 0;
    auto signal = wait.getTransfer().getDefiningOp<SignalOp>();
    for (PlannedWaitOp other : waits) {
      auto otherSignal = other.getTransfer().getDefiningOp<SignalOp>();
      if (other->getBlock() != wait->getBlock() ||
          !areEquivalentEndpoints(otherSignal.getEndpoint(), signal.getEndpoint()))
        continue;
      slots += chainSlots(otherSignal);
      if (auto bound = other.getQueueCapacity())
        capacity = std::min<int64_t>(capacity, *bound);
    }
    if (capacity == std::numeric_limits<int64_t>::max())
      return wait.emitOpError("reuse-planned completion needs the endpoints' queue capacity");
    if (capacity <= slots)
      return wait.emitOpError("the endpoint queue cannot hold one iteration");
    return (capacity - 1) / slots;
  }

  // How many iterations a reuse-planned wait may leave outstanding: the
  // largest power of two within the sources' reuse distance, the transport's
  // preferred interval, and the iterations whose commands fit the queue. Every
  // reuse-planned chain on the same endpoint in the same loop body leaves its
  // iterations outstanding alongside this one, so all of their slots count.
  FailureOr<std::int64_t> plannedInterval(PlannedWaitOp wait,
                                          ArrayRef<PlannedWaitOp> waits) {
    const std::string name = transport;
    const auto *caps =
        ::aboleth::abi::lookup_transport_caps(name.data(), name.size());
    if (!caps)
      return wait.emitOpError(
          "reuse-planned completion needs the kernel's transport");
    std::int64_t interval = *wait.getReuseDistance();
    if (caps->default_completion_interval > 0)
      interval = std::min<std::int64_t>(interval,
                                        caps->default_completion_interval);
    if (caps->queue_capacity_limited) {
      auto window = queueWindow(wait, waits);
      if (failed(window))
        return failure();
      interval = std::min(interval, *window);
    }
    return static_cast<std::int64_t>(
        llvm::bit_floor(static_cast<std::uint64_t>(interval)));
  }

  void runOnOperation() override {
    if (completionInterval < 1) {
      getOperation().emitError(
          "completion interval must be positive");
      signalPassFailure();
      return;
    }

    SmallVector<PlannedWaitOp> waits;
    getOperation().walk(
        [&](PlannedWaitOp wait) { waits.push_back(wait); });
    // Every interval is planned before any wait is rewritten: a chain's
    // queue share depends on the other planned waits on its endpoint.
    SmallVector<std::int64_t> intervals;
    for (PlannedWaitOp wait : waits) {
      std::int64_t interval = wait.getCompletionInterval().value_or(completionInterval);
      if (!planReuse) {
        interval = 1;
      } else if (wait.getReuseDistance()) {
        auto planned = plannedInterval(wait, waits);
        if (failed(planned))
          return signalPassFailure();
        interval = *planned;
      } else if (wait.getCompletionInterval() && interval > 1) {
        auto window = queueWindow(wait, waits);
        if (failed(window))
          return signalPassFailure();
        interval = std::min(interval, *window);
      }
      intervals.push_back(interval);
    }
    // Hoisting moves operations inside one block, which keeps block
    // dominance, so one analysis serves every wait.
    DominanceInfo dominance(getOperation());
    for (auto [wait, interval] : llvm::zip_equal(waits, intervals)) {
      const bool reusePlanned = wait.getReuseDistance().has_value();

      OpBuilder waitBuilder(wait);
      Value predicate;
      if (interval > 1) {
        Operation *producer = wait.getTransfer().getDefiningOp();
        if (!producer) {
          wait.emitError("planned wait transfer has no producer");
          return signalPassFailure();
        }
        llvm::DenseSet<Operation *> seen;
        SmallVector<Operation *> bounds;
        if (!collectCompletionBounds(wait.getIteration(), producer, dominance,
                                     seen, bounds) ||
            (wait.getIterations() &&
             !collectCompletionBounds(wait.getIterations(), producer,
                                      dominance, seen, bounds))) {
          if (reusePlanned) {
            wait.emitError("planned wait's iteration must be computable where "
                           "its transfer is issued");
            return signalPassFailure();
          }
          // A legacy external-plan wait whose bounds are observations keeps
          // eager completion; never move loads to manufacture a predicate.
          WaitOp::create(waitBuilder, wait.getLoc(), wait.getTransfer(), Value{},
                         /*selfReclaim=*/UnitAttr{});
          wait.erase();
          continue;
        }
        for (Operation *bound : bounds)
          bound->moveBefore(producer);
        OpBuilder builder(producer);
        auto type = dyn_cast<IntegerType>(wait.getIteration().getType());
        if (!type) {
          wait.emitError("planned wait requires integer iteration values");
          return signalPassFailure();
        }
        Location loc = wait.getLoc();
        auto constant = [&](std::int64_t value) -> Value {
          return arith::ConstantOp::create(
              builder, loc, type, builder.getIntegerAttr(type, value));
        };
        // The last iteration of each interval polls, so the first poll retires
        // a full interval rather than a single submission.
        Value position =
            reusePlanned
                ? Value(arith::AndIOp::create(builder, loc,
                                              wait.getIteration(),
                                              constant(interval - 1)))
                : Value(arith::RemUIOp::create(builder, loc,
                                               wait.getIteration(),
                                               constant(interval)));
        predicate = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::eq, position,
                                          constant(interval - 1));
        // An external plan also completes its final iteration; a kernel that
        // declares reuse drains its endpoints when it is done instead.
        // A materialized fragment carries its own compiler-planned interval;
        // the module-wide option is only the legacy unbound-wait fallback.
        if (Value iterations = wait.getIterations()) {
          Value finalIteration = arith::CmpIOp::create(
              builder, loc, arith::CmpIPredicate::eq, wait.getIteration(),
              iterations);
          predicate =
              arith::OrIOp::create(builder, loc, predicate, finalIteration);
        }
      }
      // A planned interval is bounded by the queue, so it needs no reclaim.
      WaitOp::create(
          waitBuilder, wait.getLoc(), wait.getTransfer(), predicate,
          /*selfReclaim=*/UnitAttr{});
      wait.erase();
    }
  }
};

} // namespace

} // namespace mlir::aboleth
