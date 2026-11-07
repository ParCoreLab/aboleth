#include "aboleth/Transforms/Passes.h"

#include "aboleth/Target/TransportPlan.h"
#include "mlir/IR/Builders.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHPLANPUBLICATIONPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct PlanPublication
    : public impl::AbolethPlanPublicationPassBase<PlanPublication> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void PlanPublication::runOnOperation() {
  // Without a transport the lowering keeps its default ordering point.
  bool fenced = false;
  bool completedPrefix = false;
  if (!transport.empty()) {
    const std::string name = transport;
    const auto *caps =
        ::aboleth::abi::lookup_transport_caps(name.data(), name.size());
    if (!caps) {
      getOperation().emitError("unknown transport '") << name << "'";
      return signalPassFailure();
    }
    fenced = ::aboleth::abi::legal_strong_publication(
                 *caps, /*prefer_imm=*/false, strictRegistrations) ==
             ::aboleth::abi::TransportPublicationMode::SameQueueFencedWrite;
    // Ordered queues without responder atomics can publish an unchecked
    // epoch after the sender has retired its payload prefix. This is a
    // distinct mechanism from either a checked CAS or a hardware fence bit.
    completedPrefix = caps->ordered_delivery && !caps->native_atomics &&
                      !caps->load_store_addressable && !fenced;
  }
  UnitAttr unit = UnitAttr::get(&getContext());

  getOperation().walk([&](SignalOp signal) {
    // With no transfer token, this notification publishes only its own epoch.
    // It cannot serve as the visibility boundary for payload effects.
    if (!signal.getAfter() && !signal.getGroup())
      signal.setWeakPublicationAttr(unit);
    else if (fenced)
      signal.setFencedSignalAttr(unit);
    else if (completedPrefix && !signal.getFencedSignal())
      signal.setCompletedPrefixAttr(unit);
  });
  // Queue IR written by hand (the Tier-1 programs, transport phases) arrives
  // already posted; its strong signals take the same mechanism.
  if (fenced || completedPrefix)
    getOperation().walk([&](PostSignalOp signal) {
      // An explicit expected predecessor is a checked operation. A planner
      // must never turn it into an unchecked write.
      if (signal.getWeakPublication() || signal.getExpected()) return;
      if (fenced)
        signal.setFencedSignalAttr(unit);
      else if (!signal.getFencedSignal())
        signal.setCompletedPrefixAttr(unit);
    });
}

} // namespace mlir::aboleth
