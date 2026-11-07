#include "aboleth/Transforms/Passes.h"

#include "aboleth/Target/TransportCaps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHDEFERANNOUNCEMENTPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct DeferAnnouncement
    : public impl::AbolethDeferAnnouncementPassBase<DeferAnnouncement> {
  using Base::Base;
  void runOnOperation() override;
};

bool hasPlannedPoll(SubmitOp submission) {
  for (Operation *user : submission.getToken().getUsers())
    if (auto poll = dyn_cast<PollOp>(user))
      return static_cast<bool>(poll.getPredicate());
  return false;
}

} // namespace

void DeferAnnouncement::runOnOperation() {
  // Only an announcement that costs a doorbell is worth deferring.
  const std::string name = transport;
  const auto *caps =
      ::aboleth::abi::lookup_transport_caps(name.data(), name.size());
  if (!caps || !caps->gpu_initiated || !caps->ordered_delivery ||
      caps->doorbell_model == ::aboleth::abi::TransportDoorbellModel::None)
    return;
  // Each function of the module stands on its own: its kernel either waits on
  // peers somewhere or never does.
  for (Operation &function : getOperation().getBody()->getOperations()) {
    const bool waitsOnPeers =
        function.walk([](WaitSignalOp) { return WalkResult::interrupt(); })
            .wasInterrupted();
    if (waitsOnPeers)
      continue;
    function.walk([&](SubmitOp submission) {
      if (submission.getGroup() || submission.getDeferCompletion() ||
          submission.getExternalPublication() ||
          submission.getHeadPublication() || !hasPlannedPoll(submission))
        return;
      submission.setDeferredAnnouncementAttr(
          UnitAttr::get(submission.getContext()));
    });
  }
}

} // namespace mlir::aboleth
