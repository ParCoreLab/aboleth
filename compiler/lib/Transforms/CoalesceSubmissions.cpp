#include "aboleth/Transforms/Passes.h"

#include "aboleth/Target/TransportCaps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHCOALESCESUBMISSIONSPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct CoalesceSubmissions
    : public impl::AbolethCoalesceSubmissionsPassBase<CoalesceSubmissions> {
  using Base::Base;
  void runOnOperation() override;
};

// The submission's poll, or null. A submission token feeds at most one poll,
// and nothing else may consume it for the merge to be a pure rewrite.
FailureOr<PollOp> soleConsumer(SubmitOp submission) {
  PollOp poll;
  for (Operation *user : submission.getToken().getUsers()) {
    auto candidate = dyn_cast<PollOp>(user);
    if (!candidate || poll)
      return failure();
    poll = candidate;
  }
  return poll;
}

// Two submissions merge when one batch can carry both with the obligations of
// each: the same endpoint, publication and reservation contract, and a
// completion that the merged batch's single CQE satisfies for both.
bool canMerge(SubmitOp earlier, SubmitOp later) {
  if (!areEquivalentEndpoints(earlier.getEndpoint(), later.getEndpoint()))
    return false;
  // Grouped completion retires a range per group; keep those batches whole.
  if (earlier.getGroup() || later.getGroup())
    return false;
  if (earlier.getDeferCompletionAttr() != later.getDeferCompletionAttr() ||
      earlier.getHeadPublicationAttr() != later.getHeadPublicationAttr() ||
      earlier.getPlannedReservationAttr() !=
          later.getPlannedReservationAttr() ||
      earlier.getReservationScopeAttr() != later.getReservationScopeAttr() ||
      earlier.getIssuerScopeAttr() != later.getIssuerScopeAttr() ||
      earlier.getExternalPublicationAttr() !=
          later.getExternalPublicationAttr() ||
      earlier.getDeferredAnnouncementAttr() !=
          later.getDeferredAnnouncementAttr())
    return false;
  // A completion no-op must stay the batch's last command.
  if (isa_and_nonnull<PostNopOp>(
          earlier.getTransfers().back().getDefiningOp()))
    return false;
  auto earlierPoll = soleConsumer(earlier);
  auto laterPoll = soleConsumer(later);
  if (failed(earlierPoll) || failed(laterPoll))
    return false;
  if (!*earlierPoll != !*laterPoll)
    return false;
  if (!*earlierPoll)
    return true;
  // One poll retires the merged batch, so both must be due together.
  return (*earlierPoll)->getBlock() == (*laterPoll)->getBlock() &&
         earlierPoll->getPredicate() == laterPoll->getPredicate() &&
         earlierPoll->getCoalesced() == laterPoll->getCoalesced() &&
         earlierPoll->getSelfReclaim() == laterPoll->getSelfReclaim();
}

// Replace both submissions with one at the later position. The earlier chain's
// commands come first, so each chain keeps its order and the endpoint still
// sees the earlier chain's commands ahead of the later one's.
SubmitOp merge(SubmitOp earlier, SubmitOp later) {
  PollOp earlierPoll = *soleConsumer(earlier);
  PollOp laterPoll = *soleConsumer(later);

  SmallVector<Value> transfers(earlier.getTransfers());
  llvm::append_range(transfers, later.getTransfers());
  OpBuilder builder(later);
  auto merged = SubmitOp::create(
      builder, later.getLoc(), later.getToken().getType(),
      later.getEndpoint(), transfers, /*group=*/Value{},
      builder.getI32IntegerAttr(static_cast<int32_t>(transfers.size())),
      later.getDeferCompletionAttr(), later.getHeadPublicationAttr(),
      later.getPlannedReservationAttr(), later.getReservationScopeAttr(),
      later.getIssuerScopeAttr(), later.getExternalPublicationAttr(),
      later.getProducerBarrierAttr(), later.getExclusiveIssuerAttr(),
      later.getDeferredAnnouncementAttr());
  earlier.getToken().replaceAllUsesWith(merged.getToken());
  later.getToken().replaceAllUsesWith(merged.getToken());
  // The batch's CQE retires every command, so the first poll completes both
  // chains and the second has nothing left to wait for.
  if (earlierPoll)
    (earlierPoll->isBeforeInBlock(laterPoll) ? laterPoll : earlierPoll)
        .erase();
  earlier.erase();
  later.erase();
  return merged;
}

void coalesceBlock(Block &block) {
  // The latest submission per endpoint that a following one may still join.
  SmallVector<SubmitOp> open;
  for (Operation &operation : llvm::make_early_inc_range(block)) {
    auto submission = dyn_cast<SubmitOp>(&operation);
    if (!submission) {
      if (!isCrossableBySubmission(&operation))
        open.clear();
      continue;
    }
    auto *candidate = llvm::find_if(open, [&](SubmitOp other) {
      return areEquivalentEndpoints(other.getEndpoint(),
                                    submission.getEndpoint());
    });
    if (candidate == open.end()) {
      open.push_back(submission);
      continue;
    }
    // A submission the next one cannot join still orders that endpoint's
    // queue, so it closes the earlier candidate either way.
    *candidate = canMerge(*candidate, submission)
                     ? merge(*candidate, submission)
                     : submission;
  }
}

} // namespace

void CoalesceSubmissions::runOnOperation() {
  // Merging pays where a submission has a reservation and an announcement of
  // its own and an ordered queue keeps the segments' publications sound.
  const std::string name = transport;
  const auto *caps =
      ::aboleth::abi::lookup_transport_caps(name.data(), name.size());
  if (!caps || !caps->gpu_initiated || !caps->ordered_delivery ||
      caps->load_store_addressable)
    return;
  SmallVector<Block *> blocks;
  getOperation().walk([&](Block *block) { blocks.push_back(block); });
  for (Block *block : blocks)
    coalesceBlock(*block);
}

} // namespace mlir::aboleth
