#include "aboleth/Transforms/Passes.h"
#include "aboleth/IR/LogicalRegions.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
#include <optional>

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHSIMPLIFYCOMMUNICATIONDATAFLOWPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

bool haveSameLogicalValue(const LogicalRegionMetadata &first,
                          const LogicalRegionMetadata &second) {
  return first.elementType == second.elementType &&
         first.shape == second.shape;
}

bool haveSameWorkItem(const LogicalRegionMetadata &first,
                      const LogicalRegionMetadata &second) {
  return first.elementType == second.elementType &&
         first.shape.size() == second.shape.size() &&
         std::equal(first.shape.begin() + 1, first.shape.end(),
                    second.shape.begin() + 1, second.shape.end());
}

bool haveSameLayout(const LogicalRegionMetadata &first,
                    const LogicalRegionMetadata &second) {
  return haveSameLogicalValue(first, second) &&
         first.strides == second.strides &&
         first.indexMap == second.indexMap &&
         first.visibility == second.visibility;
}

bool isCommunicationVisible(const LogicalRegionMetadata &metadata) {
  return metadata.visibility == "peer" ||
         metadata.visibility == "global";
}

bool hasOnlyReadUses(Value value) {
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (isa<UseOp, AsyncUseOp>(owner))
      continue;
    if (isa<WorksetOp>(owner))
      continue;
    if (auto stage = dyn_cast<OpaqueStageOp>(owner);
        stage && use.getOperandNumber() < stage.getReads().size())
      continue;
    if (isa<ViewOp>(owner) && use.getOperandNumber() == 0)
      continue;
    if (isa<TransferEachOp>(owner) && use.getOperandNumber() == 1)
      continue;
    if (isa<CopyEachOp>(owner) && use.getOperandNumber() == 1)
      continue;
    if (isa<TransferOp, ReductionOp, CopyOp, TransformOp>(owner) &&
        use.getOperandNumber() == 0)
      continue;
    return false;
  }
  return true;
}

bool consumersFitLifetime(
    Value value,
    const LogicalRegionMetadata &destination) {
  llvm::SmallDenseSet<Value, 8> visited;
  std::function<bool(Value)> check = [&](Value current) {
    if (!visited.insert(current).second)
      return true;
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (auto view = dyn_cast<ViewOp>(user);
          view && use.getOperandNumber() == 0) {
        if (!check(view.getResult()))
          return false;
        continue;
      }
      if (auto view = dyn_cast<IndexedViewOp>(user);
          view && use.getOperandNumber() == 0) {
        if (!check(view.getResult()))
          return false;
        continue;
      }
      if (auto asyncUse = dyn_cast<AsyncUseOp>(user)) {
        if (!asyncUse.getEventOrdered() ||
            asyncUse.getMaxInflight() > destination.poolDepth)
          return false;
        continue;
      }
      if (auto stage = dyn_cast<OpaqueStageOp>(user)) {
        if (use.getOperandNumber() >= stage.getReads().size())
          return false;
        if (stage.getAsynchronous() &&
            (!stage.getEventOrdered() ||
             stage.getMaxInflight() > destination.poolDepth))
          return false;
      }
    }
    return true;
  };
  return check(value);
}

void copyIdentityAttributes(Operation *source, Operation *destination) {
  for (StringRef name : {"id", "outputId"}) {
    if (Attribute attribute = source->getAttr(name))
      destination->setAttr(name, attribute);
  }
}

bool simplifySingletonReductions(CommunicationDataflowOp program) {
  SmallVector<ReductionOp> reductions;
  for (Operation &operation : program.getBody().front()) {
    if (auto reduction = dyn_cast<ReductionOp>(operation);
        reduction && reduction.getCardinality() == 1)
      reductions.push_back(reduction);
  }

  for (ReductionOp reduction : reductions) {
    OpBuilder builder(reduction);
    auto transfer = TransferOp::create(
        builder, reduction.getLoc(), reduction.getResult().getType(),
        reduction.getSource(), reduction.getDestination());
    copyIdentityAttributes(reduction, transfer);
    reduction.getResult().replaceAllUsesWith(transfer.getResult());
    reduction.erase();
  }
  return !reductions.empty();
}

bool forwardFinalDestinations(CommunicationDataflowOp program) {
  bool changed = false;
  SmallVector<CopyOp> copies;
  for (Operation &operation : program.getBody().front()) {
    if (auto copy = dyn_cast<CopyOp>(operation))
      copies.push_back(copy);
  }

  for (CopyOp copy : copies) {
    auto transfer = copy.getSource().getDefiningOp<TransferOp>();
    if (!transfer || !transfer.getResult().hasOneUse() ||
        !transfer.getDestination().hasOneUse() ||
        !copy.getDestination().hasOneUse())
      continue;

    auto source = getLogicalRegionMetadata(transfer.getSource());
    auto staging = getLogicalRegionMetadata(transfer.getDestination());
    auto destination = getLogicalRegionMetadata(copy.getDestination());
    if (!source || !staging || !destination)
      continue;
    if (staging->escapes || destination->immutable ||
        !isCommunicationVisible(*destination) ||
        !haveSameLogicalValue(*source, *destination) ||
        !destination->registered || !destination->donated ||
        destination->reuseInterval < 1 ||
        destination->poolDepth < 1 ||
        !consumersFitLifetime(copy.getResult(), *destination) ||
        (destination->indexed && !destination->injective))
      continue;

    Value finalDestination = copy.getDestination();
    transfer.getDestinationMutable().set(finalDestination);
    if (Attribute outputId = copy->getAttr("outputId"))
      transfer->setAttr("outputId", outputId);
    copy.getResult().replaceAllUsesWith(transfer.getResult());
    copy.erase();
    changed = true;
  }
  return changed;
}

bool forwardWorksetDestinations(CommunicationDataflowOp program) {
  bool changed = false;
  SmallVector<CopyEachOp> copies;
  for (Operation &operation : program.getBody().front()) {
    if (auto copy = dyn_cast<CopyEachOp>(operation))
      copies.push_back(copy);
  }

  for (CopyEachOp copy : copies) {
    auto transfer =
        copy.getSource().getDefiningOp<TransferEachOp>();
    if (!transfer || transfer.getWorkset() != copy.getWorkset() ||
        !transfer.getResult().hasOneUse() ||
        !transfer.getDestination().hasOneUse() ||
        !copy.getDestination().hasOneUse())
      continue;
    auto workset = transfer.getWorkset().getDefiningOp<WorksetOp>();
    if (!workset || !workset.getUniqueDestinations())
      continue;

    auto source = getLogicalRegionMetadata(transfer.getSource());
    auto staging =
        getLogicalRegionMetadata(transfer.getDestination());
    auto destination =
        getLogicalRegionMetadata(copy.getDestination());
    if (!source || !staging || !destination)
      continue;
    if (staging->escapes || destination->immutable ||
        !isCommunicationVisible(*destination) ||
        !haveSameWorkItem(*source, *destination) ||
        !destination->registered || !destination->donated ||
        destination->reuseInterval < 1 ||
        destination->poolDepth < 1 ||
        !consumersFitLifetime(copy.getResult(), *destination))
      continue;

    transfer.getDestinationMutable().set(copy.getDestination());
    if (Attribute outputId = copy->getAttr("outputId"))
      transfer->setAttr("outputId", outputId);
    copy.getResult().replaceAllUsesWith(transfer.getResult());
    copy.erase();
    changed = true;
  }
  return changed;
}

// Permission to bind an output is a declared producer capability. Donation
// gives the final destination to this value's producer; it is not permission
// to reuse an unrelated live allocation. Require the same layout and a unique
// output/copy chain, and preserve all asynchronous consumer obligations.
bool forwardOpaqueDestinations(CommunicationDataflowOp program) {
  bool changed = false;
  SmallVector<CopyOp> copies;
  for (Operation &op : program.getBody().front())
    if (auto copy = dyn_cast<CopyOp>(op))
      copies.push_back(copy);
  for (CopyOp copy : copies) {
    auto result = dyn_cast<OpResult>(copy.getSource());
    auto producer = result ? dyn_cast<OpaqueStageOp>(result.getOwner()) : OpaqueStageOp{};
    if (!producer || !producer.getBindableOutputs().value_or(false) ||
        !result.hasOneUse() || !copy.getDestination().hasOneUse())
      continue;
    Value staging = producer.getWrites()[result.getResultNumber()];
    auto finalRegion = copy.getDestination().getDefiningOp<RegionOp>();
    if (!staging.getDefiningOp<RegionOp>() || !staging.hasOneUse() ||
        !finalRegion || !finalRegion->isBeforeInBlock(producer))
      continue;
    auto source = getLogicalRegionMetadata(staging);
    auto target = getLogicalRegionMetadata(copy.getDestination());
    if (!source || !target || source->escapes || target->immutable ||
        !isCommunicationVisible(*target) || !target->registered || !target->donated ||
        !haveSameLogicalValue(*source, *target) || source->strides != target->strides ||
        source->indexMap != target->indexMap || target->indexed ||
        target->reuseInterval < 1 || target->poolDepth < 1 ||
        !consumersFitLifetime(copy.getResult(), *target) ||
        (producer.getAsynchronous() && (!producer.getEventOrdered() ||
          producer.getMaxInflight() > target->poolDepth)))
      continue;
    producer->setOperand(producer.getReads().size() + result.getResultNumber(),
                         copy.getDestination());
    copy.getResult().replaceAllUsesWith(result);
    copy.erase();
    changed = true;
  }
  return changed;
}

bool aliasImmutableCopies(CommunicationDataflowOp program) {
  bool changed = false;
  SmallVector<CopyOp> copies;
  for (Operation &operation : program.getBody().front()) {
    if (auto copy = dyn_cast<CopyOp>(operation))
      copies.push_back(copy);
  }

  for (CopyOp copy : copies) {
    if (!copy.getSource().getDefiningOp<RegionOp>() ||
        !copy.getDestination().getDefiningOp<RegionOp>())
      continue;
    auto source = getLogicalRegionMetadata(copy.getSource());
    auto destination = getLogicalRegionMetadata(copy.getDestination());
    if (!source || !destination)
      continue;
    if (!source->immutable || source->escapes || destination->escapes ||
        !copy.getDestination().hasOneUse() ||
        !haveSameLayout(*source, *destination) ||
        !hasOnlyReadUses(copy.getResult()))
      continue;

    copy.getResult().replaceAllUsesWith(copy.getSource());
    copy.erase();
    changed = true;
  }
  return changed;
}

bool eliminateDeadDataflow(CommunicationDataflowOp program) {
  bool changed = false;
  bool iterationChanged;
  do {
    iterationChanged = false;
    for (Operation &operation :
         llvm::make_early_inc_range(program.getBody().front())) {
      if (isa<TransferOp, TransferEachOp, ReductionOp, CopyOp, CopyEachOp,
              TransformOp>(operation)) {
        Value result = operation.getResult(0);
        Value destinationValue;
        if (auto transfer = dyn_cast<TransferEachOp>(operation))
          destinationValue = transfer.getDestination();
        else if (auto copy = dyn_cast<CopyEachOp>(operation))
          destinationValue = copy.getDestination();
        else
          destinationValue = operation.getOperand(1);
        auto destination =
            getLogicalRegionMetadata(destinationValue);
        if (result.use_empty() && destination &&
            !destination->escapes) {
          operation.erase();
          iterationChanged = true;
          changed = true;
        }
        continue;
      }

      if (isa<RegionOp, ViewOp, IndexedViewOp>(operation)) {
        Value result = operation.getResult(0);
        auto metadata = getLogicalRegionMetadata(result);
        if (result.use_empty() && metadata && !metadata->escapes) {
          operation.erase();
          iterationChanged = true;
          changed = true;
        }
      }
    }
  } while (iterationChanged);
  return changed;
}

struct SimplifyCommunicationDataflow
    : public impl::AbolethSimplifyCommunicationDataflowPassBase<
          SimplifyCommunicationDataflow> {
  using Base::Base;
  void runOnOperation() override {
    SmallVector<CommunicationDataflowOp> programs;
    getOperation().walk([&](CommunicationDataflowOp program) {
      programs.push_back(program);
    });

    for (CommunicationDataflowOp program : programs) {
      bool changed;
      do {
        changed = simplifySingletonReductions(program);
        changed = forwardFinalDestinations(program) || changed;
        changed = forwardWorksetDestinations(program) || changed;
        if (storageAliases)
          changed = forwardOpaqueDestinations(program) || changed;
        changed = aliasImmutableCopies(program) || changed;
        if (deadCommunication)
          changed = eliminateDeadDataflow(program) || changed;
      } while (changed);
    }
  }
};

} // namespace

} // namespace mlir::aboleth
