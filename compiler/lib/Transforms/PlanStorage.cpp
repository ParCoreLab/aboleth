#include "aboleth/Transforms/Passes.h"
#include "aboleth/IR/LogicalRegions.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <system_error>

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHPLANSTORAGEPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct StorageFacts {
  int64_t rows = 0;
  bool accessed = false;
  StringRef lastReader;
  bool eventOrdered = true;
  int64_t maxInflight = 1;
  bool prefixReadersKnown = true;
};

// A short external input is legal only when every actual reader promises the
// same runtime mask as the declaration. Unknown consumers, writes and views
// keep full storage. This permission concerns immutable input binding, never
// temporal aliasing or the retirement of an asynchronous NIC source read.
bool readsOnlyValidPrefix(Value value, RegionOp root, bool read,
                          Operation *operation) {
  if (!read || getLogicalStorageView(value) != root.getResult())
    return false;
  auto consumer = dyn_cast<OpaqueStageOp>(operation);
  if (!consumer || consumer.getAsynchronous())
    return false;
  auto counts = consumer->getAttrOfType<ArrayAttr>("prefixReadCounts");
  if (!counts || !root.getValidCount())
    return false;
  bool found = false;
  for (auto [index, operand] : llvm::enumerate(consumer.getReads())) {
    if (operand != value)
      continue;
    found = true;
    if (cast<StringAttr>(counts[index]).getValue() != *root.getValidCount())
      return false;
  }
  return found;
}

// Follow an SSA storage state back to its declared view. Only layouts whose
// first dimension is the original row coordinate are narrowed. A permutation
// or an unknown producer retains the complete allocation, never a guessed
// bounding box. All bounds here are exclusive upper bounds.
int64_t reachedRows(Value value, RegionOp root,
                    std::optional<int64_t> limit = std::nullopt) {
  auto metadata = getLogicalRegionMetadata(value);
  if (!metadata)
    return root.getShape()[0];
  Value declaration = getLogicalStorageView(value);
  Operation *op = declaration ? declaration.getDefiningOp() : nullptr;
  if (!op)
    return root.getShape()[0];
  if (isa<RegionOp>(op))
    return std::min<int64_t>(metadata->shape[0], limit.value_or(metadata->shape[0]));
  if (isa<ViewOp, IndexedViewOp>(op)) {
    Value source = op->getOperand(0);
    auto parent = getLogicalRegionMetadata(source);
    if (!parent || !metadata->indexMap.isIdentity() ||
        metadata->strides != parent->strides)
      return root.getShape()[0];
    int64_t upper = metadata->shape[0];
    if (auto indexed = dyn_cast<IndexedViewOp>(op))
      upper = indexed.getIndexUpperBounds()[0];
    else if (limit)
      upper = std::min(upper, *limit);
    return reachedRows(source, root, upper);
  }
  return root.getShape()[0];
}

std::optional<int64_t> fieldUpper(WorksetOp workset, StringRef field) {
  for (auto [index, attr] : llvm::enumerate(workset.getFieldNames()))
    if (cast<StringAttr>(attr).getValue() == field)
      return workset.getFieldUpperBounds()[index];
  return std::nullopt;
}

StringRef stageName(Operation *op) {
  for (StringRef key : {"id", "name"})
    if (auto attr = op->getAttrOfType<StringAttr>(key))
      return attr.getValue();
  return op->getName().getStringRef();
}

std::optional<int64_t> allocationBytes(const LogicalRegionMetadata &metadata,
                                      int64_t rows) {
  if (!isa<IntegerType, FloatType>(metadata.elementType))
    return std::nullopt;
  int64_t elements = 1;
  for (auto [axis, stride] : llvm::enumerate(metadata.strides)) {
    int64_t size = axis == 0 ? rows : metadata.shape[axis];
    if (stride && size - 1 > (std::numeric_limits<int64_t>::max() - elements) / stride)
      return std::nullopt;
    elements += (size - 1) * stride;
  }
  int64_t bytes = (metadata.elementType.getIntOrFloatBitWidth() + 7) / 8;
  if (bytes && elements > std::numeric_limits<int64_t>::max() / bytes)
    return std::nullopt;
  return elements * bytes;
}

LogicalResult planStorage(CommunicationDataflowOp program, bool optimize,
                           llvm::json::Array &allocations) {
  llvm::DenseMap<Value, StorageFacts> facts;
  auto touch = [&](Value value, bool read, Operation *stage,
                   std::optional<int64_t> limit = std::nullopt) {
    struct Access {
      Value value;
      bool read;
      std::optional<int64_t> limit;
    };
    SmallVector<Access> pending{{value, read, limit}};
    while (!pending.empty()) {
      Access access = pending.pop_back_val();
      Value storage = getLogicalStorageRoot(access.value);
      auto root = storage ? storage.getDefiningOp<RegionOp>() : RegionOp{};
      if (!root)
        continue;
      auto &item = facts[storage];
      if (root.getValidCount())
        item.prefixReadersKnown &= readsOnlyValidPrefix(access.value, root, access.read, stage);
      item.accessed = true;
      item.rows = std::max(item.rows, reachedRows(access.value, root, access.limit));
      if (access.read) {
        item.lastReader = stageName(stage);
        // Issuing a communication transfer is not completion of the NIC's
        // source read. Transport completion must resolve that deadline before
        // any later allocation-sharing pass may reuse this storage.
        if (isa<TransferOp, TransferEachOp, ReductionOp>(stage))
          item.eventOrdered = false;
        if (!isa<UseOp, AsyncUseOp, OpaqueStageOp, TransferOp, ReductionOp,
                 CopyOp, TransformOp, TransferEachOp, CopyEachOp, WorksetOp>(stage))
          item.eventOrdered = false;
        if (auto consumer = dyn_cast<AsyncUseOp>(stage)) {
          item.eventOrdered &= consumer.getEventOrdered();
          item.maxInflight = std::max<int64_t>(item.maxInflight, consumer.getMaxInflight());
        }
        if (auto consumer = dyn_cast<OpaqueStageOp>(stage); consumer && consumer.getAsynchronous()) {
          item.eventOrdered &= consumer.getEventOrdered();
          item.maxInflight = std::max<int64_t>(item.maxInflight, consumer.getMaxInflight());
        }
      }
      // Indexed accesses read the index table as well as payload storage.
      // Propagate that implicit effect at the real consumer, so sizing or
      // retiring an index table cannot ignore a later gather through a view.
      Value view = getLogicalStorageView(access.value);
      while (view && !view.getDefiningOp<RegionOp>()) {
        Operation *op = view.getDefiningOp();
        if (auto indexed = dyn_cast<IndexedViewOp>(op))
          pending.push_back({indexed.getIndices(), true, indexed.getShape()[0]});
        view = getLogicalStorageView(op->getOperand(0));
      }
    }
  };

  auto touchWorkset = [&](WorksetOp workset, Operation *consumer,
                          bool bounded = true) {
    for (Value field : workset.getFields())
      touch(field, true, consumer,
            bounded ? std::optional<int64_t>(workset.getMaxItems()) : std::nullopt);
    if (Value count = workset.getCount())
      touch(count, true, consumer);
  };

  for (Operation &op : program.getBody().front()) {
    if (isa<RegionOp, ViewOp, IndexedViewOp>(op))
      continue; // A declaration is not a read of its entire source.
    if (auto transfer = dyn_cast<TransferEachOp>(op)) {
      auto workset = transfer.getWorkset().getDefiningOp<WorksetOp>();
      touch(transfer.getSource(), true, &op, fieldUpper(workset, transfer.getSourceIndexField()));
      touch(transfer.getDestination(), false, &op, fieldUpper(workset, transfer.getDestinationIndexField()));
      touchWorkset(workset, &op);
    } else if (auto copy = dyn_cast<CopyEachOp>(op)) {
      auto workset = copy.getWorkset().getDefiningOp<WorksetOp>();
      touch(copy.getSource(), true, &op, fieldUpper(workset, copy.getSourceIndexField()));
      touch(copy.getDestination(), false, &op, fieldUpper(workset, copy.getDestinationIndexField()));
      touchWorkset(workset, &op);
    } else if (isa<TransferOp, ReductionOp, CopyOp, TransformOp>(op)) {
      touch(op.getOperand(0), true, &op);
      touch(op.getOperand(1), false, &op);
    } else if (auto stage = dyn_cast<OpaqueStageOp>(op)) {
      for (Value read : stage.getReads())
        touch(read, true, &op);
      for (Value write : stage.getWrites())
        touch(write, false, &op);
    } else if (isa<WorksetOp>(op)) {
      continue; // The real transfer/copy consumer reads these fields.
    } else {
      // Includes ordinary/async uses and unknown extensions to the effect
      // vocabulary. Known storage operands remain conservatively observable.
      for (Value operand : op.getOperands()) {
        if (auto workset = operand.getDefiningOp<WorksetOp>())
          touchWorkset(workset, &op, false);
        else
          touch(operand, true, &op);
      }
    }
  }

  for (Operation &op : program.getBody().front()) {
    auto region = dyn_cast<RegionOp>(op);
    if (!region)
      continue;
    auto metadata = getLogicalRegionMetadata(region.getResult());
    auto item = facts.lookup(region.getResult());
    int64_t rows = metadata->shape[0];
    if (optimize && item.accessed && !metadata->escapes)
      rows = std::max<int64_t>(1, std::min(rows, item.rows));
    SmallVector<int64_t> shape(metadata->shape);
    shape[0] = rows;
    region->setAttr("physicalShape", DenseI64ArrayAttr::get(region.getContext(), shape));
    llvm::json::Object allocation;
    allocation["name"] = region.getName();
    allocation["logical_shape"] = llvm::json::Array(metadata->shape);
    allocation["physical_shape"] = llvm::json::Array(shape);
    allocation["strides"] = llvm::json::Array(metadata->strides);
    allocation["pool_depth"] = metadata->poolDepth;
    allocation["last_reader"] = item.lastReader;
    allocation["event_ordered"] = item.eventOrdered;
    allocation["max_inflight"] = item.maxInflight;
    allocation["reader_lifetime_known"] = !item.lastReader.empty() && item.eventOrdered &&
        item.maxInflight <= std::max<int64_t>(1, metadata->poolDepth);
    if (auto bytes = allocationBytes(*metadata, rows))
      allocation["bytes"] = *bytes;
    else
      allocation["bytes"] = nullptr;
    allocation["escaped"] = metadata->escapes;
    if (region.getValidCount()) {
      bool dense = metadata->indexMap.isIdentity();
      int64_t stride = 1;
      for (int axis = metadata->shape.size() - 1; axis >= 0; --axis) {
        dense &= metadata->strides[axis] == stride;
        // The existing verifier and allocationBytes handle representability;
        // avoid overflow even for an unqualified, oversized declaration.
        if (metadata->shape[axis] > std::numeric_limits<int64_t>::max() / stride) {
          dense = false;
          break;
        }
        stride *= metadata->shape[axis];
      }
      auto rowBytes = allocationBytes(*metadata, 1);
      bool legal = optimize && item.accessed && item.prefixReadersKnown && dense && rowBytes &&
          metadata->immutable && !metadata->escapes && !metadata->registered &&
          region.getDeadPadding().value_or(false);
      if (legal) {
        region->setAttr("prefixBinding", region.getValidCountAttr());
        llvm::json::Object binding;
        binding["count"] = *region.getValidCount();
        binding["capacity"] = metadata->shape[0];
        binding["row_bytes"] = *rowBytes;
        allocation["prefix_binding"] = std::move(binding);
      } else {
        region->removeAttr("prefixBinding");
        allocation["prefix_binding"] = nullptr;
      }
    }
    allocations.push_back(std::move(allocation));
  }
  return success();
}

class PlanStorage : public impl::AbolethPlanStoragePassBase<PlanStorage> {
public:
  using Base::Base;
  void runOnOperation() override {
    llvm::json::Array programs;
    getOperation().walk([&](CommunicationDataflowOp program) {
      llvm::json::Array allocations;
      if (failed(planStorage(program, optimizeBounds, allocations))) {
        signalPassFailure();
        return;
      }
      llvm::json::Object facts;
      facts["name"] = program.getName();
      facts["allocations"] = std::move(allocations);
      programs.push_back(std::move(facts));
    });
    if (!emitPlanJson.empty()) {
      std::error_code error;
      llvm::raw_fd_ostream stream(emitPlanJson, error);
      if (error) {
        getOperation().emitError("cannot write storage plan: ") << error.message();
        return signalPassFailure();
      }
      llvm::json::Object document;
      document["programs"] = std::move(programs);
      llvm::json::OStream json(stream, 2);
      json.value(llvm::json::Value(std::move(document)));
      stream << '\n';
      stream.close();
      if (stream.has_error()) {
        getOperation().emitError("failed to write storage plan");
        signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::aboleth
