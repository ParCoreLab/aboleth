#include "aboleth/IR/Dialect.h"
#include "aboleth/IR/LogicalRegions.h"

#include "aboleth/Target/RdmaAbi.h"
#include "aboleth/Target/TransportCaps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <limits>
#include <optional>

#include "aboleth/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::aboleth;

bool mlir::aboleth::areEquivalentEndpoints(Value first, Value second) {
  if (first == second)
    return true;
  if (auto firstSelect = first.getDefiningOp<EndpointOp>()) {
    auto secondSelect = second.getDefiningOp<EndpointOp>();
    return secondSelect && firstSelect.getTable() == secondSelect.getTable() &&
           firstSelect.getIndex() == secondSelect.getIndex();
  }
  auto firstBind = first.getDefiningOp<BindEndpointOp>();
  auto secondBind = second.getDefiningOp<BindEndpointOp>();
  return firstBind && secondBind &&
         firstBind.getValue() == secondBind.getValue();
}

void AbolethDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aboleth/IR/Types.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aboleth/IR/Ops.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "aboleth/IR/Types.cpp.inc"

#define GET_OP_CLASSES
#include "aboleth/IR/Ops.cpp.inc"

namespace {

// The endpoint a semantic payload token was produced on, or a null Value when
// the token came from anything but aboleth.put or aboleth.put_inline.
Value semanticPayloadEndpoint(Value token) {
  Operation *producer = token.getDefiningOp();
  if (auto put = dyn_cast_or_null<PutOp>(producer))
    return put.getEndpoint();
  if (auto inlinePut = dyn_cast_or_null<PutInlineOp>(producer))
    return inlinePut.getEndpoint();
  return {};
}

// The same for Queue IR payload commands.
Value postedPayloadEndpoint(Value token) {
  Operation *producer = token.getDefiningOp();
  if (auto put = dyn_cast_or_null<PostPutOp>(producer))
    return put.getEndpoint();
  if (auto inlinePut = dyn_cast_or_null<PostInlineOp>(producer))
    return inlinePut.getEndpoint();
  return {};
}

// An ordered dependency names a payload command on the same endpoint. Both
// payload forms — region-sourced and inline — are valid predecessors, so the
// two spellings share one check rather than each op naming only its own kind.
LogicalResult verifyPayloadDependency(Operation *op, Value after,
                                      Value endpoint, bool posted) {
  Value producerEndpoint = posted ? postedPayloadEndpoint(after)
                                  : semanticPayloadEndpoint(after);
  if (!producerEndpoint)
    return op->emitOpError(
        posted ? "dependency must be produced by aboleth.post_put or "
                 "aboleth.post_inline"
               : "dependency must be produced by aboleth.put or "
                 "aboleth.put_inline");
  if (!areEquivalentEndpoints(producerEndpoint, endpoint))
    return op->emitOpError("dependency must use the same endpoint");
  return success();
}

// Widths the payload packer places into 32-bit words: one byte, two, three,
// four, and eight. A wider or intermediate integer (i40, i48, i56, i128) has
// no packing this dialect defines — the packer would have to extend or split
// it — so the verifier refuses it here rather than letting a backend emit
// malformed arithmetic. A client with such a value splits it itself, into the
// byte widths above, in the order it wants on the wire.
bool isPackableInlineWidth(unsigned width) {
  return width == 8 || width == 16 || width == 24 || width == 32 ||
         width == 64;
}

// Inline payload operands must be a whole number of bytes wide: the payload is
// the concatenation of their little-endian representations, which is only
// defined for byte-sized values. The total is bounded by what one WQE can
// carry (see kMaxInlinePayloadBytes) — the tightest bound across the
// transports that have an inline-payload command at all.
LogicalResult verifyInlinePayload(Operation *op, ValueRange values) {
  uint64_t bytes = 0;
  for (auto [index, value] : llvm::enumerate(values)) {
    auto type = dyn_cast<IntegerType>(value.getType());
    if (!type || type.getWidth() == 0 || type.getWidth() % 8 != 0)
      return op->emitOpError("inline payload value ")
             << index << " must be a byte-sized integer, not "
             << value.getType();
    if (!isPackableInlineWidth(type.getWidth()))
      return op->emitOpError("inline payload value ")
             << index << " has width " << type.getWidth()
             << ", which has no defined inline packing; an inline payload "
                "carries i8, i16, i24, i32, or i64 values";
    bytes += type.getWidth() / 8;
  }
  if (bytes == 0)
    return op->emitOpError("requires at least one inline payload value");
  if (bytes > ::aboleth::abi::kMaxInlinePayloadBytes)
    return op->emitOpError("inline payload of ")
           << bytes << " bytes exceeds the "
           << ::aboleth::abi::kMaxInlinePayloadBytes
           << "-byte maximum one work queue entry can carry (a 6-bit ds "
              "field spans at most "
           << ::aboleth::abi::kWqeMaxDataSegments << " 16-byte segments)";
  return success();
}

// The payload byte count once the verifier has accepted the operands.
uint64_t payloadBytes(ValueRange values) {
  uint64_t bytes = 0;
  for (Value value : values)
    bytes += cast<IntegerType>(value.getType()).getWidth() / 8;
  return bytes;
}

} // namespace

uint64_t PutInlineOp::getPayloadBytes() { return payloadBytes(getValues()); }

uint64_t PostInlineOp::getPayloadBytes() { return payloadBytes(getValues()); }

LogicalResult ValidExtentOp::verify() {
  if (getCapacity() < 1)
    return emitOpError("capacity must be positive");
  return success();
}

LogicalResult BindValidLocalRegionOp::verify() {
  if (getCapacity() < 1 || getItemBytes() < 1 ||
      getCapacity() > std::numeric_limits<int32_t>::max() / getItemBytes())
    return emitOpError("capacity times itemBytes must fit positive i32 transfer bytes");
  for (Operation *user : getRegion().getUsers()) {
    auto put = dyn_cast<PutOp>(user);
    if (!put || put.getSource() != getRegion())
      return emitOpError("bounded local regions may only be sources of semantic puts");
  }
  return success();
}

LogicalResult EndpointOp::verify() {
  auto axis = getExclusiveAxis();
  auto axes = getExclusiveAxesAttr();
  auto grid = getExclusiveGridShapeAttr();
  if (axis && *axis > 2)
    return emitOpError("exclusiveAxis must name a grid axis in [0, 2]");
  if (axis && axes)
    return emitOpError("cannot combine exclusiveAxis and exclusiveAxes");
  if (axes) {
    if (axes.size() == 0 || axes.size() > 2)
      return emitOpError("exclusiveAxes requires one or two distinct grid axes");
    llvm::SmallDenseSet<int32_t, 2> seen;
    for (int32_t value : axes.asArrayRef())
      if (value < 0 || value > 2 || !seen.insert(value).second)
        return emitOpError("exclusiveAxes requires distinct grid axes in [0, 2]");
  }
  if (grid) {
    if ((!axis && !axes) || grid.size() != 3)
      return emitOpError("exclusiveGridShape requires an issuer proof and three extents");
    for (int64_t size : grid.asArrayRef())
      if (size < 0 || size > std::numeric_limits<int32_t>::max())
        return emitOpError("exclusiveGridShape extents must fit nonnegative i32");
    for (int dimension = 0; dimension < 3; ++dimension) {
      bool selected = axis && *axis == static_cast<unsigned>(dimension);
      if (axes)
        selected = llvm::is_contained(axes.asArrayRef(), dimension);
      if (!selected && grid[dimension] > 1)
        return emitOpError("an unselected exclusive grid axis must have extent one");
    }
  }
  return success();
}

LogicalResult PutOp::verify() {
  Value after = getAfter();
  if (!after)
    return success();
  return verifyPayloadDependency(
      getOperation(), after, getEndpoint(), /*posted=*/false);
}

LogicalResult PutInlineOp::verify() {
  if (failed(verifyInlinePayload(getOperation(), getValues())))
    return failure();
  Value after = getAfter();
  if (!after)
    return success();
  return verifyPayloadDependency(
      getOperation(), after, getEndpoint(), /*posted=*/false);
}

LogicalResult SignalOp::verify() {
  Value after = getAfter();
  if (getWeakPublication() && (after || getGroup()))
    return emitOpError(
        "weak publication is only valid for an ungrouped signal-only batch");
  if (getFencedSignal() && getWeakPublication())
    return emitOpError(
        "a fenced signal only applies to a strong publication");
  if (getCompletedPrefix() && (getWeakPublication() || getFencedSignal()))
    return emitOpError("completed-prefix publication needs its own strong ordering point");
  if (after &&
      failed(verifyPayloadDependency(
          getOperation(), after, getEndpoint(), /*posted=*/false)))
    return failure();
  Value group = getGroup();
  if (group) {
    auto begin = group.getDefiningOp<BeginGroupOp>();
    if (!begin)
      return emitOpError("group must be produced by aboleth.begin_group");
    if (!areEquivalentEndpoints(begin.getEndpoint(), getEndpoint()))
      return emitOpError("group must use the same endpoint");
  }
  return success();
}

LogicalResult WaitOp::verify() {
  Operation *producer = getTransfer().getDefiningOp();
  auto signal = dyn_cast_or_null<SignalOp>(producer);
  if (
      !signal
      && !isa_and_nonnull<PutOp, PutInlineOp>(
          producer
      )
  )
    return emitOpError(
        "requires a token produced by aboleth.put, "
        "aboleth.put_inline, or aboleth.signal"
    );
  if (signal && signal.getGroup())
    return emitOpError(
        "grouped signals must be completed by aboleth.wait_group");
  if (getSelfReclaim() && !getPredicate())
    return emitOpError("selfReclaim requires a predicate");
  return success();
}

LogicalResult PlannedWaitOp::verify() {
  auto signal = getTransfer().getDefiningOp<SignalOp>();
  if (!signal)
    return emitOpError("requires a token produced by aboleth.signal");
  if (signal.getGroup())
    return emitOpError(
        "grouped signals cannot use aboleth.planned_wait");
  if (static_cast<bool>(getIterations()) == getReuseDistance().has_value())
    return emitOpError(
        "requires either a final iteration or a reuse distance");
  if (auto distance = getReuseDistance(); distance && *distance < 1)
    return emitOpError("reuse distance must be positive");
  if (auto interval = getCompletionInterval()) {
    if (*interval < 1)
      return emitOpError("completion interval must be positive");
    if (getReuseDistance())
      return emitOpError("a phase interval cannot replace declared reuse planning");
  }
  if (auto capacity = getQueueCapacity(); capacity && *capacity < 1)
    return emitOpError("queue capacity must be positive");
  if (getIterations() && getIteration().getType() != getIterations().getType())
    return emitOpError(
        "iteration and iterations must have the same integer type");
  return success();
}

LogicalResult WaitGroupOp::verify() {
  auto begin = getGroup().getDefiningOp<BeginGroupOp>();
  if (!begin)
    return emitOpError("requires a group produced by aboleth.begin_group");
  for (Operation *user : getGroup().getUsers()) {
    auto otherWait = dyn_cast<WaitGroupOp>(user);
    if (otherWait && otherWait.getOperation() != getOperation())
      return emitOpError("completion group cannot be consumed by multiple waits");
  }
  Operation *root = getOperation();
  while (root->getParentOp() && !isa<ModuleOp>(root->getParentOp()))
    root = root->getParentOp();
  PostDominanceInfo postDominance(root);
  for (Operation *user : getGroup().getUsers()) {
    if (user == getOperation())
      continue;
    if (!postDominance.postDominates(getOperation(), user))
      return emitOpError(
          "must post-dominate every submission in its completion group");
  }
  return success();
}

bool PostSignalOp::isOrderedByPrevious() {
  auto previous = getAfter() ? getAfter().getDefiningOp<PostSignalOp>()
                             : PostSignalOp{};
  return previous && !previous.getWeakPublication() &&
         (!previous.getPredicate() ||
          previous.getPredicate() == getPredicate());
}

bool PostSignalOp::isChecked() {
  return !getWeakPublication() && !getFencedSignal() &&
         !getCompletedPrefix() && !isOrderedByPrevious();
}

LogicalResult PostSignalOp::verify() {
  Value after = getAfter();
  if (getWeakPublication() && (after || getGroup()))
    return emitOpError(
        "weak publication is only valid for an ungrouped signal-only batch");
  if (getFencedSignal() && getWeakPublication())
    return emitOpError(
        "a fenced signal only applies to a strong publication");
  if (getCompletedPrefix() && (getWeakPublication() || getFencedSignal()))
    return emitOpError("completed-prefix publication needs its own strong ordering point");
  if (getExpected() && !isChecked())
    return emitOpError("only a checked publication compares an expected value");
  if (getPredicate() && isChecked() && !getExpected())
    return emitOpError(
        "a predicated checked signal needs the value it replaces: after an "
        "elided epoch the cell no longer holds epoch - 1");
  // A strong signal may also follow another strong signal on its endpoint,
  // which then orders the payload for both (aboleth-order-publications).
  if (auto previous =
          after ? after.getDefiningOp<PostSignalOp>() : PostSignalOp{}) {
    if (getWeakPublication() || !isOrderedByPrevious())
      return emitOpError(
          "only a strong signal may follow, and be ordered by, a signal");
    if (!areEquivalentEndpoints(previous.getEndpoint(), getEndpoint()))
      return emitOpError("dependency must use the same endpoint");
  } else if (after &&
             failed(verifyPayloadDependency(
                 getOperation(), after, getEndpoint(), /*posted=*/true))) {
    return failure();
  }
  Value group = getGroup();
  if (group) {
    auto begin = group.getDefiningOp<BeginGroupOp>();
    if (!begin)
      return emitOpError("group must be produced by aboleth.begin_group");
    if (!areEquivalentEndpoints(begin.getEndpoint(), getEndpoint()))
      return emitOpError("group must use the same endpoint");
  }
  return success();
}

LogicalResult PostPutOp::verify() {
  Value after = getAfter();
  if (!after)
    return success();
  return verifyPayloadDependency(
      getOperation(), after, getEndpoint(), /*posted=*/true);
}

LogicalResult PostInlineOp::verify() {
  if (failed(verifyInlinePayload(getOperation(), getValues())))
    return failure();
  Value after = getAfter();
  if (!after)
    return success();
  return verifyPayloadDependency(
      getOperation(), after, getEndpoint(), /*posted=*/true);
}

LogicalResult PostNopOp::verify() {
  Value after = getAfter();
  if (!after)
    return success();
  Operation *producer = after.getDefiningOp();
  Value endpoint = postedPayloadEndpoint(after);
  if (!endpoint) {
    if (auto signal = dyn_cast_or_null<PostSignalOp>(producer))
      endpoint = signal.getEndpoint();
    else
      return emitOpError(
          "dependency must be produced by aboleth.post_put, "
          "aboleth.post_inline, or aboleth.post_signal");
  }
  if (!areEquivalentEndpoints(endpoint, getEndpoint()))
    return emitOpError("dependency must use the same endpoint");
  return success();
}

LogicalResult SubmitOp::verify() {
  const auto transferCount = getTransfers().size();
  if (transferCount == 0)
    return emitOpError("requires at least one posted transfer");
  if (getBatchSize() != transferCount)
    return emitOpError("batchSize must equal the number of transfer tokens");
  if (getDeferCompletion() && getGroup())
    return emitOpError(
        "deferred completion cannot use an in-kernel completion group");
  if (getDeferCompletion()) {
    for (Operation *user : getToken().getUsers()) {
      if (isa<PollOp>(user))
        return emitOpError(
            "deferred completion cannot have an in-kernel poll");
    }
  }
  if (auto scope = getReservationScope();
      scope && *scope != "gpu" && *scope != "cta")
    return emitOpError("reservationScope must be gpu or cta");
  if (auto scope = getIssuerScope();
      scope && *scope != "cta" && *scope != "thread")
    return emitOpError("issuerScope must be cta or thread");
  if (getDeferredAnnouncement()) {
    if (getGroup() || getDeferCompletion() || getExternalPublication() ||
        getHeadPublication())
      return emitOpError(
          "deferredAnnouncement needs an ungrouped, polled submission on the "
          "plain publication path");
    // A poll that could be due while its batch is still unannounced would
    // wait forever, so the announcement follows the poll's predicate.
    bool predicated = false;
    for (Operation *user : getToken().getUsers())
      if (auto poll = dyn_cast<PollOp>(user))
        predicated = static_cast<bool>(poll.getPredicate());
    if (!predicated)
      return emitOpError(
          "deferredAnnouncement requires a predicated poll to announce with");
  }
  if (getExternalPublication()) {
    if (!getDeferCompletion())
      return emitOpError(
          "externalPublication requires deferCompletion: the enclosing "
          "integration owns both obligations");
    if (getHeadPublication())
      return emitOpError(
          "externalPublication and headPublication are alternate publication "
          "contracts and cannot combine");
  }

  // A submission may end with one aboleth.post_nop. It carries the batch's
  // completion so no payload or signal command has to request one, which also
  // means the batch needs a poll that consumes it unconditionally.
  const bool carriesCompletionNop =
      isa_and_nonnull<PostNopOp>(getTransfers().back().getDefiningOp());
  const bool payloadCompletedByPoll =
      llvm::any_of(
          getToken().getUsers(),
          [](Operation *user) {
            return isa<PollOp>(user);
          }
      );
  if (carriesCompletionNop) {
    if (getGroup() || getDeferCompletion())
      return emitOpError(
          "a completion no-op requires an ungrouped, non-deferred submission");
    for (Operation *user : getToken().getUsers()) {
      auto poll = dyn_cast<PollOp>(user);
      if (poll && poll.getPredicate())
        return emitOpError(
            "a completion no-op requires an unpredicated poll");
    }
  }
  // Both payload forms follow the same batch rules, so the walk below asks
  // "is this a payload command" rather than naming one op.
  const auto payloadAfter = [](Value transfer) -> std::optional<Value> {
    Operation *producer = transfer.getDefiningOp();
    if (auto put = dyn_cast_or_null<PostPutOp>(producer))
      return put.getAfter();
    if (auto inlinePut = dyn_cast_or_null<PostInlineOp>(producer))
      return inlinePut.getAfter();
    return std::nullopt;
  };
  bool hasExplicitPutChain = llvm::any_of(getTransfers(), [&](Value transfer) {
    auto after = payloadAfter(transfer);
    return after && static_cast<bool>(*after);
  });
  // A submission is a sequence of segments, each an independent chain that a
  // signal may close. Within a chain every command depends on the one before
  // it; a segment's first command depends on nothing in the batch.
  std::size_t segmentStart = 0;
  llvm::SmallDenseSet<Value, 8> seen;
  for (auto [index, transfer] : llvm::enumerate(getTransfers())) {
    if (!seen.insert(transfer).second)
      return emitOpError("cannot submit the same transfer token twice");
    for (Operation *user : transfer.getUsers()) {
      auto otherSubmit = dyn_cast<SubmitOp>(user);
      if (otherSubmit && otherSubmit.getOperation() != getOperation())
        return emitOpError(
            "posted transfer token cannot belong to multiple submissions");
    }

    Operation *producer = transfer.getDefiningOp();
    if (auto nop = dyn_cast_or_null<PostNopOp>(producer)) {
      if (index + 1 != transferCount)
        return emitOpError("post_nop must be the final transfer");
      if (!areEquivalentEndpoints(nop.getEndpoint(), getEndpoint()))
        return emitOpError("all transfers must use the submit endpoint");
      if (index == 0) {
        if (nop.getAfter())
          return emitOpError(
              "completion-only submission cannot have a dependency");
      } else if (nop.getAfter() != getTransfers()[index - 1]) {
        return emitOpError(
            "completion no-op must depend on the preceding posted transfer");
      }
      continue;
    }

    if (auto after = payloadAfter(transfer)) {
      // Deferred completion is what makes a payload-final batch honest: no
      // command carries a CQE, so a later drain (or the enclosing external
      // integration) owns retirement. A self-publishing deferred batch is the
      // Tier-1 put; external publication additionally hands off the doorbell.
      if (!carriesCompletionNop
          && !getDeferCompletion()
          && !payloadCompletedByPoll
          &&
          index + 1 == transferCount)
        return emitOpError(
            "final transfer must be aboleth.post_signal unless completion "
            "is explicitly deferred");
      if (!areEquivalentEndpoints(
              postedPayloadEndpoint(transfer), getEndpoint()))
        return emitOpError("all transfers must use the submit endpoint");
      if (hasExplicitPutChain) {
        if (index == segmentStart) {
          if (*after)
            return emitOpError(
                "first posted transfer of a segment cannot have a batch "
                "dependency");
        } else if (*after != getTransfers()[index - 1]) {
          return emitOpError(
              "posted puts must follow submit operand order");
        }
      }
      continue;
    }

    auto signal = dyn_cast_or_null<PostSignalOp>(producer);
    if (!signal)
      return emitOpError(
          "transfer tokens must be produced by post_put, post_inline, or "
          "post_signal");
    if (!areEquivalentEndpoints(signal.getEndpoint(), getEndpoint()))
      return emitOpError("all transfers must use the submit endpoint");
    if (signal.getGroup() != getGroup())
      return emitOpError(
          "submission and its signals must use the same completion group");
    // A signal may also follow the signal before it: one strong publication
    // then orders the payload of every signal chained after it.
    const bool followsSignal =
        index > 0 && signal.getAfter() == getTransfers()[index - 1] &&
        isa_and_nonnull<PostSignalOp>(
            getTransfers()[index - 1].getDefiningOp());
    if (index == segmentStart) {
      if (signal.getAfter() && !followsSignal)
        return emitOpError("signal-only segment cannot have a dependency");
    } else if (signal.getAfter() != getTransfers()[index - 1]) {
      return emitOpError(
          "a signal must depend on the preceding posted transfer");
    }
    segmentStart = index + 1;
  }
  return success();
}

LogicalResult PollOp::verify() {
  auto submit = getSubmission().getDefiningOp<SubmitOp>();
  if (!submit)
    return emitOpError("requires a token produced by aboleth.submit");
  if (submit.getGroup())
    return emitOpError(
        "grouped submissions must be completed by aboleth.wait_group");
  for (Operation *user : getSubmission().getUsers()) {
    auto otherPoll = dyn_cast<PollOp>(user);
    if (otherPoll && otherPoll.getOperation() != getOperation())
      return emitOpError(
          "submission token cannot be consumed by multiple polls");
  }
  if (getSelfReclaim() && !getPredicate())
    return emitOpError("selfReclaim requires a predicate");
  return success();
}

namespace {

LogicalResult verifyRegionArray(Operation *operation, StringRef name) {
  auto values = operation->getAttrOfType<ArrayAttr>(name);
  if (!values)
    return operation->emitOpError("requires ") << name;
  for (Attribute value : values) {
    auto region = dyn_cast<DictionaryAttr>(value);
    if (!region)
      return operation->emitOpError(name)
             << " must contain only region dictionaries";
    auto unknownAttr = region.getAs<BoolAttr>("unknown");
    const bool unknown = unknownAttr && unknownAttr.getValue();
    auto allocation = region.getAs<IntegerAttr>("allocation");
    auto generation = region.getAs<IntegerAttr>("generation");
    auto offset = region.getAs<IntegerAttr>("offset");
    auto length = region.getAs<IntegerAttr>("length");
    auto stride = region.getAs<IntegerAttr>("stride");
    auto period = region.getAs<IntegerAttr>("period");
    auto count = region.getAs<IntegerAttr>("count");
    if (!allocation || !generation || !offset || !length ||
        !stride || !period || !count)
      return operation->emitOpError(name)
             << " region requires allocation, generation, offset, "
                "length, stride, period, and count";
    if (!unknown &&
        (allocation.getInt() < 0 || generation.getInt() < 0))
      return operation->emitOpError(name)
             << " region allocation and generation cannot be negative";
    if (offset.getInt() < 0 || length.getInt() <= 0 ||
        stride.getInt() < 0 || period.getInt() < 0 ||
        count.getInt() <= 0)
      return operation->emitOpError(name)
             << " region offset/stride must be non-negative and "
                "length/count must be positive";
  }
  return success();
}

LogicalResult verifyTransportEffect(Operation *operation) {
  auto name = operation->getAttrOfType<StringAttr>("name");
  if (!name || name.getValue().empty())
    return operation->emitOpError("requires a non-empty name");
  if (failed(verifyRegionArray(operation, "reads")) ||
      failed(verifyRegionArray(operation, "writes")))
    return failure();
  if (auto kernel = operation->getAttrOfType<StringAttr>("boundKernel");
      kernel && kernel.getValue().empty())
    return operation->emitOpError("boundKernel cannot be empty");
  return success();
}

} // namespace

LogicalResult TransportProgramOp::verify() {
  auto name = (*this)->getAttrOfType<StringAttr>("name");
  if (!name || name.getValue().empty())
    return emitOpError("requires a non-empty name");
  auto ranks = getPeerRanksAttr();
  auto transports = getPeerTransportsAttr();
  if (bool(ranks) != bool(transports))
    return emitOpError("peerRanks and peerTransports must be supplied together");
  if (ranks) {
    if (ranks.empty() || ranks.size() != transports.size())
      return emitOpError("requires one transport for each nonempty peer rank");
    int32_t previous = -1;
    for (auto [rank, attribute] : llvm::zip_equal(ranks.asArrayRef(), transports)) {
      if (rank <= previous)
        return emitOpError("peer ranks must be sorted, unique and nonnegative");
      previous = rank;
      auto transport = dyn_cast<StringAttr>(attribute);
      if (!transport || !::aboleth::abi::lookup_transport_caps(
                            transport.getValue().data(), transport.getValue().size()))
        return emitOpError("unknown transport in peer geography");
    }
  }
  Region &body = getBody();
  if (!llvm::hasSingleElement(body))
    return emitOpError("requires one body block");
  for (Operation &operation : body.front()) {
    if (operation.getName().getStringRef() == "ub.unreachable") {
      if (&operation != &body.front().back())
        return emitOpError("ub.unreachable must terminate the body");
      continue;
    }
    if (!isa<TransportFragmentOp, TransportKernelOp, TransportPhaseOp>(
            operation))
      return emitOpError("contains unsupported body operation ")
             << operation.getName();
  }
  return success();
}

LogicalResult TransportFragmentOp::verify() {
  if (getProgramId() < 0)
    return emitOpError("programId cannot be negative");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires one body block");

  for (Operation &operation : *getOperation()->getBlock()) {
    auto fragment = dyn_cast<TransportFragmentOp>(operation);
    if (fragment && fragment != *this &&
        fragment.getProgramId() == getProgramId())
      return emitOpError("programId must be unique within its transport program");
  }

  // A fragment is cloned into a kernel out of a plan module that is then
  // destroyed, so nothing in its body may name a value defined outside it:
  // the clone would carry a dangling operand. Every input arrives as a block
  // argument instead. (This is the IsolatedFromAbove contract, proved here
  // because the trait itself cannot cross the plugin boundary; see the op's
  // description.)
  Region &body = getBody();
  LogicalResult captured = success();
  body.walk([&](Operation *operation) {
    for (Value operand : operation->getOperands()) {
      Region *defining = operand.getParentRegion();
      if (defining && body.isAncestor(defining))
        continue;
      captured = operation->emitOpError(
          "uses a value defined outside its transport fragment; a fragment "
          "takes every input as a block argument, because it is cloned out "
          "of a plan module that does not outlive the compilation");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (failed(captured))
    return failure();

  ArrayAttr argumentNames = getArgumentNames();
  if (argumentNames.size() != getBody().front().getNumArguments())
    return emitOpError("requires one argument name per block argument");
  llvm::DenseSet<StringRef> names;
  for (Attribute value : argumentNames) {
    auto name = dyn_cast<StringAttr>(value);
    if (!name || name.getValue().empty())
      return emitOpError(
          "argumentNames must contain non-empty strings");
    if (!names.insert(name.getValue()).second)
      return emitOpError("argument names must be unique");
  }
  return success();
}

LogicalResult InvokeTransportPhaseOp::verify() {
  if (!isa<IntegerType>(getProgramId().getType()))
    return emitOpError("programId must be an integer");
  if (getArguments().empty())
    return emitOpError("requires at least one argument");
  return success();
}

LogicalResult TransportKernelOp::verify() {
  if (failed(verifyTransportEffect(getOperation())))
    return failure();
  const bool asynchronous =
      getAsynchronousAttr() && getAsynchronousAttr().getValue();
  const bool eventOrdered =
      getEventOrderedAttr() && getEventOrderedAttr().getValue();
  if (auto maxInflight = getMaxInflightAttr();
      maxInflight && maxInflight.getInt() <= 0)
    return emitOpError("maxInflight must be positive");
  if (asynchronous && !getMaxInflightAttr())
    return emitOpError(
        "asynchronous kernels require maxInflight");
  if (asynchronous && getMaxInflightAttr().getInt() > 1 &&
      !eventOrdered)
    return emitOpError(
        "multi-generation asynchronous kernels require event ordering");
  return success();
}

LogicalResult TransportPhaseOp::verify() {
  if (failed(verifyTransportEffect(getOperation())))
    return failure();
  if (getCommandsPerIteration() <= 0)
    return emitOpError("commandsPerIteration must be positive");
  if (getIterationCount() <= 0)
    return emitOpError("iterationCount must be positive");
  if (getPayloadBytes() < 0)
    return emitOpError("payloadBytes cannot be negative");
  if (auto id = getProgramId(); id && *id < 0)
    return emitOpError("programId cannot be negative");

  if (auto interval = getCompletionIntervalAttr();
      interval && interval.getInt() <= 0)
    return emitOpError("completionInterval must be positive");
  auto transports = getPeerTransportsAttr();
  auto capacities = getPeerCapacitiesAttr();
  if (transports.size() != capacities.size())
    return emitOpError(
        "requires one endpoint capacity per peer transport");
  for (Attribute transport : transports) {
    auto name = dyn_cast<StringAttr>(transport);
    if (!name)
      return emitOpError("peer transports must be strings");
    if (!::aboleth::abi::lookup_transport_caps(name.getValue().data(),
                                    name.getValue().size()))
      return emitOpError("unknown peer transport '")
             << name.getValue() << "'";
  }

  const bool hasTiny =
      static_cast<bool>(getPolicyTinyPayloadBytesAttr());
  if (!getCompletionIntervalAttr() && !hasTiny)
    return emitOpError(
        "automatic completion requires policy metadata");
  if (auto tiny = getPolicyTinyPayloadBytesAttr();
      tiny && tiny.getInt() < 0)
    return emitOpError(
        "policyTinyPayloadBytes cannot be negative");
  if (auto intervals = getPeerCompletionIntervalsAttr()) {
    if (intervals.size() != transports.size())
      return emitOpError(
          "requires one completion interval per peer transport");
    if (llvm::any_of(intervals.asArrayRef(),
                     [](int32_t interval) { return interval < 0; }))
      return emitOpError(
          "peer completion intervals cannot be negative");
  }

  auto resolved = getResolvedCompletionIntervalAttr();
  auto reuse = getReuseDistanceAttr();
  if (static_cast<bool>(resolved) != static_cast<bool>(reuse))
    return emitOpError(
        "resolved completion and reuse distance must appear together");
  if (resolved &&
      (resolved.getInt() <= 0 || reuse.getInt() <= 0 ||
       resolved.getInt() > reuse.getInt() ||
       reuse.getInt() > getIterationCount()))
    return emitOpError("resolved completion exceeds its lifetime");
  if (auto before = getCompletionBeforeAttr();
      before && before.getValue().empty())
    return emitOpError("completionBefore cannot be empty");
  return success();
}

namespace {

LogicalResult verifyLogicalLayout(Operation *operation,
                                  unsigned expectedMapResults) {
  auto name = operation->getAttrOfType<StringAttr>("name");
  if (!name || name.getValue().empty())
    return operation->emitOpError("requires a non-empty name");

  auto elementType = operation->getAttrOfType<TypeAttr>("elementType");
  if (!elementType || isa<NoneType>(elementType.getValue()))
    return operation->emitOpError("requires a concrete element type");

  auto shape = operation->getAttrOfType<DenseI64ArrayAttr>("shape");
  auto strides = operation->getAttrOfType<DenseI64ArrayAttr>("strides");
  if (!shape || shape.empty())
    return operation->emitOpError("shape must have positive rank");
  if (!strides || strides.size() != shape.size())
    return operation->emitOpError(
        "requires one stride for every shape dimension");
  if (llvm::any_of(shape.asArrayRef(),
                   [](int64_t extent) { return extent <= 0; }))
    return operation->emitOpError("shape extents must be positive");
  if (llvm::any_of(strides.asArrayRef(),
                   [](int64_t stride) { return stride <= 0; }))
    return operation->emitOpError("strides must be positive");
  SmallVector<std::pair<uint64_t, uint64_t>> dimensions;
  for (auto [extent, stride] :
       llvm::zip(shape.asArrayRef(), strides.asArrayRef()))
    dimensions.emplace_back(
        static_cast<uint64_t>(stride),
        static_cast<uint64_t>(extent));
  llvm::sort(dimensions);
  uint64_t occupiedSpan = 1;
  for (auto [stride, extent] : dimensions) {
    if (extent <= 1)
      continue;
    if (stride < occupiedSpan)
      return operation->emitOpError(
          "strides describe an overlapping logical layout");
    if (extent - 1 >
        (std::numeric_limits<uint64_t>::max() - occupiedSpan) / stride)
      return operation->emitOpError("logical layout span overflows");
    occupiedSpan += (extent - 1) * stride;
  }

  auto indexMap = operation->getAttrOfType<AffineMapAttr>("indexMap");
  if (!indexMap)
    return operation->emitOpError("requires an indexMap");
  AffineMap map = indexMap.getValue();
  if (map.getNumSymbols() != 0)
    return operation->emitOpError("indexMap cannot contain symbols");
  if (map.getNumDims() != shape.size())
    return operation->emitOpError(
        "indexMap input rank must equal the logical shape rank");
  if (map.getNumResults() != expectedMapResults)
    return operation->emitOpError(
        "indexMap result rank is incompatible with its storage");

  auto visibility = operation->getAttrOfType<StringAttr>("visibility");
  if (!visibility ||
      !llvm::StringSwitch<bool>(visibility.getValue())
           .Case("private", true)
           .Case("local", true)
           .Case("peer", true)
           .Case("global", true)
           .Default(false))
    return operation->emitOpError(
        "visibility must be private, local, peer, or global");

  auto immutable = operation->getAttrOfType<BoolAttr>("immutable");
  auto escapes = operation->getAttrOfType<BoolAttr>("escapes");
  if (!immutable || !escapes)
    return operation->emitOpError(
        "requires immutable and escapes attributes");
  if (visibility.getValue() == "private" && escapes.getValue())
    return operation->emitOpError("a private logical region cannot escape");
  auto registered = operation->getAttrOfType<BoolAttr>("registered");
  auto donated = operation->getAttrOfType<BoolAttr>("donated");
  auto reuseInterval =
      operation->getAttrOfType<IntegerAttr>("reuseInterval");
  auto poolDepth = operation->getAttrOfType<IntegerAttr>("poolDepth");
  if (reuseInterval && reuseInterval.getInt() < 0)
    return operation->emitOpError("reuseInterval cannot be negative");
  if (poolDepth && poolDepth.getInt() < 0)
    return operation->emitOpError("poolDepth cannot be negative");
  if (donated && donated.getValue() &&
      (!registered || !registered.getValue()))
    return operation->emitOpError(
        "donated storage must also be registered");
  return success();
}

std::optional<uint64_t>
getElementCount(const LogicalRegionMetadata &metadata) {
  uint64_t result = 1;
  for (int64_t extent : metadata.shape) {
    auto unsignedExtent = static_cast<uint64_t>(extent);
    if (result > std::numeric_limits<uint64_t>::max() / unsignedExtent)
      return std::nullopt;
    result *= unsignedExtent;
  }
  return result;
}

LogicalResult verifyDestinationOperation(Operation *operation,
                                         bool communicationDestination,
                                         bool requireSameShape,
                                         bool requireSameElementType = true) {
  auto source = getLogicalRegionMetadata(operation->getOperand(0));
  auto destination = getLogicalRegionMetadata(operation->getOperand(1));
  if (!source || !destination)
    return operation->emitOpError(
        "operands must be defined by logical region dataflow");
  if (destination->immutable)
    return operation->emitOpError(
        "cannot write an immutable destination");
  if (communicationDestination &&
      destination->visibility != "peer" &&
      destination->visibility != "global")
    return operation->emitOpError(
        "communication destination must have peer or global visibility");
  // Transforms exist to change representation; every other destination
  // operation moves bytes and must preserve the element type.
  if (requireSameElementType &&
      source->elementType != destination->elementType)
    return operation->emitOpError(
        "source and destination element types must match");
  if (requireSameShape && source->shape != destination->shape)
    return operation->emitOpError(
        "source and destination logical shapes must match");
  return success();
}

LogicalResult verifyBoundedViewMap(
    Operation *operation,
    const LogicalRegionMetadata &source,
    const LogicalRegionMetadata &result) {
  AffineMap map = result.indexMap;
  if (map.getNumDims() != result.shape.size() ||
      map.getNumResults() != source.shape.size() ||
      result.shape.size() != source.shape.size())
    return operation->emitOpError(
        "view indexMap must be a rank-preserving permutation");

  llvm::DenseSet<unsigned> usedDimensions;
  for (auto [sourceDimension, expression] :
       llvm::enumerate(map.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension ||
        !usedDimensions.insert(dimension.getPosition()).second)
      return operation->emitOpError(
          "view indexMap must contain each dimension exactly once");
    if (result.shape[dimension.getPosition()] >
        source.shape[sourceDimension])
      return operation->emitOpError(
          "view indexMap bounds exceed source storage");
  }
  if (usedDimensions.size() != result.shape.size())
    return operation->emitOpError(
        "view indexMap must use every result dimension");
  return success();
}

} // namespace

LogicalResult CommunicationDataflowOp::verify() {
  if (!isa_and_nonnull<ModuleOp>(getOperation()->getParentOp()))
    return emitOpError("must be directly contained by a module");
  auto name = (*this)->getAttrOfType<StringAttr>("name");
  if (!name || name.getValue().empty())
    return emitOpError("requires a non-empty name");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires one body block");

  Block &block = getBody().front();
  if (block.empty() ||
      block.back().getName().getStringRef() != "ub.unreachable")
    return emitOpError("body must end with ub.unreachable");

  llvm::DenseSet<StringRef> identifiers;
  for (Operation &operation : block) {
    if (operation.getName().getStringRef() == "ub.unreachable") {
      if (&operation != &block.back())
        return emitOpError("ub.unreachable must terminate the body");
      continue;
    }
    if (!isa<RegionOp, ViewOp, IndexedViewOp, WorksetOp, TransferOp,
             TransferEachOp, ReductionOp, CopyOp, CopyEachOp, TransformOp,
             UseOp, AsyncUseOp, OpaqueStageOp>(operation))
      return emitOpError("contains unsupported body operation ")
             << operation.getName();

    if (isa<RegionOp, ViewOp, IndexedViewOp, WorksetOp>(operation)) {
      auto regionName = operation.getAttrOfType<StringAttr>("name");
      if (regionName &&
          !identifiers.insert(regionName.getValue()).second)
        return operation.emitOpError(
            "dataflow identifiers must be globally unique");
      if (auto operationId =
              operation.getAttrOfType<StringAttr>("id");
          operationId &&
          !identifiers.insert(operationId.getValue()).second)
        return operation.emitOpError(
            "dataflow identifiers must be globally unique");
    } else {
      auto operationId = operation.getAttrOfType<StringAttr>("id");
      if (!operationId &&
          isa<UseOp, AsyncUseOp, OpaqueStageOp>(operation))
        operationId = operation.getAttrOfType<StringAttr>("name");
      if (operationId &&
          !identifiers.insert(operationId.getValue()).second)
        return operation.emitOpError(
            "dataflow identifiers must be globally unique");
    }
    if (auto outputId =
            operation.getAttrOfType<StringAttr>("outputId");
        outputId &&
        !identifiers.insert(outputId.getValue()).second) {
      return operation.emitOpError(
          "dataflow identifiers must be globally unique");
    }
    if (auto outputIds =
            operation.getAttrOfType<ArrayAttr>("outputIds")) {
      for (Attribute value : outputIds) {
        auto outputId = dyn_cast<StringAttr>(value);
        if (!outputId || outputId.getValue().empty())
          return operation.emitOpError(
              "outputIds must contain non-empty strings");
        if (!identifiers.insert(outputId.getValue()).second)
          return operation.emitOpError(
              "dataflow identifiers must be globally unique");
      }
    }

    for (Value operand : operation.getOperands()) {
      if (!isa<LogicalRegionType>(operand.getType()))
        continue;
      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp || definingOp->getParentOp() != getOperation())
        return operation.emitOpError(
            "logical region operand must be defined in this dataflow");
    }
    for (Value result : operation.getResults()) {
      if (!isa<LogicalRegionType, WorksetType>(result.getType()))
        continue;
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return user->getParentOp() != getOperation();
          }))
        return operation.emitOpError(
            "logical region result cannot leave its dataflow");
    }
  }
  return success();
}

LogicalResult RegionOp::verify() {
  if (auto count = getValidCount(); count && count->empty())
    return emitOpError("validCount must name a non-empty runtime count");
  if (getValidCount().has_value() != getDeadPadding().value_or(false))
    return emitOpError("validCount and deadPadding must be declared together");
  auto shape = (*this)->getAttrOfType<DenseI64ArrayAttr>("shape");
  return verifyLogicalLayout(
      getOperation(), shape ? static_cast<unsigned>(shape.size()) : 0);
}

LogicalResult ViewOp::verify() {
  auto source = getLogicalRegionMetadata(getSource());
  if (!source)
    return emitOpError("source must have logical region metadata");
  if (failed(verifyLogicalLayout(
          getOperation(), static_cast<unsigned>(source->shape.size()))))
    return failure();

  auto result = getLogicalRegionMetadata(getResult());
  if (!result)
    return emitOpError("result must have logical region metadata");
  if (result->elementType != source->elementType)
    return emitOpError("view cannot change the element type");
  if (result->visibility != source->visibility)
    return emitOpError("view cannot change source visibility");
  if (result->immutable != source->immutable)
    return emitOpError("view must preserve source immutability");
  if (result->escapes != source->escapes)
    return emitOpError("view must preserve source escape behavior");
  if (result->registered != source->registered ||
      result->donated != source->donated ||
      result->reuseInterval != source->reuseInterval ||
      result->poolDepth != source->poolDepth)
    return emitOpError(
        "view must preserve registration, donation, and lifetime facts");
  return verifyBoundedViewMap(getOperation(), *source, *result);
}

LogicalResult IndexedViewOp::verify() {
  auto source = getLogicalRegionMetadata(getSource());
  auto indices = getLogicalRegionMetadata(getIndices());
  if (!source || !indices)
    return emitOpError(
        "source and indices must have logical region metadata");
  if (failed(verifyLogicalLayout(
          getOperation(), static_cast<unsigned>(source->shape.size()))))
    return failure();

  auto result = getLogicalRegionMetadata(getResult());
  if (!result)
    return emitOpError("result must have logical region metadata");
  if (result->elementType != source->elementType ||
      result->visibility != source->visibility ||
      result->immutable != source->immutable ||
      result->escapes != source->escapes ||
      result->registered != source->registered ||
      result->donated != source->donated ||
      result->reuseInterval != source->reuseInterval ||
      result->poolDepth != source->poolDepth)
    return emitOpError(
        "indexed view must preserve source storage metadata");
  if (!isa<IntegerType>(indices->elementType) ||
      !indices->immutable || indices->escapes)
    return emitOpError(
        "indices must be immutable, non-escaping integer data");
  if (indices->shape.size() != 1 ||
      result->shape.empty() ||
      indices->shape.front() != result->shape.front())
    return emitOpError(
        "indices must contain one entry per first-dimension result");

  auto lower =
      (*this)->getAttrOfType<DenseI64ArrayAttr>("indexLowerBounds");
  auto upper =
      (*this)->getAttrOfType<DenseI64ArrayAttr>("indexUpperBounds");
  if (!lower || !upper || lower.size() != 1 || upper.size() != 1)
    return emitOpError(
        "indexed view requires one proven index bound");
  if (lower[0] < 0 || upper[0] <= lower[0] ||
      upper[0] > source->shape.front())
    return emitOpError(
        "indexed view bounds are outside source storage");
  if (result->injective &&
      indices->shape.front() > upper[0] - lower[0])
    return emitOpError(
        "injective index count exceeds the proven destination range");
  if (failed(verifyBoundedViewMap(
          getOperation(), *source, *result)))
    return failure();
  return success();
}

LogicalResult WorksetOp::verify() {
  if (getName().empty())
    return emitOpError("requires a non-empty name");
  if (getFields().empty())
    return emitOpError("requires at least one field region");
  if (getMaxItems() <= 0)
    return emitOpError("maxItems must be positive");
  if (getFieldNames().size() != getFields().size())
    return emitOpError("requires one field name per field region");
  if (getFieldLowerBounds().size() != getFields().size() ||
      getFieldUpperBounds().size() != getFields().size())
    return emitOpError("requires lower and upper bounds for every field");

  llvm::DenseSet<StringRef> names;
  for (auto [index, pair] : llvm::enumerate(
           llvm::zip_equal(getFieldNames(), getFields()))) {
    auto [nameAttribute, field] = pair;
    auto name = dyn_cast<StringAttr>(nameAttribute);
    if (!name || name.getValue().empty())
      return emitOpError(
          "fieldNames must contain non-empty strings");
    if (!names.insert(name.getValue()).second)
      return emitOpError("field names must be unique");
    auto metadata = getLogicalRegionMetadata(field);
    if (!metadata)
      return emitOpError(
          "fields must be defined by logical region dataflow");
    if (!isa<IntegerType>(metadata->elementType))
      return emitOpError("fields must contain integer values");
    bool stable = metadata->immutable;
    if (!stable && field.getDefiningOp<OpaqueStageOp>()) {
      stable = true;
      for (OpOperand &use : field.getUses()) {
        Operation *user = use.getOwner();
        if (isa<WorksetOp, UseOp, AsyncUseOp, IndexedViewOp>(user))
          continue;
        if (auto stage = dyn_cast<OpaqueStageOp>(user);
            stage &&
            use.getOperandNumber() < stage.getReads().size())
          continue;
        stable = false;
        break;
      }
    }
    if (!stable || metadata->escapes)
      return emitOpError(
          "fields must be stable and non-escaping");
    if (metadata->shape.empty() ||
        metadata->shape.front() < getMaxItems())
      return emitOpError(
          "field capacity must cover maxItems");
    if (getFieldLowerBounds()[index] >=
        getFieldUpperBounds()[index])
      return emitOpError(
          "field bounds must describe a non-empty range");
  }

  if (Value count = getCount()) {
    auto metadata = getLogicalRegionMetadata(count);
    auto elements = metadata
        ? getElementCount(*metadata)
        : std::nullopt;
    bool stable = metadata && metadata->immutable;
    if (!stable && count.getDefiningOp<OpaqueStageOp>()) {
      stable = llvm::all_of(count.getUsers(), [](Operation *user) {
        return isa<WorksetOp, UseOp, AsyncUseOp>(user);
      });
    }
    if (!metadata || !isa<IntegerType>(metadata->elementType) ||
        !stable || metadata->escapes ||
        !elements || *elements != 1)
      return emitOpError(
          "count must be one immutable non-escaping integer");
  }
  return success();
}

LogicalResult TransferOp::verify() {
  return verifyDestinationOperation(
      getOperation(), /*communicationDestination=*/true,
      /*requireSameShape=*/true);
}

LogicalResult TransferEachOp::verify() {
  auto workset = getWorkset().getDefiningOp<WorksetOp>();
  if (!workset)
    return emitOpError(
        "workset must be defined by aboleth.workset");
  auto source = getLogicalRegionMetadata(getSource());
  auto destination = getLogicalRegionMetadata(getDestination());
  if (!source || !destination)
    return emitOpError(
        "source and destination require logical region metadata");
  if (destination->immutable)
    return emitOpError("cannot write an immutable destination");
  if (destination->visibility != "peer" &&
      destination->visibility != "global")
    return emitOpError(
        "destination must have peer or global visibility");
  if (source->elementType != destination->elementType)
    return emitOpError(
        "source and destination element types must match");
  if (source->shape.size() != destination->shape.size() ||
      !std::equal(source->shape.begin() + 1, source->shape.end(),
                  destination->shape.begin() + 1,
                  destination->shape.end()))
    return emitOpError(
        "source and destination item shapes must match");

  llvm::StringMap<unsigned> fields;
  for (auto [index, value] :
       llvm::enumerate(workset.getFieldNames())) {
    auto name = cast<StringAttr>(value);
    fields.insert({name.getValue(), index});
  }
  const auto requireField =
      [&](StringRef field) -> FailureOr<unsigned> {
    auto found = fields.find(field);
    if (field.empty() || found == fields.end()) {
      emitOpError("references unknown required workset field '")
          << field << "'";
      return failure();
    }
    return found->second;
  };
  auto sourceField = requireField(getSourceIndexField());
  auto peerField = requireField(getDestinationPeerField());
  auto destinationField = requireField(getDestinationIndexField());
  if (failed(sourceField) || failed(peerField) ||
      failed(destinationField))
    return failure();
  auto lower = workset.getFieldLowerBounds();
  auto upper = workset.getFieldUpperBounds();
  if (lower[*sourceField] < 0 ||
      upper[*sourceField] > source->shape.front())
    return emitOpError(
        "source index field exceeds source row bounds");
  if (lower[*destinationField] < 0 ||
      upper[*destinationField] > destination->shape.front())
    return emitOpError(
        "destination index field exceeds destination row bounds");
  if (lower[*peerField] < 0)
    return emitOpError(
        "destination peer field cannot be negative");

  for (StringAttr field : {
           getBytesFieldAttr(), getPredicateFieldAttr()}) {
    if (field && !fields.contains(field.getValue()))
      return emitOpError("references unknown optional workset field '")
             << field.getValue() << "'";
  }
  if (StringAttr bytes = getBytesFieldAttr()) {
    unsigned index = fields.lookup(bytes.getValue());
    if (lower[index] <= 0)
      return emitOpError(
          "byte-count field must be strictly positive");
    unsigned bitWidth =
        source->elementType.getIntOrFloatBitWidth();
    if (bitWidth == 0 || bitWidth % 8 != 0)
      return emitOpError(
          "byte-count fields require byte-addressable elements");
    uint64_t itemBytes = bitWidth / 8;
    for (int64_t extent :
         llvm::drop_begin(source->shape))
      itemBytes *= static_cast<uint64_t>(extent);
    if (static_cast<uint64_t>(upper[index] - 1) >
        itemBytes)
      return emitOpError(
          "byte-count field exceeds the source item size");
  }
  if (StringAttr predicate = getPredicateFieldAttr()) {
    unsigned index = fields.lookup(predicate.getValue());
    if (lower[index] < 0 || upper[index] > 2)
      return emitOpError(
          "predicate field must be bounded to zero or one");
  }
  return success();
}

LogicalResult CopyEachOp::verify() {
  auto workset = getWorkset().getDefiningOp<WorksetOp>();
  if (!workset)
    return emitOpError(
        "workset must be defined by aboleth.workset");
  auto source = getLogicalRegionMetadata(getSource());
  auto destination = getLogicalRegionMetadata(getDestination());
  if (!source || !destination)
    return emitOpError(
        "source and destination require logical region metadata");
  if (destination->immutable)
    return emitOpError("cannot write an immutable destination");
  if (source->elementType != destination->elementType ||
      source->shape.size() != destination->shape.size() ||
      !std::equal(source->shape.begin() + 1, source->shape.end(),
                  destination->shape.begin() + 1,
                  destination->shape.end()))
    return emitOpError(
        "source and destination item layouts must match");

  llvm::StringMap<unsigned> fields;
  for (auto [index, value] :
       llvm::enumerate(workset.getFieldNames())) {
    fields.insert({
        cast<StringAttr>(value).getValue(), index});
  }
  auto sourceField = fields.find(getSourceIndexField());
  auto destinationField =
      fields.find(getDestinationIndexField());
  if (sourceField == fields.end())
    return emitOpError("references unknown source workset field '")
           << getSourceIndexField() << "'";
  if (destinationField == fields.end())
    return emitOpError(
               "references unknown destination workset field '")
           << getDestinationIndexField() << "'";
  auto lower = workset.getFieldLowerBounds();
  auto upper = workset.getFieldUpperBounds();
  if (lower[sourceField->second] < 0 ||
      upper[sourceField->second] > source->shape.front())
    return emitOpError(
        "source index field exceeds source row bounds");
  if (lower[destinationField->second] < 0 ||
      upper[destinationField->second] >
          destination->shape.front())
    return emitOpError(
        "destination index field exceeds destination row bounds");
  return success();
}

LogicalResult ReductionOp::verify() {
  if (getCardinality() <= 0)
    return emitOpError("cardinality must be positive");
  if (getOperatorName().empty())
    return emitOpError("operator cannot be empty");
  return verifyDestinationOperation(
      getOperation(), /*communicationDestination=*/true,
      /*requireSameShape=*/true);
}

LogicalResult CopyOp::verify() {
  return verifyDestinationOperation(
      getOperation(), /*communicationDestination=*/false,
      /*requireSameShape=*/true);
}

LogicalResult TransformOp::verify() {
  if (getTransform().empty())
    return emitOpError("transform cannot be empty");
  if (failed(verifyDestinationOperation(
          getOperation(), /*communicationDestination=*/false,
          /*requireSameShape=*/false,
          /*requireSameElementType=*/false)))
    return failure();

  auto source = getLogicalRegionMetadata(getSource());
  auto destination = getLogicalRegionMetadata(getDestination());
  auto sourceElements = getElementCount(*source);
  auto destinationElements = getElementCount(*destination);
  if (!sourceElements || !destinationElements ||
      sourceElements != destinationElements)
    return emitOpError(
        "source and destination must have the same element count");

  AffineMap map = getIndexMap();
  if (map.getNumSymbols() != 0)
    return emitOpError("indexMap cannot contain symbols");
  if (map.getNumDims() != destination->shape.size() ||
      map.getNumResults() != source->shape.size())
    return emitOpError(
        "indexMap must map destination indices to source indices");
  return success();
}

LogicalResult UseOp::verify() {
  if (getName().empty())
    return emitOpError("requires a non-empty name");
  if (getInputs().empty())
    return emitOpError("requires at least one logical region");
  for (Value region : getInputs()) {
    if (!getLogicalRegionMetadata(region))
      return emitOpError(
          "operands must be defined by logical region dataflow");
  }
  return success();
}

LogicalResult AsyncUseOp::verify() {
  if (getName().empty())
    return emitOpError("requires a non-empty name");
  if (getInputs().empty())
    return emitOpError("requires at least one logical region");
  if (getMaxInflight() <= 0)
    return emitOpError("maxInflight must be positive");
  for (Value region : getInputs()) {
    if (!getLogicalRegionMetadata(region))
      return emitOpError(
          "operands must be defined by logical region dataflow");
  }
  return success();
}

LogicalResult OpaqueStageOp::verify() {
  if (auto counts = (*this)->getAttrOfType<ArrayAttr>("prefixReadCounts")) {
    if (counts.size() != getReads().size())
      return emitOpError("prefixReadCounts requires one string per read operand");
    for (Attribute count : counts)
      if (!isa<StringAttr>(count))
        return emitOpError("prefixReadCounts entries must be runtime-count strings");
  }
  if (getName().empty())
    return emitOpError("requires a non-empty name");
  if (getReads().empty() && getWrites().empty())
    return emitOpError("requires at least one read or write region");
  if (getWrites().size() != getResults().size())
    return emitOpError("requires one result for every write region");
  if (getOutputIds().size() != getResults().size())
    return emitOpError("requires one outputId for every result");
  if (getMaxInflight() <= 0)
    return emitOpError("maxInflight must be positive");
  if (!getAsynchronous() && getMaxInflight() != 1)
    return emitOpError(
        "synchronous stages require maxInflight equal to one");

  for (Value region : llvm::concat<Value>(
           getReads(), getWrites())) {
    if (!getLogicalRegionMetadata(region))
      return emitOpError(
          "operands must be defined by logical region dataflow");
  }
  for (auto [destination, result] :
       llvm::zip_equal(getWrites(), getResults())) {
    auto metadata = getLogicalRegionMetadata(destination);
    if (metadata && metadata->immutable)
      return emitOpError(
          "cannot write an immutable destination");
    if (destination.getType() != result.getType())
      return emitOpError(
          "result types must match write destination types");
  }
  return success();
}
