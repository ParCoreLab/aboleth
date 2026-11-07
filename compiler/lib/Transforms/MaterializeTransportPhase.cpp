#include "aboleth/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHMATERIALIZETRANSPORTPHASEPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct FragmentPlan {
  TransportFragmentOp fragment;
  std::optional<int64_t> interval;
  std::optional<int64_t> capacity;
};
using FragmentMap = llvm::DenseMap<int64_t, FragmentPlan>;

LogicalResult indexFragments(ModuleOp plan, FragmentMap &fragments) {
  llvm::DenseMap<int64_t, int64_t> intervals;
  llvm::DenseMap<int64_t, int64_t> capacities;
  auto phases = plan.walk([&](TransportPhaseOp phase) -> WalkResult {
    if (!phase.getProgramId())
      return WalkResult::advance();
    auto interval = phase.getResolvedCompletionInterval();
    if (!interval || *interval < 1) {
      phase.emitOpError("bound fragment requires a resolved completion interval");
      return WalkResult::interrupt();
    }
    auto [entry, inserted] = intervals.try_emplace(*phase.getProgramId(), *interval);
    if (!inserted)
      entry->second = std::min<int64_t>(entry->second, *interval);
    int64_t capacity = std::numeric_limits<int64_t>::max();
    for (int32_t value : phase.getPeerCapacities())
      capacity = std::min<int64_t>(capacity, value);
    auto [queue, fresh] = capacities.try_emplace(*phase.getProgramId(), capacity);
    if (!fresh)
      queue->second = std::min(queue->second, capacity);
    return WalkResult::advance();
  });
  if (phases.wasInterrupted())
    return failure();
  WalkResult result =
      plan.walk([&](TransportFragmentOp fragment) -> WalkResult {
        int64_t programId = fragment.getProgramId();
        std::optional<int64_t> interval;
        std::optional<int64_t> capacity;
        if (auto found = intervals.find(programId); found != intervals.end())
          interval = found->second;
        if (auto found = capacities.find(programId); found != capacities.end())
          capacity = found->second;
        if (!fragments.try_emplace(programId, FragmentPlan{fragment, interval, capacity}).second) {
          fragment.emitOpError("duplicate programId ")
              << programId << " in external transport program";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  return failure(result.wasInterrupted());
}

bool permitsPointerPlaceholder(BlockArgument argument, Value value) {
  if (!argument.getType().isSignlessInteger(64))
    return false;
  if (isa<EndpointType>(value.getType()))
    return llvm::all_of(argument.getUsers(), [](Operation *user) {
      return isa<BindEndpointOp>(user);
    });
  if (value.getType().getAbstractType().getName() != "tt.ptr")
    return false;
  return llvm::all_of(argument.getUsers(), [](Operation *user) {
    return isa<BindEndpointOp, BindLocalRegionOp, BindRemoteRegionOp>(user);
  });
}

LogicalResult verifyArgumentTypes(InvokeTransportPhaseOp invoke,
                                  TransportFragmentOp fragment) {
  Block &body = fragment.getBody().front();
  for (auto [index, pair] : llvm::enumerate(
           llvm::zip_equal(body.getArguments(), invoke.getArguments()))) {
    auto [parameter, argument] = pair;
    if (parameter.getType() == argument.getType() ||
        permitsPointerPlaceholder(parameter, argument))
      continue;
    return invoke.emitOpError("argument #")
           << index << " has type " << argument.getType()
           << ", but fragment '" << fragment.getName() << "' expects "
           << parameter.getType();
  }
  return success();
}

struct Materialization {
  InvokeTransportPhaseOp invoke;
  TransportFragmentOp fragment;
  std::optional<int64_t> interval;
  std::optional<int64_t> capacity;
};

LogicalResult collectMaterializations(ModuleOp module,
                                      const FragmentMap &fragments,
                                      SmallVectorImpl<Materialization> &work) {
  SmallVector<InvokeTransportPhaseOp> invokes;
  module.walk(
      [&](InvokeTransportPhaseOp invoke) { invokes.push_back(invoke); });

  for (InvokeTransportPhaseOp invoke : invokes) {
    auto constant =
        invoke.getProgramId().getDefiningOp<arith::ConstantOp>();
    auto programId = constant
                         ? dyn_cast<IntegerAttr>(constant.getValueAttr())
                         : IntegerAttr{};
    if (!programId)
      return invoke.emitOpError(
          "programId must be defined by an integer arith.constant");

    auto found = fragments.find(programId.getInt());
    if (found == fragments.end())
      return invoke.emitOpError("references missing transport fragment ")
             << programId.getInt();

    TransportFragmentOp fragment = found->second.fragment;
    if (fragment.getBody().front().getNumArguments() !=
        invoke.getArguments().size())
      return invoke.emitOpError("argument count does not match fragment '")
             << fragment.getName() << "'";
    if (failed(verifyArgumentTypes(invoke, fragment)))
      return failure();
    work.push_back({invoke, fragment, found->second.interval, found->second.capacity});
  }
  return success();
}

void materialize(Materialization &work) {
  OpBuilder builder(work.invoke);
  IRMapping mapping;
  for (auto [parameter, argument] : llvm::zip_equal(
           work.fragment.getBody().front().getArguments(),
           work.invoke.getArguments()))
    mapping.map(parameter, argument);

  for (Operation &operation : work.fragment.getBody().front()) {
    if (operation.getName().getStringRef() != "ub.unreachable") {
      if (auto bind = dyn_cast<BindEndpointOp>(operation)) {
        Value value = mapping.lookupOrDefault(bind.getValue());
        if (isa<EndpointType>(value.getType())) {
          mapping.map(bind.getResult(), value);
          continue;
        }
      }
      Operation *clone = builder.clone(operation, mapping);
      if (work.interval)
        clone->walk([&](PlannedWaitOp wait) {
          if (!wait.getReuseDistance()) {
            wait.setCompletionIntervalAttr(builder.getI64IntegerAttr(*work.interval));
            if (work.capacity)
              wait.setQueueCapacityAttr(builder.getI64IntegerAttr(*work.capacity));
          }
        });
    }
  }
  work.invoke.erase();
}

class MaterializeTransportPhase
    : public impl::AbolethMaterializeTransportPhasePassBase<
          MaterializeTransportPhase> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (planPath.empty()) {
      module.emitError(
          "aboleth-materialize-transport-phase requires plan-path");
      signalPassFailure();
      return;
    }

    OwningOpRef<ModuleOp> plan =
        parseSourceFile<ModuleOp>(planPath, module.getContext());
    if (!plan) {
      module.emitError("failed to parse external transport program ")
          << planPath;
      signalPassFailure();
      return;
    }

    FragmentMap fragments;
    SmallVector<Materialization> work;
    if (failed(indexFragments(*plan, fragments)) ||
        failed(collectMaterializations(module, fragments, work))) {
      signalPassFailure();
      return;
    }

    for (Materialization &item : work)
      materialize(item);

    bool survived = false;
    module.walk([&](InvokeTransportPhaseOp invoke) {
      invoke.emitOpError("survived transport phase materialization");
      survived = true;
    });
    if (survived)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::aboleth
