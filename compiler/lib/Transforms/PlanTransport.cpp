#include "aboleth/Transforms/Passes.h"

#include "aboleth/Target/RdmaAbi.h"
#include "aboleth/Target/TransportCaps.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <system_error>
#include <utility>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHPLANTRANSPORTPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

// Exact endpoint geography is a registration fact. Every generated launch
// lowers only its homogeneous subset; no representative backend or runtime
// transport-tag branch is permitted. Execution order and streams remain with
// the client. Programs without geography retain their previous IR verbatim.
void partitionTransports(TransportProgramOp program) {
  auto ranks = program.getPeerRanksAttr();
  if (!ranks) {
    program->removeAttr("launchPartitions");
    return;
  }
  std::map<std::string, SmallVector<int32_t>> partitions;
  for (auto [rank, attribute] :
       llvm::zip_equal(ranks.asArrayRef(), program.getPeerTransportsAttr()))
    partitions[cast<StringAttr>(attribute).getValue().str()].push_back(rank);
  Builder builder(program.getContext());
  SmallVector<Attribute> launches;
  for (const auto &[backend, peers] : partitions) {
    launches.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("backend", builder.getStringAttr(backend)),
        builder.getNamedAttr("peers", builder.getDenseI32ArrayAttr(peers)),
        builder.getNamedAttr("gridExtent", builder.getI32IntegerAttr(peers.size()))}));
  }
  program->setAttr("launchPartitions", builder.getArrayAttr(launches));
}

FailureOr<int64_t> requireInteger(Operation *operation, StringRef name) {
  auto attribute = operation->getAttrOfType<IntegerAttr>(name);
  if (!attribute) {
    operation->emitOpError("requires ") << name;
    return failure();
  }
  return attribute.getInt();
}

struct RegionEffect {
  int64_t allocation;
  int64_t generation;
  int64_t offset;
  int64_t length;
  int64_t stride;
  int64_t period;
  int64_t count;
  bool unknown;
};

FailureOr<SmallVector<RegionEffect>> readRegions(Operation *operation,
                                                StringRef name) {
  auto values = operation->getAttrOfType<ArrayAttr>(name);
  if (!values) {
    operation->emitOpError("requires ") << name;
    return failure();
  }
  SmallVector<RegionEffect> result;
  result.reserve(values.size());
  for (Attribute value : values) {
    auto region = dyn_cast<DictionaryAttr>(value);
    if (!region) {
      operation->emitOpError(name)
          << " must contain only region dictionaries";
      return failure();
    }
    auto integer = [&](StringRef field) -> FailureOr<int64_t> {
      auto attribute = region.getAs<IntegerAttr>(field);
      if (!attribute) {
        operation->emitOpError(name)
            << " region requires " << field;
        return failure();
      }
      return attribute.getInt();
    };
    auto allocation = integer("allocation");
    auto generation = integer("generation");
    auto offset = integer("offset");
    auto length = integer("length");
    auto stride = integer("stride");
    auto period = integer("period");
    auto count = integer("count");
    if (failed(allocation) || failed(generation) || failed(offset) ||
        failed(length) || failed(stride) || failed(period) ||
        failed(count))
      return failure();
    auto unknownAttr = region.getAs<BoolAttr>("unknown");
    result.push_back(
        {*allocation, *generation, *offset, *length, *stride, *period,
         *count, unknownAttr && unknownAttr.getValue()});
  }
  return result;
}

std::optional<std::pair<int64_t, int64_t>>
intervalAt(const RegionEffect &region, int64_t iteration) {
  if (region.unknown || iteration < 0 || iteration >= region.count)
    return std::nullopt;
  if (region.stride != 0 &&
      iteration >
          (std::numeric_limits<int64_t>::max() - region.offset) /
              region.stride)
    return std::nullopt;
  int64_t displacement = iteration * region.stride;
  if (region.period > 0)
    displacement %= region.period;
  const int64_t begin = region.offset + displacement;
  if (region.length >
      std::numeric_limits<int64_t>::max() - begin)
    return std::nullopt;
  return std::pair<int64_t, int64_t>{begin, begin + region.length};
}

bool overlaps(const RegionEffect &first, int64_t firstIteration,
              const RegionEffect &second, int64_t secondIteration) {
  if (first.unknown || second.unknown)
    return true;
  if (first.allocation != second.allocation ||
      first.generation != second.generation)
    return false;
  auto firstInterval = intervalAt(first, firstIteration);
  auto secondInterval = intervalAt(second, secondIteration);
  if (!firstInterval || !secondInterval)
    return true;
  return firstInterval->first < secondInterval->second &&
         secondInterval->first < firstInterval->second;
}

bool effectsOverlap(ArrayRef<RegionEffect> reads,
                    ArrayRef<RegionEffect> writes) {
  for (const RegionEffect &read : reads) {
    for (const RegionEffect &write : writes) {
      for (int64_t readIteration = 0; readIteration < read.count;
           ++readIteration) {
        for (int64_t writeIteration = 0;
             writeIteration < write.count; ++writeIteration) {
          if (overlaps(read, readIteration, write, writeIteration))
            return true;
        }
      }
    }
  }
  return false;
}

FailureOr<int64_t> inferReuseDistance(TransportPhaseOp phase,
                                      int64_t iterations) {
  auto reads = readRegions(phase, "reads");
  if (failed(reads))
    return failure();
  if (reads->empty())
    return 1;

  int64_t reuseDistance = iterations;
  for (const RegionEffect &read : *reads) {
    if (read.count != iterations)
      return phase.emitOpError(
          "phase read effect count must equal iterationCount");
    int64_t effectReuse = iterations;
    for (int64_t distance = 1; distance < iterations; ++distance) {
      bool foundOverlap = false;
      for (int64_t iteration = 0;
           iteration + distance < iterations; ++iteration) {
        if (overlaps(read, iteration, read, iteration + distance)) {
          foundOverlap = true;
          break;
        }
      }
      if (foundOverlap) {
        effectReuse = distance;
        break;
      }
    }
    reuseDistance = std::min(reuseDistance, effectReuse);
  }
  return reuseDistance;
}

// TODO: lifetime-scheduled synchronization. The reuse distance computed
// below should also schedule cross-rank credits and barriers (only on slot
// reuse); today it is recorded but unused.
LogicalResult planPhase(TransportPhaseOp phase) {
  const int64_t commands = phase.getCommandsPerIteration();
  const int64_t iterations = phase.getIterationCount();
  const int64_t payloadBytes = phase.getPayloadBytes();
  if (commands <= 0 || iterations <= 0 || payloadBytes < 0)
    return phase.emitOpError("has invalid transport planning facts");

  auto inferredReuse = inferReuseDistance(phase, iterations);
  if (failed(inferredReuse))
    return failure();
  const int64_t reuseDistance = *inferredReuse;

  auto transports = phase.getPeerTransportsAttr();
  auto capacities = phase.getPeerCapacitiesAttr();
  if (static_cast<bool>(transports) != static_cast<bool>(capacities))
    return phase.emitOpError(
        "requires peer transports and capacities together");
  if (transports && transports.size() != capacities.size())
    return phase.emitOpError(
        "requires one endpoint capacity per peer transport");

  auto explicitInterval = phase.getCompletionIntervalAttr();
  auto peerIntervals = phase.getPeerCompletionIntervalsAttr();
  if (peerIntervals && transports &&
      peerIntervals.size() != transports.size())
    return phase.emitOpError(
        "requires one completion interval per peer transport");

  int64_t policyLimit = iterations;
  int64_t capacityLimit = iterations;
  bool queued = false;
  if (transports) {
    std::optional<int64_t> tinyPayloadBytes;
    if (!explicitInterval) {
      auto tiny = requireInteger(phase, "policyTinyPayloadBytes");
      if (failed(tiny))
        return failure();
      tinyPayloadBytes = *tiny;
    }

    for (auto [index, transportAttribute] :
         llvm::enumerate(transports)) {
      auto transport = dyn_cast<StringAttr>(transportAttribute);
      if (!transport)
        return phase.emitOpError(
            "peer transports must contain only strings");
      const ::aboleth::abi::TransportCaps *caps = ::aboleth::abi::lookup_transport_caps(
          transport.getValue().data(), transport.getValue().size());
      if (!caps)
        return phase.emitOpError("unknown peer transport '")
               << transport.getValue() << "'";

      if (!explicitInterval) {
        // A transport reached by ordinary loads and stores has no queue to
        // drain, and a payload small enough to stay resident needs no early
        // completion either. Both may retire once for the whole phase.
        const int64_t configured =
            peerIntervals ? peerIntervals.asArrayRef()[index] : 0;
        const int64_t preferred =
            configured > 0
                ? configured
                : (payloadBytes <= *tinyPayloadBytes ||
                           caps->load_store_addressable
                       ? iterations
                       : (caps->default_completion_interval > 0
                              ? static_cast<int64_t>(
                                    caps->default_completion_interval)
                              : iterations));
        policyLimit =
            std::min(policyLimit, std::min(preferred, iterations));
      }

      // Source storage cannot be reused while the transport still holds the
      // submission, so a bounded queue bounds how long completion may be
      // deferred.
      if (caps->queue_capacity_limited) {
        queued = true;
        const int64_t capacity = capacities.asArrayRef()[index];
        if (capacity < commands)
          return phase.emitOpError(
              "endpoint capacity cannot hold one iteration");
        capacityLimit = std::min(
            capacityLimit,
            std::min(std::max<int64_t>(1, (capacity - 1) / commands),
                     iterations));
      }
    }
  }

  const int64_t legalMaximum =
      std::min(reuseDistance, capacityLimit);
  int64_t interval = 0;
  if (explicitInterval) {
    interval = explicitInterval.getInt();
    if (interval < 1 || interval > legalMaximum)
      return phase.emitOpError(
          "explicit completion interval exceeds region lifetime "
          "or endpoint capacity");
  } else {
    if (!transports)
      return phase.emitOpError(
          "automatic completion planning requires endpoint capabilities");
    interval = std::min(legalMaximum, policyLimit);
  }

  Builder attributes(phase.getContext());
  phase->setAttr("resolvedCompletionInterval",
                 attributes.getI32IntegerAttr(interval));
  phase->setAttr("reuseDistance",
                 attributes.getI32IntegerAttr(reuseDistance));
  if (phase.getProgramId())
    phase.setRequiresFinalDrainAttr(attributes.getBoolAttr(queued && interval > 1));
  return success();
}

LogicalResult annotateCompletionDeadlines(TransportProgramOp program) {
  SmallVector<Operation *> operations;
  for (Operation &operation : program.getBody().front()) {
    if (isa<TransportKernelOp, TransportPhaseOp>(operation))
      operations.push_back(&operation);
  }

  Builder attributes(program.getContext());
  const auto cyclic = program->getAttrOfType<BoolAttr>("cyclic");
  for (auto [index, operation] : llvm::enumerate(operations)) {
    auto phase = dyn_cast<TransportPhaseOp>(operation);
    if (!phase)
      continue;
    auto sourceRegions = readRegions(operation, "reads");
    if (failed(sourceRegions))
      return failure();

    StringRef completionBefore = "program_end";
    int64_t completionIterationDelta = 0;
    for (Operation *later : llvm::drop_begin(operations, index + 1)) {
      auto writes = readRegions(later, "writes");
      if (failed(writes))
        return failure();
      if (!effectsOverlap(*sourceRegions, *writes))
        continue;
      auto name = later->getAttrOfType<StringAttr>("name");
      if (!name)
        return later->emitOpError(
            "requires a name for completion placement");
      completionBefore = name.getValue();
      break;
    }
    if (completionBefore == "program_end" && cyclic &&
        cyclic.getValue()) {
      for (Operation *later :
           ArrayRef<Operation *>(operations)
               .take_front(index + 1)) {
        auto writes = readRegions(later, "writes");
        if (failed(writes))
          return failure();
        if (!effectsOverlap(*sourceRegions, *writes))
          continue;
        auto name = later->getAttrOfType<StringAttr>("name");
        if (!name)
          return later->emitOpError(
              "requires a name for completion placement");
        completionBefore = name.getValue();
        completionIterationDelta = 1;
        break;
      }
    }
    phase->setAttr("completionBefore",
                   attributes.getStringAttr(completionBefore));
    phase->setAttr(
        "completionIterationDelta",
        attributes.getI32IntegerAttr(
            completionIterationDelta));

    auto destinationRegions = readRegions(operation, "writes");
    if (failed(destinationRegions))
      return failure();
    StringRef publicationBefore = "program_end";
    int64_t publicationIterationDelta = 0;
    for (Operation *later : llvm::drop_begin(operations, index + 1)) {
      auto reads = readRegions(later, "reads");
      if (failed(reads))
        return failure();
      if (!effectsOverlap(*reads, *destinationRegions))
        continue;
      auto name = later->getAttrOfType<StringAttr>("name");
      if (!name)
        return later->emitOpError(
            "requires a name for publication placement");
      publicationBefore = name.getValue();
      break;
    }
    if (publicationBefore == "program_end" && cyclic &&
        cyclic.getValue()) {
      for (Operation *later :
           ArrayRef<Operation *>(operations)
               .take_front(index + 1)) {
        auto reads = readRegions(later, "reads");
        if (failed(reads))
          return failure();
        if (!effectsOverlap(*reads, *destinationRegions))
          continue;
        auto name = later->getAttrOfType<StringAttr>("name");
        if (!name)
          return later->emitOpError(
              "requires a name for publication placement");
        publicationBefore = name.getValue();
        publicationIterationDelta = 1;
        break;
      }
    }
    phase->setAttr(
        "publicationBefore",
        attributes.getStringAttr(publicationBefore));
    phase->setAttr(
        "publicationIterationDelta",
        attributes.getI32IntegerAttr(
            publicationIterationDelta));
  }
  return success();
}

LogicalResult annotateOutputLifetimes(TransportProgramOp program) {
  SmallVector<Operation *> operations;
  for (Operation &operation : program.getBody().front()) {
    if (isa<TransportKernelOp, TransportPhaseOp>(operation))
      operations.push_back(&operation);
  }

  Builder attributes(program.getContext());
  const auto cyclic = program->getAttrOfType<BoolAttr>("cyclic");
  for (auto [index, operation] : llvm::enumerate(operations)) {
    auto produced = readRegions(operation, "writes");
    if (failed(produced))
      return failure();
    if (produced->empty())
      continue;
    if (llvm::any_of(*produced,
                     [](const RegionEffect &region) {
                       return region.unknown;
                     })) {
      operation->setAttr(
          "outputReusable",
          attributes.getBoolAttr(false));
      continue;
    }

    StringRef lastConsumer = "none";
    int64_t lastConsumerIterationDelta = 0;
    StringRef reuseBefore = "program_end";
    int64_t reuseIterationDelta = 0;
    bool unknownInterference = false;

    auto inspect = [&](Operation *later,
                       int64_t iterationDelta)
        -> LogicalResult {
      auto reads = readRegions(later, "reads");
      auto writes = readRegions(later, "writes");
      if (failed(reads) || failed(writes))
        return failure();
      if (llvm::any_of(
              *reads,
              [](const RegionEffect &region) {
                return region.unknown;
              }) ||
          llvm::any_of(
              *writes,
              [](const RegionEffect &region) {
                return region.unknown;
              })) {
        unknownInterference = true;
        return success();
      }
      if (effectsOverlap(*reads, *produced)) {
        auto name =
            later->getAttrOfType<StringAttr>("name");
        if (!name) {
          later->emitOpError(
              "requires a name for output lifetime planning");
          return failure();
        }
        lastConsumer = name.getValue();
        lastConsumerIterationDelta = iterationDelta;
      }
      if (effectsOverlap(*produced, *writes)) {
        auto name =
            later->getAttrOfType<StringAttr>("name");
        if (!name) {
          later->emitOpError(
              "requires a name for output lifetime planning");
          return failure();
        }
        reuseBefore = name.getValue();
        reuseIterationDelta = iterationDelta;
      }
      return success();
    };

    for (Operation *later :
         llvm::drop_begin(operations, index + 1)) {
      if (failed(inspect(later, 0)))
        return failure();
      if (reuseBefore != "program_end")
        break;
    }
    if (reuseBefore == "program_end" && cyclic &&
        cyclic.getValue()) {
      for (Operation *later :
           ArrayRef<Operation *>(operations)
               .take_front(index + 1)) {
        if (failed(inspect(later, 1)))
          return failure();
        if (reuseBefore != "program_end")
          break;
      }
    }

    const auto asynchronous =
        operation->getAttrOfType<BoolAttr>("asynchronous");
    const auto eventOrdered =
        operation->getAttrOfType<BoolAttr>("eventOrdered");
    const auto maxInflight =
        operation->getAttrOfType<IntegerAttr>("maxInflight");
    const bool asyncReusable =
        !asynchronous || !asynchronous.getValue() ||
        (eventOrdered && eventOrdered.getValue());
    const bool reusable =
        !unknownInterference &&
        asyncReusable &&
        lastConsumer != "none" &&
        reuseBefore != "program_end" &&
        lastConsumerIterationDelta <=
            reuseIterationDelta;
    operation->setAttr(
        "outputReusable",
        attributes.getBoolAttr(reusable));
    if (!reusable)
      continue;
    operation->setAttr(
        "outputLastConsumer",
        attributes.getStringAttr(lastConsumer));
    operation->setAttr(
        "outputLastConsumerIterationDelta",
        attributes.getI32IntegerAttr(
            lastConsumerIterationDelta));
    operation->setAttr(
        "outputReuseBefore",
        attributes.getStringAttr(reuseBefore));
    operation->setAttr(
        "outputReuseIterationDelta",
        attributes.getI32IntegerAttr(
            reuseIterationDelta));
    operation->setAttr(
        "outputPoolDepth",
        attributes.getI32IntegerAttr(
            asynchronous && asynchronous.getValue() &&
                    maxInflight
                ? maxInflight.getInt()
                : 1));
  }
  return success();
}

void recordString(llvm::json::Object &facts, Operation *operation,
                  StringRef name) {
  if (auto attribute = operation->getAttrOfType<StringAttr>(name))
    facts[name] = attribute.getValue();
}

void recordInteger(llvm::json::Object &facts, Operation *operation,
                   StringRef name) {
  if (auto attribute = operation->getAttrOfType<IntegerAttr>(name))
    facts[name] = attribute.getInt();
}

void recordBoolean(llvm::json::Object &facts, Operation *operation,
                   StringRef name) {
  if (auto attribute = operation->getAttrOfType<BoolAttr>(name))
    facts[name] = attribute.getValue();
}

// The keys are the MLIR attribute names, so a planned fact reads the same in
// the printed module and in the JSON document.
llvm::json::Object plannedFacts(Operation *operation) {
  llvm::json::Object facts;
  for (StringRef name :
       {"name", "boundKernel", "completionBefore", "publicationBefore",
        "outputLastConsumer", "outputReuseBefore"})
    recordString(facts, operation, name);
  for (StringRef name :
       {"resolvedCompletionInterval", "reuseDistance",
        "completionIterationDelta", "publicationIterationDelta",
        "outputLastConsumerIterationDelta", "outputReuseIterationDelta",
        "outputPoolDepth"})
    recordInteger(facts, operation, name);
  recordBoolean(facts, operation, "outputReusable");
  recordBoolean(facts, operation, "requiresFinalDrain");
  return facts;
}

// The kernel a phase is bound to, if this module defines it.
Operation *boundFunction(ModuleOp module, Operation *phase) {
  auto bound = phase->getAttrOfType<StringAttr>("boundKernel");
  StringRef name = bound ? bound.getValue()
                         : phase->getAttrOfType<StringAttr>("name").getValue();
  for (Operation &operation : module.getBody()->getOperations()) {
    if (operation.getName().getStringRef() != "tt.func")
      continue;
    auto symbol = operation.getAttrOfType<StringAttr>("sym_name");
    if (symbol && symbol.getValue() == name)
      return &operation;
  }
  return nullptr;
}

// Planning bounds the deferral window in commands, but an endpoint's capacity
// is a count of 64-byte WQEBBs, and an inline command whose payload leaves the
// first block costs more than one. For such a command the bound above is
// optimistic: the queue can fill before the plan believes it has, and a
// deferred completion then waits for room that only that completion frees.
//
// Making the bound exact means planning in WQEBBs, which needs each phase to
// carry the sizes of the commands it posts. Until then a plan that keeps commands outstanding is
// refused when the submitting kernel — when this module holds it — carries a
// multi-block inline command.
LogicalResult refuseOptimisticInlineWindow(ModuleOp module,
                                           TransportProgramOp program) {
  for (Operation &operation : program.getBody().front()) {
    auto phase = dyn_cast<TransportPhaseOp>(operation);
    if (!phase)
      continue;
    auto interval = phase.getResolvedCompletionIntervalAttr();
    auto before = phase.getCompletionBeforeAttr();
    const bool defersCompletion =
        (interval && interval.getInt() > 1) ||
        (before && before.getValue() != "program_end");
    if (!defersCompletion)
      continue;
    Operation *function = boundFunction(module, phase);
    if (!function)
      continue;
    LogicalResult result = success();
    function->walk([&](Operation *command) {
      uint64_t bytes = 0;
      if (auto put = dyn_cast<PutInlineOp>(command))
        bytes = put.getPayloadBytes();
      else if (auto post = dyn_cast<PostInlineOp>(command))
        bytes = post.getPayloadBytes();
      else
        return;
      if (bytes <= ::aboleth::abi::kSingleBlockInlinePayloadBytes)
        return;
      result = command->emitOpError("carries a ")
               << bytes
               << "-byte inline payload, which spans more than one work "
                  "queue entry block, in a phase whose completion is "
                  "deferred; planning bounds the deferral window in commands "
                  "rather than blocks, so the bound is optimistic here";
    });
    if (failed(result))
      return failure();
  }
  return success();
}

LogicalResult writePlanJson(ModuleOp module, StringRef path) {
  llvm::json::Array programs;
  module.walk([&](TransportProgramOp program) {
    llvm::json::Array phases;
    llvm::json::Array kernels;
    for (Operation &operation : program.getBody().front()) {
      if (isa<TransportPhaseOp>(operation))
        phases.push_back(plannedFacts(&operation));
      else if (isa<TransportKernelOp>(operation))
        kernels.push_back(plannedFacts(&operation));
    }
    llvm::json::Object planned;
    planned["name"] = program.getName();
    planned["phases"] = std::move(phases);
    planned["kernels"] = std::move(kernels);
    if (auto partitions = program.getLaunchPartitionsAttr()) {
      llvm::json::Array launches;
      for (Attribute attr : partitions) {
        auto partition = cast<DictionaryAttr>(attr);
        llvm::json::Object launch;
        launch["backend"] = partition.getAs<StringAttr>("backend").getValue();
        launch["peers"] = llvm::json::Array(
            partition.getAs<DenseI32ArrayAttr>("peers").asArrayRef());
        launch["grid_extent"] = partition.getAs<IntegerAttr>("gridExtent").getInt();
        launches.push_back(std::move(launch));
      }
      planned["launch_partitions"] = std::move(launches);
    }
    programs.push_back(std::move(planned));
  });

  std::error_code error;
  llvm::raw_fd_ostream stream(path, error);
  if (error) {
    module.emitError("failed to open transport plan ")
        << path << ": " << error.message();
    return failure();
  }
  llvm::json::Object document;
  document["programs"] = std::move(programs);
  llvm::json::OStream json(stream, 2);
  json.value(llvm::json::Value(std::move(document)));
  stream << "\n";
  stream.close();
  if (stream.has_error()) {
    module.emitError("failed to write transport plan ") << path;
    return failure();
  }
  return success();
}

class PlanTransport
    : public impl::AbolethPlanTransportPassBase<PlanTransport> {
public:
  using Base::Base;

  void runOnOperation() override {
    bool failedPlanning = false;
    getOperation().walk([&](TransportProgramOp program) {
      partitionTransports(program);
      for (Operation &operation : program.getBody().front()) {
        if (auto phase = dyn_cast<TransportPhaseOp>(operation);
            phase && failed(planPhase(phase)))
          failedPlanning = true;
      }
      if (failed(annotateCompletionDeadlines(program)))
        failedPlanning = true;
      if (failed(annotateOutputLifetimes(program)))
        failedPlanning = true;
      if (failed(refuseOptimisticInlineWindow(getOperation(), program)))
        failedPlanning = true;
    });
    if (failedPlanning) {
      signalPassFailure();
      return;
    }
    if (!emitPlanJson.empty() &&
        failed(writePlanJson(getOperation(), emitPlanJson)))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::aboleth
