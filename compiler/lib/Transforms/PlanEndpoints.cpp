#include "aboleth/Transforms/Passes.h"
#include "aboleth/Target/TransportCaps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <set>
#include <system_error>
#include <tuple>

using namespace mlir;
using namespace mlir::aboleth;

namespace {
StringRef stringField(DictionaryAttr value, StringRef name) {
  auto field = value.getAs<StringAttr>(name);
  return field ? field.getValue() : StringRef{};
}
int64_t integerField(DictionaryAttr value, StringRef name) {
  auto field = value.getAs<IntegerAttr>(name);
  return field ? field.getInt() : -1;
}
bool knownBackend(StringRef backend) {
  return ::aboleth::abi::lookup_transport_caps(backend.data(), backend.size());
}
int64_t floorPowerOfTwo(int64_t value) {
  int64_t result = 1;
  while (result <= value / 2)
    result *= 2;
  return result;
}
} // namespace

LogicalResult EndpointWorkOp::verify() {
  if (getName().empty() || getPayloadBytesAttr().getInt() < 0 || getMaxItemsAttr().getInt() <= 0)
    return emitOpError("requires a name, nonnegative payloadBytes and positive maxItems");
  if (auto count = getRankCountAttr(); count &&
      (count.getInt() <= 0 || count.getInt() > std::numeric_limits<int32_t>::max()))
    return emitOpError("rankCount must be a positive signed i32 extent");
  std::set<std::tuple<std::string, std::string, std::string, std::string>> identities;
  for (Attribute attr : getProfiles()) {
    auto profile = dyn_cast<DictionaryAttr>(attr);
    if (!profile)
      return emitOpError("profiles must contain dictionaries");
    auto backend = stringField(profile, "backend");
    auto platform = stringField(profile, "platform");
    auto memory = stringField(profile, "queueMemory");
    if (!knownBackend(backend) || platform.empty() || memory.empty() ||
        stringField(profile, "identity").empty() ||
        integerField(profile, "bytesPerEndpoint") <= 0 ||
        integerField(profile, "maxLanes") <= 0)
      return emitOpError("requires complete positive measured profile facts");
    if (profile.get("peerLink") && !profile.getAs<StringAttr>("peerLink"))
      return emitOpError("profile peerLink must be a string");
    if (profile.get("senderWarps")) {
      int64_t warps = integerField(profile, "senderWarps");
      if (warps != 1 && warps != 2 && warps != 4 && warps != 8)
        return emitOpError("measured senderWarps must be 1, 2, 4 or 8");
    }
    if (!identities.emplace(backend.str(), platform.str(), memory.str(),
                            stringField(profile, "peerLink").str()).second)
      return emitOpError("duplicate platform profile");
  }
  llvm::DenseSet<int32_t> seen;
  ArrayAttr worldImage;
  ArrayAttr worldLinks;
  bool firstPartition = true;
  for (Attribute attr : getPartitions()) {
    auto partition = dyn_cast<DictionaryAttr>(attr);
    if (!partition)
      return emitOpError("partitions must contain dictionaries");
    auto peers = partition.getAs<DenseI32ArrayAttr>("peers");
    auto backend = stringField(partition, "backend");
    if (!knownBackend(backend) || !peers || peers.empty() ||
        stringField(partition, "queueMemory").empty() ||
        integerField(partition, "availableLanes") <= 0 ||
        integerField(partition, "availableLanes") > std::numeric_limits<int32_t>::max())
      return emitOpError("requires registered backend, peers, queue memory and positive availableLanes");
    int32_t previous = -1;
    for (int32_t peer : peers.asArrayRef()) {
      if (peer < 0 || peer <= previous || !seen.insert(peer).second)
        return emitOpError("requires sorted unique nonnegative peers across partitions");
      previous = peer;
      if (auto count = getRankCountAttr(); count && peer >= count.getInt())
        return emitOpError("peer is outside rankCount");
    }
    auto participants = partition.getAs<ArrayAttr>("participants");
    if (partition.get("peerLink") && !partition.getAs<StringAttr>("peerLink"))
      return emitOpError("registered peerLink must be a string");
    auto links = partition.getAs<ArrayAttr>("participantLinks");
    if (partition.get("participantLinks") && !links)
      return emitOpError("participantLinks must be a canonical string array");
    if (links) {
      if (!getRankCountAttr() || links.size() != uint64_t(getRankCountAttr().getInt()) ||
          !llvm::all_of(links, [](Attribute attr) { return isa<StringAttr>(attr); }))
        return emitOpError("participantLinks requires one canonical string per rank");
    }
    if (!firstPartition && worldLinks != links)
      return emitOpError("all transport partitions require the same participant link image");
    worldLinks = links;
    firstPartition = false;
    if (getRankCountAttr() && !participants)
      return emitOpError("rank-indexed readiness requires a canonical participant image");
    if (participants) {
      if (!getRankCountAttr() || participants.size() != uint64_t(getRankCountAttr().getInt()))
        return emitOpError("participants requires one canonical fact per rank");
      for (Attribute attr : participants) {
        auto facts = dyn_cast<DictionaryAttr>(attr);
        if (!facts || !facts.getAs<StringAttr>("platform") ||
            stringField(facts, "queueMemory").empty() ||
            integerField(facts, "availableLanes") <= 0 ||
            integerField(facts, "availableLanes") > std::numeric_limits<int32_t>::max())
          return emitOpError("participant facts require platform, queueMemory and positive availableLanes");
      }
      if (worldImage && worldImage != participants)
        return emitOpError("all transport partitions require the same participant image");
      worldImage = participants;
    }
  }
  return success();
}

namespace mlir::aboleth {
#define GEN_PASS_DEF_ABOLETHPLANENDPOINTSPASS
#include "aboleth/Transforms/Passes.h.inc"
namespace {
class PlanEndpoints : public impl::AbolethPlanEndpointsPassBase<PlanEndpoints> {
public:
  using Base::Base;
  void runOnOperation() override {
    llvm::json::Array programs;
    bool invalid = false;
    getOperation().walk([&](EndpointWorkOp work) {
      Builder builder(work.getContext());
      SmallVector<Attribute> launchAttrs;
      llvm::json::Array launches;
      int64_t readyStride = 1;
      int64_t worldLanes = work.getMaxItemsAttr().getInt();
      if (work.getRankCountAttr() && !work.getPartitions().empty()) {
        auto first = cast<DictionaryAttr>(work.getPartitions()[0]);
        for (Attribute attr : first.getAs<ArrayAttr>("participants"))
          worldLanes = std::min(worldLanes, integerField(cast<DictionaryAttr>(attr), "availableLanes"));
        worldLanes = floorPowerOfTwo(worldLanes);
        readyStride = fanout ? worldLanes : 1;
      }
      for (Attribute attr : work.getPartitions()) {
        auto partition = cast<DictionaryAttr>(attr);
        auto backend = stringField(partition, "backend");
        auto peers = partition.getAs<DenseI32ArrayAttr>("peers");
        if (work.getRankCountAttr() && integerField(partition, "availableLanes") < worldLanes) {
          work.emitOpError("partition cannot supply the participant image's common endpoint budget");
          invalid = true;
          return;
        }
        int64_t lanes = 1;
        int64_t senderWarps = 1;
        StringRef identity;
        for (Attribute p : work.getProfiles()) {
          auto profile = cast<DictionaryAttr>(p);
          if (!fanout || stringField(profile, "backend") != backend ||
              stringField(profile, "platform") != stringField(partition, "platform") ||
              stringField(profile, "queueMemory") != stringField(partition, "queueMemory"))
            continue;
          auto peerLink = stringField(profile, "peerLink");
          if (!peerLink.empty()) {
            auto links = partition.getAs<ArrayAttr>("participantLinks");
            if (stringField(partition, "peerLink") != peerLink || !links ||
                !llvm::all_of(links, [&](Attribute attr) {
                  return cast<StringAttr>(attr).getValue() == peerLink;
                }))
              continue;
          }
          // A shared receive geometry must agree at both ends. Use one
          // canonical world image, not independent local hardware choices.
          // Heterogeneous/unqualified participants retain the common one-lane
          // plan; no peer can wait for a lane its sender did not publish.
          if (auto participants = partition.getAs<ArrayAttr>("participants");
              participants && !llvm::all_of(participants, [&](Attribute attr) {
                auto facts = cast<DictionaryAttr>(attr);
                // P2P does not consume an RDMA queue. The same canonical
                // participant may have host/GPU RDMA queues in a mixed world.
                // Queue agreement remains mandatory for RDMA profiles.
                bool queueMatches = backend == "p2p" &&
                    stringField(profile, "queueMemory") == "none";
                queueMatches |= stringField(facts, "queueMemory") == stringField(profile, "queueMemory");
                return stringField(facts, "platform") == stringField(profile, "platform") && queueMatches;
              }))
            continue;
          identity = stringField(profile, "identity");
          if (profile.get("senderWarps"))
            senderWarps = integerField(profile, "senderWarps");
          int64_t budget = integerField(profile, "bytesPerEndpoint");
          int64_t bytes = work.getPayloadBytesAttr().getInt();
          int64_t needed = bytes / budget + (bytes % budget != 0);
          int64_t cap = floorPowerOfTwo(std::min<int64_t>({worldLanes, work.getMaxItemsAttr().getInt(),
              integerField(partition, "availableLanes"), integerField(profile, "maxLanes")}));
          while (lanes < needed && lanes < cap)
            lanes *= 2;
        }
        readyStride = std::max(readyStride, lanes);
        if (static_cast<uint64_t>(peers.size()) >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max() / lanes)) {
          work.emitOpError("endpoint grid exceeds signed i32 index range");
          invalid = true;
          return;
        }
        SmallVector<int64_t> boundaries;
        llvm::json::Array slices;
        // Quotient/remainder partition avoids maxItems * lane overflow.
        for (int64_t lane = 0; lane <= lanes; ++lane) {
          int64_t boundary = (work.getMaxItemsAttr().getInt() / lanes) * lane +
              ((work.getMaxItemsAttr().getInt() % lanes) * lane) / lanes;
          boundaries.push_back(boundary);
          if (lane)
            slices.push_back(llvm::json::Array{boundaries[lane-1], boundary});
        }
        bool grouped = coalesceReceives && work.getGroupedReceiverSupportedAttr() &&
                       work.getGroupedReceiverSupportedAttr().getValue();
        int64_t receiverLanes = grouped ? lanes : 1;
        NamedAttrList launchFacts({
          builder.getNamedAttr("backend", builder.getStringAttr(backend)),
          builder.getNamedAttr("peers", peers),
          builder.getNamedAttr("lanes", builder.getI64IntegerAttr(lanes)),
          builder.getNamedAttr("rowBoundaries", builder.getDenseI64ArrayAttr(boundaries)),
          builder.getNamedAttr("countLane", builder.getI64IntegerAttr(0)),
          builder.getNamedAttr("signalSlots", builder.getI64IntegerAttr(peers.size()*lanes)),
          builder.getNamedAttr("profileIdentity", builder.getStringAttr(identity))});
        if (work.getGroupedReceiverSupportedAttr())
          launchFacts.set("receiverLanesPerProgram", builder.getI64IntegerAttr(receiverLanes));
        if (senderWarps != 1)
          launchFacts.set("senderWarps", builder.getI64IntegerAttr(senderWarps));
        launchAttrs.push_back(launchFacts.getDictionary(work.getContext()));
        llvm::json::Object launch;
        launch["backend"] = backend;
        launch["peers"] = llvm::json::Array(peers.asArrayRef());
        launch["lanes"] = lanes;
        launch["slices"] = std::move(slices);
        launch["slice_quotient"] = work.getMaxItemsAttr().getInt() / lanes;
        launch["slice_remainder"] = work.getMaxItemsAttr().getInt() % lanes;
        launch["count_lane"] = 0;
        launch["signal_slots"] = int64_t(peers.size()) * lanes;
        launch["sender_grid"] = llvm::json::Array{int64_t(peers.size()), lanes};
        launch["receiver_grid"] = llvm::json::Array{int64_t(peers.size()), lanes/receiverLanes};
        launch["receiver_lanes_per_program"] = receiverLanes;
        launch["sender_warps"] = senderWarps;
        launch["receiver_warps"] = int64_t{1};
        launch["profile_identity"] = identity;
        launches.push_back(std::move(launch));
      }
      // Rank-indexed readiness uses one stride across mixed transports. A
      // P2P one-lane sender must not alias an RDMA sender's four-lane cells.
      for (auto [index, value] : llvm::enumerate(launchAttrs)) {
        auto attribute = cast<DictionaryAttr>(value);
        auto *launch = launches[index].getAsObject();
        (*launch)["ready_stride"] = readyStride;
        if (auto ranks = work.getRankCountAttr()) {
          if (ranks.getInt() > std::numeric_limits<int32_t>::max()/readyStride) {
            work.emitOpError("rank readiness extent exceeds signed i32 indexing");
            invalid = true;
            return;
          }
          int64_t slots = ranks.getInt()*readyStride;
          NamedAttrList facts(attribute.getValue());
          facts.set("readyStride", builder.getI64IntegerAttr(readyStride));
          facts.set("signalSlots", builder.getI64IntegerAttr(slots));
          launchAttrs[index] = facts.getDictionary(work.getContext());
          (*launch)["signal_slots"] = slots;
        }
      }
      work->setAttr("launchPlans", builder.getArrayAttr(launchAttrs));
      llvm::json::Object program;
      program["name"] = work.getName();
      program["payload_bytes"] = work.getPayloadBytes();
      program["max_items"] = work.getMaxItems();
      program["launches"] = std::move(launches);
      programs.push_back(std::move(program));
    });
    if (invalid)
      return signalPassFailure();
    if (!emitPlanJson.empty()) {
      std::error_code error;
      llvm::raw_fd_ostream stream(emitPlanJson, error);
      if (error) {
        getOperation().emitError("cannot write endpoint plan: ") << error.message();
        return signalPassFailure();
      }
      llvm::json::Object document;
      document["programs"] = std::move(programs);
      llvm::json::OStream json(stream, 2);
      json.value(llvm::json::Value(std::move(document)));
      stream << '\n';
      stream.close();
      if (stream.has_error()) {
        getOperation().emitError("failed to write endpoint plan");
        signalPassFailure();
      }
    }
  }
};
} // namespace
std::unique_ptr<Pass> createAbolethPlanEndpointsPass(const AbolethPlanEndpointsPassOptions &options) {
  return std::make_unique<PlanEndpoints>(options);
}
} // namespace mlir::aboleth
