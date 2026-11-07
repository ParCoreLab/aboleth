module {
  aboleth.endpoint_work {name = "qualified-p2p", maxItems = 4096 : i64,
    payloadBytes = 16777216 : i64, rankCount = 2 : i64, groupedReceiverSupported = true,
    partitions = [{backend = "p2p", peers = array<i32: 1>, availableLanes = 16 : i64,
      platform = "measured", queueMemory = "none", peerLink = "NV18",
      participants = [{platform = "measured", queueMemory = "host", availableLanes = 16 : i64},
                      {platform = "measured", queueMemory = "gpu", availableLanes = 16 : i64}],
      participantLinks = ["NV18", "NV18"]}],
    profiles = [{backend = "p2p", platform = "measured", queueMemory = "none", peerLink = "NV18",
      bytesPerEndpoint = 262144 : i64, maxLanes = 16 : i64, senderWarps = 8 : i64, identity = "fixed-policy"}]}
  aboleth.endpoint_work {name = "unknown-peer", maxItems = 4096 : i64,
    payloadBytes = 16777216 : i64, rankCount = 2 : i64, groupedReceiverSupported = true,
    partitions = [{backend = "p2p", peers = array<i32: 1>, availableLanes = 16 : i64,
      platform = "measured", queueMemory = "none", peerLink = "NV18",
      participants = [{platform = "measured", queueMemory = "host", availableLanes = 16 : i64},
                      {platform = "measured", queueMemory = "host", availableLanes = 16 : i64}],
      participantLinks = ["NV18", ""]}],
    profiles = [{backend = "p2p", platform = "measured", queueMemory = "none", peerLink = "NV18",
      bytesPerEndpoint = 262144 : i64, maxLanes = 16 : i64, identity = "fixed-policy"}]}
}
// CHECK: launchPlans = [{backend = "p2p", countLane = 0 : i64, lanes = 16 : i64
// CHECK-SAME: receiverLanesPerProgram = 16 : i64
// CHECK-SAME: senderWarps = 8 : i64
// CHECK-SAME: signalSlots = 32 : i64
// CHECK: launchPlans = [{backend = "p2p", countLane = 0 : i64, lanes = 1 : i64
// CHECK-SAME: profileIdentity = ""
// GENERIC: launchPlans = [{backend = "p2p", countLane = 0 : i64, lanes = 1 : i64
// GENERIC-SAME: receiverLanesPerProgram = 1 : i64
// GENERIC-SAME: signalSlots = 2 : i64
// UNGROUPED: launchPlans = [{backend = "p2p", countLane = 0 : i64, lanes = 16 : i64
// UNGROUPED-SAME: receiverLanesPerProgram = 1 : i64
