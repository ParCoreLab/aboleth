module {
  aboleth.endpoint_work {name = "mixed", maxItems = 512 : i64,
    payloadBytes = 2097152 : i64, rankCount = 3 : i64,
    partitions = [
      {backend = "p2p", peers = array<i32: 0>, availableLanes = 4 : i64,
       platform = "measured", queueMemory = "none", participants = [
         {platform = "measured", queueMemory = "host", availableLanes = 4 : i64},
         {platform = "measured", queueMemory = "host", availableLanes = 4 : i64},
         {platform = "measured", queueMemory = "host", availableLanes = 4 : i64}]},
      {backend = "rdma", peers = array<i32: 1>, availableLanes = 4 : i64,
       platform = "measured", queueMemory = "host", participants = [
         {platform = "measured", queueMemory = "host", availableLanes = 4 : i64},
         {platform = "measured", queueMemory = "host", availableLanes = 4 : i64},
         {platform = "measured", queueMemory = "host", availableLanes = 4 : i64}]}],
    profiles = [{backend = "rdma", platform = "measured", queueMemory = "host",
      bytesPerEndpoint = 262144 : i64, maxLanes = 4 : i64, identity = "study"}]}
}
// CHECK: launchPlans = [{backend = "p2p", countLane = 0 : i64, lanes = 1 : i64
// CHECK-SAME: readyStride = 4 : i64
// CHECK-SAME: signalSlots = 12 : i64
// CHECK-SAME: backend = "rdma", countLane = 0 : i64, lanes = 4 : i64
// CHECK-SAME: readyStride = 4 : i64
// CHECK-SAME: signalSlots = 12 : i64
// GENERIC: launchPlans = [{backend = "p2p", countLane = 0 : i64, lanes = 1 : i64
// GENERIC-SAME: readyStride = 1 : i64
// GENERIC-SAME: signalSlots = 3 : i64
// GENERIC-SAME: backend = "rdma", countLane = 0 : i64, lanes = 1 : i64
// GENERIC-SAME: readyStride = 1 : i64
// GENERIC-SAME: signalSlots = 3 : i64
