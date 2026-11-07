// RUN: aboleth-opt %s --pass plan-endpoints | FileCheck %s
module {
  aboleth.endpoint_work {name = "records", maxItems = 11 : i64,
    payloadBytes = 2097152 : i64,
    partitions = [{backend = "rdma", peers = array<i32: 1, 3>,
      availableLanes = 8 : i64, platform = "measured", queueMemory = "host"}],
    profiles = [{backend = "rdma", platform = "measured", queueMemory = "host",
      bytesPerEndpoint = 262144 : i64, maxLanes = 4 : i64, identity = "study"}]}
  aboleth.endpoint_work {name = "one-record", maxItems = 1 : i64,
    payloadBytes = 2097152 : i64,
    partitions = [{backend = "rdma", peers = array<i32: 1>,
      availableLanes = 8 : i64, platform = "measured", queueMemory = "host"}],
    profiles = [{backend = "rdma", platform = "measured", queueMemory = "host",
      bytesPerEndpoint = 262144 : i64, maxLanes = 4 : i64, identity = "study"}]}
}
// CHECK: launchPlans = [{backend = "rdma", countLane = 0 : i64, lanes = 4 : i64, peers = array<i32: 1, 3>, profileIdentity = "study", rowBoundaries = array<i64: 0, 2, 5, 8, 11>, signalSlots = 8 : i64}]
// CHECK: launchPlans = [{backend = "rdma", countLane = 0 : i64, lanes = 1 : i64, peers = array<i32: 1>, profileIdentity = "study", rowBoundaries = array<i64: 0, 1>, signalSlots = 1 : i64}]
// GENERIC: launchPlans = [{backend = "rdma", countLane = 0 : i64, lanes = 1 : i64, peers = array<i32: 1, 3>, profileIdentity = "", rowBoundaries = array<i64: 0, 11>, signalSlots = 2 : i64}]
// GENERIC: launchPlans = [{backend = "rdma", countLane = 0 : i64, lanes = 1 : i64, peers = array<i32: 1>, profileIdentity = "", rowBoundaries = array<i64: 0, 1>, signalSlots = 1 : i64}]
