// RUN: aboleth-opt %s --pass plan-transport | FileCheck %s
module {
  aboleth.transport_program {
    ub.unreachable
  } {name = "mixed", peerRanks = array<i32: 0, 2, 3, 5>,
     peerTransports = ["rdma", "p2p", "rdma", "p2p"]}
}
// CHECK: launchPartitions = [{backend = "p2p", gridExtent = 2 : i32, peers = array<i32: 2, 5>}, {backend = "rdma", gridExtent = 2 : i32, peers = array<i32: 0, 3>}]
