// RUN: aboleth-opt %s --pass plan-transport | FileCheck %s

module {
  aboleth.transport_program {
    "aboleth.transport_phase"() <{
      name = "dispatch",
      commandsPerIteration = 4 : i32,
      iterationCount = 1 : i32,
      payloadBytes = 2048 : i64,
      peerTransports = ["rdma", "p2p"],
      peerCapacities = array<i32: 2048, 1>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 0 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 2048 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}],
      writes = [{allocation = 1 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 2048 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "experts",
      reads = [{allocation = 1 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 2048 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}],
      writes = [{allocation = 2 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 2048 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_phase"() <{
      name = "combine",
      commandsPerIteration = 2 : i32,
      iterationCount = 1 : i32,
      payloadBytes = 2048 : i64,
      peerTransports = ["rdma", "p2p"],
      peerCapacities = array<i32: 2048, 1>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 2 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 2048 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}],
      writes = [{allocation = 3 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 2048 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "next_pack",
      reads = [],
      writes = [{allocation = 0 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 2048 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    ub.unreachable
  } {
    cyclic = true,
    name = "phase-program"
  }
}

// CHECK: "aboleth.transport_phase"()
// CHECK-SAME: completionBefore = "next_pack"
// CHECK-SAME: completionIterationDelta = 0 : i32
// CHECK-SAME: name = "dispatch"
// CHECK-SAME: publicationBefore = "experts"
// CHECK-SAME: publicationIterationDelta = 0 : i32
// CHECK-SAME: resolvedCompletionInterval = 1 : i32
// CHECK: "aboleth.transport_phase"()
// CHECK-SAME: completionBefore = "experts"
// CHECK-SAME: completionIterationDelta = 1 : i32
// CHECK-SAME: name = "combine"
// CHECK-SAME: publicationBefore = "program_end"
// CHECK-SAME: publicationIterationDelta = 0 : i32
// CHECK-SAME: resolvedCompletionInterval = 1 : i32
