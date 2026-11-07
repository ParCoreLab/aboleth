// RUN: aboleth-opt %s --pass plan-transport | FileCheck %s

module {
  aboleth.transport_program {
    "aboleth.transport_phase"() <{
      name = "partial",
      commandsPerIteration = 1 : i32,
      iterationCount = 1 : i32,
      payloadBytes = 64 : i64,
      peerTransports = ["p2p"],
      peerCapacities = array<i32: 1>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 10 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}],
      writes = []
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "partial_writer",
      reads = [],
      writes = [{allocation = 10 : i64, generation = 1 : i64,
        offset = 32 : i64, length = 64 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_phase"() <{
      name = "disjoint",
      commandsPerIteration = 1 : i32,
      iterationCount = 1 : i32,
      payloadBytes = 64 : i64,
      peerTransports = ["p2p"],
      peerCapacities = array<i32: 1>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 11 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}],
      writes = []
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "disjoint_writer",
      reads = [],
      writes = [{allocation = 11 : i64, generation = 1 : i64,
        offset = 128 : i64, length = 64 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_phase"() <{
      name = "unknown",
      commandsPerIteration = 1 : i32,
      iterationCount = 1 : i32,
      payloadBytes = 64 : i64,
      peerTransports = ["p2p"],
      peerCapacities = array<i32: 1>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = -1 : i64, generation = -1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32, unknown = true}],
      writes = []
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "unknown_writer",
      reads = [],
      writes = [{allocation = 12 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_phase"() <{
      name = "stale_generation",
      commandsPerIteration = 1 : i32,
      iterationCount = 1 : i32,
      payloadBytes = 64 : i64,
      peerTransports = ["p2p"],
      peerCapacities = array<i32: 1>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 13 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}],
      writes = []
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "stale_writer",
      reads = [],
      writes = [{allocation = 13 : i64, generation = 2 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    ub.unreachable
  } {
    name = "region-effects"
  }
}

// CHECK: "aboleth.transport_phase"()
// CHECK-SAME: completionBefore = "partial_writer"
// CHECK-SAME: name = "partial"
// CHECK: "aboleth.transport_phase"()
// CHECK-SAME: completionBefore = "program_end"
// CHECK-SAME: name = "disjoint"
// CHECK: "aboleth.transport_phase"()
// CHECK-SAME: completionBefore = "unknown_writer"
// CHECK-SAME: name = "unknown"
// CHECK: "aboleth.transport_phase"()
// CHECK-SAME: completionBefore = "program_end"
// CHECK-SAME: name = "stale_generation"
