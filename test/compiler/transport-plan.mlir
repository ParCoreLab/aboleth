// RUN: aboleth-opt %s --pass plan-transport | FileCheck %s

module {
  aboleth.transport_program {
    "aboleth.transport_phase"() <{
      name = "independent",
      commandsPerIteration = 4 : i32,
      iterationCount = 100 : i32,
      payloadBytes = 65536 : i64,
      peerTransports = ["rdma"],
      peerCapacities = array<i32: 33>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 0 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}],
      writes = [{allocation = 1 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}]
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "experts",
      reads = [{allocation = 1 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 6400 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}],
      writes = [{allocation = 2 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 6400 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_phase"() <{
      name = "rotating",
      commandsPerIteration = 3 : i32,
      iterationCount = 7 : i32,
      payloadBytes = 65536 : i64,
      peerTransports = ["rdma"],
      peerCapacities = array<i32: 33>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 3 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 192 : i64, count = 7 : i32}],
      writes = [{allocation = 6 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 192 : i64, count = 7 : i32}]
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "reuse-send-slot-0",
      reads = [],
      writes = [{allocation = 0 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 6400 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_phase"() <{
      name = "tiny",
      commandsPerIteration = 1 : i32,
      iterationCount = 100 : i32,
      payloadBytes = 1024 : i64,
      peerTransports = ["rdma"],
      peerCapacities = array<i32: 129>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 4 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 8 : i64, stride = 8 : i64,
        period = 0 : i64, count = 100 : i32}],
      writes = [{allocation = 5 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 8 : i64, stride = 8 : i64,
        period = 0 : i64, count = 100 : i32}]
    }> : () -> ()
    ub.unreachable
  } {
    name = "transport-plan"
  }
  aboleth.transport_program {
    "aboleth.transport_kernel"() <{
      name = "produce-output",
      reads = [],
      writes = [{allocation = 10 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 4096 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "consume-output",
      reads = [{allocation = 10 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 4096 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}],
      writes = []
    }> : () -> ()
    ub.unreachable
  } {
    name = "output-lifetime",
    cyclic = true
  }
}

// CHECK: "aboleth.transport_phase"()
// CHECK-SAME: completionBefore = "reuse-send-slot-0"
// CHECK-SAME: name = "independent"
// CHECK-SAME: resolvedCompletionInterval = 8 : i32
// CHECK-SAME: reuseDistance = 100 : i32
// CHECK: "aboleth.transport_phase"()
// CHECK-SAME: completionBefore = "program_end"
// CHECK-SAME: name = "rotating"
// CHECK-SAME: resolvedCompletionInterval = 3 : i32
// CHECK-SAME: reuseDistance = 3 : i32
// CHECK: "aboleth.transport_phase"()
// CHECK-SAME: completionBefore = "program_end"
// CHECK-SAME: name = "tiny"
// CHECK-SAME: resolvedCompletionInterval = 100 : i32
// CHECK-SAME: reuseDistance = 100 : i32
// CHECK: "aboleth.transport_kernel"()
// CHECK-SAME: name = "produce-output"
// CHECK-SAME: outputLastConsumer = "consume-output"
// CHECK-SAME: outputLastConsumerIterationDelta = 0 : i32
// CHECK-SAME: outputPoolDepth = 1 : i32
// CHECK-SAME: outputReusable = true
// CHECK-SAME: outputReuseBefore = "produce-output"
// CHECK-SAME: outputReuseIterationDelta = 1 : i32
