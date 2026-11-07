// RUN: aboleth-opt %s --pass plan-transport | FileCheck %s

// Completion policy comes from the transport capability registry rather than
// per-transport attributes. A peer reached by loads and stores needs no early
// completion; a queue-bounded peer is bounded by its own capacity; and an
// explicit per-peer interval overrides the capability default.

module {
  aboleth.transport_program {
    // p2p is load/store addressable, so the whole phase may retire once.
    "aboleth.transport_phase"() <{
      name = "peer-memory",
      commandsPerIteration = 4 : i32,
      iterationCount = 100 : i32,
      payloadBytes = 65536 : i64,
      peerTransports = ["p2p"],
      peerCapacities = array<i32: 33>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 0 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}],
      writes = [{allocation = 1 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}]
    }> : () -> ()

    // efa is queue-capacity limited like rdma: capacity 33 over 4 commands
    // bounds completion to 8 iterations.
    "aboleth.transport_phase"() <{
      name = "unordered-nic",
      commandsPerIteration = 4 : i32,
      iterationCount = 100 : i32,
      payloadBytes = 65536 : i64,
      peerTransports = ["efa"],
      peerCapacities = array<i32: 33>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 2 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}],
      writes = [{allocation = 3 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}]
    }> : () -> ()

    // An explicit per-peer interval wins over the capability default of 16.
    "aboleth.transport_phase"() <{
      name = "override",
      commandsPerIteration = 1 : i32,
      iterationCount = 100 : i32,
      payloadBytes = 65536 : i64,
      peerTransports = ["rdma"],
      peerCapacities = array<i32: 4096>,
      peerCompletionIntervals = array<i32: 4>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 4 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}],
      writes = [{allocation = 5 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}]
    }> : () -> ()

    // A zero entry defers to the capability default, here rdma's 16.
    "aboleth.transport_phase"() <{
      name = "defaulted",
      commandsPerIteration = 1 : i32,
      iterationCount = 100 : i32,
      payloadBytes = 65536 : i64,
      peerTransports = ["rdma"],
      peerCapacities = array<i32: 4096>,
      peerCompletionIntervals = array<i32: 0>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 6 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}],
      writes = [{allocation = 7 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 100 : i32}]
    }> : () -> ()
    ub.unreachable
  } {
    name = "caps"
  }
}

// CHECK: name = "peer-memory"
// CHECK-SAME: resolvedCompletionInterval = 100 : i32
// CHECK: name = "unordered-nic"
// CHECK-SAME: resolvedCompletionInterval = 8 : i32
// CHECK: name = "override"
// CHECK-SAME: resolvedCompletionInterval = 4 : i32
// CHECK: name = "defaulted"
// CHECK-SAME: resolvedCompletionInterval = 16 : i32
