// RUN: not aboleth-opt %s --pass plan-transport 2>&1 | FileCheck %s

module {
  aboleth.transport_program {
    "aboleth.transport_phase"() <{
      name = "invalid",
      commandsPerIteration = 3 : i32,
      iterationCount = 7 : i32,
      payloadBytes = 65536 : i64,
      completionInterval = 4 : i32,
      peerTransports = ["rdma"],
      peerCapacities = array<i32: 33>,
      reads = [{allocation = 0 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 192 : i64, count = 7 : i32}],
      writes = [{allocation = 1 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 192 : i64, count = 7 : i32}]
    }> : () -> ()
    ub.unreachable
  } {
    name = "invalid"
  }
}

// CHECK: error: 'aboleth.transport_phase' op explicit completion interval exceeds region lifetime or endpoint capacity
