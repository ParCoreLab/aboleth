// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

// A transport the capability registry does not know is rejected once, at the
// boundary, instead of silently planning as an unknown queue.

module {
  aboleth.transport_program {
    "aboleth.transport_phase"() <{
      name = "unknown",
      commandsPerIteration = 3 : i32,
      iterationCount = 7 : i32,
      payloadBytes = 65536 : i64,
      peerTransports = ["gin"],
      peerCapacities = array<i32: 33>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 0 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 192 : i64, count = 7 : i32}],
      writes = [{allocation = 1 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 192 : i64, count = 7 : i32}]
    }> : () -> ()
    ub.unreachable
  } {
    name = "unknown"
  }
}

// CHECK: error: 'aboleth.transport_phase' op unknown peer transport 'gin'
