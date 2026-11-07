// RUN: aboleth-opt %s --pass plan-transport | FileCheck %s

module {
  aboleth.transport_program {
    "aboleth.transport_kernel"() <{
      name = "experts",
      asynchronous = true,
      maxInflight = 2 : i32,
      eventOrdered = true,
      reads = [{allocation = 1 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 4096 : i64, stride = 4096 : i64,
        period = 8192 : i64, count = 2 : i32}],
      writes = [{allocation = 2 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 4096 : i64, stride = 4096 : i64,
        period = 8192 : i64, count = 2 : i32}]
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "combine",
      reads = [{allocation = 2 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 4096 : i64, stride = 4096 : i64,
        period = 8192 : i64, count = 2 : i32}],
      writes = []
    }> : () -> ()
    ub.unreachable
  } {
    name = "async-output-lifetime",
    cyclic = true
  }
}

// CHECK: "aboleth.transport_kernel"()
// CHECK-SAME: asynchronous = true
// CHECK-SAME: eventOrdered = true
// CHECK-SAME: maxInflight = 2 : i32
// CHECK-SAME: name = "experts"
// CHECK-SAME: outputLastConsumer = "combine"
// CHECK-SAME: outputPoolDepth = 2 : i32
// CHECK-SAME: outputReusable = true
// CHECK-SAME: outputReuseBefore = "experts"
// CHECK-SAME: outputReuseIterationDelta = 1 : i32
