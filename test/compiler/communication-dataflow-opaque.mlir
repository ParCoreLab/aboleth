// RUN: aboleth-opt %s --pass simplify-communication-dataflow | FileCheck %s

module {
  aboleth.communication_dataflow "opaque" {
    %input = aboleth.region {
      name = "input",
      elementType = bf16,
      shape = array<i64: 8, 16>,
      strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %storage = aboleth.region {
      name = "output.storage",
      elementType = bf16,
      shape = array<i64: 8, 16>,
      strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = true
    } : !aboleth.logical_region
    %output = "aboleth.opaque_stage"(%input, %storage) <{
      name = "experts",
      outputIds = ["output"],
      asynchronous = true,
      maxInflight = 2 : i64,
      eventOrdered = true,
      operandSegmentSizes = array<i32: 1, 1>
    }> : (!aboleth.logical_region, !aboleth.logical_region) -> (!aboleth.logical_region)
    aboleth.use %output {
      name = "consumer",
      id = "consumer"
    } : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: "aboleth.opaque_stage"
// CHECK-SAME: asynchronous = true
// CHECK-SAME: eventOrdered = true
// CHECK-SAME: name = "experts"
// CHECK-SAME: outputIds = ["output"]
// CHECK: aboleth.use
