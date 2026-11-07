// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "opaque-invalid" {
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
    "aboleth.opaque_stage"(%input) <{
      name = "experts",
      outputIds = [],
      asynchronous = false,
      maxInflight = 2 : i64,
      eventOrdered = false,
      operandSegmentSizes = array<i32: 1, 0>
    }> : (!aboleth.logical_region) -> ()
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.opaque_stage' op synchronous stages require maxInflight equal to one
