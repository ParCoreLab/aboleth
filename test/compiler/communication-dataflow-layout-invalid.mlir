// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "invalid-layout" {
    %region = aboleth.region {
      name = "invalid.layout",
      elementType = f32,
      shape = array<i64: 8, 8>,
      strides = array<i64: 8>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.region' op requires one stride for every shape dimension
