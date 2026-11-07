// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "invalid-escape" {
    %region = aboleth.region {
      name = "invalid.escape",
      elementType = i32,
      shape = array<i64: 8>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "private",
      immutable = true,
      escapes = true
    } : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.region' op a private logical region cannot escape
