// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "invalid-view-map" {
    %source = aboleth.region {
      name = "source",
      elementType = f32,
      shape = array<i64: 8, 8>,
      strides = array<i64: 8, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %view = aboleth.view %source {
      name = "invalid.view",
      elementType = f32,
      shape = array<i64: 8, 8>,
      strides = array<i64: 8, 1>,
      indexMap = affine_map<(d0, d1) -> (d0 + d1, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region -> !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.view' op view indexMap must contain each dimension exactly once
