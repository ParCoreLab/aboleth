// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "invalid-visibility" {
    %source = aboleth.region {
      name = "invalid.visibility.source",
      elementType = f32,
      shape = array<i64: 8>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %destination = aboleth.region {
      name = "invalid.visibility.destination",
      elementType = f32,
      shape = array<i64: 8>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = false,
      escapes = true
    } : !aboleth.logical_region
    %result = aboleth.transfer %source into %destination
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %result {name = "invalid.visibility.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.transfer' op communication destination must have peer or global visibility
