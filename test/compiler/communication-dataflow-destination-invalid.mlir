// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "invalid-destination" {
    %source = aboleth.region {
      name = "invalid.destination.source",
      elementType = f32,
      shape = array<i64: 8>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %destination = aboleth.region {
      name = "invalid.destination.target",
      elementType = f32,
      shape = array<i64: 4>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "peer",
      immutable = false,
      escapes = true
    } : !aboleth.logical_region
    %result = aboleth.transfer %source into %destination
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %result {name = "invalid.destination.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.transfer' op source and destination logical shapes must match
