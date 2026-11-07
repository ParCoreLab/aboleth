// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "duplicate-id" {
    %source = aboleth.region {
      name = "duplicate",
      elementType = i32,
      shape = array<i64: 8>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %destination = aboleth.region {
      name = "destination",
      elementType = i32,
      shape = array<i64: 8>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region
    %result = aboleth.transfer %source into %destination {
      id = "duplicate",
      outputId = "result"
    } : (!aboleth.logical_region, !aboleth.logical_region)
      -> !aboleth.logical_region
    aboleth.use %result {name = "consumer", id = "consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.transfer' op dataflow identifiers must be globally unique
