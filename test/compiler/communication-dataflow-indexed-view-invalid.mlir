// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "invalid-indexed-view" {
    %storage = aboleth.region {
      name = "storage",
      elementType = bf16,
      shape = array<i64: 16, 32>,
      strides = array<i64: 32, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region
    %indices = aboleth.region {
      name = "indices",
      elementType = i32,
      shape = array<i64: 8>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = true,
      escapes = false
    } : !aboleth.logical_region
    %view = aboleth.indexed_view %storage using %indices {
      name = "indexed",
      elementType = bf16,
      shape = array<i64: 8, 32>,
      strides = array<i64: 32, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      injective = true,
      indexLowerBounds = array<i64: 0>,
      indexUpperBounds = array<i64: 32>,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region, !aboleth.logical_region
      -> !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.indexed_view' op indexed view bounds are outside source storage
