// RUN: aboleth-opt %s --pass plan-storage | FileCheck %s
// RUN: aboleth-opt %s --pass plan-storage=,unoptimized | FileCheck %s --check-prefix=GENERIC

module {
  aboleth.communication_dataflow "bounded-indexed-reader" {
    %payload = aboleth.region {name = "payload", elementType = bf16,
      shape = array<i64: 64, 32>, strides = array<i64: 32, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>, visibility = "peer",
      immutable = false, escapes = false, registered = true,
      poolDepth = 2 : i64, reuseInterval = 1 : i64} : !aboleth.logical_region
    %indices = aboleth.region {name = "indices", elementType = i32,
      shape = array<i64: 64>, strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>, visibility = "local",
      immutable = true, escapes = false} : !aboleth.logical_region
    %indexed = aboleth.indexed_view %payload using %indices {
      name = "indexed", elementType = bf16, shape = array<i64: 64, 32>,
      strides = array<i64: 32, 1>, indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer", immutable = false, escapes = false,
      registered = true, poolDepth = 2 : i64, reuseInterval = 1 : i64,
      injective = false, indexLowerBounds = array<i64: 0>,
      indexUpperBounds = array<i64: 8>}
      : !aboleth.logical_region, !aboleth.logical_region -> !aboleth.logical_region
    aboleth.use %indexed {name = "consumer"} : !aboleth.logical_region
    ub.unreachable
  }
}

// Logical shapes stay unchanged. A bounded, noninjective indexed reader can
// reuse eight physical rows for sixty-four logical demands. The index array
// still contains one entry per demand and must not shrink with the payload.
// CHECK: name = "payload"
// CHECK-SAME: physicalShape = array<i64: 8, 32>
// CHECK-SAME: shape = array<i64: 64, 32>
// CHECK: name = "indices"
// CHECK-SAME: physicalShape = array<i64: 64>
// CHECK: aboleth.indexed_view
// CHECK-SAME: injective = false
// GENERIC: name = "payload"
// GENERIC-SAME: physicalShape = array<i64: 64, 32>
// GENERIC-SAME: shape = array<i64: 64, 32>
