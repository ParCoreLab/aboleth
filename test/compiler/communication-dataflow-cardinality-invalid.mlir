// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "invalid-cardinality" {
    %source = aboleth.region {
      name = "invalid.cardinality.source",
      elementType = f32,
      shape = array<i64: 8>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %destination = aboleth.region {
      name = "invalid.cardinality.destination",
      elementType = f32,
      shape = array<i64: 8>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "peer",
      immutable = false,
      escapes = true
    } : !aboleth.logical_region
    %result = aboleth.reduction %source into %destination {
      cardinality = 0 : i64,
      operatorName = "add"
    } : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %result {name = "invalid.cardinality.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.reduction' op cardinality must be positive
