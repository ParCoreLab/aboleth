// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  %region = aboleth.region {
    name = "invalid.containment",
    elementType = i32,
    shape = array<i64: 8>,
    strides = array<i64: 1>,
    indexMap = affine_map<(d0) -> (d0)>,
    visibility = "local",
    immutable = true,
    escapes = false
  } : !aboleth.logical_region
}

// CHECK: error: 'aboleth.region' op expects parent op 'aboleth.communication_dataflow'
