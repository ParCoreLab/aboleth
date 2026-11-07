// RUN: aboleth-opt %s --pass simplify-communication-dataflow | FileCheck %s

module {
  aboleth.communication_dataflow "opaque-alias" {
    %source = aboleth.region {
      name = "source", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local", immutable = false, escapes = false
    } : !aboleth.logical_region
    %staging = aboleth.region {
      name = "staging", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer", immutable = false, escapes = false
    } : !aboleth.logical_region
    %final = aboleth.region {
      name = "final", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer", immutable = false, escapes = true,
      registered = true, donated = true,
      reuseInterval = 1 : i64, poolDepth = 1 : i64
    } : !aboleth.logical_region
    %received = aboleth.transfer %source into %staging {
      id = "send", outputId = "received"
    } : (!aboleth.logical_region, !aboleth.logical_region) -> !aboleth.logical_region
    %output = aboleth.copy %received into %final {
      id = "copy", outputId = "output"
    } : (!aboleth.logical_region, !aboleth.logical_region) -> !aboleth.logical_region
    %view = aboleth.view %output {
      name = "output.view", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer", immutable = false, escapes = true,
      registered = true, donated = true,
      reuseInterval = 1 : i64, poolDepth = 1 : i64
    } : !aboleth.logical_region -> !aboleth.logical_region
    "aboleth.opaque_stage"(%view) <{
      name = "experts", outputIds = [],
      asynchronous = true, maxInflight = 2 : i64,
      eventOrdered = true,
      operandSegmentSizes = array<i32: 1, 0>
    }> : (!aboleth.logical_region) -> ()
    ub.unreachable
  }
}

// CHECK: aboleth.transfer
// CHECK: aboleth.copy
// CHECK: aboleth.view
// CHECK: "aboleth.opaque_stage"
