// RUN: aboleth-opt %s --pass simplify-communication-dataflow | FileCheck %s

module {
  aboleth.communication_dataflow "async-lifetime" {
    %source = aboleth.region {
      name = "source", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local", immutable = false, escapes = false,
      registered = false, donated = false,
      reuseInterval = 0 : i64, poolDepth = 0 : i64
    } : !aboleth.logical_region
    %staging = aboleth.region {
      name = "staging", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer", immutable = false, escapes = false,
      registered = false, donated = false,
      reuseInterval = 0 : i64, poolDepth = 0 : i64
    } : !aboleth.logical_region
    %final = aboleth.region {
      name = "final", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer", immutable = false, escapes = true,
      registered = true, donated = true,
      reuseInterval = 1 : i64, poolDepth = 2 : i64
    } : !aboleth.logical_region
    %received = aboleth.transfer %source into %staging {
      id = "transfer", outputId = "received"
    } : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %output = aboleth.copy %received into %final {
      id = "copy", outputId = "output"
    } : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.async_use %output {
      name = "consumer", id = "consumer",
      maxInflight = 2 : i64, eventOrdered = true
    } : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: %[[FINAL:.*]] = aboleth.region {{.*}}name = "final"
// CHECK: %[[DIRECT:.*]] = aboleth.transfer %{{.*}} into %[[FINAL]]
// CHECK-NOT: aboleth.copy
// CHECK: aboleth.async_use %[[DIRECT]] {{.*}}eventOrdered = true{{.*}}maxInflight = 2
