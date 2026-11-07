// RUN: aboleth-opt %s --pass simplify-communication-dataflow | FileCheck %s
// RUN: aboleth-opt %s --pass simplify-communication-dataflow --pass simplify-communication-dataflow | FileCheck %s

module {
  aboleth.communication_dataflow "bf16-direct" {
    %source = aboleth.region {
      name = "bf16.source",
      elementType = bf16,
      shape = array<i64: 128, 1024>,
      strides = array<i64: 1024, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %staging = aboleth.region {
      name = "bf16.staging",
      elementType = bf16,
      shape = array<i64: 128, 1024>,
      strides = array<i64: 1024, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %final = aboleth.region {
      name = "bf16.final",
      elementType = bf16,
      shape = array<i64: 128, 1024>,
      strides = array<i64: 1024, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region
    %received = aboleth.transfer %source into %staging {
      id = "bf16.transfer",
      outputId = "bf16.received"
    }
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %materialized = aboleth.copy %received into %final {
      id = "bf16.copy",
      outputId = "bf16.output"
    }
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %materialized {name = "bf16.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }

  aboleth.communication_dataflow "singleton-reduction" {
    %contribution = aboleth.region {
      name = "singleton.contribution",
      elementType = bf16,
      shape = array<i64: 64, 256>,
      strides = array<i64: 256, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %staging = aboleth.region {
      name = "singleton.staging",
      elementType = bf16,
      shape = array<i64: 64, 256>,
      strides = array<i64: 256, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %final = aboleth.region {
      name = "singleton.final",
      elementType = bf16,
      shape = array<i64: 64, 256>,
      strides = array<i64: 256, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region
    %reduced = aboleth.reduction %contribution into %staging {
      cardinality = 1 : i64,
      id = "singleton.reduction",
      outputId = "singleton.reduced",
      operatorName = "add"
    } : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %materialized = aboleth.copy %reduced into %final {
      id = "singleton.copy",
      outputId = "singleton.output"
    }
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %materialized {name = "singleton.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }

  aboleth.communication_dataflow "top-k-reduction" {
    %contributions = aboleth.region {
      name = "topk.contributions",
      elementType = f32,
      shape = array<i64: 64, 256>,
      strides = array<i64: 256, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %staging = aboleth.region {
      name = "topk.staging",
      elementType = f32,
      shape = array<i64: 64, 256>,
      strides = array<i64: 256, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %final = aboleth.region {
      name = "topk.final",
      elementType = f32,
      shape = array<i64: 64, 256>,
      strides = array<i64: 256, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = true
    } : !aboleth.logical_region
    %reduced = aboleth.reduction %contributions into %staging {
      cardinality = 2 : i64,
      operatorName = "add"
    } : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %materialized = aboleth.copy %reduced into %final
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %materialized {name = "topk.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }

  aboleth.communication_dataflow "immutable-alias" {
    %metadata = aboleth.region {
      name = "metadata.source",
      elementType = i32,
      shape = array<i64: 128>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = true,
      escapes = false
    } : !aboleth.logical_region
    %duplicate = aboleth.region {
      name = "metadata.duplicate",
      elementType = i32,
      shape = array<i64: 128>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %copied = aboleth.copy %metadata into %duplicate
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %copied {name = "metadata.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }

  aboleth.communication_dataflow "view-alias-safe" {
    %base = aboleth.region {
      name = "view_alias.base",
      elementType = i32,
      shape = array<i64: 128>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = true,
      escapes = false
    } : !aboleth.logical_region
    %view = aboleth.view %base {
      name = "view_alias.view",
      elementType = i32,
      shape = array<i64: 128>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = true,
      escapes = false
    } : !aboleth.logical_region -> !aboleth.logical_region
    %destination = aboleth.region {
      name = "view_alias.destination",
      elementType = i32,
      shape = array<i64: 128>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %copied = aboleth.copy %view into %destination
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %copied {name = "view_alias.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }

  aboleth.communication_dataflow "multi-copy-fixed-point" {
    %source = aboleth.region {
      name = "fixed.source",
      elementType = bf16,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %stage0 = aboleth.region {
      name = "fixed.stage0",
      elementType = bf16,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %stage1 = aboleth.region {
      name = "fixed.stage1",
      elementType = bf16,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = false,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region
    %final = aboleth.region {
      name = "fixed.final",
      elementType = bf16,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region
    %received = aboleth.transfer %source into %stage0
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %copied0 = aboleth.copy %received into %stage1
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %copied1 = aboleth.copy %copied0 into %final
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %copied1 {name = "fixed.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }

  aboleth.communication_dataflow "dead-route-map" {
    %route = aboleth.region {
      name = "route_map.source",
      elementType = i32,
      shape = array<i64: 128>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local",
      immutable = true,
      escapes = false
    } : !aboleth.logical_region
    %remote = aboleth.region {
      name = "route_map.output",
      elementType = i32,
      shape = array<i64: 128>,
      strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "peer",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %unused = aboleth.transfer %route into %remote
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    ub.unreachable
  }

  aboleth.communication_dataflow "fp8-multi-output" {
    %hidden_source = aboleth.region {
      name = "fp8.hidden.source",
      elementType = i8,
      shape = array<i64: 64, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %hidden_staging = aboleth.region {
      name = "fp8.hidden.staging",
      elementType = i8,
      shape = array<i64: 64, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %hidden_final = aboleth.region {
      name = "fp8.hidden.final",
      elementType = i8,
      shape = array<i64: 64, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region
    %scale_source = aboleth.region {
      name = "fp8.scale.source",
      elementType = f32,
      shape = array<i64: 64, 8>,
      strides = array<i64: 8, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %scale_staging = aboleth.region {
      name = "fp8.scale.staging",
      elementType = f32,
      shape = array<i64: 64, 8>,
      strides = array<i64: 8, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %scale_storage = aboleth.region {
      name = "fp8.scale.final.storage",
      elementType = f32,
      shape = array<i64: 64, 8>,
      strides = array<i64: 8, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region
    %scale_final = aboleth.view %scale_storage {
      name = "fp8.scale.final.view",
      elementType = f32,
      shape = array<i64: 64, 8>,
      strides = array<i64: 1, 64>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region -> !aboleth.logical_region
    %hidden_received = aboleth.transfer
      %hidden_source into %hidden_staging
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %hidden_materialized = aboleth.copy
      %hidden_received into %hidden_final
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %scale_received = aboleth.transfer
      %scale_source into %scale_staging
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %scale_materialized = aboleth.copy
      %scale_received into %scale_final
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %hidden_materialized, %scale_materialized {
      name = "fp8.consumer"
    } : !aboleth.logical_region, !aboleth.logical_region
    ub.unreachable
  }

  aboleth.communication_dataflow "partial-residual" {
    %direct_source = aboleth.region {
      name = "partial.direct.source",
      elementType = bf16,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %direct_staging = aboleth.region {
      name = "partial.direct.staging",
      elementType = bf16,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %direct_final = aboleth.region {
      name = "partial.direct.final",
      elementType = bf16,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = true,
      registered = true,
      donated = true,
      reuseInterval = 1 : i64,
      poolDepth = 1 : i64
    } : !aboleth.logical_region
    %reduce_source = aboleth.region {
      name = "partial.reduce.source",
      elementType = f32,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %reduce_staging = aboleth.region {
      name = "partial.reduce.staging",
      elementType = f32,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %reduce_final = aboleth.region {
      name = "partial.reduce.final",
      elementType = f32,
      shape = array<i64: 32, 128>,
      strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = true
    } : !aboleth.logical_region
    %direct_received = aboleth.transfer
      %direct_source into %direct_staging
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %direct_materialized = aboleth.copy
      %direct_received into %direct_final
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %reduced = aboleth.reduction %reduce_source into %reduce_staging {
      cardinality = 2 : i64,
      operatorName = "add"
    } : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    %reduce_materialized = aboleth.copy %reduced into %reduce_final
      : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %direct_materialized, %reduce_materialized {
      name = "partial.consumer"
    } : !aboleth.logical_region, !aboleth.logical_region
    ub.unreachable
  }

  aboleth.communication_dataflow "explicit-transform" {
    %source = aboleth.region {
      name = "transform.source",
      elementType = f32,
      shape = array<i64: 2, 4>,
      strides = array<i64: 4, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = false
    } : !aboleth.logical_region
    %destination = aboleth.region {
      name = "transform.destination",
      elementType = f32,
      shape = array<i64: 4, 2>,
      strides = array<i64: 2, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local",
      immutable = false,
      escapes = true
    } : !aboleth.logical_region
    %transformed = aboleth.transform %source into %destination {
      transform = "transpose",
      indexMap = affine_map<(d0, d1) -> (d1, d0)>
    } : (!aboleth.logical_region, !aboleth.logical_region)
        -> !aboleth.logical_region
    aboleth.use %transformed {name = "transform.consumer"}
      : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK-LABEL: aboleth.communication_dataflow "bf16-direct"
// CHECK: %[[BF16_SOURCE:.*]] = aboleth.region {{.*}}name = "bf16.source"
// CHECK-NOT: name = "bf16.staging"
// CHECK: %[[BF16_FINAL:.*]] = aboleth.region {{.*}}name = "bf16.final"
// CHECK: %[[BF16_DIRECT:.*]] = aboleth.transfer %[[BF16_SOURCE]] into %[[BF16_FINAL]]
// CHECK-SAME: id = "bf16.transfer"
// CHECK-SAME: outputId = "bf16.output"
// CHECK-NOT: aboleth.copy
// CHECK: aboleth.use %[[BF16_DIRECT]] {{.*}}name = "bf16.consumer"

// CHECK-LABEL: aboleth.communication_dataflow "singleton-reduction"
// CHECK: %[[SINGLE_SOURCE:.*]] = aboleth.region {{.*}}name = "singleton.contribution"
// CHECK-NOT: name = "singleton.staging"
// CHECK: %[[SINGLE_FINAL:.*]] = aboleth.region {{.*}}name = "singleton.final"
// CHECK-NOT: aboleth.reduction
// CHECK: %[[SINGLE_DIRECT:.*]] = aboleth.transfer %[[SINGLE_SOURCE]] into %[[SINGLE_FINAL]]
// CHECK-SAME: id = "singleton.reduction"
// CHECK-SAME: outputId = "singleton.output"
// CHECK-NOT: aboleth.copy
// CHECK: aboleth.use %[[SINGLE_DIRECT]]

// CHECK-LABEL: aboleth.communication_dataflow "top-k-reduction"
// CHECK: %[[TOPK_REDUCED:.*]] = aboleth.reduction
// CHECK-SAME: cardinality = 2 : i64
// CHECK: %[[TOPK_RESULT:.*]] = aboleth.copy %[[TOPK_REDUCED]]
// CHECK: aboleth.use %[[TOPK_RESULT]]

// CHECK-LABEL: aboleth.communication_dataflow "immutable-alias"
// CHECK: %[[METADATA:.*]] = aboleth.region {{.*}}name = "metadata.source"
// CHECK-NOT: name = "metadata.duplicate"
// CHECK-NOT: aboleth.copy
// CHECK: aboleth.use %[[METADATA]] {{.*}}name = "metadata.consumer"

// CHECK-LABEL: aboleth.communication_dataflow "view-alias-safe"
// CHECK: %[[VIEW_ALIAS:.*]] = aboleth.view
// CHECK: %[[VIEW_COPY:.*]] = aboleth.copy %[[VIEW_ALIAS]]
// CHECK: aboleth.use %[[VIEW_COPY]]

// CHECK-LABEL: aboleth.communication_dataflow "multi-copy-fixed-point"
// CHECK: %[[FIXED_SOURCE:.*]] = aboleth.region {{.*}}name = "fixed.source"
// CHECK-NOT: name = "fixed.stage0"
// CHECK-NOT: name = "fixed.stage1"
// CHECK: %[[FIXED_FINAL:.*]] = aboleth.region {{.*}}name = "fixed.final"
// CHECK: %[[FIXED_TRANSFER:.*]] = aboleth.transfer %[[FIXED_SOURCE]] into %[[FIXED_FINAL]]
// CHECK-NOT: aboleth.copy
// CHECK: aboleth.use %[[FIXED_TRANSFER]]

// CHECK-LABEL: aboleth.communication_dataflow "dead-route-map"
// CHECK-NEXT: ub.unreachable

// CHECK-LABEL: aboleth.communication_dataflow "fp8-multi-output"
// CHECK-NOT: name = "fp8.hidden.staging"
// CHECK: %[[HIDDEN_FINAL:.*]] = aboleth.region {{.*}}name = "fp8.hidden.final"
// CHECK-NOT: name = "fp8.scale.staging"
// CHECK: %[[SCALE_FINAL:.*]] = aboleth.view {{.*}}name = "fp8.scale.final.view"
// CHECK-SAME: strides = array<i64: 1, 64>
// CHECK: %[[HIDDEN_DIRECT:.*]] = aboleth.transfer {{.*}} into %[[HIDDEN_FINAL]]
// CHECK-NOT: aboleth.copy
// CHECK: %[[SCALE_DIRECT:.*]] = aboleth.transfer {{.*}} into %[[SCALE_FINAL]]
// CHECK-NOT: aboleth.copy
// CHECK: aboleth.use %[[HIDDEN_DIRECT]], %[[SCALE_DIRECT]]

// CHECK-LABEL: aboleth.communication_dataflow "partial-residual"
// CHECK-NOT: name = "partial.direct.staging"
// CHECK: %[[PARTIAL_DIRECT_FINAL:.*]] = aboleth.region {{.*}}name = "partial.direct.final"
// CHECK: %[[PARTIAL_DIRECT:.*]] = aboleth.transfer {{.*}} into %[[PARTIAL_DIRECT_FINAL]]
// CHECK: %[[PARTIAL_REDUCED:.*]] = aboleth.reduction
// CHECK-SAME: cardinality = 2 : i64
// CHECK: %[[PARTIAL_RESIDUAL:.*]] = aboleth.copy %[[PARTIAL_REDUCED]]
// CHECK: aboleth.use %[[PARTIAL_DIRECT]], %[[PARTIAL_RESIDUAL]]

// CHECK-LABEL: aboleth.communication_dataflow "explicit-transform"
// CHECK: %[[TRANSFORMED:.*]] = aboleth.transform
// CHECK-SAME: transform = "transpose"
// CHECK: aboleth.use %[[TRANSFORMED]]
