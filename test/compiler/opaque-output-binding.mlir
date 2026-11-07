module {
  aboleth.communication_dataflow "output-binding" {
    %input = aboleth.region {name = "input", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local", immutable = true, escapes = false} : !aboleth.logical_region
    %staging = aboleth.region {name = "staging", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local", immutable = false, escapes = false} : !aboleth.logical_region
    %return = aboleth.region {name = "return", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer", immutable = false, escapes = false,
      registered = true, donated = true, reuseInterval = 1 : i64,
      poolDepth = 1 : i64} : !aboleth.logical_region
    %expert = "aboleth.opaque_stage"(%input, %staging) <{
      name = "expert", outputIds = ["expert.output"], asynchronous = false,
      maxInflight = 1 : i64, eventOrdered = false, bindableOutputs = true,
      operandSegmentSizes = array<i32: 1, 1>
    }> : (!aboleth.logical_region, !aboleth.logical_region) -> (!aboleth.logical_region)
    %copied = aboleth.copy %expert into %return {id = "staging-copy", outputId = "ready"}
      : (!aboleth.logical_region, !aboleth.logical_region) -> !aboleth.logical_region
    aboleth.use %copied {name = "consumer", id = "consumer"} : !aboleth.logical_region
    ub.unreachable
  }
}
// CHECK-NOT: name = "staging"
// CHECK: %[[RETURN:.*]] = aboleth.region {{.*}}name = "return"
// CHECK: %[[OUTPUT:.*]] = "aboleth.opaque_stage"(%{{.*}}, %[[RETURN]])
// CHECK-SAME: bindableOutputs = true
// CHECK-NOT: aboleth.copy
// CHECK: aboleth.use %[[OUTPUT]]
// GENERIC: name = "staging"
// GENERIC: "aboleth.opaque_stage"
// GENERIC: aboleth.copy
