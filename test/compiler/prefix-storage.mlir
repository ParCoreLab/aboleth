module {
  aboleth.communication_dataflow "prefix" {
    %x = aboleth.region {name = "x", elementType = bf16,
      shape = array<i64: 16, 128>, strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0,d1)->(d0,d1)>, visibility = "local",
      immutable = true, escapes = false, validCount = "n", deadPadding = true}
      : !aboleth.logical_region
    %y = aboleth.region {name = "y", elementType = bf16,
      shape = array<i64: 16, 128>, strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0,d1)->(d0,d1)>, visibility = "local",
      immutable = false, escapes = true} : !aboleth.logical_region
    %result = "aboleth.opaque_stage"(%x,%y) <{name = "compute",
      outputIds = ["output"], asynchronous = false, maxInflight = 1 : i64,
      eventOrdered = false, prefixReadCounts = ["n"],
      operandSegmentSizes = array<i32: 1,1>}> :
      (!aboleth.logical_region,!aboleth.logical_region)->!aboleth.logical_region
    ub.unreachable
  }
  aboleth.communication_dataflow "full-reader" {
    %x = aboleth.region {name = "x", elementType = bf16,
      shape = array<i64: 16, 128>, strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0,d1)->(d0,d1)>, visibility = "local",
      immutable = true, escapes = false, validCount = "n", deadPadding = true}
      : !aboleth.logical_region
    %y = aboleth.region {name = "y", elementType = bf16,
      shape = array<i64: 16, 128>, strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0,d1)->(d0,d1)>, visibility = "local",
      immutable = false, escapes = true} : !aboleth.logical_region
    %result = "aboleth.opaque_stage"(%x,%y) <{name = "compute",
      outputIds = ["output"], asynchronous = false, maxInflight = 1 : i64,
      eventOrdered = false, prefixReadCounts = ["n"],
      operandSegmentSizes = array<i32: 1,1>}> :
      (!aboleth.logical_region,!aboleth.logical_region)->!aboleth.logical_region
    aboleth.use %x {name = "observer"} : !aboleth.logical_region
    ub.unreachable
  }
}
// CHECK-LABEL: aboleth.communication_dataflow "prefix"
// CHECK: prefixBinding = "n"
// CHECK-LABEL: aboleth.communication_dataflow "full-reader"
// CHECK-NOT: prefixBinding
// CHECK: ub.unreachable
// GENERIC-LABEL: aboleth.communication_dataflow "prefix"
// GENERIC-NOT: prefixBinding
// GENERIC: ub.unreachable
// GENERIC-LABEL: aboleth.communication_dataflow "full-reader"
// GENERIC-NOT: prefixBinding
// GENERIC: ub.unreachable
