module {
  aboleth.communication_dataflow "bad-mask-count" {
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
      eventOrdered = false, prefixReadCounts = ["n", "n"],
      operandSegmentSizes = array<i32: 1,1>}> :
      (!aboleth.logical_region,!aboleth.logical_region)->!aboleth.logical_region
    ub.unreachable
  }
}
// CHECK: prefixReadCounts requires one string per read operand
