module {
  aboleth.communication_dataflow "invalid-input" {
    %x = aboleth.region {name = "x", elementType = bf16,
      shape = array<i64: 16,128>, strides = array<i64: 128,1>,
      indexMap = affine_map<(d0,d1)->(d0,d1)>, visibility = "local",
      immutable = true, escapes = false, validCount = "n"}
      : !aboleth.logical_region
    ub.unreachable
  }
}
// CHECK: validCount and deadPadding must be declared together
