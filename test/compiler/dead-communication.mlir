module {
  aboleth.communication_dataflow "unused-result" {
    %source = aboleth.region {name = "source", elementType = bf16,
      shape = array<i64: 4, 128>, strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0,d1)->(d0,d1)>, visibility = "local",
      immutable = false, escapes = false} : !aboleth.logical_region
    %destination = aboleth.region {name = "destination", elementType = bf16,
      shape = array<i64: 4, 128>, strides = array<i64: 128, 1>,
      indexMap = affine_map<(d0,d1)->(d0,d1)>, visibility = "peer",
      immutable = false, escapes = false} : !aboleth.logical_region
    %unused = aboleth.transfer %source into %destination {id = "unused.transfer"}
      : (!aboleth.logical_region,!aboleth.logical_region)->!aboleth.logical_region
    ub.unreachable
  }
}
// CHECK: aboleth.communication_dataflow "unused-result"
// CHECK-NOT: aboleth.region
// CHECK-NOT: aboleth.transfer
// CHECK: ub.unreachable
// GENERIC: aboleth.communication_dataflow "unused-result"
// GENERIC: aboleth.region
// GENERIC: aboleth.region
// GENERIC: aboleth.transfer
