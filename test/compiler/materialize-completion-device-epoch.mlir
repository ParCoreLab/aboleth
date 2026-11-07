// A device epoch observation dominates its transfer; it must remain an
// observation, with the completion predicate computed from that loaded value.
module {
  tt.func public @device_epoch(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %epoch_ptr: !tt.ptr<i64>,
      %bytes: i32, %source_key: i32, %destination_key: i32) {
    %epoch = tt.load %epoch_ptr : !tt.ptr<i64>
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %1 = aboleth.signal %ep, %flag_dst, %epoch, %0
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    aboleth.planned_wait %1, %epoch {reuseDistance = 4 : i64} : !aboleth.token, i64
    tt.return
  }
}

// CHECK-LABEL: tt.func public @device_epoch
// CHECK: %[[EPOCH:.*]] = tt.load
// CHECK: arith.constant 3 : i64
// CHECK: arith.andi %[[EPOCH]]
// CHECK: aboleth.signal
// CHECK: aboleth.wait %{{.*}}, %{{.*}} : !aboleth.token
// EAGER-NOT: arith.andi
// EAGER: aboleth.wait %{{.*}} : !aboleth.token
