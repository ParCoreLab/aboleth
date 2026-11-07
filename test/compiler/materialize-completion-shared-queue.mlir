// Two reuse-planned chains on one endpoint leave their iterations outstanding
// on the same queue. Seven slots hold one iteration of both two-slot chains,
// so each completes every iteration (k = 1), where either alone would get
// k = 2.

module {
  tt.func public @shared(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %epoch: i64,
      %bytes: i32, %source_key: i32, %destination_key: i32) {
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
    %2 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %3 = aboleth.signal %ep, %flag_dst, %epoch, %2
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    aboleth.planned_wait %1, %epoch {reuseDistance = 12 : i64} : !aboleth.token, i64
    aboleth.planned_wait %3, %epoch {reuseDistance = 12 : i64} : !aboleth.token, i64
    tt.return
  }
}

// CHECK-LABEL: tt.func public @shared
// CHECK-NOT: arith.andi
// CHECK: aboleth.wait %{{.*}} : !aboleth.token
// CHECK-NEXT: aboleth.wait %{{.*}} : !aboleth.token
