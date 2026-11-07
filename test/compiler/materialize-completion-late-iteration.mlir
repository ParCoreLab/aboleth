// A frontend computes the iteration where the wait stands, after the signal
// whose completion request the predicate controls. The computation is free of
// side effects, so it moves in front of the signal with the predicate.

module {
  tt.func public @late_iteration(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %epoch: i64,
      %bytes: i32, %source_key: i32, %destination_key: i32) {
    %true = arith.constant true
    %one = arith.constant 1 : i64
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
    %iteration = arith.subi %epoch, %one : i64
    aboleth.planned_wait %1, %iteration {reuseDistance = 16 : i64} : !aboleth.token, i64
    tt.return
  }
}

// CHECK-LABEL: tt.func public @late_iteration
// CHECK: %[[ITERATION:.*]] = arith.subi %arg4, %{{.*}} : i64
// CHECK: arith.andi %[[ITERATION]]
// CHECK: aboleth.signal
// CHECK: aboleth.wait
