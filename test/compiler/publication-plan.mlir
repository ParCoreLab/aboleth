module {
  tt.func public @payload(
      %endpoint: !tt.ptr<i64>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %1 = aboleth.signal %ep, %signal_dst, %signal_src, %0
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %1 : !aboleth.token
    tt.return
  }

  tt.func public @credit(
      %endpoint: !tt.ptr<i64>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i64> -> !aboleth.endpoint
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %0 = aboleth.signal %ep, %signal_dst, %signal_src
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }
}

// CHECK-LABEL: tt.func public @payload
// CHECK: aboleth.post_signal
// CHECK-NOT: weakPublication
// CHECK: aboleth.submit
// CHECK-LABEL: tt.func public @credit
// CHECK: aboleth.post_signal
// CHECK-SAME: weakPublication
// CHECK: aboleth.submit

// P2P-LABEL: tt.func public @payload
// P2P: tt.atomic_rmw exch, release, sys
// P2P-LABEL: tt.func public @credit
// P2P: tt.atomic_rmw exch, release, sys

// RDMA-LABEL: tt.func public @payload
// RDMA: arith.constant 17 : i32
// RDMA-LABEL: tt.func public @credit
// RDMA-NOT: arith.constant 17 : i32

// LOWERED-LABEL: tt.func public @payload
// LOWERED-NOT: aboleth.
// LOWERED-LABEL: tt.func public @credit
// LOWERED-NOT: aboleth.
