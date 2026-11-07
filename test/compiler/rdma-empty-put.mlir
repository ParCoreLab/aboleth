// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-rdma | FileCheck %s
module {
  tt.func public @dynamic_put(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>, %bytes: i32, %key: i32, %active: i1) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %key : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %key : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %yes = arith.constant true
    %put = aboleth.put %ep, %dst, %src, %bytes, %active : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %ready = aboleth.signal %ep, %signal_dst, %signal_src, %put : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %ready : !aboleth.token
    tt.return
  }
}

module {
  tt.func public @zero_put(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>, %bytes: i32, %key: i32, %active: i1) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %key : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %key : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %yes = arith.constant true
    %zero = arith.constant 0 : i32
    %put = aboleth.put %ep, %dst, %src, %zero, %yes : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %ready = aboleth.signal %ep, %signal_dst, %signal_src, %put : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %ready : !aboleth.token
    tt.return
  }
}

// CHECK-LABEL: tt.func public @dynamic_put
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[NONEMPTY:.*]] = arith.cmpi ne, {{.*}}, %[[ZERO]] : i32
// CHECK: %[[ACTIVE:.*]] = arith.andi {{.*}}, %[[NONEMPTY]] : i1
// CHECK: arith.select %[[ACTIVE]],
// CHECK: tt.return
// CHECK-LABEL: tt.func public @zero_put
// CHECK: %[[DISABLED:.*]] = arith.constant false
// CHECK: arith.select %[[DISABLED]],
// CHECK: tt.return
