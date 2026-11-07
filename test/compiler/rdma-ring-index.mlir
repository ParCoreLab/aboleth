// RUN: aboleth-opt %s --pass form-batches --pass lower-rdma | FileCheck %s
// SQ and CQ geometry are positive powers of two, so ring positions wrap
// with a mask: any remainder here is an unintended division.

module {
  tt.func public @payload(%endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>, %bytes: i32, %key: i32, %value: i64) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %key : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %key : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %sdst = aboleth.bind_remote_region %remote_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %ssrc = aboleth.bind_local_region %local_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %yes = arith.constant true
    %put = aboleth.put %ep, %dst, %src, %bytes, %yes : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %ready = aboleth.signal %ep, %sdst, %ssrc, %put : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %ready : !aboleth.token
    tt.return
  }
  tt.func public @inline_one(%endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>, %bytes: i32, %key: i32, %value: i64) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %key : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %key : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %sdst = aboleth.bind_remote_region %remote_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %ssrc = aboleth.bind_local_region %local_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %yes = arith.constant true
    %put = aboleth.put_inline %ep, %dst, [%value, %value], %yes : (!aboleth.endpoint, !aboleth.remote_region, i64, i64, i1) -> !aboleth.token
    %ready = aboleth.signal %ep, %sdst, %ssrc, %put : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %ready : !aboleth.token
    tt.return
  }
  tt.func public @inline_wide(%endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>, %bytes: i32, %key: i32, %value: i64) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %key : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %key : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %sdst = aboleth.bind_remote_region %remote_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %ssrc = aboleth.bind_local_region %local_signal, %key : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %yes = arith.constant true
    %put = aboleth.put_inline %ep, %dst, [%value, %value, %value, %value, %value, %value, %value, %value], %yes : (!aboleth.endpoint, !aboleth.remote_region, i64, i64, i64, i64, i64, i64, i64, i64, i1) -> !aboleth.token
    %ready = aboleth.signal %ep, %sdst, %ssrc, %put : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %ready : !aboleth.token
    tt.return
  }
}
// CHECK-LABEL: tt.func public @payload
// CHECK-NOT: arith.remui
// CHECK: %[[PAYLOAD:.*]] = arith.subi %{{.*}}, %c1_i64{{[0-9_]*}} : i64
// CHECK-NEXT: %{{.*}} = arith.andi %{{.*}}, %[[PAYLOAD]] : i64
// CHECK-NOT: arith.remui
// CHECK: tt.return
// CHECK-LABEL: tt.func public @inline_one
// CHECK-NOT: arith.remui
// CHECK: %[[INLINE_ONE:.*]] = arith.subi %{{.*}}, %c1_i64{{[0-9_]*}} : i64
// CHECK-NEXT: %{{.*}} = arith.andi %{{.*}}, %[[INLINE_ONE]] : i64
// CHECK-NOT: arith.remui
// CHECK: tt.return
// CHECK-LABEL: tt.func public @inline_wide
// CHECK-NOT: arith.remui
// CHECK: %[[INLINE_WIDE:.*]] = arith.subi %{{.*}}, %c1_i64{{[0-9_]*}} : i64
// CHECK-NEXT: %{{.*}} = arith.andi %{{.*}}, %[[INLINE_WIDE]] : i64
// CHECK-NOT: arith.remui
// CHECK: tt.return
