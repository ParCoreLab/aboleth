// RUN: aboleth-opt %s --pass form-batches --pass lower-p2p | FileCheck %s
//
// Over peer memory an inline payload is written straight to the peer as
// little-endian words, with a ragged tail written by narrower stores, before
// the signal's release publishes it.

module {
  tt.func public @nine_bytes(
      %destination: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %count: i64,
      %tag: i8, %epoch: i64, %key: i32, %live: i1) {
    %endpoint = tt.int_to_ptr %epoch : i64 -> !tt.ptr<i64>
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %flag_dst = aboleth.bind_remote_region %flag, %key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %0 = aboleth.put_inline %ep, %dst, [%count, %tag], %live
      : (!aboleth.endpoint, !aboleth.remote_region, i64, i8, i1) -> !aboleth.token
    %1 = aboleth.signal %ep, %flag_dst, %epoch, %0
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    aboleth.wait %1 : !aboleth.token
    tt.return
  }

  tt.func public @three_bytes(
      %destination: !tt.ptr<i8>, %half: i16, %tag: i8, %epoch: i64,
      %key: i32) {
    %true = arith.constant true
    %endpoint = tt.int_to_ptr %epoch : i64 -> !tt.ptr<i64>
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %0 = aboleth.put_inline %ep, %dst, [%half, %tag], %true
      : (!aboleth.endpoint, !aboleth.remote_region, i16, i8, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }
}

// CHECK-LABEL: tt.func public @nine_bytes
// CHECK: scf.if %arg6
// CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i32>
// CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i32>
// CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i8>
// CHECK: tt.atomic_rmw exch, release, sys, %{{.*}}, %arg4
// CHECK-NOT: aboleth.

// CHECK-LABEL: tt.func public @three_bytes
// CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i16>
// CHECK: arith.shrui
// CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i8>
// CHECK-NOT: !tt.ptr<i32>
// CHECK-NOT: aboleth.
