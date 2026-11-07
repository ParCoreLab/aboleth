// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-p2p | FileCheck %s

module {
  tt.func public @producer(
      %endpoint: !tt.ptr<i8>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i8> -> !aboleth.endpoint
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

  tt.func public @typed_producer(
      %endpoint: !tt.ptr<i8>,
      %destination: !tt.ptr<i32>,
      %source: !tt.ptr<i32>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i8> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i32>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i32>, i32) -> !aboleth.local_region
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

  tt.func public @drain(%endpoint_ptr: !tt.ptr<i8>) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i8> -> !aboleth.endpoint
    aboleth.drain %endpoint : !aboleth.endpoint
    tt.return
  }

  // A compile-time payload size (Tier 2 bakes it) sizes the cooperative
  // chunk: 4 KiB is 512 eight-byte elements, not the 2048-element default.
  tt.func public @constant_size(
      %endpoint: !tt.ptr<i8>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %bytes = arith.constant 4096 : i32
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i8> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }

  tt.func public @consumer(
      %signal: !tt.ptr<i64>,
      %expected: i64) {
    %zero = arith.constant 0 : i32
    %local = aboleth.bind_local_region %signal, %zero
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    aboleth.wait_signal %local, %expected
      : (!aboleth.local_region, i64) -> ()
    tt.return
  }
}

// CHECK-LABEL: tt.func public @producer
// CHECK: scf.if
// CHECK: %[[SOURCE:[0-9]+]] = tt.ptr_to_int %arg2
// CHECK: %[[DESTINATION:[0-9]+]] = tt.ptr_to_int %arg1
// CHECK: %[[BYTES:[0-9]+]] = arith.extui %arg5
// CHECK: %[[POINTERS:[0-9]+]] = arith.ori %[[SOURCE]], %[[DESTINATION]]
// CHECK: %[[COMBINED:[0-9]+]] = arith.ori %[[POINTERS]], %[[BYTES]]
// CHECK: arith.andi %[[COMBINED]]
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: arith.constant 256 : i32
// CHECK: arith.cmpi uge
// CHECK: tt.make_range
// CHECK-SAME: tensor<2048xi32>
// CHECK: tt.load {{.*}} : tensor<2048x!tt.ptr<i64>>
// CHECK: tt.store {{.*}} : tensor<2048x!tt.ptr<i64>>
// The signal that follows publishes the tile: bar.sync, no membar.sys.
// CHECK: tt.elementwise_inline_asm "bar.sync 0;
// CHECK-NOT: membar.sys
// CHECK: arith.constant 8 : i32
// CHECK: tt.int_to_ptr {{.*}} -> !tt.ptr<i64>
// CHECK: tt.load {{.*}} : !tt.ptr<i64>
// CHECK: tt.store {{.*}} : !tt.ptr<i64>
// CHECK: } else {
// CHECK: arith.constant 1 : i32
// CHECK: tt.int_to_ptr {{.*}} -> !tt.ptr<i8>
// CHECK: tt.atomic_rmw exch, release, sys
// CHECK-LABEL: tt.func public @typed_producer
// CHECK: tt.ptr_to_int %arg2 : !tt.ptr<i32> -> i64
// CHECK: tt.ptr_to_int %arg1 : !tt.ptr<i32> -> i64
// CHECK: tt.int_to_ptr {{.*}} -> !tt.ptr<i64>
// CHECK: tt.int_to_ptr {{.*}} -> !tt.ptr<i8>
// CHECK: tt.atomic_rmw exch, release, sys
// A p2p put completes when its stores retire and the signal's release
// exchange publishes them, so there is no queue for quiet to drain.
// CHECK-LABEL: tt.func public @drain
// CHECK-NOT: aboleth.
// CHECK-LABEL: tt.func public @constant_size
// CHECK: tt.make_range
// CHECK-SAME: tensor<512xi32>
// CHECK-NOT: tensor<2048x
// No signal follows this put, so its copy keeps the system fence.
// CHECK: bar.sync 0;
// CHECK-SAME: membar.sys;
// CHECK-LABEL: tt.func public @consumer
// CHECK: scf.while
// CHECK: tt.elementwise_inline_asm "ld.acquire.sys.global.L1::no_allocate.u64
// CHECK-NOT: aboleth.
