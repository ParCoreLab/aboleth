// RUN: triton-opt %s --aboleth-form-batches | FileCheck %s

module {
  tt.func public @batch(
      %endpoint: !tt.ptr<i8>,
      %destination0: !tt.ptr<i8>,
      %destination1: !tt.ptr<i8>,
      %source0: !tt.ptr<i8>,
      %source1: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i8> -> !aboleth.endpoint
    %dst0 = aboleth.bind_remote_region %destination0, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %dst1 = aboleth.bind_remote_region %destination1, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src0 = aboleth.bind_local_region %source0, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %src1 = aboleth.bind_local_region %source1, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst0, %src0, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %1 = aboleth.put %ep, %dst1, %src1, %bytes, %true, %0
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1, !aboleth.token) -> !aboleth.token
    %2 = aboleth.signal %ep, %signal_dst, %signal_src, %1
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    %3 = arith.addi %bytes, %bytes : i32
    aboleth.wait %2 : !aboleth.token
    tt.return
  }

  // The unrolled body of a fused producer: each chunk is loaded, staged, and
  // put before the next is produced, and the count reaches its cell under a
  // condition. The stores produce later commands' payloads, so the submission
  // formed at the signal still follows them.
  tt.func public @produce_between(
      %endpoint: !tt.ptr<i8>,
      %destination: !tt.ptr<i8>,
      %input: !tt.ptr<i8>,
      %stage0: !tt.ptr<i8>,
      %stage1: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %live: i1,
      %source_key: i32,
      %destination_key: i32) {
    %true = arith.constant true
    %epoch = arith.constant 1 : i64
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i8> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src0 = aboleth.bind_local_region %stage0, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %src1 = aboleth.bind_local_region %stage1, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %v0 = tt.load %input : !tt.ptr<i8>
    tt.store %stage0, %v0 : !tt.ptr<i8>
    %0 = aboleth.put %ep, %dst, %src0, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %v1 = tt.load %input : !tt.ptr<i8>
    tt.store %stage1, %v1 : !tt.ptr<i8>
    %1 = aboleth.put %ep, %dst, %src1, %bytes, %live, %0
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1, !aboleth.token) -> !aboleth.token
    scf.if %live {
      tt.store %local_signal, %epoch : !tt.ptr<i64>
    }
    %2 = aboleth.signal %ep, %signal_dst, %signal_src, %1
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %2 : !aboleth.token
    tt.return
  }
}

// CHECK-LABEL: tt.func public @batch
// CHECK: %[[PUT0:.*]] = aboleth.post_put
// CHECK: %[[PUT1:.*]] = aboleth.post_put
// CHECK: %[[SIGNAL:.*]] = aboleth.post_signal
// CHECK: %[[SUBMIT:.*]] = aboleth.submit
// CHECK-SAME: %[[PUT0]], %[[PUT1]], %[[SIGNAL]]
// CHECK-SAME: batchSize = 3 : i32
// CHECK: arith.addi
// CHECK: aboleth.poll %[[SUBMIT]]
// CHECK-NOT: aboleth.put
// CHECK-NOT: aboleth.signal
// CHECK-NOT: aboleth.wait

// CHECK-LABEL: tt.func public @produce_between
// CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i8>
// CHECK: aboleth.post_put
// CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i8>
// CHECK: aboleth.post_put
// CHECK: scf.if
// CHECK: aboleth.post_signal
// CHECK: aboleth.submit
// CHECK-SAME: batchSize = 3 : i32
// CHECK: aboleth.poll
