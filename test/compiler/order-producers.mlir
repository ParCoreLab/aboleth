// RUN: aboleth-opt %s --pass form-batches --pass order-producers | FileCheck %s
// RUN: aboleth-opt %s --pass form-batches --pass order-producers --pass lower-rdma | FileCheck %s --check-prefix=RDMA
//
// The thread that builds and announces a submission is not the only one that
// wrote its payload: a tensor store is spread over the whole block. One block
// barrier separates those writes from the first submission that reads a
// region after them.

module {
  tt.func public @tensor_producer(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %stage: !tt.ptr<i8>, %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>, %value: i8, %bytes: i32,
      %source_key: i32, %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %stage, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %stage : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %pointers = tt.addptr %base, %range : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %values = tt.splat %value : i8 -> tensor<4xi8>
    tt.store %pointers, %values : tensor<4x!tt.ptr<i8>>
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %1 = aboleth.signal %ep, %signal_dst, %signal_src, %0
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %1 : !aboleth.token
    // No write since the barrier above: this submission shares it.
    %2 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %3 = aboleth.signal %ep, %signal_dst, %signal_src, %2
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %3 : !aboleth.token
    tt.return
  }

  // A scalar store is issued by the announcing thread itself.
  tt.func public @scalar_producer(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %stage: !tt.ptr<i8>, %value: i8, %bytes: i32,
      %source_key: i32, %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %stage, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    tt.store %stage, %value : !tt.ptr<i8>
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }

  // An inline payload is read from registers, not from memory.
  tt.func public @inline_after_tensor_store(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %stage: !tt.ptr<i8>, %value: i8, %count: i64,
      %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %stage : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %pointers = tt.addptr %base, %range : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %values = tt.splat %value : i8 -> tensor<4xi8>
    tt.store %pointers, %values : tensor<4x!tt.ptr<i8>>
    %0 = aboleth.put_inline %ep, %dst, [%count], %true
      : (!aboleth.endpoint, !aboleth.remote_region, i64, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }

  // A write late in a loop body reaches the submission early in the next
  // iteration.
  tt.func public @loop_carried(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %stage: !tt.ptr<i8>, %value: i8, %bytes: i32,
      %source_key: i32, %destination_key: i32) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %stage, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      %0 = aboleth.put %ep, %dst, %src, %bytes, %true
        : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
      aboleth.wait %0 : !aboleth.token
      %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %base = tt.splat %stage : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
      %pointers = tt.addptr %base, %range : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
      %values = tt.splat %value : i8 -> tensor<4xi8>
      tt.store %pointers, %values : tensor<4x!tt.ptr<i8>>
    }
    tt.return
  }

  // A private callee is not a kernel entry: its submission follows whatever
  // the caller wrote before the call (the fused MoE dispatch packs in the
  // kernel, then calls the shared send chain).
  tt.func private @send_after_caller_write(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %stage: !tt.ptr<i8>, %bytes: i32, %source_key: i32,
      %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %stage, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }

  tt.func public @caller_writes(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %stage: !tt.ptr<i8>, %value: i8, %bytes: i32,
      %source_key: i32, %destination_key: i32) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %stage : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %pointers = tt.addptr %base, %range : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %values = tt.splat %value : i8 -> tensor<4xi8>
    tt.store %pointers, %values : tensor<4x!tt.ptr<i8>>
    tt.call @send_after_caller_write(%endpoint, %destination, %stage, %bytes,
        %source_key, %destination_key)
      : (!tt.ptr<i64>, !tt.ptr<i8>, !tt.ptr<i8>, i32, i32, i32) -> ()
    tt.return
  }

  // Every call site is clean: the callee needs no barrier.
  tt.func private @send_after_clean_caller(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %stage: !tt.ptr<i8>, %bytes: i32, %source_key: i32,
      %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %stage, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }

  tt.func public @caller_clean(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %stage: !tt.ptr<i8>, %bytes: i32, %source_key: i32,
      %destination_key: i32) {
    tt.call @send_after_clean_caller(%endpoint, %destination, %stage, %bytes,
        %source_key, %destination_key)
      : (!tt.ptr<i64>, !tt.ptr<i8>, !tt.ptr<i8>, i32, i32, i32) -> ()
    tt.return
  }
}

// CHECK-LABEL: tt.func public @tensor_producer
// CHECK: aboleth.submit
// CHECK-SAME: producerBarrier
// CHECK: aboleth.submit
// CHECK-NOT: producerBarrier
// CHECK-SAME: batchSize

// CHECK-LABEL: tt.func public @scalar_producer
// CHECK-NOT: producerBarrier
// CHECK: tt.return

// CHECK-LABEL: tt.func public @inline_after_tensor_store
// CHECK-NOT: producerBarrier
// CHECK: tt.return

// CHECK-LABEL: tt.func public @loop_carried
// CHECK: aboleth.submit
// CHECK-SAME: producerBarrier

// CHECK-LABEL: tt.func private @send_after_caller_write
// CHECK: aboleth.submit
// CHECK-SAME: producerBarrier

// CHECK-LABEL: tt.func private @send_after_clean_caller
// CHECK-NOT: producerBarrier
// CHECK: tt.return

// The barrier precedes the reservation of the first submission only.
// RDMA-LABEL: tt.func public @tensor_producer
// RDMA: tt.store
// RDMA-NEXT: ttg.barrier
// RDMA: tt.atomic_rmw
// RDMA-NOT: ttg.barrier
// RDMA: tt.return
