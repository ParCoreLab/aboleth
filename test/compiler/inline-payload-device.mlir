// The mlx5 device library's inline primitive is typed on a 64-bit value, so an
// 8-byte unpredicated inline payload is the shape it can post. The direct
// emitter builds the same command from segment stores.
//
// RUN: aboleth-opt %s --pass lower-rdma | FileCheck %s --check-prefix=RDMA
// RUN: aboleth-opt %s --pass lower-rdma=test/rdma.bc \
// RUN:   | FileCheck %s --check-prefix=DEVICE

module {
  tt.func public @inline_word(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32,
      %value: i64) {
    %true = arith.constant true
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.post_inline
      %endpoint, %destination, [%value], %true
      : (!aboleth.endpoint, !aboleth.remote_region, i64, i1)
        -> !aboleth.token
    %signal = aboleth.post_signal
      %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    %submitted = aboleth.submit
      %endpoint, %put, %signal {batchSize = 2 : i32}
      : (!aboleth.endpoint, !aboleth.token,
         !aboleth.token) -> !aboleth.token
    aboleth.poll %submitted : !aboleth.token
    tt.return
  }
}

// Eight payload bytes are three segments and one WQEBB, so the batch reserves
// two slots and the inline header carries a length of eight.
// RDMA-LABEL: tt.func public @inline_word
// RDMA: %[[BLOCKS:.*]] = arith.constant 2 : i64
// RDMA: tt.atomic_rmw add, relaxed, gpu, %{{.*}}, %[[BLOCKS]]
// RDMA: arith.constant 3 : i32
// RDMA: arith.constant -2147483640 : i32
// RDMA-NOT: aboleth.

// The library call replaces the segment stores and is posted by one lane.
// DEVICE-LABEL: tt.func public @inline_word
// DEVICE: %tid.x
// DEVICE: scf.if
// DEVICE: symbol = "aboleth_rdma_reserve_batch"
// DEVICE: scf.if
// DEVICE: symbol = "aboleth_rdma_post_inline_u64"
// DEVICE: scf.if
// DEVICE: symbol = "aboleth_rdma_post_atomic_signal"
// DEVICE: scf.if
// DEVICE: symbol = "aboleth_rdma_publish"
// DEVICE: symbol = "aboleth_rdma_poll"
// DEVICE-NOT: aboleth.
