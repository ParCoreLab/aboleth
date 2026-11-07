// Register-direct payloads: aboleth.put_inline carries its bytes as operands, so
// nothing stages them and nothing holds a registration key for them. The second
// case is deliberately larger than one WQEBB, which is what makes the queue
// accounting visible: the reservation, the following command's WQE index, and
// the doorbell head all count WQEBBs rather than commands.
//
// RUN: aboleth-opt %s --pass form-batches | FileCheck %s --check-prefix=CORE
// RUN: aboleth-opt %s --pass form-batches --pass lower-rdma \
// RUN:   | FileCheck %s --check-prefix=RDMA

module {
  tt.func public @one_block(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1,
      %first: i64,
      %second: i64,
      %tail: i32) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.put_inline
      %endpoint, %destination, [%first, %second, %tail], %predicate
      : (!aboleth.endpoint, !aboleth.remote_region, i64, i64, i32, i1)
        -> !aboleth.token
    %signal = aboleth.signal %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }

  tt.func public @two_blocks(
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
    %put = aboleth.put_inline
      %endpoint, %destination,
      [%value, %value, %value, %value, %value, %value, %value, %value],
      %true
      : (!aboleth.endpoint, !aboleth.remote_region,
         i64, i64, i64, i64, i64, i64, i64, i64, i1) -> !aboleth.token
    %signal = aboleth.signal %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }

  tt.func public @polled_inline_tail(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %destination_key: i32,
      %predicate: i1,
      %value: i64) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %put = aboleth.put_inline %endpoint, %destination, [%value], %predicate
      : (!aboleth.endpoint, !aboleth.remote_region, i64, i1)
        -> !aboleth.token
    aboleth.wait %put : !aboleth.token
    tt.return
  }
}

// Batch formation treats an inline transfer exactly like a region-sourced one:
// it becomes a posted command in a two-command submission.
// CORE-LABEL: tt.func public @one_block
// CORE: aboleth.post_inline
// CORE-SAME: [%arg7, %arg8, %arg9]
// CORE: aboleth.post_signal
// CORE: aboleth.submit
// CORE-SAME: batchSize = 2 : i32
// CORE: aboleth.poll
// CORE-NOT: aboleth.put_inline

// A 20-byte payload is four 16-byte segments and still one WQEBB, so the batch
// reserves two slots exactly as a region-sourced put would.
// RDMA-LABEL: tt.func public @one_block
// RDMA: %[[BLOCKS:.*]] = arith.constant 2 : i64
// RDMA: tt.atomic_rmw add, relaxed, gpu, %{{.*}}, %[[BLOCKS]]
// RDMA: arith.constant 4 : i32
// RDMA: arith.constant -2147483628 : i32
// RDMA-NOT: aboleth.

// A 64-byte payload is seven segments and two WQEBBs. The reservation is three,
// the second WQEBB resolves its own slot so the command wraps the ring instead
// of running past its end, and the signal lands two WQEBBs after the payload.
// RDMA-LABEL: tt.func public @two_blocks
// RDMA: %[[BLOCKS:.*]] = arith.constant 3 : i64
// RDMA: tt.atomic_rmw add, relaxed, gpu, %{{.*}}, %[[BLOCKS]]
// RDMA: arith.constant 7 : i32
// RDMA: arith.constant -2147483584 : i32
// RDMA: %[[NEXT:.*]] = arith.constant 1 : i64
// RDMA: arith.addi %{{.*}}, %[[NEXT]]
// RDMA-NEXT: arith.extui
// RDMA-NEXT: arith.constant 1 : i64
// RDMA-NEXT: arith.subi
// RDMA-NEXT: arith.andi
// RDMA: %[[SIGNAL:.*]] = arith.constant 2 : i64
// RDMA: arith.addi %{{.*}}, %[[SIGNAL]]
// RDMA: arith.constant 17 : i32
// RDMA-NOT: aboleth.

// A wait that consumes an inline payload directly makes it the last command of
// a polled batch, so it must request the CQE the poll waits for, even when its
// predicate turns it into a no-op.
// RDMA-LABEL: tt.func public @polled_inline_tail
// RDMA-NOT: arith.select %false{{(_[0-9]+)?}}, %c134217728_i32
// RDMA: arith.select %true{{(_[0-9]+)?}}, %c134217728_i32
// RDMA-NOT: aboleth.
