// The selectable planned-protocol mechanisms: head publication, CTA-exclusive
// reservation, a fenced strong publication, and a completion-carrying no-op.
//
// RUN: aboleth-opt %s --pass lower-rdma | FileCheck %s --check-prefix=RDMA
// RUN: aboleth-opt %s --pass lower-rdma=test/rdma.bc | FileCheck %s --check-prefix=DEVICE
// RUN: aboleth-opt %s --pass lower-rdma | FileCheck %s --check-prefix=INLINE

module {
  tt.func public @head_publication(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %source_ptr: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.post_put
      %endpoint, %destination, %source, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.post_signal
      %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    %submitted = aboleth.submit
      %endpoint, %put, %signal
      {batchSize = 2 : i32, deferCompletion, headPublication}
      : (!aboleth.endpoint, !aboleth.token,
         !aboleth.token) -> !aboleth.token
    tt.return
  }

  tt.func public @cta_reservation(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %source_ptr: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.post_put
      %endpoint, %destination, %source, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.post_signal
      %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    %submitted = aboleth.submit
      %endpoint, %put, %signal
      {batchSize = 2 : i32, deferCompletion, headPublication,
       reservationScope = "cta"}
      : (!aboleth.endpoint, !aboleth.token,
         !aboleth.token) -> !aboleth.token
    tt.return
  }

  tt.func public @fenced_signal(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %source_ptr: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.post_put
      %endpoint, %destination, %source, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.post_signal
      %endpoint, %signal_dst, %signal_src, %put {fencedSignal}
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    %submitted = aboleth.submit
      %endpoint, %put, %signal
      {batchSize = 2 : i32, deferCompletion, headPublication}
      : (!aboleth.endpoint, !aboleth.token,
         !aboleth.token) -> !aboleth.token
    tt.return
  }

  tt.func public @completion_nop(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %source_ptr: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.post_put
      %endpoint, %destination, %source, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.post_signal
      %endpoint, %signal_dst, %signal_src, %put {fencedSignal}
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    %carrier = aboleth.post_nop %endpoint, %signal
      : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token
    %submitted = aboleth.submit
      %endpoint, %put, %signal, %carrier {batchSize = 3 : i32}
      : (!aboleth.endpoint, !aboleth.token, !aboleth.token,
         !aboleth.token) -> !aboleth.token
    aboleth.poll %submitted : !aboleth.token
    tt.return
  }
}

// Head publication reads the producer index and rings there; no batch-order
// wait on the ready word remains.
// RDMA-LABEL: tt.func public @head_publication
// RDMA: tt.atomic_rmw add, relaxed, gpu
// RDMA: tt.elementwise_inline_asm {{.*}}ld.acquire.gpu.global
// RDMA-NOT: scf.while
// RDMA: tt.elementwise_inline_asm {{.*}}st.relaxed.sys.global.L1::no_allocate.b32

// A CTA-exclusive reservation is block-scoped and does not wait for capacity.
// RDMA-LABEL: tt.func public @cta_reservation
// RDMA: tt.atomic_rmw add, relaxed, cta
// RDMA-NOT: scf.while

// A fenced strong publication is a write carrying fm = 3 (0x60000000) instead
// of a compare-and-swap.
// RDMA-LABEL: tt.func public @fenced_signal
// RDMA-NOT: arith.constant 17 : i32
// RDMA: arith.constant 1610612736 : i32

// The no-op carries the CQE, so the fenced signal keeps a bare fence word.
// RDMA-LABEL: tt.func public @completion_nop
// RDMA: arith.constant 1610612736 : i32
// RDMA: arith.constant 134217728 : i32
// RDMA: %clock64
// RDMA-NOT: aboleth.

// Every device-library call is posted by one lane.
// DEVICE-LABEL: tt.func public @head_publication
// DEVICE: %tid.x
// DEVICE: scf.if
// DEVICE: symbol = "aboleth_rdma_reserve_batch"
// DEVICE: scf.if
// DEVICE: symbol = "aboleth_rdma_post_write"
// DEVICE: scf.if
// DEVICE: symbol = "aboleth_rdma_post_atomic_signal"
// DEVICE: scf.if
// DEVICE: symbol = "aboleth_rdma_publish_head"
// DEVICE-LABEL: tt.func public @cta_reservation
// DEVICE: symbol = "aboleth_rdma_reserve_planned"
// DEVICE-LABEL: tt.func public @fenced_signal
// DEVICE: arith.constant 1610612736 : i32
// DEVICE: symbol = "aboleth_rdma_post_inline_u64"
// DEVICE-LABEL: tt.func public @completion_nop
// DEVICE: symbol = "aboleth_rdma_post_nop"
// DEVICE: %tid.x
// DEVICE: scf.if
// DEVICE: symbol = "aboleth_rdma_poll"
// DEVICE-NOT: aboleth.

// With inline signals the fence rides on the inline-data publication WQE.
// INLINE-LABEL: tt.func public @fenced_signal
// INLINE: arith.constant -2147483640 : i32
// INLINE: arith.constant 1610612736 : i32
// INLINE-NOT: aboleth.
