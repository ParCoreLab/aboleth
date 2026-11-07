// RUN: aboleth-opt %s --pass form-batches --pass lower-efa | FileCheck %s

// EFA lowering calls the efa-dp-direct device primitives, which link into the
// kernel as extern bitcode. SRD is unordered and has no remote atomics, so the
// batch retires by counting completions and a strong signal is published only
// after the sender has retired the payload ordered before it.

module {
  tt.func public @producer(
      %endpoint: !tt.ptr<i64>,
      %remote: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %signal: !tt.ptr<i64>,
      %bytes: i32,
      %lkey: i32,
      %rkey: i32,
      %enabled: i1) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %remote, %rkey
        : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %lkey
        : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %sig_dst = aboleth.bind_remote_region %remote_signal, %rkey
        : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %sig_src = aboleth.bind_local_region %signal, %lkey
        : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %transfer = aboleth.put %ep, %dst, %src, %bytes, %enabled
        : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region,
           i32, i1) -> !aboleth.token
    %published = aboleth.signal %ep, %sig_dst, %sig_src, %transfer
        : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region,
           !aboleth.token) -> !aboleth.token
    aboleth.wait %published : !aboleth.token
    tt.return
  }

  tt.func public @consumer(%signal: !tt.ptr<i64>, %expected: i64) {
    %zero = arith.constant 0 : i32
    %local = aboleth.bind_local_region %signal, %zero
        : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    aboleth.wait_signal %local, %expected
        : (!aboleth.local_region, i64) -> ()
    tt.return
  }
}

// A batch reserves its range, then waits for the send queue to drain enough of
// its retired head to hold the batch.
// CHECK-LABEL: tt.func public @producer
// CHECK: tt.atomic_rmw add, relaxed, gpu
// CHECK: scf.while
// CHECK: arith.cmpi ugt

// CHECK: symbol = "aboleth_efa_begin_batch"

// A disabled transfer keeps its slot as a zero-length write, so the batch
// still produces the completion count the compiler planned.
// CHECK: arith.select
// CHECK: symbol = "aboleth_efa_post_write"

// Sender-fenced publication: the payload is flushed and retired before the
// epoch is posted, because SRD would otherwise let the signal overtake it.
// CHECK: symbol = "aboleth_efa_flush"
// CHECK: symbol = "aboleth_efa_completion_advance"
// CHECK: symbol = "aboleth_efa_begin_batch"

// The epoch travels from its registered source cell rather than by value —
// EFA has no inline data on RDMA writes — so the signal call carries two
// pointer/key pairs: (endpoint, slot, source, source key, destination,
// destination key, request id).
// CHECK: symbol = "aboleth_efa_post_inline"
// CHECK-SAME: (i64, i32, i64, i32, i64, i32, i32)

// Batches ring the doorbell in reservation order.
// CHECK: scf.while
// CHECK: arith.cmpi ne
// CHECK: symbol = "aboleth_efa_flush"
// CHECK: st.release.sys

// Completion retires by count rather than by queue index.
// CHECK: symbol = "aboleth_efa_completion_advance"
// CHECK-NOT: aboleth.

// The receiver contract is unchanged: an acquire load observes the epoch.
// CHECK-LABEL: tt.func public @consumer
// CHECK: ld.acquire.sys
// CHECK: arith.cmpi ult
// CHECK-NOT: aboleth.
