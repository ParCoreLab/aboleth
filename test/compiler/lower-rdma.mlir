// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-rdma | FileCheck %s

module {
  tt.func public @producer(
      %endpoint: !tt.ptr<i64>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i64> -> !aboleth.endpoint
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
// CHECK: tt.atomic_rmw add, relaxed, gpu
// CHECK: arith.select
// CHECK: tt.store
// CHECK: tt.elementwise_inline_asm {{.*}}setp.eq.u32 first, tid, 0
// CHECK: scf.while
// CHECK-LABEL: tt.func public @consumer
// CHECK: tt.elementwise_inline_asm {{.*}}ld.acquire.sys.global
// CHECK-NOT: aboleth.

// invariant-descriptors and vector-wqe-stores: the descriptor's words are pure
// non-coherent loads the batch shares, and the RDMA-write WQE is three 16-byte
// stores from the issuing thread.
// WQE-LABEL: tt.func public @producer
// WQE: ld.global.nc.b64
// WQE: st.global.v4.b32
// WQE: st.global.v4.b32
// WQE: st.global.v4.b32
// WQE-LABEL: tt.func public @consumer

// Without them, descriptor words are plain loads and the WQE scalar stores.
// SCALAR-LABEL: tt.func public @producer
// SCALAR-NOT: ld.global.nc
// SCALAR-NOT: st.global.v4
// SCALAR-LABEL: tt.func public @consumer
