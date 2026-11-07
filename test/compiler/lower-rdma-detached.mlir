// Detached fragments run the direct emitter with no CTA context: the
// reservation is a per-thread inline atomic (no scalar tt.atomic_rmw whose
// live result Triton would broadcast through shared memory), and no
// publishing store carries the implicit tid.x==0 predicate — a caller's lane
// id must not decide whether the doorbell rings.
//
// RUN: aboleth-opt %s --pass form-batches --pass lower-rdma=,detached | FileCheck %s

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

  tt.func public @drain(%endpoint_ptr: !tt.ptr<i64>) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %endpoint {issuerScope = "thread"} : !aboleth.endpoint
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
// CHECK-NOT: tt.atomic_rmw
// CHECK: atom.relaxed.gpu.global.add.u64
// CHECK-NOT: setp.eq.u32 first, tid, 0
// CHECK: st.relaxed.sys.global.L1::no_allocate.b32
// CHECK-NOT: setp.eq.u32 first, tid, 0
// CHECK: st.release.sys.global.L1::no_allocate.b64
// CHECK-LABEL: tt.func public @drain
// CHECK-NOT: tt.atomic_rmw
// CHECK: atom.relaxed.gpu.global.add.u64
// CHECK-NOT: setp.eq.u32 first, tid, 0
// CHECK: st.release.sys.global.L1::no_allocate.b64
// CHECK-LABEL: tt.func public @consumer
// CHECK: ld.acquire.sys.global
// CHECK-NOT: aboleth.
// CHECK-NOT: tt.atomic_rmw
// CHECK-NOT: setp.eq.u32 first, tid, 0

// The exclusive-issuer contract: no reservation atomic (plain load/add/
// store), the reservation-order spin gone, GPU-scope ready publication —
// and still exactly one system-release doorbell.
// EXCL-LABEL: tt.func public @producer
// EXCL-NOT: tt.atomic_rmw
// EXCL-NOT: atom.relaxed.gpu.global.add.u64
// EXCL: st.relaxed.sys.global.L1::no_allocate.b32
// EXCL: st.release.sys.global.L1::no_allocate.b64
// EXCL: st.release.gpu.global.L1::no_allocate.b64
// EXCL-LABEL: tt.func public @drain
// EXCL-NOT: atom.relaxed.gpu.global.add.u64
// EXCL: st.release.sys.global.L1::no_allocate.b64
