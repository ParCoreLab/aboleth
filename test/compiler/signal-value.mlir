// RUN: aboleth-opt %s --pass plan-publication --pass form-batches --pass lower-rdma | FileCheck %s --check-prefix=RDMA
// RUN: aboleth-opt %s --pass plan-publication --pass form-batches --pass lower-p2p | FileCheck %s --check-prefix=P2P
// RUN: aboleth-opt %s --pass plan-publication --pass form-batches --pass lower-proxy | FileCheck %s --check-prefix=PROXY
//
// A signal publishes its epoch by value: no local cell, no key, and no load.

module {
  tt.func public @strong(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %epoch: i64,
      %bytes: i32, %source_key: i32, %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %1 = aboleth.signal %ep, %flag_dst, %epoch, %0
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    aboleth.wait %1 : !aboleth.token
    tt.return
  }

  tt.func public @weak(
      %endpoint: !tt.ptr<i64>, %flag: !tt.ptr<i64>, %epoch: i64,
      %destination_key: i32) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %0 = aboleth.signal %ep, %flag_dst, %epoch
      : (!aboleth.endpoint, !aboleth.remote_region, i64) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }
}

// The strong signal is a responder compare-and-swap (opcode 0x11) whose swap
// value is the epoch operand itself.
// RDMA-LABEL: tt.func public @strong
// RDMA-NOT: tt.load %arg4
// RDMA: arith.constant 17 : i32
// RDMA: arith.subi %arg4
// RDMA-NOT: aboleth.

// The weak signal is an inline write of the epoch: an 8-byte inline segment.
// RDMA-LABEL: tt.func public @weak
// RDMA: arith.trunci %arg2 : i64 to i32
// RDMA: arith.constant -2147483640 : i32
// RDMA-NOT: aboleth.

// P2P-LABEL: tt.func public @strong
// P2P: tt.atomic_rmw exch, release, sys, %{{.*}}, %arg4
// P2P-LABEL: tt.func public @weak
// P2P: tt.atomic_rmw exch, release, sys, %{{.*}}, %arg2

// PROXY-LABEL: tt.func public @weak
// PROXY: tt.store %{{.*}}, %arg2

// With every queue in GPU memory the doorbell needs only a GPU-scope release,
// and nothing else in a posting sequence is system scope.
// GPUDB-LABEL: tt.func public @strong
// GPUDB-NOT: st.release.sys
// GPUDB: st.release.gpu
// GPUDB-NOT: st.release.sys
// GPUDB-LABEL: tt.func public @weak
// GPUDB-NOT: st.release.sys
