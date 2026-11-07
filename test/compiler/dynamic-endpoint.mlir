// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-rdma | FileCheck %s --check-prefix=RDMA
// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-proxy | FileCheck %s --check-prefix=PROXY

module {
  tt.func public @dynamic_endpoint(
      %endpoint_table: !tt.ptr<i64>,
      %peer: i32,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32) {
    %endpoint_slot = tt.addptr %endpoint_table, %peer
      : !tt.ptr<i64>, i32
    %endpoint_address = tt.load %endpoint_slot
      : !tt.ptr<i64>
    %endpoint = tt.int_to_ptr %endpoint_address
      : i64 -> !tt.ptr<i64>
    %true = arith.constant true
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
    %put = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.signal %ep, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }
}

// RDMA-LABEL: tt.func public @dynamic_endpoint
// RDMA: tt.load
// RDMA: tt.int_to_ptr
// RDMA: tt.atomic_rmw add, relaxed, gpu
// RDMA: tt.elementwise_inline_asm
// RDMA-NOT: aboleth.

// PROXY-LABEL: tt.func public @dynamic_endpoint
// PROXY: tt.load
// PROXY: tt.int_to_ptr
// PROXY: tt.atomic_rmw add, relaxed, gpu
// PROXY: st.release.sys.global
// PROXY-NOT: aboleth.
