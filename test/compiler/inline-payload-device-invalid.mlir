// A payload the device library's 64-bit inline primitive cannot carry. The
// lowering says so rather than truncating the command or silently writing eight
// bytes where sixteen were asked for.
//
// RUN: not aboleth-opt %s --pass lower-rdma=test/rdma.bc 2>&1 | FileCheck %s

module {
  tt.func public @too_wide(
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
      %endpoint, %destination, [%value, %value], %true
      : (!aboleth.endpoint, !aboleth.remote_region, i64, i64, i1)
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

// CHECK: error: the mlx5 device library posts only an 8-byte inline payload; this command carries 16 bytes and needs the direct emitter
