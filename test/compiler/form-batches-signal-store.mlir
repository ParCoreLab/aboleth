// RUN: triton-opt %s --aboleth-form-batches | FileCheck %s

module {
  tt.func public @signal_store(
      %endpoint_ptr: !tt.ptr<i8>,
      %destination_ptr: !tt.ptr<i8>,
      %source_ptr: !tt.ptr<i8>,
      %remote_signal_ptr: !tt.ptr<i64>,
      %local_signal_ptr: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %value: i64) {
    %true = arith.constant true
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i8> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %put = aboleth.put
      %endpoint, %destination, %source, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    tt.store %local_signal_ptr, %value : !tt.ptr<i64>
    %remote_signal = aboleth.bind_remote_region
      %remote_signal_ptr, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %local_signal = aboleth.bind_local_region
      %local_signal_ptr, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %signal = aboleth.signal
      %endpoint, %remote_signal, %local_signal, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }
}

// CHECK: aboleth.post_put
// CHECK: tt.store
// CHECK: aboleth.post_signal
// CHECK: aboleth.submit
// CHECK: aboleth.poll
