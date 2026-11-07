// RUN: not triton-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @multiple_submit(
      %endpoint_ptr: !tt.ptr<i8>,
      %destination_ptr: !tt.ptr<i8>,
      %source_ptr: !tt.ptr<i8>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32) {
    %true = arith.constant true
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i8> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %put = aboleth.post_put
      %endpoint, %destination, %source, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.post_signal
      %endpoint, %destination, %source, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    %first = aboleth.submit
      %endpoint, %put, %signal {batchSize = 2 : i32}
      : (!aboleth.endpoint, !aboleth.token,
         !aboleth.token) -> !aboleth.token
    %second = aboleth.submit
      %endpoint, %put, %signal {batchSize = 2 : i32}
      : (!aboleth.endpoint, !aboleth.token,
         !aboleth.token) -> !aboleth.token
    aboleth.poll %first : !aboleth.token
    aboleth.poll %second : !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.submit' op posted transfer token cannot belong to multiple submissions
