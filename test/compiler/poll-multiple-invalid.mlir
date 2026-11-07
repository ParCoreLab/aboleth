// RUN: not triton-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @multiple_poll(
      %endpoint_ptr: !tt.ptr<i8>,
      %destination_ptr: !tt.ptr<i64>,
      %source_ptr: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i8> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %signal = aboleth.post_signal
      %endpoint, %destination, %source
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region) -> !aboleth.token
    %submission = aboleth.submit
      %endpoint, %signal {batchSize = 1 : i32}
      : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token
    aboleth.poll %submission : !aboleth.token
    aboleth.poll %submission : !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.poll' op submission token cannot be consumed by multiple polls
