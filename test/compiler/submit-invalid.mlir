// RUN: not triton-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @wrong_batch_size(
      %endpoint: !tt.ptr<i8>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i8> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %0 = aboleth.post_put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %1 = aboleth.submit %ep, %0 {batchSize = 2 : i32}
      : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token
    aboleth.poll %1 : !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.submit' op batchSize must equal the number of transfer tokens
