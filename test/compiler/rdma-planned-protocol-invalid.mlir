// A completion no-op carries the batch's only CQE, so it cannot ride on a
// submission whose completion is deferred to a later drain.
//
// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @deferred_completion_nop(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %source_ptr: !tt.ptr<i8>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32) {
    %true = arith.constant true
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %put = aboleth.post_put
      %endpoint, %destination, %source, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %carrier = aboleth.post_nop %endpoint, %put
      : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token
    %submitted = aboleth.submit
      %endpoint, %put, %carrier {batchSize = 2 : i32, deferCompletion}
      : (!aboleth.endpoint, !aboleth.token,
         !aboleth.token) -> !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.submit' op a completion no-op requires an ungrouped, non-deferred submission
