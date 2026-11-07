// A put completed by aboleth.wait with no signal is a payload batch: the
// submission polls its own completion instead of publishing one.
//
// RUN: triton-opt %s --aboleth-form-batches | FileCheck %s

module {
  tt.func public @payload_only(
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
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }
}

// CHECK: aboleth.post_put
// CHECK: aboleth.submit
// CHECK-SAME: batchSize = 1
// CHECK: aboleth.poll
// CHECK-NOT: aboleth.wait
