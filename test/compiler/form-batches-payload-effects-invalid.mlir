// RUN: not triton-opt %s --aboleth-form-batches 2>&1 | FileCheck %s
//
// A payload chain completed by a wait forms a submission at its last put, and
// it may no more cross a blocking wait than a signal chain may.

module {
  tt.func public @wait_between_payloads(
      %endpoint: !tt.ptr<i8>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %bytes: i32,
      %value: i8,
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
    %expected = arith.extui %value : i8 to i64
    aboleth.wait_signal %src, %expected : (!aboleth.local_region, i64) -> ()
    %1 = aboleth.put %ep, %dst, %src, %bytes, %true, %0
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1, !aboleth.token) -> !aboleth.token
    aboleth.wait %1 : !aboleth.token
    tt.return
  }
}

// CHECK: error: operation cannot be crossed by an Aboleth submission batch
