// RUN: not triton-opt %s --aboleth-form-batches 2>&1 | FileCheck %s

module {
  tt.func public @wait_between_put_and_signal(
      %endpoint: !tt.ptr<i8>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
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
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %expected = arith.extui %value : i8 to i64
    aboleth.wait_signal %signal_src, %expected
      : (!aboleth.local_region, i64) -> ()
    %1 = aboleth.signal %ep, %signal_dst, %signal_src, %0
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %1 : !aboleth.token
    tt.return
  }
}

// CHECK: error: operation cannot be crossed by an Aboleth submission batch
