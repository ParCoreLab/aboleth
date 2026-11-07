// RUN: triton-opt %s --aboleth-form-batches | FileCheck %s

module {
  tt.func public @acknowledge(
      %endpoint: !tt.ptr<i64>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i64> -> !aboleth.endpoint
    %remote = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %local = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %0 = aboleth.signal %ep, %remote, %local
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }
}

// CHECK-LABEL: tt.func public @acknowledge
// CHECK: %[[SIGNAL:.*]] = aboleth.post_signal
// CHECK: %[[SUBMIT:.*]] = aboleth.submit
// CHECK-SAME: %[[SIGNAL]]
// CHECK-SAME: batchSize = 1 : i32
// CHECK: aboleth.poll %[[SUBMIT]]
