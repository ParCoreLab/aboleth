// RUN: triton-opt %s --aboleth-form-batches | FileCheck %s

module {
  tt.func public @core_batch(
      %endpoint: !aboleth.endpoint,
      %destination: !aboleth.remote_region,
      %source: !aboleth.local_region,
      %remote_signal: !aboleth.remote_region,
      %local_signal: !aboleth.local_region,
      %bytes: i32,
      %predicate: i1) {
    %put = aboleth.put
      %endpoint, %destination, %source, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.signal
      %endpoint, %remote_signal, %local_signal, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }
}

// CHECK-LABEL: tt.func public @core_batch
// CHECK: aboleth.post_put
// CHECK: aboleth.post_signal
// CHECK: aboleth.submit
// CHECK: aboleth.poll
// CHECK-NOT: !tt.ptr
// CHECK-NOT: source_key
// CHECK-NOT: destination_key
