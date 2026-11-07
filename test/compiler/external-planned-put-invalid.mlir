// A payload-only submission is legal only when its completion is explicitly
// deferred — to a later drain, or to an enclosing integration that owns both
// publication and completion. This one defers nothing.
//
// RUN: not aboleth-opt %s --pass lower-rdma 2>&1 | FileCheck %s

module {
  tt.func public @missing_external_contract(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %source_ptr: !tt.ptr<i8>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %put = aboleth.post_put
      %endpoint, %destination, %source, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %posted = aboleth.submit %endpoint, %put
      {batchSize = 1 : i32, plannedReservation}
      : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.submit' op final transfer must be
// CHECK-SAME: aboleth.post_signal unless completion is explicitly deferred
