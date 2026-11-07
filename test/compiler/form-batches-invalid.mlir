// A put-only chain completed by aboleth.wait is legal since payload batches
// poll their own completion, so the surviving refusal is a grouped signal
// consumed by the single-token wait: the group's members retire together or
// not at all.
//
// RUN: not triton-opt %s --aboleth-form-batches 2>&1 | FileCheck %s

module {
  tt.func public @grouped_signal_single_wait(
      %endpoint: !tt.ptr<i8>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i8> -> !aboleth.endpoint
    %c8 = arith.constant 8 : i32
    %dst = aboleth.bind_remote_region %destination, %c8
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %c8
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %group = aboleth.begin_group %ep
      : !aboleth.endpoint -> !aboleth.group
    %0 = aboleth.signal %ep, %dst, %src group %group
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.group) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    aboleth.wait_group %group : !aboleth.group
    tt.return
  }
}

// CHECK: error: 'aboleth.wait' op grouped signals must be completed by aboleth.wait_group
