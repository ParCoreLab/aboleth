// RUN: triton-opt %s --aboleth-form-batches | FileCheck %s --check-prefix=CORE
// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-p2p | FileCheck %s --check-prefix=LOWERED
// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-proxy | FileCheck %s --check-prefix=LOWERED
// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-rdma | FileCheck %s --check-prefix=LOWERED

module {
  tt.func public @grouped(
      %endpoint: !tt.ptr<i64>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %group = aboleth.begin_group %ep
      : !aboleth.endpoint -> !aboleth.group
    %put = aboleth.put %ep, %dst, %src, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.signal
      %ep, %signal_dst, %signal_src, %put group %group
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token,
         !aboleth.group) -> !aboleth.token
    aboleth.wait_group %group : !aboleth.group
    tt.return
  }
}

// CORE-LABEL: tt.func public @grouped
// CORE: %[[GROUP:.*]] = aboleth.begin_group
// CORE: %[[PUT:.*]] = aboleth.post_put
// CORE: %[[SIGNAL:.*]] = aboleth.post_signal
// CORE-SAME: group %[[GROUP]]
// CORE: %[[SUBMIT:.*]] = aboleth.submit
// CORE-SAME: %[[PUT]], %[[SIGNAL]] group %[[GROUP]]
// CORE: aboleth.wait_group %[[GROUP]]
// CORE-NOT: aboleth.poll

// LOWERED-LABEL: tt.func public @grouped
// LOWERED-NOT: aboleth.

// A detached fragment may not impose the group's shared-memory broadcast on
// its caller, so the RDMA lowering refuses groups under detached emission.
// DETACHED: completion groups are not yet available in detached fragments
