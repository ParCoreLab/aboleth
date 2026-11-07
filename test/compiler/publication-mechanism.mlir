// RUN: aboleth-opt %s --pass plan-publication=rdma,strict --pass form-batches | FileCheck %s --check-prefix=STRICT
// RUN: aboleth-opt %s --pass plan-publication=rdma --pass form-batches | FileCheck %s --check-prefix=RELAXED
// RUN: aboleth-opt %s --pass plan-publication=proxy,strict --pass form-batches | FileCheck %s --check-prefix=RELAXED
// RUN: aboleth-opt %s --pass plan-publication=rdma,strict --pass form-batches --pass lower-rdma | FileCheck %s --check-prefix=LOWERED
//
// Publication planning picks a strong signal's ordering point. A fenced write
// replaces the responder compare-and-swap only on a transport that has one and
// only when every receiver registration is strictly ordered. Signals posted by
// hand get the same mechanism; weak ones get none.

module {
  tt.func public @payload(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %epoch: i64,
      %bytes: i32, %source_key: i32, %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %1 = aboleth.signal %ep, %flag_dst, %epoch, %0
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    aboleth.wait %1 : !aboleth.token
    tt.return
  }

  tt.func public @credit(
      %endpoint: !tt.ptr<i64>, %flag: !tt.ptr<i64>, %epoch: i64,
      %destination_key: i32) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %0 = aboleth.signal %ep, %flag_dst, %epoch
      : (!aboleth.endpoint, !aboleth.remote_region, i64) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }

  tt.func public @posted(
      %endpoint: !tt.ptr<i64>, %flag: !tt.ptr<i64>, %epoch: i64,
      %destination_key: i32) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %0 = aboleth.post_signal %ep, %flag_dst, %epoch
      : (!aboleth.endpoint, !aboleth.remote_region, i64) -> !aboleth.token
    %1 = aboleth.submit %ep, %0 {batchSize = 1 : i32, deferCompletion}
      : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token
    tt.return
  }
}

// STRICT-LABEL: tt.func public @payload
// STRICT: aboleth.post_signal
// STRICT-SAME: fencedSignal
// STRICT-LABEL: tt.func public @credit
// STRICT: aboleth.post_signal
// STRICT-NOT: fencedSignal
// STRICT-SAME: weakPublication
// STRICT-LABEL: tt.func public @posted
// STRICT: aboleth.post_signal
// STRICT-SAME: fencedSignal

// RELAXED-LABEL: tt.func public @payload
// RELAXED-NOT: fencedSignal
// RELAXED: aboleth.submit
// RELAXED-LABEL: tt.func public @posted
// RELAXED-NOT: fencedSignal
// RELAXED: tt.return

// The payload's signal is a write carrying fm = 3, not a compare-and-swap.
// LOWERED-LABEL: tt.func public @payload
// LOWERED-NOT: arith.constant 17 : i32
// LOWERED: arith.constant 1610612736 : i32
// LOWERED-LABEL: tt.func public @credit
// LOWERED-NOT: arith.constant 1610612736 : i32
// LOWERED-LABEL: tt.func public @posted
// LOWERED-NOT: arith.constant 17 : i32
// LOWERED: arith.constant 1610612736 : i32
// LOWERED-NOT: aboleth.
