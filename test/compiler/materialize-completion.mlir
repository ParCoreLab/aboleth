// RUN: aboleth-opt %s --pass materialize-completion=8 | FileCheck %s --check-prefix=INTERVAL
// RUN: aboleth-opt %s --pass materialize-completion=1 | FileCheck %s --check-prefix=EAGER
// RUN: aboleth-opt %s --pass materialize-completion=8 --pass form-batches | FileCheck %s --check-prefix=QUEUE

module {
  tt.func public @planned(
      %endpoint: !tt.ptr<i64>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32,
      %iteration: i32,
      %iterations: i32) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i64> -> !aboleth.endpoint
    %signal_dst = aboleth.bind_remote_region
      %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %ready = aboleth.signal %ep, %signal_dst, %signal_src
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region) -> !aboleth.token
    aboleth.planned_wait %ready, %iteration, %iterations
      : !aboleth.token, i32, i32
    tt.return
  }
}

// The last iteration of each interval polls.
// INTERVAL: %[[REMAINDER:.*]] = arith.remui
// INTERVAL: %[[LAST:.*]] = arith.constant 7 : i32
// INTERVAL: arith.cmpi eq, %[[REMAINDER]], %[[LAST]]
// INTERVAL: arith.ori
// INTERVAL: aboleth.wait %{{.*}}, %{{.*}}
// INTERVAL-NOT: aboleth.planned_wait

// EAGER-NOT: arith.remui
// EAGER: aboleth.wait %{{.*}} : !aboleth.token
// EAGER-NOT: aboleth.planned_wait

// QUEUE: aboleth.submit
// QUEUE: aboleth.poll %{{.*}}, %{{.*}} {coalesced}
// QUEUE-NOT: aboleth.planned_wait
