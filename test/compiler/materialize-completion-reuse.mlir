// RUN: aboleth-opt %s --pass materialize-completion=,rdma,1024 | FileCheck %s --check-prefix=PREFERRED
// RUN: aboleth-opt %s --pass materialize-completion=,rdma,7 | FileCheck %s --check-prefix=QUEUE
// RUN: aboleth-opt %s --pass materialize-completion=,p2p | FileCheck %s --check-prefix=REUSE
// RUN: aboleth-opt %s --pass materialize-completion=,rdma,1024,unplanned | FileCheck %s --check-prefix=UNPLANNED
//
// A kernel that overwrites its sources twelve iterations later completes every
// k-th iteration, for the largest power of two k within the reuse distance,
// the transport's preferred interval, and the iterations the queue holds.

module {
  tt.func public @reuse(
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
    aboleth.planned_wait %1, %epoch {reuseDistance = 12 : i64} : !aboleth.token, i64
    tt.return
  }
}

// mlx5 prefers 16, the queue holds 511 two-slot iterations: k = 8.
// PREFERRED: %[[MASK:.*]] = arith.constant 7 : i64
// PREFERRED: %[[POSITION:.*]] = arith.andi %arg4, %[[MASK]]
// PREFERRED: arith.cmpi eq, %[[POSITION]]
// PREFERRED-NOT: arith.ori
// PREFERRED: aboleth.wait %{{.*}}, %{{.*}} : !aboleth.token

// Seven slots hold three two-slot iterations: k = 2.
// QUEUE: arith.constant 1 : i64
// QUEUE: arith.andi %arg4
// QUEUE: aboleth.wait %{{.*}}, %{{.*}} : !aboleth.token

// Peer memory has no queue and no preference: k = 8, the largest power of two
// within the reuse distance.
// REUSE: arith.constant 7 : i64
// REUSE: aboleth.wait %{{.*}}, %{{.*}} : !aboleth.token

// Planning off (the ablation baseline): every iteration waits.
// UNPLANNED-NOT: arith.andi
// UNPLANNED: aboleth.wait %{{.*}} : !aboleth.token
