// In a loop that completes every sixteenth iteration, a signal is published
// only on those iterations and on the loop's last. A checked one
// (compare-and-swap) compares against the epoch last published, which the loop
// carries: epoch - 1 on its first iteration, then whatever it last published.

module {
  tt.func public @planned(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %epoch: i64,
      %bytes: i32, %source_key: i32, %destination_key: i32, %n: i32) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c15 = arith.constant 15 : i32
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    scf.for %i = %c0 to %n step %c1 : i32 {
      %position = arith.andi %i, %c15 : i32
      %due = arith.cmpi eq, %position, %c15 : i32
      %0 = aboleth.put %ep, %dst, %src, %bytes, %true
        : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
      %1 = aboleth.signal %ep, %flag_dst, %epoch, %0 {fencedSignal}
        : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
      aboleth.wait %1, %due : !aboleth.token
    }
    tt.return
  }

  tt.func public @checked(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %epoch: i64,
      %bytes: i32, %source_key: i32, %destination_key: i32, %n: i32) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c15 = arith.constant 15 : i32
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    scf.for %i = %c0 to %n step %c1 : i32 {
      %position = arith.andi %i, %c15 : i32
      %due = arith.cmpi eq, %position, %c15 : i32
      %0 = aboleth.put %ep, %dst, %src, %bytes, %true
        : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
      %1 = aboleth.signal %ep, %flag_dst, %epoch, %0
        : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
      aboleth.wait %1, %due : !aboleth.token
    }
    tt.return
  }
}

// MARKED-LABEL: tt.func public @planned
// MARKED: %[[NEXT:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// MARKED: %[[LAST:.*]] = arith.cmpi sge, %[[NEXT]], %arg8 : i32
// MARKED: %[[WHEN:.*]] = arith.ori %{{.*}}, %[[LAST]] : i1
// MARKED: aboleth.post_signal {{.*}} if %[[WHEN]] {fencedSignal}
// MARKED-LABEL: tt.func public @checked
// MARKED: %[[ZERO:.*]] = arith.constant 0 : i64
// MARKED: scf.for %[[I:.*]] = %[[LB:.*]] to %{{.*}} step %{{.*}} iter_args(%[[LAST:.*]] = %[[ZERO]]) -> (i64)
// MARKED: %[[WHEN:.*]] = arith.ori
// MARKED: %[[FIRST:.*]] = arith.cmpi eq, %[[I]], %[[LB]] : i32
// MARKED: %[[PREVIOUS:.*]] = arith.subi %arg4, %{{.*}} : i64
// MARKED: %[[EXPECTED:.*]] = arith.select %[[FIRST]], %[[PREVIOUS]], %[[LAST]] : i64
// MARKED: %[[NEXT:.*]] = arith.select %[[WHEN]], %arg4, %[[EXPECTED]] : i64
// MARKED: aboleth.post_signal {{.*}} if %[[WHEN]] expect %[[EXPECTED]] :
// MARKED: scf.yield %[[NEXT]] : i64

// An elided signal is a no-op of the same size: its opcode word selects between
// the write and the no-op, and its control word between the fence and none.
// LOWERED-LABEL: tt.func public @planned
// LOWERED: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// LOWERED: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// LOWERED-NOT: aboleth.
// The compare-and-swap's opcode word selects between it and the no-op, and its
// compare operand is the carried value rather than epoch - 1.
// LOWERED-LABEL: tt.func public @checked
// LOWERED: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// LOWERED-NOT: aboleth.
