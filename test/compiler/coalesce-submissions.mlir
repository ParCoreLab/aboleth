// Two independent put+signal chains to one endpoint whose waits both follow
// them merge into one submission of two segments with one poll. A wait between
// two chains ends the region, so those stay apart; a kernel lowering to peer
// memory has no submission to save and is left alone.

module {
  tt.func public @adjacent(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %flags: !tt.ptr<i64>, %epoch: i64,
      %bytes: i32, %source_key: i32, %destination_key: i32) {
    %true = arith.constant true
    %c1 = arith.constant 1 : i32
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %flag0 = aboleth.bind_remote_region %flags, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %second = tt.addptr %flags, %c1 : !tt.ptr<i64>, i32
    %flag1 = aboleth.bind_remote_region %second, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %t0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %s0 = aboleth.signal %ep, %flag0, %epoch, %t0
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    %t1 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %s1 = aboleth.signal %ep, %flag1, %epoch, %t1
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    aboleth.wait %s0 : !aboleth.token
    aboleth.wait %s1 : !aboleth.token
    tt.return
  }

  tt.func public @separated(
      %endpoint: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %flags: !tt.ptr<i64>, %epoch: i64,
      %bytes: i32, %source_key: i32, %destination_key: i32) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %flag = aboleth.bind_remote_region %flags, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %t0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %s0 = aboleth.signal %ep, %flag, %epoch, %t0
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    aboleth.wait %s0 : !aboleth.token
    %t1 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %s1 = aboleth.signal %ep, %flag, %epoch, %t1
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    aboleth.wait %s1 : !aboleth.token
    tt.return
  }
}

// MERGED-LABEL: tt.func public @adjacent
// MERGED: %[[P0:.*]] = aboleth.post_put
// MERGED: %[[S0:.*]] = aboleth.post_signal {{.*}}, %[[P0]]
// MERGED: %[[P1:.*]] = aboleth.post_put
// MERGED: %[[S1:.*]] = aboleth.post_signal {{.*}}, %[[P1]]
// MERGED: %[[SUB:.*]] = aboleth.submit %{{.*}}, %[[P0]], %[[S0]], %[[P1]], %[[S1]] {batchSize = 4 : i32}
// MERGED-NEXT: aboleth.poll %[[SUB]]
// MERGED-NOT: aboleth.poll
// MERGED-LABEL: tt.func public @separated
// MERGED: aboleth.submit {{.*}} {batchSize = 2 : i32}
// MERGED: aboleth.poll
// MERGED: aboleth.submit {{.*}} {batchSize = 2 : i32}
// MERGED: aboleth.poll

// UNMERGED-LABEL: tt.func public @adjacent
// UNMERGED: aboleth.submit {{.*}} {batchSize = 2 : i32}
// UNMERGED: aboleth.submit {{.*}} {batchSize = 2 : i32}

// One reservation and one doorbell for both chains: a single relaxed atomic on
// the producer index and a single BlueFlame release.
// LOWERED-LABEL: tt.func public @adjacent
// LOWERED-COUNT-1: tt.atomic_rmw add, relaxed, gpu
// LOWERED-NOT: tt.atomic_rmw add, relaxed, gpu
// LOWERED-LABEL: tt.func public @separated
// LOWERED-COUNT-2: tt.atomic_rmw add, relaxed, gpu

// Ordering the merged batch's publications puts both payload commands first and
// chains the second signal after the first.
// ORDERED-LABEL: tt.func public @adjacent
// ORDERED: %[[P0:.*]] = aboleth.post_put
// ORDERED-NOT: aboleth.post_signal
// ORDERED: %[[P1:.*]] = aboleth.post_put {{.*}}, %[[P0]] :
// ORDERED: %[[S0:.*]] = aboleth.post_signal {{.*}}, %[[P1]] :
// ORDERED: %[[S1:.*]] = aboleth.post_signal {{.*}}, %[[S0]] :
// ORDERED: aboleth.submit %{{.*}}, %[[P0]], %[[P1]], %[[S0]], %[[S1]]

// Only the first signal is a compare-and-swap (opcode 0x11 in its index
// word); the second is a plain inline write.
// ORDERED-LOWERED-LABEL: tt.func public @adjacent
// ORDERED-LOWERED-COUNT-1: arith.constant 17 : i32
// ORDERED-LOWERED-NOT: arith.constant 17 : i32
// ORDERED-LOWERED-LABEL: tt.func public @separated
