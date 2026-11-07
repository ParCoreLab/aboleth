// A wait whose predicate is the kernel's own (`wait(predicate=)`, marked
// selfReclaim) may defer more submissions than the queue holds. With the
// `self-reclaim` lowering option, a module whose endpoints each have one
// issuing program lowers its capacity wait and its poll as CQ-consuming loops
// over register-carried (consumer, completed, failed) state, and the batch
// requests its CQE whether or not the poll is due. Without the option, or
// with endpoints the module cannot prove single-issuer, it is lowered as any
// predicated wait is.

module {
  tt.func public @deferred_window(
      %endpoint: !tt.ptr<i64>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %last: i1) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %1 = aboleth.put %ep, %dst, %src, %bytes, %true, %0
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1, !aboleth.token) -> !aboleth.token
    aboleth.wait %1, %last {selfReclaim} : !aboleth.token
    tt.return
  }
}

// CORE-LABEL: tt.func public @deferred_window
// CORE: %[[SUBMIT:.*]] = aboleth.submit
// CORE: aboleth.poll %[[SUBMIT]], %arg6 {coalesced, selfReclaim}

// The capacity wait and the due poll each consume CQEs in a loop; the poll's
// loop sits under its predicate.
// RECLAIM-LABEL: tt.func public @deferred_window
// RECLAIM: scf.while {{.*}} : (i64, i64, i1) -> (i64, i64, i1)
// RECLAIM: scf.if %arg6
// RECLAIM: scf.while {{.*}} : (i64, i64, i1) -> (i64, i64, i1)
// RECLAIM-NOT: aboleth.

// PLAIN-LABEL: tt.func public @deferred_window
// PLAIN-NOT: (i64, i64, i1) -> (i64, i64, i1)
// PLAIN: scf.if %arg6
// PLAIN-NOT: (i64, i64, i1) -> (i64, i64, i1)
// PLAIN-NOT: aboleth.
