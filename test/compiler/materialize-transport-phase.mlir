// RUN: triton-opt %s --aboleth-materialize-transport-phase="plan-path=%S/external-transport-fragment.mlir" | FileCheck %s

module {
  tt.func public @dispatch(
      %endpoint: !tt.ptr<i64>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %remoteSignal: !tt.ptr<i64>,
      %localSignal: !tt.ptr<i64>,
      %bytes: i32,
      %sourceKey: i32,
      %destinationKey: i32,
      %predicate: i1,
      %iteration: i32,
      %iterations: i32) {
    %programId = arith.constant 7 : i32
    "aboleth.invoke_transport_phase"(
        %programId, %endpoint, %destination, %source, %remoteSignal,
        %localSignal, %bytes, %sourceKey, %destinationKey, %predicate,
        %iteration, %iterations)
      : (i32, !tt.ptr<i64>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i64>,
         !tt.ptr<i64>, i32, i32, i32, i1, i32, i32) -> ()
    tt.return
  }
}

// CHECK-NOT: aboleth.transport_program
// CHECK-NOT: aboleth.invoke_transport_phase
// CHECK-LABEL: tt.func public @dispatch
// CHECK: aboleth.bind_endpoint
// CHECK-SAME: !tt.ptr<i64> -> !aboleth.endpoint
// CHECK: aboleth.bind_remote_region
// CHECK-SAME: (!tt.ptr<i8>, i32) -> !aboleth.remote_region
// CHECK: aboleth.bind_local_region
// CHECK-SAME: (!tt.ptr<i8>, i32) -> !aboleth.local_region
// CHECK: aboleth.put
// CHECK: aboleth.signal
// CHECK: aboleth.planned_wait
// CHECK-NOT: aboleth.invoke_transport_phase
// CHECK: tt.return
