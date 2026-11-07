// RUN: aboleth-opt %s --pass materialize-completion=8 | FileCheck %s
module {
  tt.func public @later_pure_bounds(%ep: !aboleth.endpoint,
      %dst: !aboleth.remote_region, %src: !aboleth.local_region,
      %i: i32, %n: i32) {
    %s = aboleth.signal %ep, %dst, %src
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region) -> !aboleth.token
    %one = arith.constant 1 : i32
    %next = arith.addi %i, %one : i32
    %iteration = arith.extui %next : i32 to i64
    %iterations = arith.extui %n : i32 to i64
    aboleth.planned_wait %s, %iteration, %iterations : !aboleth.token, i64, i64
    tt.return
  }
  tt.func public @later_observed_bound(%ep: !aboleth.endpoint,
      %dst: !aboleth.remote_region, %src: !aboleth.local_region,
      %ptr: !tt.ptr<i32>, %n: i32) {
    %s = aboleth.signal %ep, %dst, %src
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region) -> !aboleth.token
    %observed = tt.load %ptr : !tt.ptr<i32>
    aboleth.planned_wait %s, %observed, %n : !aboleth.token, i32, i32
    tt.return
  }
  tt.func public @later_unsafe_to_speculate(%ep: !aboleth.endpoint,
      %dst: !aboleth.remote_region, %src: !aboleth.local_region,
      %i: i32, %divisor: i32, %n: i32) {
    %s = aboleth.signal %ep, %dst, %src
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region) -> !aboleth.token
    %quotient = arith.divui %i, %divisor : i32
    aboleth.planned_wait %s, %quotient, %n : !aboleth.token, i32, i32
    tt.return
  }
}
// CHECK-LABEL: tt.func public @later_pure_bounds
// CHECK: arith.addi
// CHECK: arith.extui
// CHECK: arith.extui
// CHECK: arith.remui
// CHECK: arith.ori
// CHECK: %[[PURE:.*]] = aboleth.signal
// CHECK: aboleth.wait %[[PURE]], %{{.*}}
// CHECK-LABEL: tt.func public @later_observed_bound
// CHECK-NOT: arith.remui
// CHECK: %[[OBSERVED:.*]] = aboleth.signal
// CHECK-NEXT: %{{.*}} = tt.load
// CHECK-NEXT: aboleth.wait %[[OBSERVED]] : !aboleth.token
// CHECK-LABEL: tt.func public @later_unsafe_to_speculate
// CHECK-NOT: arith.remui
// CHECK: %[[UNSAFE:.*]] = aboleth.signal
// CHECK-NEXT: %{{.*}} = arith.divui
// CHECK-NEXT: aboleth.wait %[[UNSAFE]] : !aboleth.token
// CHECK-NOT: aboleth.planned_wait
