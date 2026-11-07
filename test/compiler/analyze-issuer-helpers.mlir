// Private JIT arguments carry call-site values, including nonuniform lanes.
module {
  tt.func public @entry(%table: !tt.ptr<i64>) {
    %lane = tt.get_program_id y : i32
    tt.call @count_lane(%table, %lane) : (!tt.ptr<i64>, i32) -> ()
    tt.call @other_lanes(%table, %lane) : (!tt.ptr<i64>, i32) -> ()
    tt.return
  }
  tt.func private @count_lane(%table: !tt.ptr<i64>, %lane: i32) {
    %peer = tt.get_program_id x : i32
    %index = arith.addi %lane, %peer : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  tt.func private @other_lanes(%table: !tt.ptr<i64>, %lane: i32) {
    %peer = tt.get_program_id x : i32
    %index = arith.addi %lane, %peer : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  tt.func public @loaded_entry(%table: !tt.ptr<i64>, %indices: !tt.ptr<i32>) {
    %lane = tt.load %indices : !tt.ptr<i32>
    tt.call @unknown_lane(%table, %lane) : (!tt.ptr<i64>, i32) -> ()
    tt.return
  }
  tt.func private @unknown_lane(%table: !tt.ptr<i64>, %lane: i32) {
    %peer = tt.get_program_id x : i32
    %index = arith.addi %lane, %peer : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  // An uncalled helper has no kernel-uniformity contract.
  tt.func private @uncalled(%table: !tt.ptr<i64>, %lane: i32) {
    %peer = tt.get_program_id x : i32
    %index = arith.addi %lane, %peer : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  // Distinct helper indices can alias across CTAs on the same actual table.
  tt.func public @alias_entry(%table: !tt.ptr<i64>) {
    tt.call @first(%table) : (!tt.ptr<i64>) -> ()
    tt.call @second(%table) : (!tt.ptr<i64>) -> ()
    tt.return
  }
  tt.func private @first(%table: !tt.ptr<i64>) {
    %index = tt.get_program_id x : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  tt.func private @second(%table: !tt.ptr<i64>) {
    %peer = tt.get_program_id x : i32
    %c1 = arith.constant 1 : i32
    %index = arith.addi %peer, %c1 : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
}
// CHECK-LABEL: tt.func private @count_lane(
// CHECK: exclusiveAxes = array<i32: 0, 1>, exclusiveGridShape = array<i64: 1, 0, 0>
// CHECK: aboleth.drain
// CHECK-SAME: exclusiveIssuer
// CHECK-LABEL: tt.func private @other_lanes(
// CHECK: exclusiveAxes = array<i32: 0, 1>, exclusiveGridShape = array<i64: 1, 0, 0>
// CHECK-LABEL: tt.func private @unknown_lane(
// CHECK-NOT: exclusive
// CHECK: tt.return
// CHECK-LABEL: tt.func private @uncalled(
// CHECK-NOT: exclusive
// CHECK: tt.return
// CHECK-LABEL: tt.func private @first(
// CHECK-NOT: exclusive
// CHECK: tt.return
// CHECK-LABEL: tt.func private @second(
// CHECK-NOT: exclusive
// CHECK: tt.return
