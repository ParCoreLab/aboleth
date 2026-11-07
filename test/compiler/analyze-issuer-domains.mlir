// RUN: aboleth-opt %s --pass analyze-issuers | FileCheck %s
// RUN: aboleth-opt %s --pass analyze-issuers --pass lower-rdma | FileCheck %s --check-prefix=GRID

module {
  tt.func public @rotated(%table: !tt.ptr<i64>) {
    %peer = tt.get_program_id x : i32
    %c3 = arith.constant 3 : i32
    %c8 = arith.constant 8 : i32
    %shifted = arith.addi %peer, %c3 : i32
    %index = arith.remsi %shifted, %c8 : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  tt.func public @peer_lane(%table: !tt.ptr<i64>) {
    %peer = tt.get_program_id x : i32
    %lane = tt.get_program_id y : i32
    %c8 = arith.constant 8 : i32
    %base = arith.muli %lane, %c8 : i32
    %index = arith.addi %base, %peer : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  tt.func public @rotated_peer_lane(%table: !tt.ptr<i64>) {
    %peer = tt.get_program_id z : i32
    %lane = tt.get_program_id x : i32
    %c3 = arith.constant 3 : i32
    %c8 = arith.constant 8 : i32
    %shifted = arith.addi %c3, %peer : i32
    %rotated = arith.remui %shifted, %c8 : i32
    %base = arith.muli %c8, %lane : i32
    %index = arith.addi %rotated, %base : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  // A signed negative remainder does not select a valid endpoint row.
  tt.func public @negative_rotation(%table: !tt.ptr<i64>) {
    %peer = tt.get_program_id x : i32
    %negative = arith.constant -1 : i32
    %c8 = arith.constant 8 : i32
    %shifted = arith.addi %peer, %negative : i32
    %index = arith.remsi %shifted, %c8 : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  // Both terms use the same axis; no two-axis ownership fact is available.
  tt.func public @same_axis(%table: !tt.ptr<i64>) {
    %peer = tt.get_program_id x : i32
    %c8 = arith.constant 8 : i32
    %base = arith.muli %peer, %c8 : i32
    %index = arith.addi %base, %peer : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  // An arbitrary loaded ordering may repeat endpoint indices.
  tt.func public @loaded_order(%table: !tt.ptr<i64>, %order: !tt.ptr<i32>) {
    %peer = tt.get_program_id x : i32
    %slot = tt.addptr %order, %peer : !tt.ptr<i32>, i32
    %index = tt.load %slot : !tt.ptr<i32>
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
  // A four-peer rotation cannot prove an eight-peer linearization domain.
  tt.func public @mismatched_period(%table: !tt.ptr<i64>) {
    %peer = tt.get_program_id x : i32
    %lane = tt.get_program_id y : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %rotated = arith.remui %peer, %c4 : i32
    %base = arith.muli %lane, %c8 : i32
    %index = arith.addi %base, %rotated : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %ep : !aboleth.endpoint
    tt.return
  }
}

// CHECK-LABEL: @rotated(
// CHECK: exclusiveAxis = 0 : i32, exclusiveGridShape = array<i64: 8, 0, 0>
// CHECK: aboleth.drain
// CHECK-SAME: exclusiveIssuer
// CHECK-LABEL: @peer_lane(
// CHECK: exclusiveAxes = array<i32: 0, 1>, exclusiveGridShape = array<i64: 8, 0, 0>
// CHECK-LABEL: @rotated_peer_lane(
// CHECK: exclusiveAxes = array<i32: 0, 2>, exclusiveGridShape = array<i64: 0, 0, 8>
// CHECK-LABEL: @negative_rotation(
// CHECK-NOT: exclusive
// CHECK: tt.return
// CHECK-LABEL: @same_axis(
// CHECK-NOT: exclusive
// CHECK: tt.return
// CHECK-LABEL: @loaded_order(
// CHECK-NOT: exclusive
// CHECK: tt.return
// CHECK-LABEL: @mismatched_period(
// CHECK-NOT: exclusive
// CHECK: tt.return

// Every constrained extent is checked before resolving the endpoint handle.
// GRID-LABEL: @rotated(
// GRID: tt.get_num_programs x
// GRID: arith.cmpi eq
// GRID: tt.get_num_programs y
// GRID: tt.get_num_programs z
// GRID: tt.assert
// GRID-SAME: violates its proven launch grid
// GRID: tt.load
// GRID-LABEL: @peer_lane(
// GRID: tt.get_num_programs x
// GRID: arith.cmpi eq
// GRID: tt.get_num_programs z
// GRID: arith.extui
// GRID: tt.get_num_programs y
// GRID: arith.extui
// GRID: arith.muli
// GRID: arith.constant 2147483648 : i64
// GRID: arith.cmpi ule
// GRID: tt.assert
// GRID: tt.load
