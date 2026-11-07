// A phase fence executes once per producer only when that producer posted
// deferred WQEs.
//
// RUN: aboleth-opt %s --pass lower-rdma | FileCheck %s

module {
  tt.func public @fence(%predicate: i1) {
    aboleth.system_fence %predicate : i1
    tt.return
  }
}

// CHECK-LABEL: tt.func public @fence
// CHECK: scf.if
// CHECK: fence.acq_rel.sys
// CHECK-NOT: aboleth.
