// RUN: aboleth-opt %s --pass form-batches --pass analyze-issuers | FileCheck %s
// RUN: aboleth-opt %s --pass form-batches --pass analyze-issuers --pass lower-rdma | FileCheck %s --check-prefix=RDMA
//
// Each program selects its own endpoint when the table index is an injective
// function of its program id, so the endpoint has one issuer.

module {
  tt.func public @per_program(
      %table: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %first: i32, %bytes: i32,
      %source_key: i32, %destination_key: i32) {
    %true = arith.constant true
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %scaled = arith.muli %pid, %c2 : i32
    %index = arith.addi %scaled, %first : i32
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }

  // An index read from memory may repeat across programs.
  tt.func public @loaded_index(
      %table: !tt.ptr<i64>, %indices: !tt.ptr<i32>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %bytes: i32, %source_key: i32,
      %destination_key: i32) {
    %true = arith.constant true
    %pid = tt.get_program_id x : i32
    %slot = tt.addptr %indices, %pid : !tt.ptr<i32>, i32
    %index = tt.load %slot : !tt.ptr<i32>
    %ep = aboleth.endpoint %table[%index] : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }

  // Program p's second selection is program p+1's first.
  tt.func public @two_indices(
      %table: !tt.ptr<i64>, %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>, %bytes: i32, %source_key: i32,
      %destination_key: i32) {
    %true = arith.constant true
    %c1 = arith.constant 1 : i32
    %pid = tt.get_program_id x : i32
    %next = arith.addi %pid, %c1 : i32
    %ep0 = aboleth.endpoint %table[%pid] : !tt.ptr<i64> -> !aboleth.endpoint
    %ep1 = aboleth.endpoint %table[%next] : !tt.ptr<i64> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep0, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    %1 = aboleth.put %ep1, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    aboleth.wait %1 : !aboleth.token
    tt.return
  }
}

// CHECK-LABEL: tt.func public @per_program
// CHECK: aboleth.endpoint %{{.*}}[%{{.*}}] {exclusiveAxis = 0 : i32}
// CHECK: aboleth.submit
// CHECK-SAME: exclusiveIssuer

// CHECK-LABEL: tt.func public @loaded_index
// CHECK-NOT: exclusive
// CHECK: tt.return

// CHECK-LABEL: tt.func public @two_indices
// CHECK-NOT: exclusive
// CHECK: tt.return

// The reservation is a plain increment, publication does not wait for other
// producers, and the grid is checked once. The ready head, which only this
// issuer reads, is stored relaxed ahead of the doorbell, whose release covers
// it; with several producers it is released after the ring.
// RDMA-LABEL: tt.func public @per_program
// RDMA: tt.get_num_programs y
// RDMA: tt.get_num_programs z
// RDMA: tt.assert
// RDMA-NOT: tt.atomic_rmw
// RDMA: st.relaxed.gpu.global.L1::no_allocate.b64
// RDMA: st.release.sys.global.L1::no_allocate.b64
// RDMA: tt.return
// RDMA-LABEL: tt.func public @loaded_index
// RDMA-NOT: tt.assert
// RDMA: tt.atomic_rmw
// RDMA-NOT: st.relaxed.gpu
// RDMA: st.release.sys.global.L1::no_allocate.b64
// RDMA: st.release.gpu.global.L1::no_allocate.b64
