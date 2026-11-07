// One externally published planned put: reserve and write one WQE, but leave
// publication, completion, and phase-plan selection to the enclosing adapter.
//
// RUN: aboleth-opt %s --pass lower-rdma | FileCheck %s --check-prefix=RDMA
// RUN: aboleth-opt %s --pass lower-rdma=test/rdma.bc | FileCheck %s --check-prefix=DEVICE

module {
  tt.func public @planned_put(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %source_ptr: !tt.ptr<i8>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %source_ptr, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %put = aboleth.post_put
      %endpoint, %destination, %source, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %posted = aboleth.submit %endpoint, %put {
      batchSize = 1 : i32,
      deferCompletion,
      externalPublication,
      plannedReservation,
      reservationScope = "cta",
      issuerScope = "thread"
    } : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token
    tt.return
  }
}

// The direct lowering reserves at CTA scope and performs no capacity wait,
// doorbell publication, or completion polling.
// RDMA-LABEL: tt.func public @planned_put
// RDMA: tt.atomic_rmw add, relaxed, cta
// RDMA-NOT: scf.while
// RDMA-NOT: %clock64
// RDMA-NOT: aboleth.

// The device-library lowering calls only the planned reservation and WQE
// constructor. Publication and completion remain external.
// DEVICE-LABEL: tt.func public @planned_put
// DEVICE-NOT: %tid.x
// DEVICE: symbol = "aboleth_rdma_reserve_planned"
// DEVICE: symbol = "aboleth_rdma_post_write"
// DEVICE-NOT: symbol = "aboleth_rdma_publish"
// DEVICE-NOT: symbol = "aboleth_rdma_poll"
// DEVICE-NOT: aboleth.
