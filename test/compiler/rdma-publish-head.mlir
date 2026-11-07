// Queue-head publication is a standalone phase operation: it posts no WQE and
// may skip the doorbell when no producer advanced.
//
// RUN: aboleth-opt %s --pass lower-rdma | FileCheck %s --check-prefix=RDMA
// RUN: aboleth-opt %s --pass lower-rdma=test/rdma.bc | FileCheck %s --check-prefix=DEVICE

module {
  tt.func public @publish(
      %endpoint_ptr: !tt.ptr<i64>,
      %only_if_advanced: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.publish_head %endpoint, %only_if_advanced
      : !aboleth.endpoint, i1
    tt.return
  }
}

// RDMA-LABEL: tt.func public @publish
// RDMA: ld.acquire.gpu.global
// RDMA: ld.acquire.gpu.global
// RDMA: arith.cmpi ugt
// RDMA: scf.if
// RDMA: st.relaxed.sys.global.L1::no_allocate.b32
// RDMA-NOT: aboleth.

// DEVICE-LABEL: tt.func public @publish
// DEVICE: symbol = "aboleth_rdma_publish_head"
// DEVICE-NOT: symbol = "aboleth_rdma_reserve"
// DEVICE-NOT: symbol = "aboleth_rdma_poll"
// DEVICE-NOT: aboleth.
