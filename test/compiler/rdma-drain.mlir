// Capacity reclamation is a dedicated completion NOP followed by one CQ poll.
//
// RUN: aboleth-opt %s --pass lower-rdma | FileCheck %s --check-prefix=RDMA
// RUN: aboleth-opt %s --pass lower-rdma=test/rdma.bc | FileCheck %s --check-prefix=DEVICE

module {
  tt.func public @drain(%endpoint_ptr: !tt.ptr<i64>) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %endpoint : !aboleth.endpoint
    tt.return
  }
}

// RDMA-LABEL: tt.func public @drain
// RDMA-NOT: tt.atomic_rmw
// RDMA: tt.elementwise_inline_asm "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;"
// RDMA-NOT: tt.atomic_rmw
// RDMA: arith.constant 134217728 : i32
// RDMA: %clock64
// RDMA-NOT: aboleth.

// DEVICE-LABEL: tt.func public @drain
// DEVICE: symbol = "aboleth_rdma_drain"
// DEVICE-NOT: aboleth.
