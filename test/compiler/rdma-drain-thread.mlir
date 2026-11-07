// Thread-scoped artifacts keep their original scalar reservation lowering.
// RUN: aboleth-opt %s --pass lower-rdma | FileCheck %s --check-prefix=RDMA
// RUN: aboleth-opt %s --pass lower-rdma=test/rdma.bc | FileCheck %s --check-prefix=DEVICE
module {
  tt.func public @drain_thread(%endpoint_ptr: !tt.ptr<i64>) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %endpoint {issuerScope = "thread"} : !aboleth.endpoint
    tt.return
  }
}
// RDMA-LABEL: tt.func public @drain_thread
// RDMA: tt.atomic_rmw add, relaxed, gpu
// RDMA: arith.constant 134217728 : i32
// RDMA: %clock64
// RDMA-NOT: aboleth.
// DEVICE-LABEL: tt.func public @drain_thread
// DEVICE: symbol = "aboleth_rdma_drain"
// DEVICE-NOT: aboleth.
