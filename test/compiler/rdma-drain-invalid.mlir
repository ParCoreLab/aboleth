// RUN: not aboleth-opt %s --pass lower-rdma 2>&1 | FileCheck %s

module {
  tt.func public @drain(%endpoint_ptr: !tt.ptr<i64>) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    aboleth.drain %endpoint {issuerScope = "warp"}
      : !aboleth.endpoint
    tt.return
  }
}

// CHECK: unsupported drain issuer scope warp
