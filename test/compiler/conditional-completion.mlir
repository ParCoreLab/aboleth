// RUN: triton-opt %s --aboleth-form-batches | FileCheck %s --check-prefix=CORE
// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-proxy | FileCheck %s --check-prefix=LOWERED
// RUN: triton-opt %s --aboleth-form-batches --aboleth-lower-rdma | FileCheck %s --check-prefix=LOWERED

module {
  tt.func public @conditional_completion(
      %endpoint: !tt.ptr<i64>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32,
      %complete: i1) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i64> -> !aboleth.endpoint
    %signal_dst = aboleth.bind_remote_region
      %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region
      %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %signal = aboleth.signal
      %ep, %signal_dst, %signal_src
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region) -> !aboleth.token
    aboleth.wait %signal, %complete : !aboleth.token
    tt.return
  }
}

// CORE-LABEL: tt.func public @conditional_completion
// CORE: %[[SUBMIT:.*]] = aboleth.submit
// CORE: aboleth.poll %[[SUBMIT]], %arg5
// CORE-SAME: coalesced

// LOWERED-LABEL: tt.func public @conditional_completion
// LOWERED: scf.if %arg5
// LOWERED-NOT: aboleth.

// INLINE-LABEL: tt.func public @conditional_completion
// INLINE: tt.load %arg2
// INLINE: arith.constant -2147483640 : i32
// INLINE-NOT: aboleth.
