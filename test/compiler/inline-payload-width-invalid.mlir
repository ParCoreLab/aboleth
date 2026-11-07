// The inline payload is packed into 32-bit words, and the packing is defined
// for one-, two-, three-, four-, and eight-byte values. An i48 is a whole
// number of bytes but has no packing: the lowering would have to split or
// extend it, and no rule says which. The verifier refuses it here rather than
// letting a backend emit arithmetic on mismatched widths.
//
// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @odd_width_value(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %destination_key: i32,
      %value: i48,
      %predicate: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %put = aboleth.put_inline
      %endpoint, %destination, [%value], %predicate
      : (!aboleth.endpoint, !aboleth.remote_region, i48, i1)
        -> !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.put_inline' op inline payload value 0 has width 48, which has no defined inline packing; an inline payload carries i8, i16, i24, i32, or i64 values
