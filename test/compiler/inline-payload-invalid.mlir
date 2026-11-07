// An inline payload is the concatenation of its values' little-endian bytes, so
// a value that is not a whole number of bytes wide has no defined placement.
//
// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @sub_byte_value(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %destination_key: i32,
      %flag: i1,
      %predicate: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %put = aboleth.put_inline
      %endpoint, %destination, [%flag], %predicate
      : (!aboleth.endpoint, !aboleth.remote_region, i1, i1)
        -> !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.put_inline' op inline payload value 0 must be a byte-sized integer, not 'i1'
