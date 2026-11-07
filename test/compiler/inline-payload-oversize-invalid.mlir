// An inline payload rides inside the work queue entry, and the entry's control
// segment counts its 16-byte segments in six bits: at most 63. Two go to the
// control and remote-address segments, leaving the inline-data segment its
// 4-byte header plus 972 bytes of payload. One byte more wraps the count and
// the responder reads an entry that was never written, so the verifier refuses
// it here rather than letting a lowering build it.
//
// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @oversized_inline_payload(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %destination_key: i32,
      %value: i64,
      %predicate: i1) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    // 122 eight-byte values: 976 bytes.
    %put = aboleth.put_inline
      %endpoint, %destination, [%value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value, %value], %predicate
      : (!aboleth.endpoint, !aboleth.remote_region, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i1)
        -> !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.put_inline' op inline payload of 976 bytes exceeds the 972-byte maximum one work queue entry can carry
