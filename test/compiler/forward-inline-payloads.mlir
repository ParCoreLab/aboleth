// Forwarding a staged scalar payload into an inline transfer. The positive case
// is the shape a Tier 3 kernel produces: constant-offset scalar stores into a
// scratch buffer, then a put of exactly those bytes. Every other function states
// a fact the pass cannot prove and must therefore leave alone.
//
// RUN: aboleth-opt %s --pass forward-inline-payloads \
// RUN:   | FileCheck %s --check-prefix=FORWARD
// RUN: aboleth-opt %s --pass forward-inline-payloads=16 \
// RUN:   | FileCheck %s --check-prefix=CAP16
// RUN: aboleth-opt %s --pass forward-inline-payloads --pass form-batches \
// RUN:   --pass lower-rdma | FileCheck %s --check-prefix=RDMA

module {
  // 2 x i64 + 1 x i32 = 20 bytes, tiling [0, 20) of the scratch buffer.
  tt.func public @forwarded(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %scratch: !tt.ptr<i64>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32,
      %first: i64,
      %second: i64,
      %tail: i32) {
    %true = arith.constant true
    %bytes = arith.constant 20 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    tt.store %scratch, %first : !tt.ptr<i64>
    %at8 = tt.addptr %scratch, %one : !tt.ptr<i64>, i32
    tt.store %at8, %second : !tt.ptr<i64>
    %at16 = tt.addptr %scratch, %two : !tt.ptr<i64>, i32
    %at16_i32 = tt.bitcast %at16 : !tt.ptr<i64> -> !tt.ptr<i32>
    tt.store %at16_i32, %tail : !tt.ptr<i32>
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %scratch, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.put %endpoint, %destination, %source, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.signal %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }

  // An intervening load of the staged range: the buffer is observed, so its
  // stores are not dead and the transfer keeps reading it.
  tt.func public @aliasing_read(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %scratch: !tt.ptr<i64>,
      %observed: !tt.ptr<i64>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32,
      %first: i64) {
    %true = arith.constant true
    %bytes = arith.constant 8 : i32
    tt.store %scratch, %first : !tt.ptr<i64>
    %seen = tt.load %scratch : !tt.ptr<i64>
    tt.store %observed, %seen : !tt.ptr<i64>
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %scratch, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.put %endpoint, %destination, %source, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.signal %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }

  // 32 bytes: over the default cap, and over the 16-byte cap as well.
  tt.func public @over_cap(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %scratch: !tt.ptr<i64>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32,
      %value: i64) {
    %true = arith.constant true
    %bytes = arith.constant 32 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %three = arith.constant 3 : i32
    tt.store %scratch, %value : !tt.ptr<i64>
    %at8 = tt.addptr %scratch, %one : !tt.ptr<i64>, i32
    tt.store %at8, %value : !tt.ptr<i64>
    %at16 = tt.addptr %scratch, %two : !tt.ptr<i64>, i32
    tt.store %at16, %value : !tt.ptr<i64>
    %at24 = tt.addptr %scratch, %three : !tt.ptr<i64>, i32
    tt.store %at24, %value : !tt.ptr<i64>
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %scratch, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.put %endpoint, %destination, %source, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.signal %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }

  // A runtime store offset: the pass cannot say which bytes the store covers.
  tt.func public @dynamic_offset(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %scratch: !tt.ptr<i64>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32,
      %slot: i32,
      %value: i64) {
    %true = arith.constant true
    %bytes = arith.constant 8 : i32
    %moving = tt.addptr %scratch, %slot : !tt.ptr<i64>, i32
    tt.store %moving, %value : !tt.ptr<i64>
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %moving, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.put %endpoint, %destination, %source, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.signal %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }

  // Only 8 of the 16 transferred bytes are stored; the rest would come from the
  // buffer's previous contents.
  tt.func public @partial_tiling(
      %endpoint_ptr: !tt.ptr<i64>,
      %destination_ptr: !tt.ptr<i8>,
      %scratch: !tt.ptr<i64>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %source_key: i32,
      %destination_key: i32,
      %value: i64) {
    %true = arith.constant true
    %bytes = arith.constant 16 : i32
    tt.store %scratch, %value : !tt.ptr<i64>
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i64> -> !aboleth.endpoint
    %destination = aboleth.bind_remote_region
      %destination_ptr, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %source = aboleth.bind_local_region %scratch, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %put = aboleth.put %endpoint, %destination, %source, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, i32, i1) -> !aboleth.token
    %signal = aboleth.signal %endpoint, %signal_dst, %signal_src, %put
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %signal : !aboleth.token
    tt.return
  }
}

// The three stores become the payload operands, in address order, and neither
// the stores nor the staging binding survive.
// FORWARD-LABEL: tt.func public @forwarded
// FORWARD-NOT: tt.store
// FORWARD-NOT: tt.addptr
// FORWARD: aboleth.put_inline
// FORWARD-SAME: [%arg7, %arg8, %arg9]
// FORWARD-NOT: aboleth.put %

// FORWARD-LABEL: tt.func public @aliasing_read
// FORWARD: tt.store
// FORWARD: aboleth.put %
// FORWARD-NOT: aboleth.put_inline

// FORWARD-LABEL: tt.func public @over_cap
// FORWARD: tt.store
// FORWARD: aboleth.put %
// FORWARD-NOT: aboleth.put_inline

// FORWARD-LABEL: tt.func public @dynamic_offset
// FORWARD: tt.store
// FORWARD: aboleth.put %
// FORWARD-NOT: aboleth.put_inline

// FORWARD-LABEL: tt.func public @partial_tiling
// FORWARD: tt.store
// FORWARD: aboleth.put %
// FORWARD-NOT: aboleth.put_inline

// A 16-byte cap rejects the 20-byte payload the default accepts, so the cap is
// what decides and not the shape.
// CAP16-LABEL: tt.func public @forwarded
// CAP16: tt.store
// CAP16: aboleth.put %
// CAP16-NOT: aboleth.put_inline

// End to end: the forwarded transfer reaches the queue as an inline-data WQE
// carrying 20 bytes, and nothing reads a staging buffer.
// RDMA-LABEL: tt.func public @forwarded
// RDMA: arith.constant -2147483628 : i32
// RDMA-NOT: aboleth.
