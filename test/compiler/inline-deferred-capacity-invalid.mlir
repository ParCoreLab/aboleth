// Transport planning bounds a deferral window in commands, but an endpoint's
// capacity counts 64-byte work queue entry blocks, and an inline command whose
// payload leaves the first block costs more than one. Here `sender` posts a
// 32-byte inline payload — two blocks — and its completion is deferred into
// `reuse`, so the queue can fill before the plan believes it has and the
// deferred completion would wait for room only it can free. Planning refuses
// rather than emitting an optimistic bound.
//
// RUN: not aboleth-opt %s --pass plan-transport 2>&1 | FileCheck %s

module {
  aboleth.transport_program {
    "aboleth.transport_phase"() <{
      name = "send",
      boundKernel = "sender",
      commandsPerIteration = 1 : i32,
      iterationCount = 1 : i32,
      payloadBytes = 32 : i64,
      peerTransports = ["rdma"],
      peerCapacities = array<i32: 33>,
      policyTinyPayloadBytes = 4096 : i64,
      reads = [{allocation = 0 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 1 : i32}],
      writes = [{allocation = 1 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 64 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    "aboleth.transport_kernel"() <{
      name = "reuse",
      boundKernel = "reuse",
      reads = [],
      writes = [{allocation = 0 : i64, generation = 1 : i64,
        offset = 0 : i64, length = 64 : i64, stride = 0 : i64,
        period = 0 : i64, count = 1 : i32}]
    }> : () -> ()
    ub.unreachable
  } {
    name = "inline-deferred"
  }

  tt.func public @sender(
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
    %put = aboleth.put_inline
      %endpoint, %destination, [%value, %value, %value, %value], %predicate
      : (!aboleth.endpoint, !aboleth.remote_region, i64, i64, i64, i64, i1)
        -> !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.put_inline' op carries a 32-byte inline payload, which spans more than one work queue entry block, in a phase whose completion is deferred
