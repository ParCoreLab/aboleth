module {
  aboleth.transport_program {
    "aboleth.transport_fragment"() <{
      programId = 7 : i32,
      name = "dispatch",
      argumentNames = [
        "endpoint", "destination", "source", "remote_signal",
        "local_signal", "bytes", "source_key", "destination_key",
        "predicate", "iteration", "iterations"
      ]
    }> ({
    ^bb0(
        %endpoint: i64,
        %destination: i64,
        %source: i64,
        %remoteSignal: i64,
        %localSignal: i64,
        %bytes: i32,
        %sourceKey: i32,
        %destinationKey: i32,
        %predicate: i1,
        %iteration: i32,
        %iterations: i32):
      %ep = aboleth.bind_endpoint %endpoint
        : i64 -> !aboleth.endpoint
      %dst = aboleth.bind_remote_region %destination, %destinationKey
        : (i64, i32) -> !aboleth.remote_region
      %src = aboleth.bind_local_region %source, %sourceKey
        : (i64, i32) -> !aboleth.local_region
      %signalDst = aboleth.bind_remote_region
        %remoteSignal, %destinationKey
        : (i64, i32) -> !aboleth.remote_region
      %signalSrc = aboleth.bind_local_region %localSignal, %sourceKey
        : (i64, i32) -> !aboleth.local_region
      %put = aboleth.put %ep, %dst, %src, %bytes, %predicate
        : (!aboleth.endpoint, !aboleth.remote_region,
           !aboleth.local_region, i32, i1) -> !aboleth.token
      %ready = aboleth.signal %ep, %signalDst, %signalSrc, %put
        : (!aboleth.endpoint, !aboleth.remote_region,
           !aboleth.local_region, !aboleth.token) -> !aboleth.token
      aboleth.planned_wait %ready, %iteration, %iterations
        : !aboleth.token, i32, i32
      ub.unreachable
    }) : () -> ()
    ub.unreachable
  } {
    name = "external-inline-fragments"
  }
}
