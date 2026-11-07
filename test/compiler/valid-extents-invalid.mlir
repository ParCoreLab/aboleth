module {
  tt.func public @bounded(%endpoint: !tt.ptr<i64>, %source: !tt.ptr<i8>,
      %destination: !tt.ptr<i8>, %flag: !tt.ptr<i64>, %key: i32,
      %count: i64, %offset: i64, %bytes: i32, %epoch: i64) {
    %true = arith.constant true
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %live = aboleth.valid_extent %count {capacity = 2147483647 : i64} : i64
    %src = aboleth.bind_valid_local_region %source, %key, %live, %offset
      {capacity = 2147483647 : i64, itemBytes = 16 : i64}
      : (!tt.ptr<i8>, i32, i64, i64) -> !aboleth.local_region
    %dst = aboleth.bind_remote_region %destination, %key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %ready = aboleth.bind_remote_region %flag, %key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %put = aboleth.put %ep, %dst, %src, %bytes, %true
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %published = aboleth.signal %ep, %ready, %epoch, %put
      : (!aboleth.endpoint, !aboleth.remote_region, i64, !aboleth.token) -> !aboleth.token
    aboleth.wait %published : !aboleth.token
    tt.return
  }
}
// CHECK: error: 'aboleth.bind_valid_local_region' op capacity times itemBytes must fit positive i32 transfer bytes
