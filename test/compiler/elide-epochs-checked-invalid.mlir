module {
  tt.func public @elided_compare_and_swap(
      %endpoint: !tt.ptr<i64>, %flag: !tt.ptr<i64>, %epoch: i64,
      %destination_key: i32, %due: i1) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %0 = aboleth.post_signal %ep, %flag_dst, %epoch if %due
      : (!aboleth.endpoint, !aboleth.remote_region, i64, i1) -> !aboleth.token
    %1 = aboleth.submit %ep, %0 {batchSize = 1 : i32}
      : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.post_signal' op a predicated checked signal needs the value it replaces
