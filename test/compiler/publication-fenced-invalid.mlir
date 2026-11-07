module {
  tt.func public @fenced_credit(
      %endpoint: !tt.ptr<i64>, %flag: !tt.ptr<i64>, %epoch: i64,
      %destination_key: i32) {
    %ep = aboleth.bind_endpoint %endpoint : !tt.ptr<i64> -> !aboleth.endpoint
    %flag_dst = aboleth.bind_remote_region %flag, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %0 = aboleth.signal %ep, %flag_dst, %epoch
      {weakPublication, fencedSignal}
      : (!aboleth.endpoint, !aboleth.remote_region, i64) -> !aboleth.token
    aboleth.wait %0 : !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.signal' op a fenced signal only applies to a strong publication
