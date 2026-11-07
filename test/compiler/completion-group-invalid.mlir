// RUN: not triton-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @wait_before_submission(
      %endpoint: !aboleth.endpoint,
      %remote_signal: !aboleth.remote_region,
      %local_signal: !aboleth.local_region) {
    %group = aboleth.begin_group %endpoint
      : !aboleth.endpoint -> !aboleth.group
    aboleth.wait_group %group : !aboleth.group
    %signal = aboleth.signal
      %endpoint, %remote_signal, %local_signal group %group
      : (!aboleth.endpoint, !aboleth.remote_region,
         !aboleth.local_region, !aboleth.group) -> !aboleth.token
    tt.return
  }
}

// CHECK: error: 'aboleth.wait_group' op must post-dominate every submission in its completion group
