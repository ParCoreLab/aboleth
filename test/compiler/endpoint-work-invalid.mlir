module {
  aboleth.endpoint_work {name = "bad", maxItems = 1 : i64, payloadBytes = 16 : i64,
    partitions = [{backend = "rdma", peers = array<i32: 1, 1>,
      availableLanes = 8 : i64, platform = "measured", queueMemory = "host"}], profiles = []}
}
// CHECK: error: 'aboleth.endpoint_work' op requires sorted unique nonnegative peers across partitions
