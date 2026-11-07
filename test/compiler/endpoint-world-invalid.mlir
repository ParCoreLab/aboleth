module {
  aboleth.endpoint_work {name = "missing-image", maxItems = 512 : i64,
    payloadBytes = 2097152 : i64, rankCount = 3 : i64,
    partitions = [{backend = "rdma", peers = array<i32: 1>, availableLanes = 4 : i64,
                   platform = "measured", queueMemory = "host"}], profiles = []}
}
// CHECK: error: 'aboleth.endpoint_work' op rank-indexed readiness requires a canonical participant image
