module {
  aboleth.endpoint_work {name = "invalid-warp-count", maxItems = 1 : i64, payloadBytes = 16 : i64,
    partitions = [], profiles = [{backend = "p2p", platform = "measured", queueMemory = "none",
      bytesPerEndpoint = 262144 : i64, maxLanes = 16 : i64, senderWarps = 3 : i64, identity = "bad"}]}
}
// CHECK: error: 'aboleth.endpoint_work' op measured senderWarps must be 1, 2, 4 or 8
