module {
  aboleth.endpoint_work {name = "missing-rank", maxItems = 1 : i64,
    payloadBytes = 16 : i64, rankCount = 2 : i64,
    partitions = [{backend = "p2p", peers = array<i32: 1>, availableLanes = 16 : i64,
      platform = "measured", queueMemory = "none", participantLinks = ["NV18"],
      participants = [{platform = "measured", queueMemory = "host", availableLanes = 16 : i64},
                      {platform = "measured", queueMemory = "host", availableLanes = 16 : i64}]}],
    profiles = []}
}
// CHECK: error: 'aboleth.endpoint_work' op participantLinks requires one canonical string per rank
