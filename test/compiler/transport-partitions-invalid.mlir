module {
  aboleth.transport_program {
    ub.unreachable
  } {name = "duplicate", peerRanks = array<i32: 2, 2>,
     peerTransports = ["rdma", "p2p"]}
}
// CHECK: error: 'aboleth.transport_program' op peer ranks must be sorted, unique and nonnegative
