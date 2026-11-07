#pragma once

// Turning two endpoints into a connection.
//
// The queue-pair state machine is the same whoever owns the endpoints — a
// direct kernel endpoint, a proxy endpoint, or the two halves of a local pair
// — so the INIT/RTR/RTS transitions, the packet sequence numbers and the
// path-MTU negotiation live here rather than in any one endpoint module.
//
// Internal to the runtime; see aboleth_runtime_internal.h.

#include "../aboleth_runtime_internal.h"
#include "aboleth_endpoint_rc.h"

#include "aboleth/runtime_c.h"

#include <cstdint>

namespace aboleth::runtime {

// A fresh 24-bit packet sequence number. Seeded per process so two runs
// against the same peer do not open with the same numbers.
std::uint32_t allocate_psn();

// The MTU to put on the path: the port's active MTU unless ABOLETH_PATH_MTU
// asks for a smaller one.
ibv_mtu select_path_mtu(ibv_mtu active_mtu);

// log2 of the RDMA reads and atomics a queue pair keeps in flight, as initiator
// and as responder. One by default; ABOLETH_RD_ATOMIC_DEPTH (a power of two up
// to 16) asks for more. With one, every strong signal (a compare-and-swap)
// waits a round trip for the previous one.
std::uint32_t log_rd_atomic_depth();

// The three transitions an RC queue pair makes before it can carry traffic.
// RTR needs both ends' port addresses because IB-to-IB paths address by LID
// and everything else by GRH.
void qp_to_init(
    ibv_qp* qp,
    const PortAddress& local);
void qp_to_rtr(
    ibv_qp* qp,
    std::uint32_t remote_qpn,
    std::uint32_t remote_psn,
    const PortAddress& local,
    const PortAddress& remote,
    ibv_mtu path_mtu);
void qp_to_rts(
    ibv_qp* qp,
    std::uint32_t psn);

// The remote port as the exchanged connection blob describes it. Throws when
// the blob carries an MTU this build does not recognize.
PortAddress port_address(
    const AbolethConnectionInfo& info);

// Which of the three endpoint shapes a local handle carries.
enum class OwnedEndpointBackend {
    Direct,
    Proxy,
    Efa,
};

// Cross-connects two endpoints owned by this process, as the loopback and
// differential paths need.
void connect_owned_endpoints(
    OwnedEndpointContext* first,
    OwnedEndpointContext* second);

} // namespace aboleth::runtime
