#pragma once

// The CPU-proxied datapath.
//
// A proxy endpoint is a queue the kernel writes commands into and a host
// thread drains: the GPU never touches the adapter. The same endpoint shape
// serves both providers — an mlx5 RC queue pair or an EFA-OFI endpoint — so
// which one is in play is a field, not a type.
//
// Internal to the runtime; see aboleth_runtime_internal.h.

#include "../aboleth_runtime_internal.h"

#include "aboleth/Target/ProxyAbi.h"
#include "aboleth/runtime_c.h"

#if ABOLETH_HAS_EFA
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace aboleth::runtime {

// Which provider ABOLETH_PROXY_PROVIDER asked for. Resolved once, when the
// shared runtime is created.
ProxyProviderKind configured_proxy_provider();

// The progress thread that drains proxy endpoints, and the per-shard EFA
// completion queues it polls. Both are private to the proxy implementation;
// an endpoint only holds them.
struct ProxyWorker;
#if ABOLETH_HAS_EFA
struct EfaProxyCqShard;
#endif

// Where an EFA batch got to. The libfabric path posts a batch's commands over
// several progress iterations, so the position has to survive between them.
struct EfaProxyBatchState {
    std::uint64_t target{};
    std::uint32_t command_count{};
    std::uint32_t next_command{};
    std::uint32_t pending_completions{};
    std::uint32_t completed_commands{};
    std::uint64_t counter_base{};
    std::uint64_t counter_target{};
    bool active{};
    bool force_flush{};
    std::uint32_t posts_since_flush{};
};

// One proxied endpoint: the shared command queue, the provider objects behind
// it, and the scratch the worker builds work requests in. The work-request
// arrays are members rather than locals so a batch costs no allocation.
struct ProxyEndpointContext {
    std::shared_ptr<SharedRuntimeContext> runtime;
    int gpu_index{};
    int worker_shard{};
    ProxyProviderKind provider{
        ProxyProviderKind::VerbsRc};
    ibv_cq* send_cq{};
    ibv_qp* sender{};
    std::uint32_t psn{};
#if ABOLETH_HAS_EFA
    fid_av* efa_av{};
    fid_cq* efa_tx_cq{};
    fid_cq* efa_rx_cq{};
    fid_ep* efa_endpoint{};
    fid_cntr* efa_tx_counter{};
    fid_mr* efa_signal_mr{};
    std::vector<std::uint64_t>
        efa_signal_values;
    std::shared_ptr<EfaProxyCqShard>
        efa_cq_shard;
    fi_addr_t efa_peer{FI_ADDR_NOTAVAIL};
#endif
    std::array<
        std::uint8_t,
        kAbolethConnectionAddressBytes>
        efa_address{};
    std::uint32_t efa_address_length{};
    std::uint64_t* producer_index{};
    aboleth::abi::ProxyQueueState* queue_host{};
    aboleth::abi::ProxyQueueState* queue_device{};
    std::array<
        ibv_sge,
        aboleth::abi::kProxyMaximumCommands>
        scatter_entries{};
    std::array<
        ibv_send_wr,
        aboleth::abi::kProxyMaximumCommands>
        work_requests{};
    std::array<
        std::uint64_t,
        aboleth::abi::kProxyMaximumCommands>
        inline_values{};
    std::uint64_t atomic_result{};
    ibv_mr* atomic_result_mr{};
    bool debug{};
    std::uint64_t next_sequence{};
    std::uint64_t outstanding_target{};
    bool outstanding{};
    std::uint64_t failure_status{};
    std::chrono::steady_clock::time_point
        completion_deadline{};
    std::uint64_t efa_accepted_sequence{};
    std::array<
        EfaProxyBatchState,
        aboleth::abi::kProxySlotCount>
        efa_batches{};
    std::uint64_t efa_batch_count{};
    std::uint64_t efa_command_count{};
    std::uint64_t efa_payload_bytes{};
    std::uint64_t efa_adjacent_destinations{};
    std::uint64_t efa_adjacent_source_and_destination{};
    std::uint32_t efa_post_batch{8};
    std::uint32_t efa_post_budget{8};
    std::shared_ptr<ProxyWorker> worker;

    ~ProxyEndpointContext();
};

// Creates the endpoint, joins it to the progress thread for its shard, and
// fills `descriptor` with the queue the kernel writes into.
std::unique_ptr<ProxyEndpointContext> create_proxy_endpoint(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    aboleth::abi::ProxyEndpointDescriptor& descriptor,
    int worker_shard = 0);

// Cross-connects two proxy endpoints owned by this process. Both must be on
// the same provider.
void connect_proxy_endpoints(
    ProxyEndpointContext* first,
    ProxyEndpointContext* second);

} // namespace aboleth::runtime
