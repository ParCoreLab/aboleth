#pragma once

// The RC endpoint a generated kernel drives directly.
//
// An owned endpoint is one send queue pair plus the mlx5 objects a kernel
// needs mapped into its address space: the send queue, its doorbell record,
// the blueflame page, the completion queue. Creating it publishes an
// RdmaEndpointDescriptor; connecting it is the connection module's job.
//
// Internal to the runtime; see aboleth_runtime_internal.h.

#include "../aboleth_runtime_internal.h"
#include "aboleth_doorbell_agent.h"

#include "aboleth/Target/RdmaAbi.h"

#include <infiniband/mlx5dv.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aboleth::runtime {

// The GPU-resident staged queue state: send queue, both doorbell records and
// the completion queue live in cudaMalloc memory the NIC reaches through
// DEVX dmabuf umems, so the GPU builds WQEs and polls CQEs at HBM latency
// and the NIC fetches them with one peer read. The QP/CQ are DEVX objects
// (no ibv_qp exists); only the UAR doorbell store stays with the host agent.
// Created by create_devx_staged_queues (aboleth_endpoint_devx.cu).
struct DevxQueueState {
    mlx5dv_devx_uar* uar{};
    mlx5dv_devx_umem* send_umem{};
    mlx5dv_devx_umem* dbr_umem{};
    mlx5dv_devx_umem* completion_umem{};
    mlx5dv_devx_obj* completion_obj{};
    mlx5dv_devx_obj* queue_obj{};
    // Raw cudaMalloc bases (freed) and their page-aligned views (used).
    void* send_raw{};
    void* dbr_raw{};
    void* completion_raw{};
    std::uint64_t send_buffer{};
    std::uint64_t dbr_buffer{};
    std::uint64_t completion_buffer{};
    std::uint32_t qpn{};
    std::uint32_t cqn{};
    std::uint8_t port_num{};

    ~DevxQueueState();
};

// A reliable-connected queue pair on `pd`, using `cq` for both directions.
// `granted_inline_bytes` receives what the adapter actually allowed, which is
// only reported by ibv_create_qp rewriting the capability it was handed.
// `send_wqe_requests` is the send-queue work-request count to ask for, or zero
// for the runtime's own kMaximumTransfers + 1.
ibv_qp* create_rc_qp(
    ibv_pd* pd,
    ibv_cq* cq,
    std::uint32_t maximum_inline_bytes = 0,
    std::uint32_t* granted_inline_bytes = nullptr,
    std::uint32_t send_wqe_requests = 0);

// The adapter may refuse a large max_inline_data request outright; the
// no-inline queue pair is the floor every compiled kernel already assumes.
ibv_qp* create_rc_qp_optional_inline(
    ibv_pd* pd,
    ibv_cq* cq,
    std::uint32_t maximum_inline_bytes,
    std::uint32_t* granted_inline_bytes,
    std::uint32_t send_wqe_requests = 0);

// What one endpoint asks of its queue pair beyond the runtime's defaults —
// the honored half of AbolethEndpointConfig, already checked by the C entry
// point. Zero means the default: kMaximumTransfers + 1 work requests, and
// ABOLETH_RDMA_INLINE_BYTES (or kRdmaInlineBytes) of inline capacity.
struct OwnedEndpointRequest {
    std::uint32_t send_wqe_requests{};
    std::uint32_t max_inline_bytes{};
};

// A host registration the endpoint took out of the runtime's shared registry
// and has to give back when it dies.
struct MappingRequest {
    void* pointer{};
    std::size_t bytes{};
    unsigned int flags{};
};

// One kernel-driven send endpoint. Everything the descriptor points at is
// owned here, so destroying the context tears the mappings down in the order
// they were taken.
struct OwnedEndpointContext {
    std::shared_ptr<SharedRuntimeContext> runtime;
    int gpu_index{};
    ibv_td* thread_domain{};
    ibv_pd* parent_domain{};
    ibv_cq* send_cq{};
    ibv_qp* sender{};
    std::uint32_t psn{};
    ibv_mr* completion_mr{};
    std::uint64_t* producer_index{};
    std::uint64_t* completion_consumer{};
    // The provider's own send queue and doorbell record, unmapped: the
    // descriptor carries the GPU-visible aliases, and reading the wire bytes
    // back on the host is what the differential tests compare.
    void* send_queue_host{};
    std::uint32_t* send_dbrec_host{};
    std::uint32_t send_stride{};
    std::uint32_t send_wqe_count{};
    std::uint32_t max_inline_bytes{};
    // Staged submission: the descriptor's blueflame word points at this
    // host-pinned cell instead of the UAR, and the doorbell agent forwards
    // what the kernel release-publishes there. Direct endpoints leave all
    // four fields empty. See docs/RDMA_ABI.md § Submission modes.
    bool staged_submit{};
    std::uint64_t* doorbell_cell{};
    std::shared_ptr<DoorbellAgent> agent;
    std::uint64_t agent_token{};
    std::vector<MappingRequest> mappings;
    // Non-null on the DEVX-backed staged path: queues in GPU memory, no
    // ibv_qp (sender stays null; qpn/psn live here and below).
    std::unique_ptr<DevxQueueState> devx;
    std::uint32_t queue_pair_number{};

    ~OwnedEndpointContext() {
        cudaSetDevice(gpu_index);
        // The agent must stop touching the doorbell cell and the UAR before
        // either goes away; remove() blocks until its loop has let go.
        if (agent_token != 0 && agent != nullptr) {
            agent->remove(agent_token);
        }
        agent.reset();
        // After the agent lets go of the UAR: the DEVX objects, umems, UAR
        // and GPU queue buffers can come down.
        devx.reset();
        if (doorbell_cell != nullptr) {
            cudaFreeHost(doorbell_cell);
        }
        if (completion_mr != nullptr) {
            aboleth::dl::verbs().ibv_dereg_mr(completion_mr);
        }
        if (completion_consumer != nullptr) {
            cudaFree(completion_consumer);
        }
        if (producer_index != nullptr) {
            cudaFree(producer_index);
        }
        for (auto mapping = mappings.rbegin();
             mapping != mappings.rend();
             ++mapping) {
            runtime->mappings.unmap(
                mapping->pointer,
                mapping->bytes,
                mapping->flags);
        }
        if (sender != nullptr) {
            aboleth::dl::verbs().ibv_destroy_qp(sender);
        }
        if (send_cq != nullptr) {
            aboleth::dl::verbs().ibv_destroy_cq(send_cq);
        }
        if (parent_domain != nullptr) {
            aboleth::dl::verbs().ibv_dealloc_pd(parent_domain);
        }
        if (thread_domain != nullptr) {
            ibv_dealloc_td(thread_domain);
        }
    }
};

// Creates the endpoint and fills `descriptor` with what the kernel reads. A
// dedicated thread domain gives the queue pair its own lock domain, which is
// what a per-thread endpoint wants and what a shared one must not have. A
// default-constructed `request` is the runtime's own sizing, which is what
// every caller that has no per-endpoint config to honor passes.
std::unique_ptr<OwnedEndpointContext> create_owned_endpoint(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    aboleth::abi::RdmaEndpointDescriptor& descriptor,
    bool dedicated_thread_domain = false,
    const OwnedEndpointRequest& request = {});

// The GPU-resident staged queue bring-up (aboleth_endpoint_devx.cu).
// Builds the DEVX CQ/QP over cudaMalloc queues and fills the queue-related
// descriptor and endpoint fields (send/completion queues, dbrecs, qpn, the
// staged doorbell cell and agent). Returns false with `reason` set when the
// platform cannot do it (no DEVX exports, no dmabuf export, RoCE link,
// firmware refusal) — the caller then falls back to host-pinned queues.
bool create_devx_staged_queues(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    aboleth::abi::RdmaEndpointDescriptor& descriptor,
    OwnedEndpointContext& endpoint,
    const OwnedEndpointRequest& request,
    std::string& reason);

// Whether staged endpoints should place their queues in GPU memory:
// ABOLETH_RDMA_STAGED_QUEUES=gpu (default; probe with fallback) or =host.
bool devx_staged_queues_requested();

// DEVX state transitions for an endpoint without an ibv_qp. The RST->INIT
// step runs at creation; these mirror qp_to_rtr/qp_to_rts field for field.
void devx_qp_to_rtr(
    DevxQueueState& state,
    ibv_context* context,
    std::uint32_t remote_qpn,
    std::uint32_t remote_psn,
    std::uint16_t remote_lid,
    ibv_mtu path_mtu);
// ABOLETH_QPC_DUMP: print the queue pair's firmware context (QUERY_QP),
// for comparing a verbs QP with a DEVX one. At most the first few per
// process.
void dump_queue_pair_context(
    DevxQueueState* devx, ibv_qp* verbs_qp, int gpu_index);

void devx_qp_to_rts(
    DevxQueueState& state,
    std::uint32_t psn);

} // namespace aboleth::runtime
