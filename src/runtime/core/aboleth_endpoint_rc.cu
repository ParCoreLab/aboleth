// Building the RC endpoint a generated kernel drives directly.
//
// One send queue pair, whatever inline capacity the adapter grants, and the
// mlx5 objects behind it mapped into the GPU's address space: the send queue,
// its doorbell record, the blueflame page, the completion queue and the
// completion cell the kernel spins on. What comes out is the
// RdmaEndpointDescriptor compiled code reads. Connecting the endpoint is
// core/aboleth_connection.cu's job.
//
// The helpers aboleth_runtime_internal.h declares for every module are defined
// here too - the two provider status checks, the environment integer, and the
// acquire/release pair the host reaches ABI-described cells through. They are
// the layer this file's own bodies sit directly on, and exactly one
// translation unit has to hold them.
//
// The types are declared in aboleth_endpoint_rc.h.

#include "aboleth_endpoint_rc.h"

#include "../aboleth_runtime_common.h"
#include "../aboleth_runtime_internal.h"
#include "aboleth_connection.h"
#include "aboleth_memory.h"

#include "aboleth/Target/RdmaAbi.h"

#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include "../dl/shims.h"

namespace dl = aboleth::dl;

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>

namespace aboleth::runtime {

void verbs_check(int status, const char* operation) {
    if (status != 0) {
        throw_error(
            std::string(operation) + ": " +
            std::strerror(errno));
    }
}

#if ABOLETH_HAS_EFA
void fabric_check(ssize_t status, const char* operation) {
    if (status < 0) {
        throw_error(
            std::string(operation) + ": " +
            dl::fabric().fi_strerror(static_cast<int>(-status)));
    }
}
#endif

int configured_index(
    const char* name,
    int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' ||
        parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        throw_error(
            std::string(name) +
            " must be a non-negative integer");
    }
    return static_cast<int>(parsed);
}

std::uint64_t atomic_load_acquire(
    const std::uint64_t* pointer) {
    return __atomic_load_n(
        pointer,
        __ATOMIC_ACQUIRE);
}

void atomic_store_release(
    std::uint64_t* pointer,
    std::uint64_t value) {
    __atomic_store_n(
        pointer,
        value,
        __ATOMIC_RELEASE);
}

namespace {

// ABOLETH_RDMA_SUBMIT selects who performs the UAR doorbell store: "direct"
// insists on the GPU's IO mapping and propagates its failure, "staged" skips
// the attempt and hands the store to a doorbell agent. Unset lets the first
// endpoint probe — the mapping attempt IS the capability check, there is no
// sysfs shortcut — and the process remembers a refusal so later endpoints
// do not repeat the failing call. See docs/RDMA_ABI.md, Submission modes.
enum class RdmaSubmitMode { Probe, Direct, Staged };

RdmaSubmitMode configured_rdma_submit() {
    const char* configured =
        std::getenv("ABOLETH_RDMA_SUBMIT");
    const std::string submit =
        configured == nullptr ? "" : configured;
    if (submit.empty()) {
        return RdmaSubmitMode::Probe;
    }
    if (submit == "direct") {
        return RdmaSubmitMode::Direct;
    }
    if (submit == "staged") {
        return RdmaSubmitMode::Staged;
    }
    throw_error(
        "ABOLETH_RDMA_SUBMIT must be direct or staged");
}

// The probe verdict is a property of the process's privileges and driver
// configuration, not of any one endpoint.
std::atomic<bool> uar_mapping_refused{false};
std::atomic<bool> uar_probe_completed{false};

void note_staged_fallback_once(
    const std::string& reason) {
    static std::atomic<bool> noted{false};
    if (noted.exchange(true)) {
        return;
    }
    std::fprintf(
        stderr,
        "aboleth: GPU mapping of the NIC doorbell register was refused "
        "(%s); falling back to staged submission — kernels build and "
        "publish work exactly as before and a host doorbell agent "
        "performs the one MMIO store this configuration withholds from "
        "the GPU. Set ABOLETH_RDMA_SUBMIT=direct to make this an error, "
        "or =staged to skip the probe. See docs/RDMA_ABI.md.\n",
        reason.c_str());
    std::fflush(stderr);
}

void note_devx_fallback_once(
    const std::string& reason) {
    static std::atomic<bool> noted{false};
    if (noted.exchange(true)) {
        return;
    }
    std::fprintf(
        stderr,
        "aboleth: GPU-resident staged queues are unavailable (%s); "
        "staged endpoints keep their queues in host-pinned memory. Set "
        "ABOLETH_RDMA_STAGED_QUEUES=host to silence the attempt. See "
        "docs/RDMA_ABI.md.\n",
        reason.c_str());
    std::fflush(stderr);
}

} // namespace

ibv_qp* create_rc_qp(
    ibv_pd* pd,
    ibv_cq* cq,
    std::uint32_t maximum_inline_bytes,
    std::uint32_t* granted_inline_bytes,
    std::uint32_t send_wqe_requests) {
    ibv_qp_init_attr attributes{};
    attributes.send_cq = cq;
    attributes.recv_cq = cq;
    attributes.cap.max_send_wr =
        send_wqe_requests != 0
            ? send_wqe_requests
            : kMaximumTransfers + 1;
    attributes.cap.max_recv_wr = 1;
    attributes.cap.max_send_sge = 1;
    attributes.cap.max_recv_sge = 1;
    attributes.cap.max_inline_data =
        maximum_inline_bytes;
    attributes.qp_type = IBV_QPT_RC;
    ibv_qp* qp = dl::verbs().ibv_create_qp(pd, &attributes);
    if (qp == nullptr) {
        throw_error(
            std::string("ibv_create_qp: ") +
            std::strerror(errno));
    }
    // ibv_create_qp rewrites cap with what the adapter granted, which is the
    // only place the effective inline limit is reported.
    if (granted_inline_bytes != nullptr) {
        *granted_inline_bytes =
            attributes.cap.max_inline_data;
    }
    return qp;
}

// The adapter may refuse a large max_inline_data request outright; the
// no-inline queue pair is the floor every compiled kernel already assumes.
ibv_qp* create_rc_qp_optional_inline(
    ibv_pd* pd,
    ibv_cq* cq,
    std::uint32_t maximum_inline_bytes,
    std::uint32_t* granted_inline_bytes,
    std::uint32_t send_wqe_requests) {
    if (maximum_inline_bytes != 0) {
        try {
            return create_rc_qp(
                pd,
                cq,
                maximum_inline_bytes,
                granted_inline_bytes,
                send_wqe_requests);
        } catch (const std::exception&) {
        }
    }
    return create_rc_qp(
        pd,
        cq,
        0,
        granted_inline_bytes,
        send_wqe_requests);
}

namespace {

// Settles the UAR-probe verdict once per process without leaving a
// half-built endpoint behind: a throwaway CQ+QP, one mapping attempt, torn
// down again. Needed when GPU-resident staged queues are on the table — the
// DEVX path must be chosen BEFORE any libmlx5 objects exist, but in probe
// mode "staged" is only known after a mapping attempt.
void settle_uar_probe(
    const std::shared_ptr<SharedRuntimeContext>& runtime) {
    if (uar_probe_completed.load(std::memory_order_acquire)) {
        return;
    }
    ibv_cq* probe_cq =
        dl::verbs().ibv_create_cq(
            runtime->verbs_context, 1, nullptr, nullptr, 0);
    if (probe_cq == nullptr) {
        throw_error("ibv_create_cq(uar probe) failed");
    }
    ibv_qp* probe_qp = nullptr;
    try {
        probe_qp = create_rc_qp(runtime->pd, probe_cq, 0, nullptr, 1);
        mlx5dv_qp direct_qp{};
        mlx5dv_obj direct_object{};
        direct_object.qp.in = probe_qp;
        direct_object.qp.out = &direct_qp;
        verbs_check(
            dl::mlx5dv().mlx5dv_init_obj(&direct_object, MLX5DV_OBJ_QP),
            "mlx5dv_init_obj(uar probe)");
        try {
            void* mapped = runtime->mappings.map(
                direct_qp.bf.reg,
                sizeof(std::uint64_t),
                cudaHostRegisterMapped | cudaHostRegisterIoMemory);
            (void)mapped;
            runtime->mappings.unmap(
                direct_qp.bf.reg,
                sizeof(std::uint64_t),
                cudaHostRegisterMapped | cudaHostRegisterIoMemory);
        } catch (const std::exception& error) {
            cudaGetLastError();
            uar_mapping_refused.store(true, std::memory_order_relaxed);
            note_staged_fallback_once(error.what());
        }
    } catch (...) {
        if (probe_qp != nullptr) {
            dl::verbs().ibv_destroy_qp(probe_qp);
        }
        dl::verbs().ibv_destroy_cq(probe_cq);
        throw;
    }
    dl::verbs().ibv_destroy_qp(probe_qp);
    dl::verbs().ibv_destroy_cq(probe_cq);
    uar_probe_completed.store(true, std::memory_order_release);
}

// The endpoint state every queue layout shares: the producer index, the
// completion state block and its MR, and the descriptor words derived from
// them.
void finish_endpoint_common(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    aboleth::abi::RdmaEndpointDescriptor& descriptor,
    OwnedEndpointContext& endpoint) {
    cuda_check(
        cudaMalloc(
            &endpoint.producer_index, sizeof(std::uint64_t)),
        "cudaMalloc(producer_index)");
    cuda_check(
        cudaMalloc(
            &endpoint.completion_consumer, kCompletionAllocationBytes),
        "cudaMalloc(completion_consumer)");
    cuda_check(
        cudaMemset(
            endpoint.producer_index, 0, sizeof(std::uint64_t)),
        "cudaMemset(producer_index)");
    cuda_check(
        cudaMemset(
            endpoint.completion_consumer, 0, kCompletionAllocationBytes),
        "cudaMemset(completion_consumer)");
    endpoint.completion_mr =
        register_memory(
            runtime->pd,
            endpoint.completion_consumer,
            kCompletionAllocationBytes,
            IBV_ACCESS_LOCAL_WRITE,
            /*allow_relaxed=*/false);
    if (endpoint.completion_mr == nullptr) {
        throw_error("ibv_reg_mr(completion state) failed");
    }
    descriptor.producer_index_address =
        reinterpret_cast<std::uint64_t>(endpoint.producer_index);
    descriptor.completion_consumer_address =
        reinterpret_cast<std::uint64_t>(endpoint.completion_consumer);
    descriptor.completion_timeout_cycles =
        cuda_clock_rate_khz(gpu_index) * 1000ULL * kTimeoutSeconds;
    descriptor.abi_version = aboleth::abi::kRdmaEndpointVersion;
    descriptor.atomic_result_lkey = endpoint.completion_mr->lkey;
    descriptor.reserved1 = 0;
}

}  // namespace

std::unique_ptr<OwnedEndpointContext>
create_owned_endpoint(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    aboleth::abi::RdmaEndpointDescriptor& descriptor,
    bool dedicated_thread_domain,
    const OwnedEndpointRequest& request) {
    // A depth the adapter cannot give is worth naming here: ibv_create_qp
    // would answer ENOMEM, which says nothing about which request was too
    // large.
    if (request.send_wqe_requests != 0) {
        ibv_device_attr device_attributes{};
        verbs_check(
            dl::verbs().ibv_query_device(
                runtime->verbs_context,
                &device_attributes),
            "ibv_query_device(endpoint queue depth)");
        if (request.send_wqe_requests >
            static_cast<std::uint32_t>(
                device_attributes.max_qp_wr)) {
            throw_error(
                "AbolethEndpointConfig.qp_depth=" +
                std::to_string(
                    request.send_wqe_requests) +
                " exceeds the adapter's maximum send-queue depth of " +
                std::to_string(
                    device_attributes.max_qp_wr) +
                " (see docs/transport-control.md)");
        }
    }
    auto endpoint =
        std::make_unique<OwnedEndpointContext>();
    endpoint->runtime = runtime;
    endpoint->gpu_index = gpu_index;
    cuda_check(
        cudaSetDevice(gpu_index),
        "cudaSetDevice");

    // GPU-resident staged queues: when submission is (or probes to) staged,
    // build the endpoint through DEVX with every queue in GPU memory and no
    // ibv_qp at all. The probe must settle first — the DEVX path has to be
    // chosen before any libmlx5 queue objects exist.
    {
        const RdmaSubmitMode early_mode = configured_rdma_submit();
        if (early_mode != RdmaSubmitMode::Direct
            && devx_staged_queues_requested()) {
            if (early_mode == RdmaSubmitMode::Probe) {
                settle_uar_probe(runtime);
            }
            const bool early_staged =
                early_mode == RdmaSubmitMode::Staged ||
                uar_mapping_refused.load(std::memory_order_relaxed);
            if (early_staged) {
                std::string reason;
                if (create_devx_staged_queues(
                        runtime, gpu_index, descriptor, *endpoint,
                        request, reason)) {
                    endpoint->psn = allocate_psn();
                    finish_endpoint_common(
                        runtime, gpu_index, descriptor, *endpoint);
                    return endpoint;
                }
                if (std::getenv("ABOLETH_DEVX_REQUIRE") != nullptr) {
                    throw_error("DEVX staged queues required: " + reason);
                }
                note_devx_fallback_once(reason);
                // A failed attempt may have left partial DEVX state; drop
                // it before the classic path builds its own objects.
                endpoint->devx.reset();
            }
        }
    }

    ibv_pd* qp_domain = runtime->pd;
    if (dedicated_thread_domain) {
        ibv_td_init_attr td_attributes{};
        endpoint->thread_domain =
            ibv_alloc_td(
                runtime->verbs_context,
                &td_attributes);
        if (endpoint->thread_domain == nullptr) {
            throw_error("ibv_alloc_td failed");
        }
        ibv_parent_domain_init_attr
            parent_attributes{};
        parent_attributes.pd = runtime->pd;
        parent_attributes.td =
            endpoint->thread_domain;
        endpoint->parent_domain =
            ibv_alloc_parent_domain(
                runtime->verbs_context,
                &parent_attributes);
        if (endpoint->parent_domain == nullptr) {
            throw_error(
                "ibv_alloc_parent_domain failed");
        }
        qp_domain = endpoint->parent_domain;
    }

    // A deeper send queue needs a completion queue that can hold what it
    // retires, so the requested depth carries into both.
    endpoint->send_cq =
        dl::verbs().ibv_create_cq(
            runtime->verbs_context,
            static_cast<int>(
                std::max(
                    kMaximumTransfers,
                    request.send_wqe_requests)),
            nullptr,
            nullptr,
            0);
    if (endpoint->send_cq == nullptr) {
        throw_error("ibv_create_cq failed");
    }

    // The config's inline request wins over ABOLETH_RDMA_INLINE_BYTES, which
    // wins over the compiled-in floor.
    endpoint->sender =
        create_rc_qp_optional_inline(
            qp_domain,
            endpoint->send_cq,
            request.max_inline_bytes != 0
                ? request.max_inline_bytes
                : static_cast<std::uint32_t>(
                      configured_index(
                          "ABOLETH_RDMA_INLINE_BYTES",
                          kRdmaInlineBytes)),
            &endpoint->max_inline_bytes,
            request.send_wqe_requests);
    qp_to_init(
        endpoint->sender,
        runtime->port);
    endpoint->psn = allocate_psn();

    mlx5dv_qp direct_qp{};
    mlx5dv_cq direct_cq{};
    mlx5dv_obj direct_object{};
    direct_object.qp.in = endpoint->sender;
    direct_object.qp.out = &direct_qp;
    direct_object.cq.in = endpoint->send_cq;
    direct_object.cq.out = &direct_cq;
    verbs_check(
        dl::mlx5dv().mlx5dv_init_obj(
            &direct_object,
            MLX5DV_OBJ_QP |
                MLX5DV_OBJ_CQ),
        "mlx5dv_init_obj");
    if (direct_qp.sq.wqe_cnt == 0 ||
        (direct_qp.sq.wqe_cnt &
         (direct_qp.sq.wqe_cnt - 1)) != 0) {
        throw_error(
            "mlx5 send queue length is not a power of two");
    }

    const auto send_queue_bytes =
        static_cast<std::size_t>(
            direct_qp.sq.wqe_cnt) *
        direct_qp.sq.stride;
    const auto completion_queue_bytes =
        static_cast<std::size_t>(
            direct_cq.cqe_cnt) *
        direct_cq.cqe_size;
    auto map_resource =
        [&](void* pointer,
            std::size_t bytes,
            unsigned int flags) {
            void* mapped =
                runtime->mappings.map(
                    pointer,
                    bytes,
                    flags);
            endpoint->mappings.push_back(
                {pointer, bytes, flags});
            return mapped;
        };
    void* send_queue =
        map_resource(
            direct_qp.sq.buf,
            send_queue_bytes,
            cudaHostRegisterMapped);
    auto* send_dbrec =
        static_cast<std::uint32_t*>(
            map_resource(
                direct_qp.dbrec,
                2 * sizeof(std::uint32_t),
                cudaHostRegisterMapped)) +
        MLX5_SND_DBR;
    // Who stores to the UAR decides the mode. Direct maps the register into
    // the GPU's address space; staged points the descriptor's blueflame word
    // at a host-pinned cell instead, and a doorbell agent forwards what the
    // kernel release-publishes there. Device code is identical either way.
    const RdmaSubmitMode submit_mode =
        configured_rdma_submit();
    bool staged =
        submit_mode == RdmaSubmitMode::Staged ||
        (submit_mode == RdmaSubmitMode::Probe &&
         uar_mapping_refused.load(
             std::memory_order_relaxed));
    void* blueflame = nullptr;
    if (!staged) {
        if (submit_mode == RdmaSubmitMode::Probe) {
            try {
                blueflame = map_resource(
                    direct_qp.bf.reg,
                    sizeof(std::uint64_t),
                    cudaHostRegisterMapped |
                        cudaHostRegisterIoMemory);
            } catch (const std::exception& error) {
                cudaGetLastError();
                uar_mapping_refused.store(
                    true, std::memory_order_relaxed);
                note_staged_fallback_once(error.what());
                staged = true;
            }
        } else {
            blueflame = map_resource(
                direct_qp.bf.reg,
                sizeof(std::uint64_t),
                cudaHostRegisterMapped |
                    cudaHostRegisterIoMemory);
        }
    }
    if (staged) {
        // The staged head is 16 bits on the wire; a queue this deep could
        // advance a full head period between two agent observations, which
        // level-triggered forwarding cannot distinguish from no progress.
        if (direct_qp.sq.wqe_cnt >= 65536) {
            throw_error(
                "staged submission caps qp_depth below 65536 entries");
        }
        endpoint->staged_submit = true;
        cuda_check(
            cudaHostAlloc(
                &endpoint->doorbell_cell,
                sizeof(std::uint64_t),
                cudaHostAllocPortable |
                    cudaHostAllocMapped),
            "cudaHostAlloc(doorbell cell)");
        *endpoint->doorbell_cell = 0;
        void* cell_device = nullptr;
        cuda_check(
            cudaHostGetDevicePointer(
                &cell_device,
                endpoint->doorbell_cell,
                0),
            "cudaHostGetDevicePointer(doorbell cell)");
        blueflame = cell_device;
    }
    void* completion_queue =
        map_resource(
            direct_cq.buf,
            completion_queue_bytes,
            cudaHostRegisterMapped);
    void* completion_dbrec =
        map_resource(
            direct_cq.dbrec,
            2 * sizeof(std::uint32_t),
            cudaHostRegisterMapped);

    finish_endpoint_common(
        runtime, gpu_index, descriptor, *endpoint);

    descriptor.send_queue =
        reinterpret_cast<std::uint64_t>(
            send_queue);
    descriptor.send_dbrec =
        reinterpret_cast<std::uint64_t>(
            send_dbrec);
    descriptor.blueflame =
        reinterpret_cast<std::uint64_t>(
            blueflame);
    descriptor.completion_queue =
        reinterpret_cast<std::uint64_t>(
            completion_queue);
    descriptor.completion_dbrec =
        reinterpret_cast<std::uint64_t>(
            completion_dbrec);
    descriptor.send_stride =
        direct_qp.sq.stride;
    descriptor.send_wqe_count =
        direct_qp.sq.wqe_cnt;
    endpoint->send_queue_host =
        direct_qp.sq.buf;
    endpoint->send_dbrec_host =
        direct_qp.dbrec + MLX5_SND_DBR;
    endpoint->send_stride =
        direct_qp.sq.stride;
    endpoint->send_wqe_count =
        direct_qp.sq.wqe_cnt;
    descriptor.qpn =
        endpoint->sender->qp_num;
    endpoint->queue_pair_number =
        endpoint->sender->qp_num;
    descriptor.completion_entry_size =
        direct_cq.cqe_size;
    descriptor.completion_entry_count =
        direct_cq.cqe_cnt;
    if (endpoint->staged_submit) {
        // Only the low half of the published doorbell word ever changes —
        // the high half is the endpoint's byte-swapped QPN — so the agent
        // watches the cell's first 32 bits and the hook rebuilds the word.
        endpoint->agent = acquire_doorbell_agent(
            runtime.get(),
            gpu_index,
            pin_numa_local);
        endpoint->agent_token = endpoint->agent->add(
            reinterpret_cast<const volatile std::uint32_t*>(
                endpoint->doorbell_cell),
            make_mlx5_uar_submit(
                reinterpret_cast<volatile std::uint64_t*>(
                    direct_qp.bf.reg),
                endpoint->sender->qp_num));
    }
    return endpoint;
}

}  // namespace aboleth::runtime
