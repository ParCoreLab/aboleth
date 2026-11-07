#include "aboleth_efa_runtime.h"

#include "../core/aboleth_doorbell_agent.h"
#include "../aboleth_runtime_common.h"
#include "aboleth/Target/EfaAbi.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include "../dl/shims.h"

namespace dl = aboleth::dl;

#include <cerrno>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include "efa_cuda_dp.h"
#include "efa_cuda_dp_types.h"
}

// efa_io_defs.h spells its field masks with the kernel's BIT/GENMASK macros
// but does not define them; the device library gets them from
// efa_cuda_dp_impl.cuh, which is device-only code.
#define BIT(nr) (1UL << (nr))
#define BITS_PER_LONG (8 * sizeof(long))
#define GENMASK(h, l) \
    (((~0UL) - (1UL << (l)) + 1) & (~0UL >> (BITS_PER_LONG - 1 - (h))))
#include "efa_io_defs.h"

namespace {

using namespace aboleth::runtime;
namespace abi = aboleth::abi;

// Both peers must agree; SRD carries no connection state to negotiate it.
constexpr std::uint32_t kQueueKey = 0x11111111u;
// The device library places send descriptors on a fixed 64-byte stride
// (sizeof(struct efa_io_tx_wqe)); a queue pair whose slots are wider is
// unusable through it.
constexpr std::uint32_t kSendSlotBytes = 64;
constexpr std::uint64_t kTimeoutSeconds = 5;
constexpr std::uint8_t kPortNumber = 1;
constexpr std::uint32_t kGidIndex = 0;

thread_local std::string last_error_message;

void verbs_check(int status, const char* operation) {
    if (status != 0) {
        throw_error(
            std::string(operation) + ": " +
            std::strerror(errno));
    }
}

// GPU-initiated posting needs the platform to route GPU MMIO stores into the
// adapter's BAR. EFA generations 1 and 2 (1d0f:efa0 / 1d0f:efa1 — the p4 and
// p5 families) drop those stores in the Nitro fabric: descriptors place and
// doorbells ring from the GPU's point of view, but the adapter never fetches
// anything and no error surfaces anywhere. Warn rather than fail — the
// control plane, registration, and the host-posted diagnostic path all still
// work — but a GPU-posted transfer on such a device can only time out.
// Submit-mode resolution (ABOLETH_EFA_SUBMIT, device generation) lives in
// aboleth_runtime_common.h so both EFA control planes share it.

std::uint32_t round_up_power_of_two(std::uint32_t value) {
    std::uint32_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

// Queue memory and doorbells come back from efadv as host mappings; the kernel
// reaches them through the CUDA address space.
class MappingRegistry {
public:
    void* map(
        void* pointer,
        std::size_t bytes,
        unsigned int flags) {
        std::lock_guard<std::mutex> lock(mutex_);
        flags |= cudaHostRegisterPortable;
        const auto page_size =
            static_cast<std::uintptr_t>(
                sysconf(_SC_PAGESIZE));
        const auto start =
            reinterpret_cast<std::uintptr_t>(pointer);
        const auto base = start & ~(page_size - 1);
        const auto end =
            (start + bytes + page_size - 1) &
            ~(page_size - 1);

        for (auto& mapping : mappings_) {
            const auto mapping_base =
                reinterpret_cast<std::uintptr_t>(
                    mapping.host_base);
            const auto mapping_end =
                mapping_base + mapping.bytes;
            if (base >= mapping_end ||
                end <= mapping_base) {
                continue;
            }
            if (flags != mapping.flags ||
                base < mapping_base ||
                end > mapping_end) {
                throw_error(
                    "incompatible overlapping CUDA host mappings");
            }
            ++mapping.references;
            return reinterpret_cast<void*>(
                reinterpret_cast<std::uintptr_t>(
                    mapping.device_base) +
                start - mapping_base);
        }

        Mapping mapping{
            reinterpret_cast<void*>(base),
            static_cast<std::size_t>(end - base),
            nullptr,
            flags,
            1,
        };
        cuda_check(
            cudaHostRegister(
                mapping.host_base,
                mapping.bytes,
                flags),
            "cudaHostRegister(EFA queue)");
        cuda_check(
            cudaHostGetDevicePointer(
                &mapping.device_base,
                mapping.host_base,
                0),
            "cudaHostGetDevicePointer(EFA queue)");
        mappings_.push_back(mapping);
        return reinterpret_cast<void*>(
            reinterpret_cast<std::uintptr_t>(
                mapping.device_base) +
            start - base);
    }

    void clear() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto mapping = mappings_.rbegin();
             mapping != mappings_.rend();
             ++mapping) {
            cudaHostUnregister(mapping->host_base);
        }
        mappings_.clear();
    }

    ~MappingRegistry() { clear(); }

private:
    struct Mapping {
        void* host_base;
        std::size_t bytes;
        void* device_base;
        unsigned int flags;
        int references;
    };
    std::mutex mutex_;
    std::vector<Mapping> mappings_;
};

} // namespace

struct EfaRuntimeHandle {
    int gpu_index{-1};
    std::string device_name;
    ibv_context* context{};
    ibv_pd* protection_domain{};
    efadv_device_attr device_attributes{};
    union ibv_gid gid {};
    MappingRegistry mappings;
    // Staged submission: the GPU builds descriptors into host-pinned staging
    // rings and this agent moves them into the adapter's queue.
    bool staged_submit{};
    std::unique_ptr<aboleth::runtime::DoorbellAgent> agent;

    ~EfaRuntimeHandle() {
        agent.reset();
        mappings.clear();
        if (protection_domain != nullptr) {
            dl::verbs().ibv_dealloc_pd(protection_domain);
        }
        if (context != nullptr) {
            dl::verbs().ibv_close_device(context);
        }
    }
};

struct EfaBufferHandle {
    ibv_mr* memory_region{};

    ~EfaBufferHandle() {
        if (memory_region != nullptr) {
            dl::verbs().ibv_dereg_mr(memory_region);
        }
    }
};

// A completion queue backed by GPU memory. EFA can DMA completions straight
// into a dmabuf-exported device allocation, which is what lets the kernel poll
// them without any host mapping.
struct EfaCompletionQueue {
    ibv_cq* queue{};
    void* memory{};

    ~EfaCompletionQueue() {
        if (queue != nullptr) {
            dl::verbs().ibv_destroy_cq(queue);
        }
        if (memory != nullptr) {
            cudaFree(memory);
        }
    }
};

struct EfaEndpointHandle {
    EfaRuntimeHandle* runtime{};
    ibv_qp* queue_pair{};
    EfaCompletionQueue send_cq;
    EfaCompletionQueue recv_cq;
    efa_cuda_qp* device_queue_pair{};
    efa_cuda_cq* device_send_cq{};
    efa_cuda_cq* device_recv_cq{};
    std::uint64_t* completion_state{};
    ibv_ah* address_handle{};
    std::uint32_t send_depth{};
    std::uint32_t recv_depth{};
    abi::EfaEndpointDescriptor descriptor{};
    void* descriptor_device{};
    // Host mapping of the send queue, kept for the diagnostic host-posted
    // datapath on platforms that drop GPU stores into the adapter BAR.
    std::uint8_t* host_send_buffer{};
    std::uint32_t* host_send_doorbell{};
    std::uint32_t host_send_entry_bytes{};
    std::uint32_t host_producer{};
    // Staged submission: the GPU posts into this host-pinned ring and
    // advances the producer cell; the runtime's doorbell agent copies the
    // new slots into the adapter's queue and rings the real doorbell.
    std::uint8_t* staging_ring{};
    std::uint32_t* staging_cells{};
    std::uint64_t agent_token{};

    ~EfaEndpointHandle() {
        // The agent must stop touching the staging ring and the adapter
        // queue before either goes away.
        if (agent_token != 0 && runtime != nullptr &&
            runtime->agent != nullptr) {
            runtime->agent->remove(agent_token);
        }
        if (staging_ring != nullptr) {
            cudaFreeHost(staging_ring);
        }
        if (staging_cells != nullptr) {
            cudaFreeHost(staging_cells);
        }
        if (address_handle != nullptr) {
            dl::verbs().ibv_destroy_ah(address_handle);
        }
        if (device_queue_pair != nullptr) {
            cudaFree(device_queue_pair);
        }
        if (device_send_cq != nullptr) {
            efa_cuda_destroy_cq(device_send_cq);
        }
        if (device_recv_cq != nullptr) {
            efa_cuda_destroy_cq(device_recv_cq);
        }
        if (completion_state != nullptr) {
            cudaFree(completion_state);
        }
        if (queue_pair != nullptr) {
            dl::verbs().ibv_destroy_qp(queue_pair);
        }
    }
};

namespace aboleth::efa {

namespace {

// SRD is connectionless: the transitions carry no peer, only the queue key the
// sender will quote on every work request.
void queue_pair_to_ready(ibv_qp* queue_pair) {
    ibv_qp_attr attributes{};
    attributes.qp_state = IBV_QPS_INIT;
    attributes.pkey_index = 0;
    attributes.port_num = kPortNumber;
    attributes.qkey = kQueueKey;
    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX |
        IBV_QP_PORT | IBV_QP_QKEY;
    if (dl::verbs().ibv_modify_qp(queue_pair, &attributes, mask) != 0) {
        // Some providers reject a pkey index on a driver queue pair.
        attributes = {};
        attributes.qp_state = IBV_QPS_INIT;
        attributes.port_num = kPortNumber;
        attributes.qkey = kQueueKey;
        mask = IBV_QP_STATE | IBV_QP_PORT | IBV_QP_QKEY;
        verbs_check(
            dl::verbs().ibv_modify_qp(queue_pair, &attributes, mask),
            "ibv_modify_qp(EFA INIT)");
    }

    attributes = {};
    attributes.qp_state = IBV_QPS_RTR;
    verbs_check(
        dl::verbs().ibv_modify_qp(queue_pair, &attributes, IBV_QP_STATE),
        "ibv_modify_qp(EFA RTR)");

    attributes = {};
    attributes.qp_state = IBV_QPS_RTS;
    attributes.sq_psn = 0;
    verbs_check(
        dl::verbs().ibv_modify_qp(
            queue_pair,
            &attributes,
            IBV_QP_STATE | IBV_QP_SQ_PSN),
        "ibv_modify_qp(EFA RTS)");
}

// The completion queue is created over device memory the driver reaches
// through dmabuf, so the kernel polls completions in its own address space.
void create_completion_queue(
    EfaRuntimeHandle* runtime,
    EfaCompletionQueue& completion,
    std::uint32_t entries) {
    // dmabuf export works on whole allocation granules, so a queue smaller
    // than one granule is rounded up rather than sub-allocated.
    constexpr std::size_t kDmabufGranule = 2u << 20;
    const std::size_t requested =
        static_cast<std::size_t>(entries) * 64;
    const std::size_t bytes =
        (requested + kDmabufGranule - 1) &
        ~(kDmabufGranule - 1);
    cuda_check(
        cudaMalloc(&completion.memory, bytes),
        "cudaMalloc(EFA completion queue)");
    cuda_check(
        cudaMemset(completion.memory, 0, bytes),
        "cudaMemset(EFA completion queue)");

    int descriptor = -1;
    if (dl::cuda().cuMemGetHandleForAddressRange(
            &descriptor,
            reinterpret_cast<CUdeviceptr>(completion.memory),
            bytes,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            0) != CUDA_SUCCESS) {
        throw_error(
            "could not export the completion queue as a dmabuf");
    }

    ibv_cq_init_attr_ex attributes{};
    attributes.cqe = entries;
    attributes.wc_flags = IBV_WC_EX_WITH_BYTE_LEN;
    efadv_cq_init_attr efa_attributes{};
    efa_attributes.flags =
        EFADV_CQ_INIT_FLAGS_EXT_MEM_DMABUF;
    efa_attributes.ext_mem_dmabuf.buffer =
        static_cast<std::uint8_t*>(completion.memory);
    efa_attributes.ext_mem_dmabuf.length = bytes;
    efa_attributes.ext_mem_dmabuf.offset = 0;
    efa_attributes.ext_mem_dmabuf.fd = descriptor;
    ibv_cq_ex* extended = dl::efadv().efadv_create_cq(
        runtime->context,
        &attributes,
        &efa_attributes,
        sizeof(efa_attributes));
    close(descriptor);
    if (extended == nullptr) {
        throw_error(
            std::string("efadv_create_cq: ") +
            std::strerror(errno));
    }
    completion.queue = ibv_cq_ex_to_cq(extended);
}

void publish_descriptor(EfaEndpointHandle* endpoint) {
    cuda_check(
        cudaMemcpy(
            endpoint->descriptor_device,
            &endpoint->descriptor,
            sizeof(endpoint->descriptor),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(EFA descriptor)");
}

} // namespace

EfaRuntimeHandle* runtime_create(
    int gpu_index,
    const char* device_name) {
    last_error_message.clear();
    try {
        cuda_check(
            cudaSetDevice(gpu_index),
            "cudaSetDevice(EFA runtime)");
        auto runtime =
            std::make_unique<EfaRuntimeHandle>();
        runtime->gpu_index = gpu_index;
        runtime->device_name =
            select_efa_device_name(gpu_index, device_name);

        int device_count = 0;
        ibv_device** devices =
            dl::verbs().ibv_get_device_list(&device_count);
        if (devices == nullptr) {
            throw_error("ibv_get_device_list found no devices");
        }
        ibv_device* selected = nullptr;
        for (int index = 0; index < device_count; ++index) {
            if (runtime->device_name ==
                dl::verbs().ibv_get_device_name(devices[index])) {
                selected = devices[index];
                break;
            }
        }
        if (selected == nullptr) {
            dl::verbs().ibv_free_device_list(devices);
            throw_error(
                "EFA device is not available through verbs: " +
                runtime->device_name);
        }
        runtime->context = dl::verbs().ibv_open_device(selected);
        dl::verbs().ibv_free_device_list(devices);
        if (runtime->context == nullptr) {
            throw_error(
                "ibv_open_device failed for " +
                runtime->device_name);
        }
        runtime->staged_submit =
            resolve_staged_submit(runtime->device_name);
        if (runtime->staged_submit) {
            runtime->agent = std::make_unique<
                aboleth::runtime::DoorbellAgent>(gpu_index);
        }

        if (dl::efadv().efadv_query_device(
                runtime->context,
                &runtime->device_attributes,
                sizeof(runtime->device_attributes)) != 0) {
            throw_error(
                runtime->device_name +
                " does not expose the EFA direct-verbs interface");
        }
        if ((runtime->device_attributes.device_caps &
             EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE) == 0) {
            throw_error(
                runtime->device_name +
                " does not support RDMA write");
        }

        runtime->protection_domain =
            dl::verbs().ibv_alloc_pd(runtime->context);
        if (runtime->protection_domain == nullptr) {
            throw_error("ibv_alloc_pd failed");
        }
        verbs_check(
            dl::verbs().ibv_query_gid(
                runtime->context,
                kPortNumber,
                kGidIndex,
                &runtime->gid),
            "ibv_query_gid(EFA)");
        return runtime.release();
    } catch (const std::exception& error) {
        last_error_message = error.what();
        return nullptr;
    }
}

void runtime_destroy(EfaRuntimeHandle* runtime) noexcept {
    delete runtime;
}

const char* runtime_device_name(EfaRuntimeHandle* runtime) {
    return runtime == nullptr
        ? ""
        : runtime->device_name.c_str();
}

bool runtime_staged_submit(EfaRuntimeHandle* runtime) {
    return runtime != nullptr && runtime->staged_submit;
}

EfaEndpointHandle* endpoint_create(
    EfaRuntimeHandle* runtime,
    std::uint32_t send_depth,
    std::uint32_t recv_depth,
    void* descriptor_device,
    std::size_t descriptor_bytes) {
    last_error_message.clear();
    try {
        if (runtime == nullptr ||
            descriptor_device == nullptr) {
            throw_error("null EFA endpoint argument");
        }
        if (descriptor_bytes <
            sizeof(abi::EfaEndpointDescriptor)) {
            throw_error(
                "EFA endpoint descriptor buffer is too small");
        }
        cuda_check(
            cudaSetDevice(runtime->gpu_index),
            "cudaSetDevice(EFA endpoint)");

        auto endpoint =
            std::make_unique<EfaEndpointHandle>();
        endpoint->runtime = runtime;
        endpoint->descriptor_device = descriptor_device;
        endpoint->send_depth =
            round_up_power_of_two(send_depth);
        endpoint->recv_depth =
            round_up_power_of_two(recv_depth);

        // Nothing rings a consumer doorbell on the completion queue, so it
        // must be able to hold every completion a full send queue produces.
        const std::uint32_t completion_depth =
            round_up_power_of_two(
                endpoint->send_depth * 2);
        create_completion_queue(
            runtime, endpoint->send_cq, completion_depth);
        create_completion_queue(
            runtime, endpoint->recv_cq, completion_depth);

        ibv_qp_init_attr_ex attributes{};
        attributes.pd = runtime->protection_domain;
        attributes.comp_mask = IBV_QP_INIT_ATTR_PD;
        attributes.qp_type = IBV_QPT_DRIVER;
        attributes.send_cq = endpoint->send_cq.queue;
        attributes.recv_cq = endpoint->recv_cq.queue;
        attributes.cap.max_send_wr = endpoint->send_depth;
        attributes.cap.max_recv_wr = endpoint->recv_depth;
        attributes.cap.max_send_sge = 1;
        attributes.cap.max_recv_sge = 1;
        // Requesting more inline data than fits a 64-byte descriptor — or the
        // inline-write capability — makes the provider size every send slot
        // at 128 bytes, and the device library hard-codes 64-byte slots. The
        // adapter silently ignores descriptors placed on the wrong stride, so
        // both are pinned here and the geometry is checked after creation.
        attributes.cap.max_inline_data =
            runtime->device_attributes.inline_buf_size < kSendSlotBytes / 2
                ? runtime->device_attributes.inline_buf_size
                : kSendSlotBytes / 2;
        attributes.sq_sig_all = 1;

        efadv_qp_init_attr efa_attributes{};
        efa_attributes.driver_qp_type =
            EFADV_QP_DRIVER_TYPE_SRD;
        endpoint->queue_pair = dl::efadv().efadv_create_qp_ex(
            runtime->context,
            &attributes,
            &efa_attributes,
            sizeof(efa_attributes));
        if (endpoint->queue_pair == nullptr) {
            throw_error(
                std::string("efadv_create_qp_ex: ") +
                std::strerror(errno));
        }
        queue_pair_to_ready(endpoint->queue_pair);

        efadv_wq_attr send_wq{};
        efadv_wq_attr recv_wq{};
        verbs_check(
            dl::efadv().efadv_query_qp_wqs(
                endpoint->queue_pair,
                &send_wq,
                &recv_wq,
                sizeof(send_wq)),
            "efadv_query_qp_wqs");
        if (send_wq.entry_size != kSendSlotBytes) {
            throw_error(
                "EFA send queue slots are " +
                std::to_string(send_wq.entry_size) +
                " bytes; the device library requires " +
                std::to_string(kSendSlotBytes));
        }
        endpoint->host_send_buffer =
            static_cast<std::uint8_t*>(send_wq.buffer);
        endpoint->host_send_doorbell = send_wq.doorbell;
        endpoint->host_send_entry_bytes = send_wq.entry_size;
        efadv_cq_attr send_cq_attr{};
        efadv_cq_attr recv_cq_attr{};
        verbs_check(
            dl::efadv().efadv_query_cq(
                endpoint->send_cq.queue,
                &send_cq_attr,
                sizeof(send_cq_attr)),
            "efadv_query_cq(send)");
        verbs_check(
            dl::efadv().efadv_query_cq(
                endpoint->recv_cq.queue,
                &recv_cq_attr,
                sizeof(recv_cq_attr)),
            "efadv_query_cq(recv)");

        // Every queue the provider hands back — work queues, doorbells, and
        // completion queues alike — lives on driver-mapped I/O pages rather
        // than ordinary host memory, so all of them register as I/O. That also
        // means the process needs the privilege to map device memory, the same
        // requirement the direct-mlx5 path carries.
        auto& mappings = runtime->mappings;
        const auto map_queue =
            [&](void* pointer, std::size_t bytes) {
                return mappings.map(
                    pointer,
                    bytes,
                    cudaHostRegisterMapped |
                        cudaHostRegisterIoMemory);
            };
        const auto map_doorbell =
            [&](void* pointer) {
                return map_queue(
                    pointer, sizeof(std::uint32_t));
            };

        efa_cuda_qp_attrs queue_attributes{};
        // Sender-fenced publication posts no receive work requests, so the
        // receive queue stays host-resident and unmapped. Immediate counting
        // would have to place it in device memory first.
        queue_attributes.rq_buffer = nullptr;
        if (runtime->staged_submit) {
            // Staged submission: the GPU never touches the adapter's queue.
            // It posts descriptors into a host-pinned staging ring shaped
            // exactly like the send queue, and its "doorbell" is a producer
            // cell the runtime's agent watches. Slot indices, phase bits, and
            // batching are identical to GPU-direct posting, so the device
            // library and the generated code do not change. No I/O-memory
            // mapping happens, which also drops the root requirement.
            const std::size_t ring_bytes =
                static_cast<std::size_t>(send_wq.entry_size) *
                send_wq.num_entries;
            cuda_check(
                cudaHostAlloc(
                    &endpoint->staging_ring,
                    ring_bytes,
                    cudaHostAllocPortable |
                        cudaHostAllocMapped),
                "cudaHostAlloc(EFA staging ring)");
            std::memset(
                endpoint->staging_ring, 0, ring_bytes);
            // Two cells: the send producer the agent watches, and a sink for
            // the receive doorbell nothing rings.
            cuda_check(
                cudaHostAlloc(
                    &endpoint->staging_cells,
                    2 * sizeof(std::uint32_t),
                    cudaHostAllocPortable |
                        cudaHostAllocMapped),
                "cudaHostAlloc(EFA staging cells)");
            std::memset(
                endpoint->staging_cells,
                0,
                2 * sizeof(std::uint32_t));
            void* ring_device = nullptr;
            void* cells_device = nullptr;
            cuda_check(
                cudaHostGetDevicePointer(
                    &ring_device, endpoint->staging_ring, 0),
                "cudaHostGetDevicePointer(EFA staging ring)");
            cuda_check(
                cudaHostGetDevicePointer(
                    &cells_device, endpoint->staging_cells, 0),
                "cudaHostGetDevicePointer(EFA staging cells)");
            queue_attributes.sq_buffer =
                static_cast<std::uint8_t*>(ring_device);
            queue_attributes.sq_doorbell =
                static_cast<std::uint32_t*>(cells_device);
            queue_attributes.rq_doorbell =
                static_cast<std::uint32_t*>(cells_device) + 1;
        } else {
            queue_attributes.sq_buffer =
                static_cast<std::uint8_t*>(map_queue(
                    send_wq.buffer,
                    static_cast<std::size_t>(
                        send_wq.entry_size) *
                        send_wq.num_entries));
            queue_attributes.sq_doorbell =
                static_cast<std::uint32_t*>(
                    map_doorbell(send_wq.doorbell));
            queue_attributes.rq_doorbell =
                static_cast<std::uint32_t*>(
                    map_doorbell(recv_wq.doorbell));
        }
        (void)recv_wq.entry_size;
        queue_attributes.sq_num_entries = send_wq.num_entries;
        queue_attributes.sq_entry_size = send_wq.entry_size;
        queue_attributes.sq_max_batch = send_wq.max_batch;
        queue_attributes.rq_num_entries = recv_wq.num_entries;
        queue_attributes.rq_entry_size = recv_wq.entry_size;

        efa_cuda_cq_attrs send_completion{};
        send_completion.buffer = send_cq_attr.buffer;
        send_completion.num_entries =
            send_cq_attr.num_entries;
        send_completion.entry_size =
            send_cq_attr.entry_size;

        efa_cuda_cq_attrs recv_completion{};
        recv_completion.buffer = recv_cq_attr.buffer;
        recv_completion.num_entries =
            recv_cq_attr.num_entries;
        recv_completion.entry_size =
            recv_cq_attr.entry_size;

        // The device ignores a descriptor whose phase bit does not match what
        // it expects for that pass over the queue. The send queue's first pass
        // is phase 0 — rdma-core's provider posts its first descriptors from a
        // zero-initialized queue pair, and a CPU-posted descriptor carrying
        // phase 1 is silently never fetched — so efa_cuda_init_qp leaving
        // sq.wq.phase at 0 is correct and must not be overridden.
        efa_cuda_qp host_queue_pair{};
        if (efa_cuda_init_qp(
                &host_queue_pair,
                &queue_attributes,
                sizeof(queue_attributes)) != 0) {
            throw_error(
                "efa_cuda_init_qp rejected the queue geometry");
        }
        cuda_check(
            cudaMalloc(
                &endpoint->device_queue_pair,
                sizeof(efa_cuda_qp)),
            "cudaMalloc(EFA queue pair)");
        cuda_check(
            cudaMemcpy(
                endpoint->device_queue_pair,
                &host_queue_pair,
                sizeof(efa_cuda_qp),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(EFA queue pair)");
        endpoint->device_send_cq = efa_cuda_create_cq(
            &send_completion, sizeof(send_completion));
        endpoint->device_recv_cq = efa_cuda_create_cq(
            &recv_completion, sizeof(recv_completion));
        if (endpoint->device_send_cq == nullptr ||
            endpoint->device_recv_cq == nullptr) {
            throw_error(
                "efa_cuda_create_qp/cq rejected the queue geometry");
        }

        cuda_check(
            cudaMalloc(
                &endpoint->completion_state,
                abi::kEfaCompletionStateWords *
                    sizeof(std::uint64_t)),
            "cudaMalloc(EFA completion state)");
        cuda_check(
            cudaMemset(
                endpoint->completion_state,
                0,
                abi::kEfaCompletionStateWords *
                    sizeof(std::uint64_t)),
            "cudaMemset(EFA completion state)");

        auto& descriptor = endpoint->descriptor;
        descriptor.queue_pair =
            reinterpret_cast<std::uint64_t>(
                endpoint->device_queue_pair);
        descriptor.send_completion_queue =
            reinterpret_cast<std::uint64_t>(
                endpoint->device_send_cq);
        descriptor.recv_completion_queue =
            reinterpret_cast<std::uint64_t>(
                endpoint->device_recv_cq);
        descriptor.completion_state =
            reinterpret_cast<std::uint64_t>(
                endpoint->completion_state);
        descriptor.completion_timeout_cycles =
            cuda_clock_rate_khz(runtime->gpu_index) *
            1000ULL * kTimeoutSeconds;
        descriptor.send_depth = send_wq.num_entries;
        descriptor.recv_depth = recv_wq.num_entries;
        // Filled in by connect; an unconnected endpoint cannot address a peer.
        descriptor.address_handle = 0;
        descriptor.remote_queue_pair_number = 0;
        descriptor.queue_key = kQueueKey;
        descriptor.publication_scheme =
            static_cast<std::uint32_t>(
                abi::EfaPublicationScheme::FenceThenSignal);
        descriptor.max_inline_bytes =
            runtime->device_attributes.inline_buf_size < 32
                ? runtime->device_attributes.inline_buf_size
                : 32;
        descriptor.abi_version = abi::kEfaEndpointVersion;
        publish_descriptor(endpoint.get());
        if (runtime->staged_submit) {
            endpoint->agent_token = runtime->agent->add(
                endpoint->staging_cells,
                aboleth::runtime::make_efa_llq_submit(
                    endpoint->host_send_buffer,
                    endpoint->host_send_doorbell,
                    endpoint->staging_ring,
                    endpoint->host_send_entry_bytes,
                    send_wq.num_entries - 1));
        }
        return endpoint.release();
    } catch (const std::exception& error) {
        last_error_message = error.what();
        return nullptr;
    }
}

void endpoint_destroy(EfaEndpointHandle* endpoint) noexcept {
    delete endpoint;
}

Address endpoint_address(EfaEndpointHandle* endpoint) {
    Address address{};
    if (endpoint == nullptr) {
        return address;
    }
    std::memcpy(
        address.gid,
        endpoint->runtime->gid.raw,
        sizeof(address.gid));
    address.queue_pair_number =
        endpoint->queue_pair->qp_num;
    address.queue_key = kQueueKey;
    return address;
}

void endpoint_connect(
    EfaEndpointHandle* endpoint,
    const Address& peer) {
    last_error_message.clear();
    if (endpoint == nullptr) {
        throw_error("null EFA endpoint");
    }
    ibv_ah_attr attributes{};
    attributes.is_global = 1;
    attributes.port_num = kPortNumber;
    attributes.grh.sgid_index = kGidIndex;
    attributes.grh.hop_limit = 0xff;
    std::memcpy(
        attributes.grh.dgid.raw,
        peer.gid,
        sizeof(attributes.grh.dgid.raw));
    ibv_ah* handle = dl::verbs().ibv_create_ah(
        endpoint->runtime->protection_domain,
        &attributes);
    if (handle == nullptr) {
        throw_error(
            std::string("ibv_create_ah: ") +
            std::strerror(errno));
    }
    if (endpoint->address_handle != nullptr) {
        dl::verbs().ibv_destroy_ah(endpoint->address_handle);
    }
    endpoint->address_handle = handle;

    efadv_ah_attr address_handle_attributes{};
    verbs_check(
        dl::efadv().efadv_query_ah(
            handle,
            &address_handle_attributes,
            sizeof(address_handle_attributes)),
        "efadv_query_ah");

    endpoint->descriptor.address_handle =
        address_handle_attributes.ahn;
    endpoint->descriptor.remote_queue_pair_number =
        peer.queue_pair_number;
    endpoint->descriptor.queue_key = peer.queue_key;
    publish_descriptor(endpoint);
}

std::uint64_t endpoint_descriptor_address(
    EfaEndpointHandle* endpoint) {
    return endpoint == nullptr
        ? 0
        : reinterpret_cast<std::uint64_t>(
              endpoint->descriptor_device);
}

std::uint32_t endpoint_capacity(EfaEndpointHandle* endpoint) {
    return endpoint == nullptr
        ? 0
        : endpoint->descriptor.send_depth;
}

void endpoint_state(
    EfaEndpointHandle* endpoint,
    std::uint64_t* producer,
    std::uint64_t* retired,
    std::uint64_t* status) {
    std::uint64_t words[abi::kEfaCompletionStateWords]{};
    if (endpoint != nullptr) {
        cuda_check(
            cudaMemcpy(
                words,
                endpoint->completion_state,
                sizeof(words),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(EFA completion state)");
    }
    if (producer != nullptr) {
        *producer = words[abi::kEfaStateProducer];
    }
    if (retired != nullptr) {
        *retired = words[abi::kEfaStateRetired];
    }
    if (status != nullptr) {
        *status = words[abi::kEfaStateError];
    }
}

std::uint64_t endpoint_host_post_write(
    EfaEndpointHandle* endpoint,
    std::uint64_t local_address,
    std::uint32_t local_key,
    std::uint64_t remote_address,
    std::uint32_t remote_key,
    std::uint32_t bytes,
    std::uint32_t request_id) {
    last_error_message.clear();
    if (endpoint == nullptr ||
        endpoint->host_send_buffer == nullptr) {
        last_error_message =
            "endpoint has no host send-queue mapping";
        return abi::kEfaStatusInvalid;
    }
    if (endpoint->runtime != nullptr &&
        endpoint->runtime->staged_submit) {
        // The doorbell agent owns the adapter queue's producer count; a
        // second producer would corrupt it. Diagnostics that need this path
        // run with ABOLETH_EFA_SUBMIT=gpu.
        last_error_message =
            "host-posted writes bypass the staged submitter; "
            "run with ABOLETH_EFA_SUBMIT=gpu";
        return abi::kEfaStatusInvalid;
    }
    // A zero address handle and a zero queue-pair number are both real
    // values on EFA, so connectivity is judged by the verbs object.
    if (endpoint->address_handle == nullptr) {
        last_error_message = "endpoint is not connected";
        return abi::kEfaStatusInvalid;
    }
    const auto& descriptor = endpoint->descriptor;

    // The same fields efa_cuda_init_rdma_write_wr, efa_cuda_wr_set_remote,
    // and efa_cuda_wr_set_sge store, built host-side. Kept in lockstep with
    // the device library; the CPU-posted descriptor is the reference the
    // GPU-posted one is diffed against when the datapath misbehaves.
    efa_io_tx_wqe request{};
    request.meta.req_id =
        static_cast<std::uint16_t>(request_id);
    request.meta.ctrl1 = static_cast<std::uint8_t>(
        EFA_IO_TX_META_DESC_META_DESC_MASK |
        (EFA_IO_RDMA_WRITE &
         EFA_IO_TX_META_DESC_OP_TYPE_MASK));
    // First pass over the queue is phase 0; the bit flips every wrap.
    const std::uint32_t depth = descriptor.send_depth;
    const std::uint32_t phase =
        (endpoint->host_producer / depth) & 1u;
    request.meta.ctrl2 = static_cast<std::uint8_t>(
        EFA_IO_TX_META_DESC_FIRST_MASK |
        EFA_IO_TX_META_DESC_LAST_MASK |
        EFA_IO_TX_META_DESC_COMP_REQ_MASK |
        (phase ? EFA_IO_TX_META_DESC_PHASE_MASK : 0));
    request.meta.dest_qp_num = static_cast<std::uint16_t>(
        descriptor.remote_queue_pair_number);
    request.meta.length = 1; // SGE count, not bytes
    request.meta.ah =
        static_cast<std::uint16_t>(descriptor.address_handle);
    request.meta.qkey = descriptor.queue_key;
    auto& remote = request.data.rdma_req.remote_mem;
    remote.length = bytes;
    remote.rkey = remote_key;
    remote.buf_addr_lo =
        static_cast<std::uint32_t>(remote_address);
    remote.buf_addr_hi =
        static_cast<std::uint32_t>(remote_address >> 32);
    auto& local = request.data.rdma_req.local_mem[0];
    local.length = bytes;
    local.lkey = local_key & EFA_IO_TX_BUF_DESC_LKEY_MASK;
    local.buf_addr_lo =
        static_cast<std::uint32_t>(local_address);
    local.buf_addr_hi =
        static_cast<std::uint32_t>(local_address >> 32);

    // Place and ring, like efa_cuda_sq_batch_place_wr + flush_sq_wrs. The
    // send queue is write-only I/O memory, so the descriptor goes out as
    // eight 64-bit stores and the doorbell carries the new producer count.
    const std::uint32_t slot =
        endpoint->host_producer & (depth - 1);
    volatile std::uint64_t* destination =
        reinterpret_cast<volatile std::uint64_t*>(
            endpoint->host_send_buffer +
            static_cast<std::size_t>(slot) *
                endpoint->host_send_entry_bytes);
    const std::uint64_t* source =
        reinterpret_cast<const std::uint64_t*>(&request);
    for (int word = 0; word < 8; ++word) {
        destination[word] = source[word];
    }
    __sync_synchronize();
    ++endpoint->host_producer;
    *reinterpret_cast<volatile std::uint32_t*>(
        endpoint->host_send_doorbell) = endpoint->host_producer;
    __sync_synchronize();
    return abi::kEfaStatusOk;
}

EfaBufferHandle* register_buffer(
    EfaRuntimeHandle* runtime,
    void* address,
    std::size_t bytes) {
    last_error_message.clear();
    try {
        if (runtime == nullptr || address == nullptr ||
            bytes == 0) {
            throw_error("null EFA registration argument");
        }
        const int access =
            IBV_ACCESS_LOCAL_WRITE |
            IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ;
        auto buffer = std::make_unique<EfaBufferHandle>();

        const auto page_size =
            static_cast<std::uintptr_t>(
                sysconf(_SC_PAGESIZE));
        const auto start =
            reinterpret_cast<std::uintptr_t>(address);
        const auto base = start & ~(page_size - 1);
        const auto end =
            (start + bytes + page_size - 1) &
            ~(page_size - 1);

        // EFA reaches GPU memory only through dmabuf.
        int descriptor = -1;
        if (dl::cuda().cuMemGetHandleForAddressRange(
                &descriptor,
                static_cast<CUdeviceptr>(base),
                end - base,
                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                0) == CUDA_SUCCESS) {
            buffer->memory_region = dl::verbs().ibv_reg_dmabuf_mr(
                runtime->protection_domain,
                /*offset=*/0,
                end - base,
                base,
                descriptor,
                access);
            close(descriptor);
        } else {
            buffer->memory_region = dl::verbs().reg_mr(
                runtime->protection_domain,
                address,
                bytes,
                access);
        }
        if (buffer->memory_region == nullptr) {
            throw_error(
                std::string("EFA memory registration failed: ") +
                std::strerror(errno));
        }
        return buffer.release();
    } catch (const std::exception& error) {
        last_error_message = error.what();
        return nullptr;
    }
}

void unregister_buffer(EfaBufferHandle* buffer) noexcept {
    delete buffer;
}

std::uint32_t buffer_local_key(EfaBufferHandle* buffer) {
    return buffer == nullptr || buffer->memory_region == nullptr
        ? 0
        : buffer->memory_region->lkey;
}

std::uint32_t buffer_remote_key(EfaBufferHandle* buffer) {
    return buffer == nullptr || buffer->memory_region == nullptr
        ? 0
        : buffer->memory_region->rkey;
}

std::size_t pack_address(
    const Address& address,
    std::uint8_t* bytes,
    std::size_t capacity) {
    if (bytes == nullptr || capacity < kAddressBytes) {
        return 0;
    }
    std::memcpy(bytes, address.gid, 16);
    std::memcpy(
        bytes + 16, &address.queue_pair_number, 4);
    std::memcpy(bytes + 20, &address.queue_key, 4);
    return kAddressBytes;
}

Address unpack_address(
    const std::uint8_t* bytes,
    std::size_t length) {
    Address address{};
    if (bytes == nullptr || length < kAddressBytes) {
        throw_error("EFA peer address is malformed");
    }
    std::memcpy(address.gid, bytes, 16);
    std::memcpy(
        &address.queue_pair_number, bytes + 16, 4);
    std::memcpy(&address.queue_key, bytes + 20, 4);
    return address;
}

const char* last_error() {
    return last_error_message.c_str();
}

} // namespace aboleth::efa

