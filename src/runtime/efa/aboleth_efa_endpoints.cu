// EFA endpoints: the proxy provider's libfabric endpoint, the GPU-initiated
// GDA endpoint, and the Transport.EFA endpoint the compiler backend targets.
//
// All three are built on the fabric context efa/aboleth_efa_control.cu opens,
// and all three end the same way: an address a peer inserts into the shared
// address vector, and a descriptor a kernel reads. The local GDA smoke entry
// point lives here too — it is the only host-side consumer of the vendored
// device primitives, and what it exercises is exactly the endpoints above it.

#include "aboleth_efa_internal.h"

#if ABOLETH_HAS_EFA

#include "../core/aboleth_doorbell_agent.h"
#include "../aboleth_runtime_common.h"
#include "../aboleth_runtime_internal.h"
#include "../core/aboleth_memory.h"
#include "../core/aboleth_runtime.h"
#include "../proxy/aboleth_proxy.h"

#include "aboleth/Target/EfaAbi.h"
#include "aboleth/Target/EfaGdaAbi.h"
#include "aboleth/runtime_c.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_ext_efa.h>
#include <rdma/fi_rma.h>

#include "../dl/shims.h"

namespace dl = aboleth::dl;

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

// The efa-dp-direct declaration header declares each primitive non-static
// while the implementation defines it `static inline`, which the host
// compiler rejects in one translation unit. Suppressing the declaration
// header leaves only the definitions, the same arrangement
// aboleth_efa_device.cu uses; nothing here needs the two enumerations it
// also provides.
#define EFA_CUDA_DP_CUH
#include "efa_cuda_dp_types.h"
#include "efa_io_defs.h"

// The suppressed declaration header also provides these two enumerations,
// which the implementation uses; restated verbatim, as in
// aboleth_efa_device.cu.
enum efa_cuda_wc_opcode {
    EFA_CUDA_WC_SEND,
    EFA_CUDA_WC_RDMA_WRITE,
    EFA_CUDA_WC_RDMA_READ,
    EFA_CUDA_WC_RECV = 1 << 7,
    EFA_CUDA_WC_RECV_RDMA_WITH_IMM,
};

enum efa_cuda_processing_hint {
    EFA_CUDA_PROCESSING_HINT_BURST_PPS_SENSITIVE = 1 << 0,
};

#include "efa_cuda_dp_impl.cuh"

// The efa-gda branch patches the vendored device library with proper MMIO
// store semantics; the submodule pin does not carry that patch, so the GDA
// kernels in this translation unit define the store locally.
__device__ static inline void efa_cuda_mmio_write32(
    volatile uint32_t* address,
    uint32_t value) {
    asm volatile(
        "st.relaxed.mmio.sys.u32 [%0], %1;"
        :
        : "l"(address), "r"(value)
        : "memory");
}

// aboleth::runtime is where the shared internals live (see
// aboleth_runtime_internal.h). The anonymous-namespace helpers and the
// extern "C" arm below reach them unqualified, as the runtime translation
// unit this code came from does.
using namespace aboleth::runtime;

namespace {

std::shared_ptr<EfaProxyCqShard>
get_efa_proxy_cq_shard(
    EfaFabricContext& fabric,
    int worker_shard) {
    std::lock_guard<std::mutex> lock(
        fabric.proxy_shard_mutex);
    auto& existing =
        fabric.proxy_shards[worker_shard];
    if (existing) {
        return existing;
    }
    auto shard =
        std::make_shared<EfaProxyCqShard>();
    fi_cq_attr tx_attributes{};
    tx_attributes.size = 16384;
    tx_attributes.format =
        FI_CQ_FORMAT_CONTEXT;
    tx_attributes.wait_obj = FI_WAIT_NONE;
    fabric_check(
        fi_cq_open(
            fabric.domain,
            &tx_attributes,
            &shard->tx,
            nullptr),
        "fi_cq_open(EFA proxy shard TX)");
    fi_cq_attr rx_attributes =
        tx_attributes;
    rx_attributes.format =
        FI_CQ_FORMAT_DATA;
    fabric_check(
        fi_cq_open(
            fabric.domain,
            &rx_attributes,
            &shard->rx,
            nullptr),
        "fi_cq_open(EFA proxy shard RX)");
    existing = shard;
    return shard;
}

}  // namespace

namespace aboleth::runtime {

void create_efa_proxy_endpoint_resources(
    ProxyEndpointContext* endpoint) {
    auto& fabric = *endpoint->runtime->efa;
    endpoint->efa_cq_shard =
        get_efa_proxy_cq_shard(
            fabric,
            endpoint->worker_shard);
    endpoint->efa_av =
        fabric.proxy_av;
    endpoint->efa_tx_cq =
        endpoint->efa_cq_shard->tx;
    endpoint->efa_rx_cq =
        endpoint->efa_cq_shard->rx;
    fabric_check(
        fi_endpoint(
            fabric.domain,
            fabric.info,
            &endpoint->efa_endpoint,
            nullptr),
        "fi_endpoint(EFA proxy)");
    fi_cntr_attr counter_attributes{};
    counter_attributes.events =
        FI_CNTR_EVENTS_COMP;
    counter_attributes.wait_obj =
        FI_WAIT_NONE;
    fabric_check(
        fi_cntr_open(
            fabric.domain,
            &counter_attributes,
            &endpoint->efa_tx_counter,
            nullptr),
        "fi_cntr_open(EFA proxy TX)");

    bool disabled = false;
    bool enabled = true;
    int p2p_mode = FI_HMEM_P2P_REQUIRED;
    const bool direct =
        fabric.fabric_name ==
        "efa-direct";
    if (!direct) {
        fabric_check(
            fi_setopt(
                &endpoint->efa_endpoint->fid,
                FI_OPT_ENDPOINT,
                FI_OPT_SHARED_MEMORY_PERMITTED,
                &disabled,
                sizeof(disabled)),
            "fi_setopt(EFA disable SHM)");
        fabric_check(
            fi_setopt(
                &endpoint->efa_endpoint->fid,
                FI_OPT_ENDPOINT,
                FI_OPT_FI_HMEM_P2P,
                &p2p_mode,
                sizeof(p2p_mode)),
            "fi_setopt(EFA require HMEM P2P)");
        fabric_check(
            fi_setopt(
                &endpoint->efa_endpoint->fid,
                FI_OPT_ENDPOINT,
                FI_OPT_EFA_USE_DEVICE_RDMA,
                &enabled,
                sizeof(enabled)),
            "fi_setopt(EFA device RDMA)");
        fabric_check(
            fi_setopt(
                &endpoint->efa_endpoint->fid,
                FI_OPT_ENDPOINT,
                FI_OPT_EFA_HOMOGENEOUS_PEERS,
                &enabled,
                sizeof(enabled)),
            "fi_setopt(EFA homogeneous peers)");
    }
    fabric_check(
        fi_ep_bind(
            endpoint->efa_endpoint,
            &endpoint->efa_av->fid,
            0),
        "fi_ep_bind(EFA AV)");
    fabric_check(
        fi_ep_bind(
            endpoint->efa_endpoint,
            &endpoint->efa_tx_cq->fid,
            FI_TRANSMIT |
                FI_SELECTIVE_COMPLETION),
        "fi_ep_bind(EFA TX CQ)");
    fabric_check(
        fi_ep_bind(
            endpoint->efa_endpoint,
            &endpoint->efa_rx_cq->fid,
            FI_RECV),
        "fi_ep_bind(EFA RX CQ)");
    fabric_check(
        fi_ep_bind(
            endpoint->efa_endpoint,
            &endpoint->efa_tx_counter->fid,
            FI_WRITE),
        "fi_ep_bind(EFA TX counter)");
    fabric_check(
        fi_enable(endpoint->efa_endpoint),
        "fi_enable(EFA proxy)");

    if (direct) {
        endpoint->efa_signal_values.resize(
            static_cast<std::size_t>(
                aboleth::abi::
                    kProxySlotCount) *
            aboleth::abi::
                kProxyMaximumCommands);
        fabric_check(
            fi_mr_reg(
                fabric.domain,
                endpoint->
                    efa_signal_values.data(),
                endpoint->
                    efa_signal_values.size() *
                    sizeof(std::uint64_t),
                FI_SEND |
                    FI_RECV |
                    FI_READ |
                    FI_WRITE,
                0,
                0,
                0,
                &endpoint->efa_signal_mr,
                nullptr),
            "fi_mr_reg(EFA direct signal pool)");
    }

    if (!direct) {
        bool emulated_write = true;
        std::size_t option_bytes =
            sizeof(emulated_write);
        fabric_check(
            fi_getopt(
                &endpoint->efa_endpoint->fid,
                FI_OPT_ENDPOINT,
                FI_OPT_EFA_EMULATED_WRITE,
                &emulated_write,
                &option_bytes),
            "fi_getopt(EFA emulated write)");
        if (emulated_write) {
            throw_error(
                "EFA proxy requires native device RDMA writes");
        }
    }

    std::size_t address_bytes =
        endpoint->efa_address.size();
    fabric_check(
        fi_getname(
            &endpoint->efa_endpoint->fid,
            endpoint->efa_address.data(),
            &address_bytes),
        "fi_getname(EFA proxy)");
    if (address_bytes == 0 ||
        address_bytes >
            endpoint->efa_address.size()) {
        throw_error(
            "EFA endpoint address exceeds connection ABI");
    }
    endpoint->efa_address_length =
        static_cast<std::uint32_t>(
            address_bytes);
}

void connect_efa_proxy_endpoint(
    ProxyEndpointContext* endpoint,
    const std::uint8_t* address,
    std::size_t address_bytes) {
    if (address == nullptr ||
        address_bytes == 0 ||
        address_bytes >
            kAbolethConnectionAddressBytes) {
        throw_error(
            "invalid remote EFA endpoint address");
    }
    fi_addr_t peer = FI_ADDR_NOTAVAIL;
    const int inserted =
        fi_av_insert(
            endpoint->efa_av,
            address,
            1,
            &peer,
            0,
            nullptr);
    if (inserted != 1 ||
        peer == FI_ADDR_NOTAVAIL) {
        throw_error(
            "fi_av_insert(EFA peer) failed");
    }
    endpoint->efa_peer = peer;
}

std::unique_ptr<EfaGdaEndpointContext>
create_efa_gda_endpoint(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    std::uint32_t local_window_lkey,
    std::uint32_t local_control_lkey,
    aboleth::abi::EfaGdaEndpointDescriptor&
        descriptor) {
    if (runtime == nullptr ||
        runtime->efa == nullptr ||
        runtime->efa->fabric_name !=
            "efa-direct" ||
        runtime->efa->gda_ops == nullptr) {
        throw_error(
            "EFA GDA requires an efa-direct runtime");
    }
    cuda_check(
        cudaSetDevice(gpu_index),
        "cudaSetDevice(EFA GDA endpoint)");
    auto result =
        std::make_unique<
            EfaGdaEndpointContext>();
    result->runtime = runtime;
    result->gpu_index = gpu_index;
    result->av = runtime->efa->proxy_av;

    fi_cq_attr cq_attributes{};
    cq_attributes.size = 4096;
    cq_attributes.format =
        FI_CQ_FORMAT_DATA;
    cq_attributes.wait_obj =
        FI_WAIT_NONE;
    const bool debug =
        std::getenv(
            "ABOLETH_EFA_GDA_DEBUG") !=
        nullptr;
    if (debug) {
        std::fprintf(
            stderr,
            "EFA GDA create gpu=%d cq-vmm-begin\n",
            gpu_index);
        std::fflush(stderr);
    }
    result->cq_memory.allocate(
        cq_attributes.size * 64);
    if (debug) {
        std::fprintf(
            stderr,
            "EFA GDA create gpu=%d cq-vmm=%p bytes=%zu fd=%d\n",
            gpu_index,
            result->cq_memory.pointer,
            result->cq_memory.bytes,
            result->cq_memory.dmabuf_fd);
        std::fflush(stderr);
    }
    fi_efa_cq_init_attr efa_cq_attributes{};
    efa_cq_attributes.flags =
        FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF;
    efa_cq_attributes.ext_mem_dmabuf.buffer =
        nullptr;
    efa_cq_attributes.ext_mem_dmabuf.length =
        result->cq_memory.bytes;
    efa_cq_attributes.ext_mem_dmabuf.offset = 0;
    efa_cq_attributes.ext_mem_dmabuf.fd =
        result->cq_memory.dmabuf_fd;
    fabric_check(
        runtime->efa->gda_ops->cq_open_ext(
            runtime->efa->domain,
            &cq_attributes,
            &efa_cq_attributes,
            &result->cq,
            nullptr),
        "cq_open_ext(EFA GDA)");
    if (debug) {
        std::fprintf(
            stderr,
            "EFA GDA create gpu=%d cq-open=%p\n",
            gpu_index,
            static_cast<void*>(result->cq));
        std::fflush(stderr);
    }
    fabric_check(
        fi_endpoint(
            runtime->efa->domain,
            runtime->efa->info,
            &result->endpoint,
            nullptr),
        "fi_endpoint(EFA GDA)");
    fabric_check(
        fi_ep_bind(
            result->endpoint,
            &result->av->fid,
            0),
        "fi_ep_bind(EFA GDA AV)");
    fabric_check(
        fi_ep_bind(
            result->endpoint,
            &result->cq->fid,
            FI_TRANSMIT | FI_RECV),
        "fi_ep_bind(EFA GDA CQ)");
    fabric_check(
        fi_enable(result->endpoint),
        "fi_enable(EFA GDA)");
    if (debug) {
        std::fprintf(
            stderr,
            "EFA GDA create gpu=%d endpoint-enabled\n",
            gpu_index);
        std::fflush(stderr);
    }

    fabric_check(
        runtime->efa->gda_ops->
            query_qp_wqs(
                result->endpoint,
                &result->sq_attributes,
                &result->rq_attributes),
        "query_qp_wqs(EFA GDA)");
    fabric_check(
        runtime->efa->gda_ops->
            query_cq(
                result->cq,
                &result->cq_attributes),
        "query_cq(EFA GDA)");
    if (debug) {
        std::fprintf(
            stderr,
            "EFA GDA create gpu=%d queues sq=%u/%u batch=%u cq=%u/%u buffer=%p\n",
            gpu_index,
            result->sq_attributes.num_entries,
            result->sq_attributes.entry_size,
            result->sq_attributes.max_batch,
            result->cq_attributes.num_entries,
            result->cq_attributes.entry_size,
            result->cq_attributes.buffer);
        std::fflush(stderr);
    }
    if (result->sq_attributes.entry_size !=
            sizeof(efa_io_tx_wqe) ||
        result->sq_attributes.num_entries == 0 ||
        (result->sq_attributes.num_entries &
         (result->sq_attributes.num_entries - 1)) !=
            0 ||
        result->cq_attributes.num_entries == 0 ||
        (result->cq_attributes.num_entries &
         (result->cq_attributes.num_entries - 1)) !=
            0) {
        throw_error(
            "unsupported EFA GDA queue layout");
    }

    efa_cuda_qp host_qp{};
    result->staged = aboleth::runtime::resolve_staged_submit(
        runtime->efa->device_name);
    if (result->staged) {
        // Staged submission: the GPU never touches the adapter's queue. It
        // posts into a host-pinned staging ring shaped exactly like the send
        // queue, and its "doorbell" is a producer cell the doorbell agent
        // watches. No I/O-memory mapping happens, so no privileges either.
        const std::size_t ring_bytes =
            static_cast<std::size_t>(
                result->sq_attributes.num_entries) *
            result->sq_attributes.entry_size;
        cuda_check(
            cudaHostAlloc(
                &result->staging_ring,
                ring_bytes,
                cudaHostAllocPortable |
                    cudaHostAllocMapped),
            "cudaHostAlloc(EFA GDA staging ring)");
        std::memset(result->staging_ring, 0, ring_bytes);
        cuda_check(
            cudaHostAlloc(
                &result->staging_cells,
                2 * sizeof(std::uint32_t),
                cudaHostAllocPortable |
                    cudaHostAllocMapped),
            "cudaHostAlloc(EFA GDA staging cells)");
        std::memset(
            result->staging_cells,
            0,
            2 * sizeof(std::uint32_t));
        void* ring_device = nullptr;
        void* cells_device = nullptr;
        cuda_check(
            cudaHostGetDevicePointer(
                &ring_device, result->staging_ring, 0),
            "cudaHostGetDevicePointer(EFA GDA staging)");
        cuda_check(
            cudaHostGetDevicePointer(
                &cells_device, result->staging_cells, 0),
            "cudaHostGetDevicePointer(EFA GDA cells)");
        host_qp.sq.wq.buf =
            static_cast<std::uint8_t*>(ring_device);
        host_qp.sq.wq.db =
            static_cast<std::uint32_t*>(cells_device);
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA create gpu=%d staged\n",
                gpu_index);
            std::fflush(stderr);
        }
    } else {
        result->sq_buffer.map(
            result->sq_attributes.buffer,
            static_cast<std::size_t>(
                result->sq_attributes.num_entries) *
                result->sq_attributes.entry_size,
            "EFA GDA SQ");
        result->sq_doorbell.map(
            result->sq_attributes.doorbell,
            static_cast<std::size_t>(
                sysconf(_SC_PAGESIZE)),
            "EFA GDA doorbell");
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA create gpu=%d mmio-mapped\n",
                gpu_index);
            std::fflush(stderr);
        }
        host_qp.sq.wq.buf =
            static_cast<std::uint8_t*>(
                result->sq_buffer.device);
        host_qp.sq.wq.db =
            static_cast<std::uint32_t*>(
                result->sq_doorbell.device);
    }
    host_qp.sq.wq.max_wqes =
        result->sq_attributes.num_entries;
    host_qp.sq.wq.max_batch =
        result->sq_attributes.max_batch;
    host_qp.sq.wq.queue_mask =
        result->sq_attributes.num_entries - 1;
    host_qp.sq.wq.queue_size_shift =
        static_cast<std::uint32_t>(
            __builtin_ctz(
                result->
                    sq_attributes.num_entries));
    host_qp.sq.max_inline_data = 32;
    host_qp.sq.max_rdma_sges = 2;
    host_qp.rq.wq.buf =
        result->rq_attributes.buffer;
    host_qp.rq.wq.db =
        result->rq_attributes.doorbell;
    host_qp.rq.wq.max_wqes =
        result->rq_attributes.num_entries;
    host_qp.rq.wq.max_batch =
        result->rq_attributes.max_batch;
    host_qp.rq.wq.queue_mask =
        result->rq_attributes.num_entries - 1;
    host_qp.rq.wq.queue_size_shift =
        static_cast<std::uint32_t>(
            __builtin_ctz(
                result->
                    rq_attributes.num_entries));
    host_qp.rq.wq.phase = 1;
    cuda_check(
        cudaMalloc(
            &result->device_qp,
            sizeof(host_qp)),
        "cudaMalloc(EFA GDA QP)");
    cuda_check(
        cudaMemcpy(
            result->device_qp,
            &host_qp,
            sizeof(host_qp),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(EFA GDA QP)");

    efa_cuda_cq host_cq{};
    host_cq.buf =
        static_cast<std::uint8_t*>(
            result->cq_memory.pointer);
    host_cq.entry_size =
        result->cq_attributes.entry_size;
    host_cq.num_entries =
        result->cq_attributes.num_entries;
    host_cq.queue_mask =
        result->cq_attributes.num_entries - 1;
    host_cq.queue_size_shift =
        static_cast<std::uint32_t>(
            __builtin_ctz(
                result->
                    cq_attributes.num_entries));
    host_cq.phase = 1;
    cuda_check(
        cudaMalloc(
            &result->device_cq,
            sizeof(host_cq)),
        "cudaMalloc(EFA GDA CQ)");
    cuda_check(
        cudaMemcpy(
            result->device_cq,
            &host_cq,
            sizeof(host_cq),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(EFA GDA CQ)");

    std::size_t address_bytes =
        result->address.size();
    fabric_check(
        fi_getname(
            &result->endpoint->fid,
            result->address.data(),
            &address_bytes),
        "fi_getname(EFA GDA)");
    if (address_bytes == 0 ||
        address_bytes >
            result->address.size()) {
        throw_error(
            "EFA GDA address exceeds connection ABI");
    }
    result->address_length =
        static_cast<std::uint32_t>(
            address_bytes);
    fi_addr_t self_address =
        FI_ADDR_NOTAVAIL;
    const int self_inserted =
        fi_av_insert(
            result->av,
            result->address.data(),
            1,
            &self_address,
            0,
            nullptr);
    if (self_inserted != 1 ||
        self_address ==
            FI_ADDR_NOTAVAIL) {
        throw_error(
            "fi_av_insert(EFA GDA self) failed");
    }
    std::uint16_t self_ah = 0;
    std::uint16_t self_qpn = 0;
    std::uint32_t self_qkey = 0;
    fabric_check(
        runtime->efa->gda_ops->
            query_addr(
                result->endpoint,
                self_address,
                &self_ah,
                &self_qpn,
                &self_qkey),
        "query_addr(EFA GDA self)");

    descriptor.qp =
        reinterpret_cast<std::uint64_t>(
            result->device_qp);
    descriptor.cq =
        reinterpret_cast<std::uint64_t>(
            result->device_cq);
    descriptor.completion_timeout_cycles =
        cuda_clock_rate_khz(gpu_index) *
        1000ULL *
        kTimeoutSeconds;
    descriptor.local_window_lkey =
        local_window_lkey;
    descriptor.local_control_lkey =
        local_control_lkey;
    descriptor.abi_version =
        aboleth::abi::
            kEfaGdaEndpointVersion;
    if (result->staged) {
        result->agent =
            aboleth::runtime::acquire_doorbell_agent(
                runtime.get(), gpu_index);
        result->agent_token = result->agent->add(
            result->staging_cells,
            aboleth::runtime::make_efa_llq_submit(
                result->sq_attributes.buffer,
                result->sq_attributes.doorbell,
                result->staging_ring,
                result->sq_attributes.entry_size,
                result->sq_attributes.num_entries - 1));
    }
    return result;
}

void connect_efa_gda_endpoint(
    EfaGdaEndpointContext& endpoint,
    const std::uint8_t* address,
    std::size_t address_bytes,
    std::uint32_t remote_window_rkey,
    std::uint32_t remote_barrier_rkey,
    aboleth::abi::EfaGdaEndpointDescriptor&
        descriptor) {
    if (address == nullptr ||
        address_bytes == 0 ||
        address_bytes >
            kAbolethConnectionAddressBytes) {
        throw_error(
            "invalid remote EFA GDA address");
    }
    fi_addr_t peer = FI_ADDR_NOTAVAIL;
    const int inserted =
        fi_av_insert(
            endpoint.av,
            address,
            1,
            &peer,
            0,
            nullptr);
    if (inserted != 1 ||
        peer == FI_ADDR_NOTAVAIL) {
        throw_error(
            "fi_av_insert(EFA GDA peer) failed");
    }
    std::uint16_t ah = 0;
    std::uint16_t qpn = 0;
    std::uint32_t qkey = 0;
    fabric_check(
        endpoint.runtime->efa->
            gda_ops->query_addr(
                endpoint.endpoint,
                peer,
                &ah,
                &qpn,
                &qkey),
        "query_addr(EFA GDA peer)");
    endpoint.peer = peer;
    descriptor.peer_ah = ah;
    descriptor.peer_qpn = qpn;
    descriptor.peer_qkey = qkey;
    descriptor.remote_window_rkey =
        remote_window_rkey;
    descriptor.remote_barrier_rkey =
        remote_barrier_rkey;
}

}  // namespace aboleth::runtime

namespace {

void publish_efa_abi_descriptor(
    EfaAbiEndpointState& endpoint) {
    cuda_check(
        cudaSetDevice(endpoint.gpu_index),
        "cudaSetDevice(EFA descriptor)");
    cuda_check(
        cudaMemcpy(
            endpoint.descriptor_device,
            &endpoint.descriptor,
            sizeof(endpoint.descriptor),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(EFA descriptor)");
}

}  // namespace

namespace aboleth::runtime {

std::unique_ptr<EfaAbiEndpointState>
create_efa_abi_endpoint(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    void* descriptor_device,
    std::size_t descriptor_bytes) {
    if (descriptor_device == nullptr ||
        descriptor_bytes <
            sizeof(aboleth::abi::EfaEndpointDescriptor)) {
        throw_error(
            "EFA endpoint descriptor buffer is too small");
    }
    auto endpoint =
        std::make_unique<EfaAbiEndpointState>();
    endpoint->gpu_index = gpu_index;
    endpoint->descriptor_device = descriptor_device;
    aboleth::abi::EfaGdaEndpointDescriptor scratch{};
    endpoint->context = create_efa_gda_endpoint(
        runtime,
        gpu_index,
        /*local_window_lkey=*/0,
        /*local_control_lkey=*/0,
        scratch);
    cuda_check(
        cudaMalloc(
            &endpoint->completion_state,
            aboleth::abi::kEfaCompletionStateWords *
                sizeof(std::uint64_t)),
        "cudaMalloc(EFA completion state)");
    cuda_check(
        cudaMemset(
            endpoint->completion_state,
            0,
            aboleth::abi::kEfaCompletionStateWords *
                sizeof(std::uint64_t)),
        "cudaMemset(EFA completion state)");

    auto& descriptor = endpoint->descriptor;
    descriptor.queue_pair = scratch.qp;
    descriptor.send_completion_queue = scratch.cq;
    // Sender-fenced publication consumes no receive completions.
    descriptor.recv_completion_queue = 0;
    descriptor.completion_state =
        reinterpret_cast<std::uint64_t>(
            endpoint->completion_state);
    descriptor.completion_timeout_cycles =
        scratch.completion_timeout_cycles;
    descriptor.send_depth =
        endpoint->context->sq_attributes.num_entries;
    descriptor.recv_depth =
        endpoint->context->rq_attributes.num_entries;
    // Filled by connect; an unconnected endpoint cannot address a peer.
    descriptor.address_handle = 0;
    descriptor.remote_queue_pair_number = 0;
    descriptor.queue_key = 0;
    descriptor.publication_scheme =
        static_cast<std::uint32_t>(
            aboleth::abi::EfaPublicationScheme::
                FenceThenSignal);
    descriptor.max_inline_bytes = 32;
    descriptor.abi_version =
        aboleth::abi::kEfaEndpointVersion;
    publish_efa_abi_descriptor(*endpoint);
    return endpoint;
}

void connect_efa_abi_endpoint(
    EfaAbiEndpointState& endpoint,
    const std::uint8_t* address,
    std::size_t address_bytes) {
    if (address == nullptr || address_bytes == 0 ||
        address_bytes > kAbolethConnectionAddressBytes) {
        throw_error("invalid remote EFA address");
    }
    fi_addr_t peer = FI_ADDR_NOTAVAIL;
    const int inserted = fi_av_insert(
        endpoint.context->av,
        address,
        1,
        &peer,
        0,
        nullptr);
    if (inserted != 1 || peer == FI_ADDR_NOTAVAIL) {
        throw_error("fi_av_insert(EFA peer) failed");
    }
    std::uint16_t ah = 0;
    std::uint16_t qpn = 0;
    std::uint32_t qkey = 0;
    fabric_check(
        endpoint.context->runtime->efa->gda_ops->query_addr(
            endpoint.context->endpoint,
            peer,
            &ah,
            &qpn,
            &qkey),
        "query_addr(EFA peer)");
    endpoint.context->peer = peer;
    endpoint.descriptor.address_handle = ah;
    endpoint.descriptor.remote_queue_pair_number = qpn;
    endpoint.descriptor.queue_key = qkey;
    publish_efa_abi_descriptor(endpoint);
}

}  // namespace aboleth::runtime

namespace {

struct EfaGdaSmokeResult {
    std::uint32_t status{};
    std::uint32_t vendor_error{};
    std::uint32_t cq_flags{};
    std::uint32_t qp_pc{};
};

void efa_gda_cpu_post(
    EfaGdaEndpointContext& endpoint,
    const aboleth::abi::
        EfaGdaEndpointDescriptor& descriptor,
    std::uint64_t source_address,
    std::uint64_t destination_address,
    std::uint32_t bytes,
    bool ring_doorbell = true) {
    efa_io_tx_wqe wqe{};
    wqe.meta.req_id = 1;
    wqe.meta.ctrl1 =
        static_cast<std::uint8_t>(
            (1u << 7) |
            EFA_IO_RDMA_WRITE);
    wqe.meta.ctrl2 =
        static_cast<std::uint8_t>(
            (1u << 2) |
            (1u << 3) |
            (1u << 4));
    wqe.meta.dest_qp_num =
        descriptor.peer_qpn;
    wqe.meta.length = 1;
    wqe.meta.ah = descriptor.peer_ah;
    wqe.meta.qkey =
        descriptor.peer_qkey;
    wqe.data.rdma_req.remote_mem.length =
        bytes;
    wqe.data.rdma_req.remote_mem.rkey =
        descriptor.remote_window_rkey;
    wqe.data.rdma_req.remote_mem.
        buf_addr_lo =
        static_cast<std::uint32_t>(
            destination_address);
    wqe.data.rdma_req.remote_mem.
        buf_addr_hi =
        static_cast<std::uint32_t>(
            destination_address >> 32);
    wqe.data.rdma_req.local_mem[0].
        length = bytes;
    wqe.data.rdma_req.local_mem[0].
        lkey =
        descriptor.local_window_lkey &
        0x00ffffffu;
    wqe.data.rdma_req.local_mem[0].
        buf_addr_lo =
        static_cast<std::uint32_t>(
            source_address);
    wqe.data.rdma_req.local_mem[0].
        buf_addr_hi =
        static_cast<std::uint32_t>(
            source_address >> 32);
    const auto* source_words =
        reinterpret_cast<
            const std::uint64_t*>(&wqe);
    auto* destination_words =
        reinterpret_cast<
            volatile std::uint64_t*>(
            endpoint.sq_attributes.buffer);
    for (int index = 0;
         index < 8;
         ++index) {
        destination_words[index] =
            source_words[index];
    }
    asm volatile("sfence" ::: "memory");
    if (ring_doorbell) {
        *reinterpret_cast<
            volatile std::uint32_t*>(
            endpoint.sq_attributes.doorbell) =
            1;
        asm volatile(
            "sfence" ::: "memory");
    }
}

__global__ void efa_gda_smoke_kernel(
    aboleth::abi::EfaGdaEndpointDescriptor
        descriptor,
    std::uint64_t source_address,
    std::uint64_t destination_address,
    std::uint32_t bytes,
    int mode,
    EfaGdaSmokeResult* result) {
    if (blockIdx.x != 0 ||
        threadIdx.x != 0) {
        return;
    }
    auto* qp =
        reinterpret_cast<efa_cuda_qp*>(
            descriptor.qp);
    auto* cq =
        reinterpret_cast<efa_cuda_cq*>(
            descriptor.cq);
    if (mode != 2) {
        efa_io_tx_wqe wqe{};
        efa_cuda_init_rdma_write_wr(
            &wqe,
            1,
            descriptor.remote_window_rkey,
            destination_address);
        efa_cuda_wr_set_remote(
            &wqe,
            descriptor.peer_ah,
            descriptor.peer_qpn,
            descriptor.peer_qkey);
        efa_cuda_wr_set_sge(
            &wqe,
            descriptor.local_window_lkey,
            source_address,
            bytes);
        efa_cuda_start_sq_batch(qp, 1);
        efa_cuda_sq_batch_place_wr(
            qp,
            0,
            &wqe);
    }
    if (mode == 1) {
        __threadfence_system();
        result->status = 3;
        return;
    }
    if (mode == 2) {
        qp->sq.wq.wqes_pending = 0;
        qp->sq.wq.pc = 1;
        __threadfence_system();
        efa_cuda_mmio_write32(
            qp->sq.wq.db,
            1);
        __threadfence_system();
    } else {
        efa_cuda_flush_sq_wrs(qp);
    }
    result->qp_pc = qp->sq.wq.pc;
    const auto start = clock64();
    void* completion = nullptr;
    while ((completion =
                efa_cuda_cq_poll(cq, 0)) ==
           nullptr &&
           clock64() - start <
               descriptor.
                   completion_timeout_cycles) {
        __nanosleep(64);
    }
    if (completion == nullptr) {
        result->status = 1;
        auto* raw =
            efa_cuda_get_cqe(cq, 0);
        result->cq_flags = raw->flags;
        return;
    }
    result->vendor_error =
        efa_cuda_wc_read_vendor_err(
            completion);
    result->cq_flags =
        static_cast<
            efa_io_cdesc_common*>(
            completion)->flags;
    result->status =
        result->vendor_error == 0
            ? 0
            : 2;
    efa_cuda_cq_pop(cq, 1);
}

}  // namespace

extern "C" {

int aboleth_efa_gda_local_smoke(
    int left_gpu,
    int right_gpu,
    std::size_t bytes) {
    last_error.clear();
    void* source = nullptr;
    void* destination = nullptr;
    EfaGdaSmokeResult* device_result =
        nullptr;
    try {
        if (left_gpu < 0 ||
            right_gpu < 0 ||
            left_gpu == right_gpu ||
            bytes == 0 ||
            bytes >
                std::numeric_limits<
                    std::uint32_t>::max()) {
            throw_error(
                "invalid EFA GDA smoke arguments");
        }
        setenv(
            "ABOLETH_PROXY_PROVIDER",
            "efa",
            1);
        setenv(
            "ABOLETH_EFA_FABRIC",
            "efa-direct",
            1);
        auto left_runtime =
            create_shared_runtime(left_gpu);
        auto right_runtime =
            create_shared_runtime(right_gpu);
        cuda_check(
            cudaSetDevice(left_gpu),
            "cudaSetDevice(EFA GDA smoke source)");
        cuda_check(
            cudaMalloc(&source, bytes),
            "cudaMalloc(EFA GDA smoke source)");
        cuda_check(
            cudaMemset(source, 0xab, bytes),
            "cudaMemset(EFA GDA smoke source)");
        auto source_buffer =
            register_buffer(
                left_runtime,
                left_gpu,
                source,
                bytes,
                91,
                false);
        cuda_check(
            cudaSetDevice(right_gpu),
            "cudaSetDevice(EFA GDA smoke destination)");
        cuda_check(
            cudaMalloc(&destination, bytes),
            "cudaMalloc(EFA GDA smoke destination)");
        cuda_check(
            cudaMemset(destination, 0, bytes),
            "cudaMemset(EFA GDA smoke destination)");
        auto destination_buffer =
            register_buffer(
                right_runtime,
                right_gpu,
                destination,
                bytes,
                92,
                false);

        aboleth::abi::EfaGdaEndpointDescriptor
            left_descriptor{};
        aboleth::abi::EfaGdaEndpointDescriptor
            right_descriptor{};
        auto left_endpoint =
            create_efa_gda_endpoint(
                left_runtime,
                left_gpu,
                source_buffer->local_key,
                source_buffer->local_key,
                left_descriptor);
        auto right_endpoint =
            create_efa_gda_endpoint(
                right_runtime,
                right_gpu,
                destination_buffer->local_key,
                destination_buffer->local_key,
                right_descriptor);
        const bool debug =
            std::getenv(
                "ABOLETH_EFA_GDA_DEBUG") !=
            nullptr;
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA smoke endpoints-created\n");
            std::fflush(stderr);
        }
        connect_efa_gda_endpoint(
            *right_endpoint,
            left_endpoint->address.data(),
            left_endpoint->address_length,
            source_buffer->remote_key,
            source_buffer->remote_key,
            right_descriptor);
        connect_efa_gda_endpoint(
            *left_endpoint,
            right_endpoint->address.data(),
            right_endpoint->address_length,
            destination_buffer->remote_key,
            destination_buffer->remote_key,
            left_descriptor);
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA smoke endpoint-connected ah=%u qpn=%u qkey=%u lkey=%u rkey=%u\n",
                left_descriptor.peer_ah,
                left_descriptor.peer_qpn,
                left_descriptor.peer_qkey,
                left_descriptor.local_window_lkey,
                left_descriptor.remote_window_rkey);
            std::fflush(stderr);
        }

        cuda_check(
            cudaSetDevice(left_gpu),
            "cudaSetDevice(EFA GDA smoke launch)");
        EfaGdaSmokeResult result{};
        const bool direct_queue_arm =
            std::getenv(
                "ABOLETH_EFA_GDA_CPU_POST") !=
                nullptr ||
            std::getenv(
                "ABOLETH_EFA_GDA_STREAM_DB") !=
                nullptr ||
            std::getenv(
                "ABOLETH_EFA_GDA_MEMCPY_DB") !=
                nullptr;
        if (direct_queue_arm &&
            aboleth::runtime::resolve_staged_submit(
                left_runtime->efa->device_name)) {
            // The diagnostic arms write the adapter queue directly, which
            // would race the doorbell agent's producer accounting.
            throw_error(
                "the CPU-post/stream/memcpy probe arms need "
                "ABOLETH_EFA_SUBMIT=gpu");
        }
        if (direct_queue_arm) {
            const bool stream_doorbell =
                std::getenv(
                    "ABOLETH_EFA_GDA_STREAM_DB") !=
                nullptr;
            const bool memcpy_doorbell =
                std::getenv(
                    "ABOLETH_EFA_GDA_MEMCPY_DB") !=
                nullptr;
            efa_gda_cpu_post(
                *left_endpoint,
                left_descriptor,
                reinterpret_cast<
                    std::uint64_t>(source),
                reinterpret_cast<
                    std::uint64_t>(
                    destination),
                static_cast<std::uint32_t>(
                    bytes),
                !stream_doorbell &&
                    !memcpy_doorbell);
            if (stream_doorbell) {
                cudaStream_t stream{};
                cuda_check(
                    cudaStreamCreate(&stream),
                    "cudaStreamCreate(EFA GDA)");
                const auto stream_status =
                    dl::cuda().cuStreamWriteValue32(
                        reinterpret_cast<
                            CUstream>(stream),
                        static_cast<CUdeviceptr>(
                            reinterpret_cast<
                                std::uintptr_t>(
                                left_endpoint->
                                    sq_doorbell.
                                    device)),
                        1,
                        CU_STREAM_WRITE_VALUE_DEFAULT);
                if (stream_status !=
                    CUDA_SUCCESS) {
                    cudaStreamDestroy(stream);
                    throw_error(
                        "cuStreamWriteValue32(EFA GDA doorbell) failed");
                }
                cuda_check(
                    cudaStreamSynchronize(
                        stream),
                    "cudaStreamSynchronize(EFA GDA)");
                cudaStreamDestroy(stream);
            } else if (memcpy_doorbell) {
                const std::uint32_t value = 1;
                cuda_check(
                    cudaMemcpy(
                        left_endpoint->
                            sq_doorbell.device,
                        &value,
                        sizeof(value),
                        cudaMemcpyHostToDevice),
                    "cudaMemcpy(EFA GDA doorbell)");
            }
            efa_io_cdesc_common completion{};
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() <
                   deadline) {
                cuda_check(
                    cudaMemcpy(
                        &completion,
                        left_endpoint->
                            cq_memory.pointer,
                        sizeof(completion),
                        cudaMemcpyDeviceToHost),
                    "cudaMemcpy(EFA GDA CPU-post CQ)");
                if ((completion.flags & 1u) ==
                    1u) {
                    result.status =
                        completion.status == 0
                            ? 0
                            : 2;
                    result.vendor_error =
                        completion.status;
                    result.cq_flags =
                        completion.flags;
                    result.qp_pc = 1;
                    break;
                }
                std::this_thread::yield();
            }
            if ((completion.flags & 1u) ==
                0u) {
                result.status = 1;
            }
        } else {
            int kernel_mode = 0;
            if (std::getenv(
                    "ABOLETH_EFA_GDA_GPU_SQ_ONLY") !=
                nullptr) {
                kernel_mode = 1;
            } else if (std::getenv(
                           "ABOLETH_EFA_GDA_GPU_DB_ONLY") !=
                       nullptr) {
                kernel_mode = 2;
                efa_gda_cpu_post(
                    *left_endpoint,
                    left_descriptor,
                    reinterpret_cast<
                        std::uint64_t>(
                        source),
                    reinterpret_cast<
                        std::uint64_t>(
                        destination),
                    static_cast<
                        std::uint32_t>(
                        bytes),
                    false);
            }
            cuda_check(
                cudaMalloc(
                    &device_result,
                    sizeof(EfaGdaSmokeResult)),
                "cudaMalloc(EFA GDA smoke result)");
            cuda_check(
                cudaMemset(
                    device_result,
                    0,
                    sizeof(EfaGdaSmokeResult)),
                "cudaMemset(EFA GDA smoke result)");
            efa_gda_smoke_kernel<<<1, 1>>>(
                left_descriptor,
                reinterpret_cast<
                    std::uint64_t>(source),
                reinterpret_cast<
                    std::uint64_t>(
                    destination),
                static_cast<std::uint32_t>(
                    bytes),
                kernel_mode,
                device_result);
            if (debug) {
                std::fprintf(
                    stderr,
                    "EFA GDA smoke kernel-launched\n");
                std::fflush(stderr);
            }
            cuda_check(
                cudaGetLastError(),
                "efa_gda_smoke_kernel launch");
            cuda_check(
                cudaDeviceSynchronize(),
                "efa_gda_smoke_kernel synchronize");
            if (debug) {
                std::fprintf(
                    stderr,
                    "EFA GDA smoke kernel-synchronized\n");
                std::fflush(stderr);
            }
            cuda_check(
                cudaMemcpy(
                    &result,
                    device_result,
                    sizeof(result),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(EFA GDA smoke result)");
            cudaFree(device_result);
            device_result = nullptr;
            if (kernel_mode == 1) {
                *reinterpret_cast<
                    volatile std::uint32_t*>(
                    left_endpoint->
                        sq_attributes.
                        doorbell) = 1;
                asm volatile(
                    "sfence" ::: "memory");
                efa_io_cdesc_common
                    completion{};
                const auto deadline =
                    std::chrono::
                        steady_clock::now() +
                    std::chrono::seconds(5);
                while (
                    std::chrono::
                        steady_clock::now() <
                    deadline) {
                    cuda_check(
                        cudaMemcpy(
                            &completion,
                            left_endpoint->
                                cq_memory.pointer,
                            sizeof(completion),
                            cudaMemcpyDeviceToHost),
                        "cudaMemcpy(EFA GDA GPU-SQ CQ)");
                    if ((completion.flags &
                         1u) == 1u) {
                        result.status =
                            completion.status ==
                                    0
                                ? 0
                                : 2;
                        result.vendor_error =
                            completion.status;
                        result.cq_flags =
                            completion.flags;
                        result.qp_pc = 1;
                        break;
                    }
                    std::this_thread::yield();
                }
                if ((completion.flags & 1u) ==
                    0u) {
                    result.status = 1;
                }
            }
        }
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA smoke result status=%u vendor=%u flags=%u pc=%u\n",
                result.status,
                result.vendor_error,
                result.cq_flags,
                result.qp_pc);
            std::fflush(stderr);
        }

        cuda_check(
            cudaSetDevice(right_gpu),
            "cudaSetDevice(EFA GDA smoke verify)");
        const auto flush_status =
            cudaDeviceFlushGPUDirectRDMAWrites(
                cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                cudaFlushGPUDirectRDMAWritesToOwner);
        if (flush_status != cudaSuccess &&
            flush_status !=
                cudaErrorNotSupported) {
            cuda_check(
                flush_status,
                "cudaDeviceFlushGPUDirectRDMAWrites(EFA GDA smoke)");
        }
        std::vector<std::uint8_t>
            output(bytes);
        cuda_check(
            cudaMemcpy(
                output.data(),
                destination,
                bytes,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(EFA GDA smoke output)");
        const bool payload_ok =
            std::all_of(
                output.begin(),
                output.end(),
                [](std::uint8_t value) {
                    return value == 0xab;
                });
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA smoke payload-copied ok=%d\n",
                payload_ok ? 1 : 0);
            std::fflush(stderr);
        }
        std::fprintf(
            stderr,
            "EFA GDA smoke status=%u vendor=%u flags=%u pc=%u payload=%d\n",
            result.status,
            result.vendor_error,
            result.cq_flags,
            result.qp_pc,
            payload_ok ? 1 : 0);
        std::fflush(stderr);

        const int final_status =
            result.status == 0 &&
                    payload_ok
                ? 0
                : -1;
        left_endpoint.reset();
        right_endpoint.reset();
        source_buffer.reset();
        destination_buffer.reset();
        cuda_check(
            cudaSetDevice(left_gpu),
            "cudaSetDevice(EFA GDA smoke source free)");
        cudaFree(source);
        source = nullptr;
        cuda_check(
            cudaSetDevice(right_gpu),
            "cudaSetDevice(EFA GDA smoke destination free)");
        cudaFree(destination);
        destination = nullptr;
        return final_status;
    } catch (const std::exception& error) {
        if (device_result != nullptr ||
            source != nullptr) {
            cudaSetDevice(left_gpu);
            if (device_result != nullptr)
                cudaFree(device_result);
            if (source != nullptr)
                cudaFree(source);
        }
        if (destination != nullptr) {
            cudaSetDevice(right_gpu);
            cudaFree(destination);
        }
        last_error = error.what();
        return -1;
    }
}

}  // extern "C"

#endif  // ABOLETH_HAS_EFA
