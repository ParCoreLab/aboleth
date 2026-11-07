#pragma once

// The EFA control plane's shared internals.
//
// One libfabric fabric per opened device and the endpoints that hang off it:
// the proxy provider's, the GPU-initiated GDA one, and the Transport.EFA one
// the compiler backend targets. The provider-neutral runtime reaches all of
// them — the proxy datapath walks the fabric's memory map, client integrations
// own GDA endpoints, and the C ABI hands out Transport.EFA endpoints — so the
// types and the entry points that build them are declared here rather than
// inside either EFA translation unit.
//
// The bodies live in efa/aboleth_efa_control.cu (the fabric) and
// efa/aboleth_efa_endpoints.cu (the endpoints).
//
// Internal to the runtime; see aboleth_runtime_internal.h.

#include "../core/aboleth_doorbell_agent.h"
#include "../aboleth_runtime_internal.h"
#include "../proxy/aboleth_proxy.h"

#include "aboleth/Target/EfaAbi.h"
#include "aboleth/Target/EfaGdaAbi.h"
#include "aboleth/runtime_c.h"

#if ABOLETH_HAS_EFA

#include "../dl/shims.h"

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

// Only for the two queue-state types a GDA endpoint hands the device; the
// primitives that act on them are device code and stay out of this header.
#include "efa_cuda_dp_types.h"

#include <unistd.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

// Every runtime translation unit names the dlopen shims `dl::`, and so do the
// struct bodies below. Redeclaring an alias that already refers to the same
// namespace is allowed and changes no lookup, so naming it here keeps this
// header independent of where it is included.
namespace dl = aboleth::dl;

namespace aboleth::runtime {

struct EfaMemoryRecord {
    void* address{};
    std::size_t bytes{};
    void* descriptor{};
};

struct EfaProxyCqShard {
    fid_cq* tx{};
    fid_cq* rx{};
    std::atomic<std::uint64_t>
        failure_status{0};

    ~EfaProxyCqShard() {
        if (tx != nullptr) {
            fi_close(&tx->fid);
        }
        if (rx != nullptr) {
            fi_close(&rx->fid);
        }
    }
};

struct EfaFabricContext {
    fi_info* info{};
    fid_fabric* fabric{};
    fid_domain* domain{};
    fid_av* proxy_av{};
    fi_efa_ops_gda* gda_ops{};
    std::mutex proxy_shard_mutex;
    std::map<
        int,
        std::shared_ptr<EfaProxyCqShard>>
        proxy_shards;
    std::string device_name;
    std::string domain_name;
    std::string fabric_name;
    std::mutex memory_mutex;
    std::unordered_map<
        std::uint32_t,
        EfaMemoryRecord>
        memory;

    ~EfaFabricContext() {
        proxy_shards.clear();
        if (proxy_av != nullptr) {
            fi_close(&proxy_av->fid);
        }
        if (domain != nullptr) {
            fi_close(&domain->fid);
        }
        if (fabric != nullptr) {
            fi_close(&fabric->fid);
        }
        if (info != nullptr) {
            dl::fabric().fi_freeinfo(info);
        }
    }
};

struct EfaGdaMmioRegion {
    void* host{};
    void* device{};

    void map(
        void* pointer,
        std::size_t bytes,
        const char* operation) {
        if (host != nullptr ||
            pointer == nullptr ||
            bytes == 0) {
            throw_error(
                "invalid EFA GDA MMIO mapping");
        }
        const auto register_status =
            dl::cuda().cuMemHostRegister(
                pointer,
                bytes,
                CU_MEMHOSTREGISTER_IOMEMORY |
                    CU_MEMHOSTREGISTER_DEVICEMAP);
        if (register_status != CUDA_SUCCESS) {
            const char* error_name = nullptr;
            const char* error_text = nullptr;
            dl::cuda().cuGetErrorName(
                register_status,
                &error_name);
            dl::cuda().cuGetErrorString(
                register_status,
                &error_text);
            throw_error(
                std::string(
                    "cuMemHostRegister(") +
                operation + ") failed: " +
                (error_name == nullptr
                     ? "unknown"
                     : error_name) +
                " (" +
                (error_text == nullptr
                     ? "unknown"
                     : error_text) +
                ")");
        }
        CUdeviceptr mapped = 0;
        const auto map_status =
            dl::cuda().cuMemHostGetDevicePointer(
                &mapped,
                pointer,
                0);
        if (map_status != CUDA_SUCCESS) {
            dl::cuda().cuMemHostUnregister(pointer);
            throw_error(
                "cuMemHostGetDevicePointer(EFA GDA MMIO) failed");
        }
        host = pointer;
        device =
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(
                    mapped));
    }

    void reset() {
        if (host != nullptr) {
            dl::cuda().cuMemHostUnregister(host);
            host = nullptr;
            device = nullptr;
        }
    }

    ~EfaGdaMmioRegion() {
        reset();
    }
};

struct EfaGdaGpuMemory {
    void* pointer{};
    std::size_t bytes{};
    int dmabuf_fd{-1};

    void allocate(std::size_t requested_bytes) {
        if (pointer != nullptr ||
            requested_bytes == 0) {
            throw_error(
                "invalid EFA GDA GPU allocation");
        }
        CUdevice device = 0;
        if (dl::cuda().cuCtxGetDevice(&device) !=
            CUDA_SUCCESS) {
            throw_error(
                "cuCtxGetDevice(EFA GDA) failed");
        }
        CUmemAllocationProp properties{};
        properties.type =
            CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type =
            CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device;
        int gdr_vmm = 0;
        dl::cuda().cuDeviceGetAttribute(
            &gdr_vmm,
            CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
            device);
        if (gdr_vmm != 0) {
            properties.allocFlags.
                gpuDirectRDMACapable = 1;
        }
        std::size_t granularity = 0;
        if (dl::cuda().cuMemGetAllocationGranularity(
                &granularity,
                &properties,
                CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            CUDA_SUCCESS) {
            throw_error(
                "cuMemGetAllocationGranularity(EFA GDA) failed");
        }
        bytes =
            (requested_bytes +
             granularity - 1) /
            granularity *
            granularity;
        CUmemGenericAllocationHandle handle{};
        if (dl::cuda().cuMemCreate(
                &handle,
                bytes,
                &properties,
                0) != CUDA_SUCCESS) {
            throw_error(
                "cuMemCreate(EFA GDA) failed");
        }
        CUdeviceptr address = 0;
        if (dl::cuda().cuMemAddressReserve(
                &address,
                bytes,
                granularity,
                0,
                0) != CUDA_SUCCESS) {
            dl::cuda().cuMemRelease(handle);
            throw_error(
                "cuMemAddressReserve(EFA GDA) failed");
        }
        if (dl::cuda().cuMemMap(
                address,
                bytes,
                0,
                handle,
                0) != CUDA_SUCCESS) {
            dl::cuda().cuMemAddressFree(address, bytes);
            dl::cuda().cuMemRelease(handle);
            throw_error(
                "cuMemMap(EFA GDA) failed");
        }
        CUmemAccessDesc access{};
        access.location = properties.location;
        access.flags =
            CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (dl::cuda().cuMemSetAccess(
                address,
                bytes,
                &access,
                1) != CUDA_SUCCESS) {
            dl::cuda().cuMemUnmap(address, bytes);
            dl::cuda().cuMemAddressFree(address, bytes);
            dl::cuda().cuMemRelease(handle);
            throw_error(
                "cuMemSetAccess(EFA GDA) failed");
        }
        dl::cuda().cuMemRelease(handle);
        pointer =
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(
                    address));
        cuda_check(
            cudaMemset(pointer, 0, bytes),
            "cudaMemset(EFA GDA GPU allocation)");
        unsigned long long flags =
            CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
        auto export_status =
            dl::cuda().cuMemGetHandleForAddressRange(
                &dmabuf_fd,
                address,
                bytes,
                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                flags);
        if ((export_status ==
                 CUDA_ERROR_INVALID_VALUE ||
             export_status ==
                 CUDA_ERROR_NOT_SUPPORTED) &&
            flags != 0) {
            export_status =
                dl::cuda().cuMemGetHandleForAddressRange(
                    &dmabuf_fd,
                    address,
                    bytes,
                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                    0);
        }
        if (export_status != CUDA_SUCCESS) {
            reset();
            throw_error(
                "cuMemGetHandleForAddressRange(EFA GDA) failed");
        }
    }

    void reset() {
        const bool debug =
            std::getenv(
                "ABOLETH_EFA_GDA_DEBUG") !=
            nullptr;
        if (dmabuf_fd >= 0) {
            close(dmabuf_fd);
            dmabuf_fd = -1;
        }
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA GPU memory fd-closed pointer=%p bytes=%zu\n",
                pointer,
                bytes);
            std::fflush(stderr);
        }
        if (pointer != nullptr) {
            const auto address =
                static_cast<CUdeviceptr>(
                    reinterpret_cast<
                        std::uintptr_t>(
                        pointer));
            dl::cuda().cuMemUnmap(address, bytes);
            dl::cuda().cuMemAddressFree(address, bytes);
            pointer = nullptr;
            bytes = 0;
        }
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA GPU memory unmapped\n");
            std::fflush(stderr);
        }
    }

    ~EfaGdaGpuMemory() {
        reset();
    }
};

struct EfaGdaEndpointContext {
    std::shared_ptr<SharedRuntimeContext> runtime;
    int gpu_index{};
    fid_cq* cq{};
    fid_ep* endpoint{};
    fid_av* av{};
    fi_addr_t peer{FI_ADDR_NOTAVAIL};
    fi_efa_wq_attr sq_attributes{};
    fi_efa_wq_attr rq_attributes{};
    fi_efa_cq_attr cq_attributes{};
    EfaGdaGpuMemory cq_memory;
    EfaGdaMmioRegion sq_buffer;
    EfaGdaMmioRegion sq_doorbell;
    efa_cuda_qp* device_qp{};
    efa_cuda_cq* device_cq{};
    // Staged submission: the GPU posts into this host-pinned ring and
    // advances the producer cell; the doorbell agent copies the new slots
    // into the adapter's queue and rings the real doorbell.
    bool staged{};
    std::uint8_t* staging_ring{};
    std::uint32_t* staging_cells{};
    std::shared_ptr<aboleth::runtime::DoorbellAgent> agent;
    std::uint64_t agent_token{};
    std::array<
        std::uint8_t,
        kAbolethConnectionAddressBytes>
        address{};
    std::uint32_t address_length{};

    ~EfaGdaEndpointContext() {
        cudaSetDevice(gpu_index);
        const bool debug =
            std::getenv(
                "ABOLETH_EFA_GDA_DEBUG") !=
            nullptr;
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA destroy gpu=%d begin\n",
                gpu_index);
            std::fflush(stderr);
        }
        // The agent must stop touching the staging ring and the adapter
        // queue before either goes away.
        if (agent_token != 0 && agent != nullptr) {
            agent->remove(agent_token);
        }
        agent.reset();
        if (staging_ring != nullptr) {
            cudaFreeHost(staging_ring);
        }
        if (staging_cells != nullptr) {
            cudaFreeHost(staging_cells);
        }
        if (device_qp != nullptr) {
            cudaFree(device_qp);
        }
        if (device_cq != nullptr) {
            cudaFree(device_cq);
        }
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA destroy gpu=%d device-state-freed\n",
                gpu_index);
            std::fflush(stderr);
        }
        sq_doorbell.reset();
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA destroy gpu=%d doorbell-unmapped\n",
                gpu_index);
            std::fflush(stderr);
        }
        sq_buffer.reset();
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA destroy gpu=%d sq-unmapped\n",
                gpu_index);
            std::fflush(stderr);
        }
        if (endpoint != nullptr) {
            fi_close(&endpoint->fid);
            endpoint = nullptr;
        }
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA destroy gpu=%d endpoint-closed\n",
                gpu_index);
            std::fflush(stderr);
        }
        if (cq != nullptr) {
            fi_close(&cq->fid);
            cq = nullptr;
        }
        if (debug) {
            std::fprintf(
                stderr,
                "EFA GDA destroy gpu=%d cq-closed\n",
                gpu_index);
            std::fflush(stderr);
        }
    }
};

// Opens the EFA device closest to `gpu_index` — or the one `requested_name` /
// ABOLETH_EFA_DEVICE names — as a libfabric fabric, domain and shared address
// vector. ABOLETH_EFA_FABRIC picks between "efa" and "efa-direct"; only the
// latter carries the FI_EFA_GDA_OPS handle the GPU-initiated path needs.
std::shared_ptr<EfaFabricContext>
create_efa_fabric_context(
    int gpu_index,
    const char* requested_name);

// Builds the libfabric objects behind a proxy endpoint on the EFA provider —
// the endpoint, its shard's completion queues, the transmit counter — and
// records the address a peer connects to.
void create_efa_proxy_endpoint_resources(
    ProxyEndpointContext* endpoint);

// Inserts a remote EFA proxy address into the shared address vector.
void connect_efa_proxy_endpoint(
    ProxyEndpointContext* endpoint,
    const std::uint8_t* address,
    std::size_t address_bytes);

// Creates a GPU-initiated endpoint on an efa-direct runtime: a completion
// queue in GPU memory, a send queue either mapped into the GPU's address space
// or staged behind a doorbell agent, and the device-side queue state
// `descriptor` points the kernel at.
std::unique_ptr<EfaGdaEndpointContext>
create_efa_gda_endpoint(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    std::uint32_t local_window_lkey,
    std::uint32_t local_control_lkey,
    aboleth::abi::EfaGdaEndpointDescriptor&
        descriptor);

// Resolves a remote GDA address into the address handle, queue-pair number and
// queue key a kernel addresses it with, and fills them into `descriptor`.
void connect_efa_gda_endpoint(
    EfaGdaEndpointContext& endpoint,
    const std::uint8_t* address,
    std::size_t address_bytes,
    std::uint32_t remote_window_rkey,
    std::uint32_t remote_barrier_rkey,
    aboleth::abi::EfaGdaEndpointDescriptor&
        descriptor);

// Transport.EFA (kAbolethTransportEfa): the compiler backend's endpoint, a
// GDA-ops endpoint published through the nine-word EfaAbi descriptor that
// generated kernels read. Keys travel per operation from registered regions,
// so the GDA descriptor's window keys stay zero.
struct EfaAbiEndpointState {
    std::unique_ptr<EfaGdaEndpointContext> context;
    int gpu_index{};
    std::uint64_t* completion_state{};
    aboleth::abi::EfaEndpointDescriptor descriptor{};
    void* descriptor_device{};

    ~EfaAbiEndpointState() {
        if (completion_state != nullptr) {
            cudaSetDevice(gpu_index);
            cudaFree(completion_state);
        }
    }
};

// Creates a Transport.EFA endpoint and publishes its descriptor into
// `descriptor_device`, which must be at least `descriptor_bytes` wide.
std::unique_ptr<EfaAbiEndpointState>
create_efa_abi_endpoint(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    void* descriptor_device,
    std::size_t descriptor_bytes);

// Points a Transport.EFA endpoint at a peer and republishes its descriptor.
void connect_efa_abi_endpoint(
    EfaAbiEndpointState& endpoint,
    const std::uint8_t* address,
    std::size_t address_bytes);

} // namespace aboleth::runtime

#endif  // ABOLETH_HAS_EFA
