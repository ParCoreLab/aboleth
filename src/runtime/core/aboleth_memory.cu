// Registering memory with the provider, and the buffers the C ABI hands out.
//
// Two layers live here. register_memory() is the verbs primitive every other
// module registers through: a direct registration, a dmabuf fallback for the
// device pointers a driver will not pin, and the relaxed-ordering negotiation
// the two environment switches govern. Above it, register_buffer() is what a
// caller of the C ABI gets - one record per region, keyed for whichever
// provider is in play and holding the runtime alive - and the accessors at the
// end of the file are how that record is read back.
//
// The types are declared in aboleth_memory.h.

#include "aboleth_memory.h"

#include "../aboleth_runtime_common.h"
#include "../aboleth_runtime_internal.h"
#include "../efa/aboleth_efa_internal.h"
#include "aboleth_runtime.h"

#include "aboleth/runtime_c.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#if ABOLETH_HAS_EFA
// The EFA arm of register_buffer() is the only libfabric consumer here.
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#endif

#include "../dl/shims.h"

namespace dl = aboleth::dl;

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <unistd.h>

// aboleth::runtime is where the shared internals live (see
// aboleth_runtime_internal.h). The extern "C" arm below reaches them
// unqualified, as the runtime translation unit this code came from does.
using namespace aboleth::runtime;

namespace {

bool relaxed_ordering_enabled() {
    const char* configured =
        std::getenv(
            "ABOLETH_RELAXED_ORDERING");
    if (configured == nullptr ||
        configured[0] == '\0') {
        return true;
    }
    return std::strcmp(configured, "0") != 0 &&
        std::strcmp(configured, "false") != 0 &&
        std::strcmp(configured, "off") != 0;
}

bool relaxed_ordering_required() {
    const char* configured =
        std::getenv(
            "ABOLETH_REQUIRE_RELAXED_ORDERING");
    if (configured == nullptr ||
        configured[0] == '\0') {
        return false;
    }
    return std::strcmp(configured, "0") != 0 &&
        std::strcmp(configured, "false") != 0 &&
        std::strcmp(configured, "off") != 0;
}

bool sync_memops_enabled() {
    const char* configured =
        std::getenv(
            "ABOLETH_SYNC_MEMOPS");
    if (configured == nullptr ||
        configured[0] == '\0') {
        return true;
    }
    return std::strcmp(configured, "0") != 0 &&
        std::strcmp(configured, "false") != 0 &&
        std::strcmp(configured, "off") != 0;
}

// GPUDirect RDMA §2.7: the legacy peermem path promised the NIC that copy
// engine work against a registered allocation had landed before the BAR
// mapping was read, and made that promise implicitly. The dmabuf path makes
// no such promise, so an allocation the NIC gathers from has to ask for it
// by name or a stream-ordered copy can still be in flight when the payload
// is pulled. The attribute is a property of the CUDA allocation and not of
// the registration, so every device pointer we register gets it, whichever
// MR flavour ends up carrying it. Non-device pointers are left alone.
void enable_sync_memops(void* address) {
    if (!sync_memops_enabled()) {
        return;
    }
    const auto pointer =
        static_cast<CUdeviceptr>(
            reinterpret_cast<std::uintptr_t>(
                address));
    unsigned int memory_type = 0;
    if (dl::cuda().cuPointerGetAttribute(
            &memory_type,
            CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            pointer) != CUDA_SUCCESS ||
        memory_type != CU_MEMORYTYPE_DEVICE) {
        return;
    }
    unsigned int sync_memops = 1;
    const auto status =
        dl::cuda().cuPointerSetAttribute(
            &sync_memops,
            CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
            pointer);
    if (status != CUDA_SUCCESS) {
        std::fprintf(
            stderr,
            "aboleth: cuPointerSetAttribute(SYNC_MEMOPS) failed on %p (CUresult %d)\n",
            address,
            static_cast<int>(status));
        std::fflush(stderr);
    }
}

}  // namespace

namespace aboleth::runtime {

ibv_mr* register_memory(
    ibv_pd* pd,
    void* address,
    std::size_t bytes,
    int access,
    bool allow_relaxed,
    bool* relaxed) {
    if (relaxed != nullptr) {
        *relaxed = false;
    }
    enable_sync_memops(address);
    const auto register_dmabuf =
        [&](int requested_access) -> ibv_mr* {
            const auto page_size =
                static_cast<std::uintptr_t>(
                    sysconf(_SC_PAGESIZE));
            const auto start =
                reinterpret_cast<std::uintptr_t>(
                    address);
            const auto base =
                start & ~(page_size - 1);
            const auto end =
                (start + bytes + page_size - 1) &
                ~(page_size - 1);
            int fd = -1;
            const auto status =
                dl::cuda().cuMemGetHandleForAddressRange(
                    &fd,
                    static_cast<CUdeviceptr>(base),
                    end - base,
                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                    0);
            if (status != CUDA_SUCCESS) {
                return nullptr;
            }
            ibv_mr* mr =
                dl::verbs().ibv_reg_dmabuf_mr(
                    pd,
                    /*offset=*/0,
                    end - base,
                    base,
                    fd,
                    requested_access);
            close(fd);
            return mr;
        };
    const auto register_any =
        [&](int requested_access) -> ibv_mr* {
            ibv_mr* mr =
                dl::verbs().reg_mr(
                    pd,
                    address,
                    bytes,
                    requested_access);
            return mr != nullptr
                ? mr
                : register_dmabuf(
                      requested_access);
        };
    if (!allow_relaxed)
        return register_any(access);
    if (!relaxed_ordering_enabled()) {
        if (relaxed_ordering_required()) {
            throw_error(
                "relaxed ordering is required but disabled");
        }
        return register_any(access);
    }
    ibv_mr* mr =
        register_any(
            access |
            IBV_ACCESS_RELAXED_ORDERING);
    if (mr != nullptr) {
        if (relaxed != nullptr) {
            *relaxed = true;
        }
        return mr;
    }
    if (relaxed_ordering_required()) {
        throw_error(
            "IBV_ACCESS_RELAXED_ORDERING registration failed");
    }
    return register_any(access);
}

RegisteredBufferContext::~RegisteredBufferContext() {
#if ABOLETH_HAS_EFA
    if (efa_mr != nullptr) {
        {
            std::lock_guard<std::mutex> lock(
                runtime->efa->memory_mutex);
            runtime->efa->memory.erase(
                local_key);
        }
        fi_close(&efa_mr->fid);
    }
#endif
    if (mr != nullptr) {
        dl::verbs().ibv_dereg_mr(mr);
    }
}

std::shared_ptr<RegisteredBufferContext>
register_buffer(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    void* address,
    std::size_t bytes,
    std::uint64_t generation,
    bool allow_relaxed) {
    if (address == nullptr || bytes == 0) {
        throw_error("registered buffer must be non-empty");
    }
    auto buffer =
        std::make_shared<RegisteredBufferContext>();
    buffer->runtime = runtime;
    buffer->address = address;
    buffer->bytes = bytes;
    buffer->gpu_index = gpu_index;
    buffer->generation = generation;
#if ABOLETH_HAS_EFA
    if (runtime->efa != nullptr) {
        cuda_check(
            cudaSetDevice(gpu_index),
            "cudaSetDevice(EFA registration)");
        enable_sync_memops(address);
        iovec local_iov{};
        local_iov.iov_base = address;
        local_iov.iov_len = bytes;
        fi_mr_attr attributes{};
        attributes.mr_iov = &local_iov;
        attributes.iov_count = 1;
        attributes.access =
            FI_READ |
            FI_WRITE |
            FI_REMOTE_READ |
            FI_REMOTE_WRITE;
        attributes.iface = FI_HMEM_CUDA;
        attributes.device.cuda = gpu_index;
        fabric_check(
            fi_mr_regattr(
                runtime->efa->domain,
                &attributes,
                0,
                &buffer->efa_mr),
            "fi_mr_regattr(EFA CUDA)");
        const std::uint64_t key =
            fi_mr_key(buffer->efa_mr);
        const std::uint64_t local_key =
            runtime->efa->gda_ops != nullptr
                ? runtime->efa->gda_ops->
                      get_mr_lkey(
                          buffer->efa_mr)
                : key;
        if (key >
                std::numeric_limits<
                    std::uint32_t>::max() ||
            local_key >
            std::numeric_limits<
                std::uint32_t>::max()) {
            throw_error(
                "EFA memory key exceeds proxy ABI width");
        }
        if (runtime->efa->gda_ops != nullptr &&
            local_key > 0x00ffffffULL) {
            throw_error(
                "EFA GDA local key exceeds WQE width");
        }
        buffer->local_key =
            static_cast<std::uint32_t>(
                local_key);
        buffer->remote_key =
            static_cast<std::uint32_t>(key);
        {
            std::lock_guard<std::mutex> lock(
                runtime->efa->memory_mutex);
            const auto inserted =
                runtime->efa->memory.emplace(
                    buffer->local_key,
                    EfaMemoryRecord{
                        address,
                        bytes,
                        fi_mr_desc(
                            buffer->efa_mr)});
            if (!inserted.second) {
                throw_error(
                    "duplicate EFA memory key");
            }
        }
        if (std::getenv(
                "ABOLETH_PROXY_DEBUG") !=
            nullptr) {
            std::fprintf(
                stderr,
                "aboleth EFA MR gpu=%d device=%s base=%p bytes=%zu key=%u\n",
                gpu_index,
                runtime->device_name.c_str(),
                address,
                bytes,
                buffer->remote_key);
            std::fflush(stderr);
        }
        return buffer;
    }
#endif
    bool relaxed = false;
    buffer->mr =
        register_memory(
            runtime->pd,
            address,
            bytes,
            IBV_ACCESS_LOCAL_WRITE |
                IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ |
                IBV_ACCESS_REMOTE_ATOMIC,
            allow_relaxed,
            &relaxed);
    if (buffer->mr == nullptr) {
        throw_error("ibv_reg_mr failed");
    }
    buffer->strictly_ordered = !relaxed;
    buffer->local_key = buffer->mr->lkey;
    buffer->remote_key = buffer->mr->rkey;
    return buffer;
}

}  // namespace aboleth::runtime

extern "C" {

void* aboleth_runtime_register_buffer(
    void* runtime_handle,
    int gpu_index,
    void* address,
    std::size_t bytes,
    std::uint64_t generation) {
    last_error.clear();
    try {
        auto* runtime =
            static_cast<RuntimeHandle*>(
                runtime_handle);
        if (runtime == nullptr) {
            throw_error("null runtime handle");
        }
        if (runtime->runtime->gpu_index >= 0 &&
            runtime->runtime->gpu_index !=
                gpu_index) {
            throw_error(
                "buffer GPU does not match the rank-local runtime");
        }
        auto handle =
            std::make_unique<BufferHandle>();
        handle->buffer = register_buffer(
            runtime->runtime,
            gpu_index,
            address,
            bytes,
            generation);
        return handle.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

int aboleth_runtime_unregister_buffer(
    void* buffer_handle) {
    last_error.clear();
    try {
        delete static_cast<BufferHandle*>(
            buffer_handle);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

std::uint64_t aboleth_runtime_buffer_address(
    void* buffer_handle) {
    auto* handle =
        static_cast<BufferHandle*>(
            buffer_handle);
    if (handle == nullptr) {
        return 0;
    }
    return reinterpret_cast<std::uint64_t>(
        handle->buffer->address);
}

std::uint64_t aboleth_runtime_buffer_length(
    void* buffer_handle) {
    auto* handle =
        static_cast<BufferHandle*>(
            buffer_handle);
    if (handle == nullptr) {
        return 0;
    }
    return handle->buffer->bytes;
}

std::uint32_t aboleth_runtime_buffer_lkey(
    void* buffer_handle) {
    auto* handle =
        static_cast<BufferHandle*>(
            buffer_handle);
    if (handle == nullptr) {
        return 0;
    }
    return handle->buffer->local_key;
}

std::uint32_t aboleth_runtime_buffer_rkey(
    void* buffer_handle) {
    auto* handle =
        static_cast<BufferHandle*>(
            buffer_handle);
    if (handle == nullptr) {
        return 0;
    }
    return handle->buffer->remote_key;
}

int aboleth_runtime_buffer_device(
    void* buffer_handle) {
    auto* handle =
        static_cast<BufferHandle*>(
            buffer_handle);
    if (handle == nullptr) {
        return -1;
    }
    return handle->buffer->gpu_index;
}

int aboleth_runtime_buffer_strictly_ordered(
    void* buffer_handle) {
    auto* handle =
        static_cast<BufferHandle*>(
            buffer_handle);
    if (handle == nullptr) {
        return 0;
    }
    return handle->buffer->strictly_ordered ? 1 : 0;
}

std::uint64_t aboleth_runtime_buffer_generation(
    void* buffer_handle) {
    auto* handle =
        static_cast<BufferHandle*>(
            buffer_handle);
    if (handle == nullptr) {
        return 0;
    }
    return handle->buffer->generation;
}

}  // extern "C"
