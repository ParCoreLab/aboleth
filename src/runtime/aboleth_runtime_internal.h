#pragma once

// Internals shared by the host runtime's translation units.
//
// The provider-neutral runtime is one shared object assembled from several
// pieces — device and connection setup, the proxy progress thread, and the
// EFA control plane. Everything those pieces have to
// agree on lives here: the open-device context they all hold, the error
// plumbing behind the C ABI, the queue and timeout constants baked into both
// the descriptors and the kernels, and the two verbs helpers every module
// calls.
//
// Internal. This header is not installed and is deliberately not part of
// include/aboleth/: nothing declared here is a stable interface, and the ABI
// the outside world sees is runtime_c.h.

#include "aboleth_runtime_common.h"

#include "aboleth/Target/RdmaAbi.h"
#include "aboleth/runtime_c.h"

#include "dl/shims.h"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace aboleth::runtime {

// Send-queue depth every RC queue pair is created with, and the completion
// queue depth that goes with it.
inline constexpr std::uint32_t kMaximumTransfers = 256;

// How long a host-side wait for a completion may run before it is called a
// failure, and how long a kernel is told to wait in the descriptors.
inline constexpr std::uint32_t kTimeoutSeconds = 5;

// The completion cell a kernel polls: a consumer index and the scratch the
// fetch-add lands in, allocated as one device object.
inline constexpr std::size_t kCompletionStateWords = 4;
inline constexpr std::size_t kCompletionAllocationBytes =
    aboleth::abi::kRdmaAtomicResultOffset +
    sizeof(std::uint64_t);

// Inline-data capacity requested when the queue pair is created. The adapter
// may grant less; what it granted is what the descriptor reports.
inline constexpr std::uint32_t kProxyInlineBytes = 128;
inline constexpr std::uint32_t kRdmaInlineBytes = 256;

#if !ABOLETH_HAS_EFA
// What every EFA entry point reports when the provider was compiled out.
inline constexpr const char* kEfaUnavailable =
    "EFA support was not compiled into this runtime";
#endif

// The message behind aboleth_runtime_last_error() and its siblings. Every C
// entry point catches, records here, and returns a status; the string is only
// valid on the thread that made the failing call.
extern thread_local std::string last_error;

// throw_error(), cuda_check() and the GPU/EFA topology helpers come from
// aboleth_runtime_common.h, which this header includes.

// ibverbs reports failure as a nonzero return with the reason in errno.
void verbs_check(
    int status,
    const char* operation);

#if ABOLETH_HAS_EFA
// libfabric reports failure as a negated errno in the return value.
void fabric_check(
    ssize_t status,
    const char* operation);
#endif

// Reads a non-negative integer out of the environment, or returns
// `default_value` when the variable is unset or empty.
int configured_index(
    const char* name,
    int default_value);

// The 64-bit cells the GPU and the host share are described by the ABI rather
// than by std::atomic, so the host reaches them through plain acquire/release
// builtins.
std::uint64_t atomic_load_acquire(
    const std::uint64_t* pointer);
void atomic_store_release(
    std::uint64_t* pointer,
    std::uint64_t value);

// One end of an RC path as the local port describes it. Both halves of a
// connection are needed to decide between LID and GRH addressing, so this
// travels alongside the remote's connection info.
struct PortAddress {
    std::uint8_t port_num{};
    std::uint8_t link_layer{};
    std::uint16_t lid{};
    std::uint32_t gid_index{};
    ibv_gid gid{};
    ibv_mtu active_mtu{};
};

// Which provider carries a proxy endpoint's traffic. The values are the ones
// the C ABI publishes, so a kind can be handed across it unchanged.
enum class ProxyProviderKind : std::uint32_t {
    VerbsRc = kAbolethProxyProviderVerbsRc,
    EfaOfi = kAbolethProxyProviderEfaOfi,
};

// One cudaHostRegister'd span and the device pointer it maps to.
struct RegisteredMapping {
    void* host_base{};
    std::size_t bytes{};
    void* device_base{};
    unsigned int flags{};
    std::size_t references{};
};

// Reference-counted host registrations, page-aligned and deduplicated.
//
// Several endpoints on one device end up mapping overlapping pieces of the
// same provider allocation — doorbell records and blueflame pages in
// particular — and CUDA refuses a second registration of a page it already
// owns. The registry hands out the existing device pointer instead, and
// unregisters only when the last reference goes away.
class MappingRegistry {
public:
    void* map(
        void* pointer,
        std::size_t bytes,
        unsigned int flags) {
        std::lock_guard<std::mutex> lock(mutex_);
        flags |= cudaHostRegisterPortable;
        const auto page_size =
            static_cast<std::size_t>(
                sysconf(_SC_PAGESIZE));
        const auto start =
            reinterpret_cast<std::uintptr_t>(
                pointer);
        const auto base =
            start & ~(page_size - 1);
        const auto end =
            (start + bytes + page_size - 1) &
            ~(page_size - 1);

        for (auto& mapping : mappings_) {
            const auto mapping_base =
                reinterpret_cast<std::uintptr_t>(
                    mapping.host_base);
            const auto mapping_end =
                mapping_base + mapping.bytes;
            const bool overlaps =
                base < mapping_end &&
                end > mapping_base;
            if (!overlaps) {
                continue;
            }
            if (flags == mapping.flags &&
                base >= mapping_base &&
                end <= mapping_end) {
                ++mapping.references;
                return reinterpret_cast<void*>(
                    reinterpret_cast<std::uintptr_t>(
                        mapping.device_base) +
                    start -
                    mapping_base);
            }
            throw_error(
                "incompatible overlapping CUDA host mappings");
        }

        RegisteredMapping mapping{
            reinterpret_cast<void*>(base),
            end - base,
            nullptr,
            flags,
            1,
        };
        cuda_check(
            cudaHostRegister(
                mapping.host_base,
                mapping.bytes,
                flags),
            "cudaHostRegister");
        cuda_check(
            cudaHostGetDevicePointer(
                &mapping.device_base,
                mapping.host_base,
                0),
            "cudaHostGetDevicePointer");
        mappings_.push_back(mapping);
        return reinterpret_cast<void*>(
            reinterpret_cast<std::uintptr_t>(
                mapping.device_base) +
            start -
            base);
    }

    void unmap(
        void* pointer,
        std::size_t bytes,
        unsigned int flags) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        flags |= cudaHostRegisterPortable;
        const auto page_size =
            static_cast<std::size_t>(
                sysconf(_SC_PAGESIZE));
        const auto start =
            reinterpret_cast<std::uintptr_t>(
                pointer);
        const auto base =
            start & ~(page_size - 1);
        const auto end =
            (start + bytes + page_size - 1) &
            ~(page_size - 1);
        for (auto mapping = mappings_.begin();
             mapping != mappings_.end();
             ++mapping) {
            const auto mapping_base =
                reinterpret_cast<std::uintptr_t>(
                    mapping->host_base);
            const auto mapping_end =
                mapping_base + mapping->bytes;
            if (mapping->flags != flags ||
                base < mapping_base ||
                end > mapping_end) {
                continue;
            }
            if (mapping->references > 1) {
                --mapping->references;
            } else {
                cudaHostUnregister(
                    mapping->host_base);
                mappings_.erase(mapping);
            }
            return;
        }
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

private:
    std::mutex mutex_;
    std::vector<RegisteredMapping> mappings_;
};

#if ABOLETH_HAS_EFA
// Defined by the EFA control plane; held here only by shared_ptr, so the
// declaration is enough.
struct EfaFabricContext;
#endif

// The open device every endpoint on this process hangs off: one ibv_context,
// one protection domain, one port, one host-mapping registry. Endpoints,
// registered buffers and client integrations all keep it alive by shared_ptr, so
// it outlives whichever of them was created first.
struct SharedRuntimeContext {
    ibv_device** devices{};
    ibv_context* verbs_context{};
    ibv_pd* pd{};
    int gpu_index{-1};
    std::string device_name;
    PortAddress port{};
    ibv_mtu path_mtu{};
    MappingRegistry mappings;
    ProxyProviderKind proxy_provider{
        ProxyProviderKind::VerbsRc};
#if ABOLETH_HAS_EFA
    std::shared_ptr<EfaFabricContext> efa;
#endif

    ~SharedRuntimeContext() {
        mappings.clear();
        if (pd != nullptr) {
            aboleth::dl::verbs().ibv_dealloc_pd(pd);
        }
        if (verbs_context != nullptr) {
            aboleth::dl::verbs().ibv_close_device(verbs_context);
        }
        if (devices != nullptr) {
            aboleth::dl::verbs().ibv_free_device_list(devices);
        }
    }
};

} // namespace aboleth::runtime
