#pragma once

#include <cstddef>
#include <cstdint>

// One registry of transport capabilities, shared by the scheduling passes, the
// host runtime, and client integration layers. Passes above Queue IR ask
// what a transport can do instead of testing which transport it is, so adding a
// backend means adding a row here rather than a branch in every planner.
//
// This header stays free of LLVM, Triton, and CUDA runtime dependencies so the
// Triton-free `AbolethScheduling` library, the `.cu` runtime translation units,
// and transport adapters can all include it.

namespace aboleth::abi {

#if defined(__CUDACC__)
#define ABOLETH_CAPS_HOST_DEVICE __host__ __device__
#else
#define ABOLETH_CAPS_HOST_DEVICE
#endif

// How a sender learns that its own submissions retired.
enum class TransportCompletionModel : std::uint32_t {
    // Completion is established by program order alone; no queue to poll.
    None = 0,
    // Ordered queue: one completion entry retires every submission before it,
    // so polling the last entry of a batch retires the batch.
    OrderedCounter = 1,
    // Unordered queue: completions may arrive in any order, so a batch retires
    // only once the number of completions consumed reaches its length.
    UnorderedCount = 2,
};

// How a submission is announced to the device that executes it.
enum class TransportDoorbellModel : std::uint32_t {
    None = 0,
    // mlx5 BlueFlame/UAR register written with a single ordered 64-bit store.
    BlueFlame = 1,
    // Host-visible request ring consumed by a progress thread.
    HostRing = 2,
    // Ordinary doorbell register published after the queue entries.
    Doorbell = 3,
};

struct TransportCaps {
    const char *name;

    // The submitting agent is the GPU kernel itself rather than a host thread.
    bool gpu_initiated;
    // Remote memory is reachable with ordinary loads and stores, so transfers
    // need no queue, doorbell, or completion at all.
    bool load_store_addressable;
    // Submissions on one endpoint are delivered in submission order. Strong
    // publication may then rely on a later command being observed after
    // payload ordered before it.
    bool ordered_delivery;
    // Remote atomics (compare-and-swap) are available, which lets a signal be
    // both an ordering point and a checked epoch update.
    bool native_atomics;
    // A write can carry a fence that holds it until every earlier command on
    // its queue has completed. With ordered delivery this is an unchecked
    // ordering point, legal for publication only where the receiver's
    // registrations are strictly ordered (relaxed-ordering payload writes may
    // otherwise land after it).
    bool fenced_write;
    // Write-with-immediate delivers a receiver-visible completion carrying a
    // small value alongside the payload write.
    bool imm_data;
    // Outstanding submissions are bounded by a queue the compiler must respect
    // when planning how long source storage stays live.
    bool queue_capacity_limited;
    // One completion request can cover a whole batch. When false, every
    // submission produces its own completion and polling cost scales with the
    // number of commands rather than the number of batches.
    bool selective_completion;

    // Largest payload that may travel inside a command instead of referencing
    // registered memory. Zero when the transport has no inline form.
    std::uint32_t max_inline_bytes;

    // Iterations of a phase that may retire together when the compiler is left
    // to choose. Zero means the whole phase may retire once, which is the
    // right answer for transports with no queue to drain.
    std::uint32_t default_completion_interval;

    TransportCompletionModel completion_model;
    TransportDoorbellModel doorbell_model;

    // Byte-aware costing inputs. `path_mtu` is the packetization unit on the
    // wire; zero means the caller supplies one. `max_data_queues` bounds the
    // queue-count search, with zero meaning the search is bounded by the
    // resources the runtime actually allocated.
    std::uint32_t path_mtu;
    std::uint32_t max_data_queues;
};

inline constexpr TransportCaps kTransportCaps[] = {
    // Peer memory over CUDA IPC/NVLink. No queue, no completion, and
    // publication is an ordinary release store.
    TransportCaps{
        /*name=*/"p2p",
        /*gpu_initiated=*/true,
        /*load_store_addressable=*/true,
        /*ordered_delivery=*/true,
        /*native_atomics=*/true,
        /*fenced_write=*/false,
        /*imm_data=*/false,
        /*queue_capacity_limited=*/false,
        /*selective_completion=*/true,
        /*max_inline_bytes=*/0,
        /*default_completion_interval=*/0,
        /*completion_model=*/TransportCompletionModel::None,
        /*doorbell_model=*/TransportDoorbellModel::None,
        /*path_mtu=*/0,
        /*max_data_queues=*/0,
    },
    // Compiler-formed batches published to a host progress thread. The ring
    // applies its own backpressure, so the compiler does not bound completion
    // by queue capacity here.
    TransportCaps{
        /*name=*/"proxy",
        /*gpu_initiated=*/false,
        /*load_store_addressable=*/false,
        /*ordered_delivery=*/true,
        /*native_atomics=*/true,
        /*fenced_write=*/false,
        /*imm_data=*/false,
        /*queue_capacity_limited=*/false,
        /*selective_completion=*/true,
        /*max_inline_bytes=*/8,
        /*default_completion_interval=*/8,
        /*completion_model=*/TransportCompletionModel::OrderedCounter,
        /*doorbell_model=*/TransportDoorbellModel::HostRing,
        /*path_mtu=*/4096,
        /*max_data_queues=*/0,
    },
    // Direct mlx5: GPU-built WQEs, BlueFlame doorbell, ordered RC delivery,
    // responder atomics for strong publication.
    TransportCaps{
        /*name=*/"rdma",
        /*gpu_initiated=*/true,
        /*load_store_addressable=*/false,
        /*ordered_delivery=*/true,
        /*native_atomics=*/true,
        /*fenced_write=*/true,
        /*imm_data=*/false,
        /*queue_capacity_limited=*/true,
        /*selective_completion=*/true,
        /*max_inline_bytes=*/28,
        /*default_completion_interval=*/16,
        /*completion_model=*/TransportCompletionModel::OrderedCounter,
        /*doorbell_model=*/TransportDoorbellModel::BlueFlame,
        /*path_mtu=*/4096,
        /*max_data_queues=*/0,
    },
    // Ionic direct RC: GPU-built descriptors and MMIO, cumulative CCQE.
    // A fenced write is legal only with proven strict receiver registrations;
    // unknown registrations retain the completed-prefix publication fallback.
    TransportCaps{
        "ionic", true, false, true, false, true, false, true, false,
        16, 8, TransportCompletionModel::OrderedCounter,
        TransportDoorbellModel::Doorbell, 4096, 0,
    },
    // AWS EFA through efa-dp-direct: GPU-posted SRD work requests. SRD is
    // reliable but unordered and offers no remote atomics, so strong
    // publication cannot ride on same-queue ordering and every work request
    // carries its own completion.
    TransportCaps{
        /*name=*/"efa",
        /*gpu_initiated=*/true,
        /*load_store_addressable=*/false,
        /*ordered_delivery=*/false,
        /*native_atomics=*/false,
        /*fenced_write=*/false,
        /*imm_data=*/true,
        /*queue_capacity_limited=*/true,
        /*selective_completion=*/false,
        // EFA inlines only SENDs; SRD RDMA writes carry no inline data.
        /*max_inline_bytes=*/0,
        /*default_completion_interval=*/16,
        /*completion_model=*/TransportCompletionModel::UnorderedCount,
        /*doorbell_model=*/TransportDoorbellModel::Doorbell,
        /*path_mtu=*/4096,
        /*max_data_queues=*/0,
    },
};

inline constexpr std::size_t kTransportCapsCount =
    sizeof(kTransportCaps) / sizeof(kTransportCaps[0]);

ABOLETH_CAPS_HOST_DEVICE
inline constexpr bool transport_name_equals(
    const char *candidate,
    const char *name,
    std::size_t length) {
    for (std::size_t index = 0; index < length; ++index) {
        if (candidate[index] == '\0' ||
            candidate[index] != name[index]) {
            return false;
        }
    }
    return candidate[length] == '\0';
}

// Names arrive from MLIR string attributes, which are not null terminated.
ABOLETH_CAPS_HOST_DEVICE
inline constexpr const TransportCaps *lookup_transport_caps(
    const char *name,
    std::size_t length) {
    for (std::size_t index = 0; index < kTransportCapsCount;
         ++index) {
        if (transport_name_equals(
                kTransportCaps[index].name, name, length)) {
            return &kTransportCaps[index];
        }
    }
    return nullptr;
}

ABOLETH_CAPS_HOST_DEVICE
inline constexpr const TransportCaps *lookup_transport_caps(
    const char *name) {
    std::size_t length = 0;
    while (name[length] != '\0') {
        ++length;
    }
    return lookup_transport_caps(name, length);
}

// Byte-aware costing inputs, so the planner scores a candidate queue count the
// same way for every transport while still reflecting how that transport
// packetizes payload and pays for completion.
struct TransportCostParams {
    std::uint32_t path_mtu;
    // Completion entries consumed per submitted command. Selective completion
    // amortizes this over a batch and therefore contributes zero here.
    std::uint32_t completion_units_per_command;
    std::uint32_t max_data_queues;
};

ABOLETH_CAPS_HOST_DEVICE
inline constexpr TransportCostParams transport_cost_params(
    const TransportCaps &caps) {
    return TransportCostParams{
        caps.path_mtu,
        caps.selective_completion ? 0u : 1u,
        caps.max_data_queues,
    };
}

ABOLETH_CAPS_HOST_DEVICE
inline constexpr TransportCostParams default_transport_cost_params() {
    return transport_cost_params(kTransportCaps[2]);
}

static_assert(
    lookup_transport_caps("rdma") != nullptr &&
    lookup_transport_caps("rdma")->ordered_delivery &&
    lookup_transport_caps("rdma")->native_atomics);
static_assert(
    lookup_transport_caps("efa") != nullptr &&
    !lookup_transport_caps("efa")->ordered_delivery &&
    !lookup_transport_caps("efa")->native_atomics &&
    lookup_transport_caps("efa")->imm_data);
static_assert(lookup_transport_caps("p2p") != nullptr);
static_assert(lookup_transport_caps("proxy") != nullptr);
static_assert(lookup_transport_caps("gin") == nullptr);
static_assert(lookup_transport_caps("rdm", 3) == nullptr);
static_assert(lookup_transport_caps("rdmax", 4) != nullptr);
static_assert(
    default_transport_cost_params().path_mtu == 4096 &&
    default_transport_cost_params().completion_units_per_command == 0);

#undef ABOLETH_CAPS_HOST_DEVICE

} // namespace aboleth::abi
