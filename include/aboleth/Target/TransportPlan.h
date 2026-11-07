#pragma once

#include <cstdint>
#include <type_traits>

#include "aboleth/Target/TransportCaps.h"

namespace aboleth::abi {

#if defined(__CUDACC__)
#define ABOLETH_TRANSPORT_HOST_DEVICE __host__ __device__
#else
#define ABOLETH_TRANSPORT_HOST_DEVICE
#endif

inline constexpr std::uint32_t
    kTransportPhasePlanVersion = 4;
inline constexpr std::uint32_t
    kTransportNoPhase = 0xffffffffu;

enum class TransportTopologyKind : std::uint32_t {
    AllNIC = 0,
    Mixed = 1,
    AllP2P = 2,
};

enum class TransportQp0Role : std::uint32_t {
    Data = 0,
    Control = 1,
};

// How a strong publication makes payload visible before its epoch.
enum class TransportPublicationMode : std::uint32_t {
    // Ordered queue: the signal cannot overtake payload posted before it, so a
    // responder atomic on the same queue is both the ordering point and a
    // checked epoch update.
    SameQueueStrong = 0,
    // Peer memory: a release store orders prior stores.
    P2PRelease = 1,
    GinSignal = 2,
    // The sender retires its own payload completions before
    // posting the signal, so placement precedes the epoch.
    FenceThenSignal = 3,
    // Unordered queue with immediate data: payload arrives as
    // write-with-immediate and the receiver advances the epoch once it has
    // counted the writes belonging to that publication.
    ImmCounting = 4,
    // Ordered queue whose receiver registrations are all strictly ordered: a
    // fenced write is the ordering point. It does not check the predecessor
    // epoch.
    SameQueueFencedWrite = 5,
};

enum class TransportBarrierKind : std::uint32_t {
    None = 0,
    Transport = 1,
    NativeP2P = 2,
};

enum class TransportEffectPhase : std::uint32_t {
    Dispatch = 0,
    Experts = 1,
    Combine = 2,
    NextPack = 3,
    Count = 4,
};

inline constexpr std::uint32_t
    kTransportPlanFirstUse = 1u << 0;
inline constexpr std::uint32_t
    kTransportPlanReuseSafe = 1u << 1;
inline constexpr std::uint32_t
    kTransportPlanQpExclusive = 1u << 2;
inline constexpr std::uint32_t
    kTransportPlanDrainAtEpilogue = 1u << 3;
inline constexpr std::uint32_t
    kTransportPlanGroupedCompletion = 1u << 4;
inline constexpr std::uint32_t
    kTransportPlanCompletionLeader = 1u << 5;

// The publication a transport is capable of, rather than the one a particular
// backend prefers. `prefer_imm` selects between the two unordered schemes for
// transports that can do either. `strict_registrations` states that every
// receiver registration is strictly ordered; only then may a fenced write
// replace the responder atomic, because relaxed-ordering payload writes are
// not known to stay behind a fence, while an atomic's read flushes them.
ABOLETH_TRANSPORT_HOST_DEVICE
inline constexpr TransportPublicationMode legal_strong_publication(
    const TransportCaps &caps,
    bool prefer_imm,
    bool strict_registrations = false) {
    if (caps.load_store_addressable) {
        return TransportPublicationMode::P2PRelease;
    }
    if (caps.ordered_delivery) {
        if (strict_registrations && caps.fenced_write)
            return TransportPublicationMode::SameQueueFencedWrite;
        return caps.native_atomics ? TransportPublicationMode::SameQueueStrong
                                   : TransportPublicationMode::FenceThenSignal;
    }
    return prefer_imm && caps.imm_data
        ? TransportPublicationMode::ImmCounting
        : TransportPublicationMode::FenceThenSignal;
}

// A checked compare-and-swap detects a lost or duplicated publication in
// hardware. Transports without remote atomics still publish monotonically
// because one sender owns each signal cell, but they cannot verify the
// predecessor.
ABOLETH_TRANSPORT_HOST_DEVICE
inline constexpr bool publication_is_checked(
    const TransportCaps &caps) {
    return caps.native_atomics && !caps.load_store_addressable;
}

static_assert(
    legal_strong_publication(
        *lookup_transport_caps("rdma"), false) ==
    TransportPublicationMode::SameQueueStrong);
static_assert(
    legal_strong_publication(
        *lookup_transport_caps("rdma"), false, true) ==
    TransportPublicationMode::SameQueueFencedWrite);
static_assert(
    legal_strong_publication(
        *lookup_transport_caps("proxy"), false, true) ==
    TransportPublicationMode::SameQueueStrong);
static_assert(
    legal_strong_publication(
        *lookup_transport_caps("p2p"), true) ==
    TransportPublicationMode::P2PRelease);
static_assert(
    legal_strong_publication(
        *lookup_transport_caps("efa"), false) ==
    TransportPublicationMode::FenceThenSignal);
static_assert(
    legal_strong_publication(
        *lookup_transport_caps("ionic"), false) ==
    TransportPublicationMode::FenceThenSignal);
static_assert(
    legal_strong_publication(
        *lookup_transport_caps("efa"), true) ==
    TransportPublicationMode::ImmCounting);
static_assert(
    publication_is_checked(*lookup_transport_caps("rdma")));
static_assert(
    !publication_is_checked(*lookup_transport_caps("efa")));

ABOLETH_TRANSPORT_HOST_DEVICE
inline constexpr bool transport_deadline_due(
    TransportEffectPhase entering_phase,
    std::uint64_t entering_iteration,
    std::uint32_t before_phase,
    std::uint32_t iteration_delta) {
    if (before_phase == kTransportNoPhase ||
        before_phase !=
            static_cast<std::uint32_t>(
                entering_phase)) {
        return false;
    }
    return iteration_delta == 0 ||
        entering_iteration > iteration_delta;
}

static_assert(transport_deadline_due(
    TransportEffectPhase::NextPack,
    1,
    static_cast<std::uint32_t>(
        TransportEffectPhase::NextPack),
    0));
static_assert(!transport_deadline_due(
    TransportEffectPhase::Experts,
    1,
    static_cast<std::uint32_t>(
        TransportEffectPhase::Experts),
    1));
static_assert(transport_deadline_due(
    TransportEffectPhase::Experts,
    2,
    static_cast<std::uint32_t>(
        TransportEffectPhase::Experts),
    1));

struct alignas(8) TransportPhasePlan {
    std::uint32_t abi_version;
    std::uint32_t byte_size;

    std::uint32_t active_qps;
    std::uint32_t data_qps;

    std::uint32_t data_wqe_bound;
    std::uint32_t data_reclaim_interval;
    std::uint32_t control_wqe_bound;
    std::uint32_t control_reclaim_interval;

    std::uint32_t topology_kind;
    std::uint32_t qp0_role;
    std::uint32_t publication_mode;
    std::uint32_t prologue_barrier_kind;
    std::uint32_t epilogue_barrier_kind;
    std::uint32_t flags;

    std::uint32_t publication_before_phase;
    std::uint32_t publication_iteration_delta;
    std::uint32_t completion_before_phase;
    std::uint32_t completion_iteration_delta;
    std::uint32_t first_overwrite_phase;
    std::uint32_t first_overwrite_iteration_delta;
    std::uint32_t reclaim_before_phase;
    std::uint32_t reclaim_iteration_delta;

    std::uint32_t payload_bytes;
    std::uint32_t path_mtu;
    std::uint32_t completion_interval;
    std::uint32_t buffer_slots;
    std::uint32_t payload_commit_tokens;
    std::uint32_t reserved;
};

static_assert(
    std::is_standard_layout_v<
        TransportPhasePlan>);
static_assert(
    std::is_trivially_copyable_v<
        TransportPhasePlan>);
static_assert(
    sizeof(TransportPhasePlan) == 112);

#undef ABOLETH_TRANSPORT_HOST_DEVICE

} // namespace aboleth::abi
