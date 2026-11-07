#pragma once

#include <cstdint>
#include <type_traits>

// Device ABI for the AWS EFA backend.
//
// Like the direct-mlx5 descriptor, this is an array of 64-bit words that the
// generated kernel loads by index. Unlike mlx5, the kernel does not build
// hardware descriptors itself: it calls the efa-dp-direct device library, so
// the descriptor carries queue-object pointers rather than raw queue memory.
//
// SRD is reliable but unordered and offers no remote atomics, which changes
// two contracts relative to the mlx5 backend:
//
//   * Completion is counted, not indexed. A batch has retired once the number
//     of consumed completions reaches its end, because completions may arrive
//     in any order.
//   * Strong publication cannot ride on same-queue ordering. Either the sender
//     retires its own payload completions before posting the signal
//     (FenceThenSignal) or the payload travels as write-with-immediate and the
//     receiver counts arrivals before advancing the epoch (ImmCounting).
//
// See docs/EFA_ABI.md.

namespace aboleth::abi {

inline constexpr std::uint32_t kEfaEndpointVersion = 1;

// Word indices the generated code loads from the descriptor.
enum EfaEndpointWord : std::uint32_t {
    kEfaWordQueuePair = 0,
    kEfaWordSendCq = 1,
    kEfaWordRecvCq = 2,
    kEfaWordCompletionState = 3,
    kEfaWordTimeoutCycles = 4,
    // Packed 32-bit pairs, low half first.
    kEfaWordDepths = 5,          // send depth | recv depth
    kEfaWordAddressHandle = 6,   // address handle | remote queue-pair number
    kEfaWordQueueKey = 7,        // queue key | publication scheme
    kEfaWordLimits = 8,          // max inline bytes | abi version
    kEfaEndpointWordCount = 9,
};

enum class EfaPublicationScheme : std::uint32_t {
    // Sender retires payload completions, then posts an ordinary signal write.
    FenceThenSignal = 0,
    // Payload carries immediate data; the receiver counts arrivals.
    ImmCounting = 1,
};

struct alignas(8) EfaEndpointDescriptor {
    // Device pointers to the efa-dp-direct queue objects.
    std::uint64_t queue_pair;
    std::uint64_t send_completion_queue;
    // Zero unless the endpoint publishes by counting receive completions.
    std::uint64_t recv_completion_queue;
    // Four 64-bit words: producer index, retired head, publish index, error.
    std::uint64_t completion_state;
    std::uint64_t completion_timeout_cycles;

    std::uint32_t send_depth;
    std::uint32_t recv_depth;

    // SRD addressing: an address handle plus the peer's queue-pair number and
    // queue key, rather than a connected queue pair.
    std::uint32_t address_handle;
    std::uint32_t remote_queue_pair_number;
    std::uint32_t queue_key;
    std::uint32_t publication_scheme;

    std::uint32_t max_inline_bytes;
    std::uint32_t abi_version;
};

static_assert(std::is_standard_layout_v<EfaEndpointDescriptor>);
static_assert(std::is_trivially_copyable_v<EfaEndpointDescriptor>);
static_assert(sizeof(EfaEndpointDescriptor) == 72);
// The generated kernel reads the descriptor as an array of 64-bit words, so
// the word indices above must cover the struct exactly.
static_assert(sizeof(EfaEndpointDescriptor) ==
              kEfaEndpointWordCount * sizeof(std::uint64_t));

// Completion-state word indices, shared with the mlx5 backend's block layout.
enum EfaCompletionStateWord : std::uint32_t {
    kEfaStateProducer = 0,
    kEfaStateRetired = 1,
    kEfaStatePublish = 2,
    kEfaStateError = 3,
    kEfaCompletionStateWords = 4,
};

// Status values returned by the device wrappers.
inline constexpr std::uint64_t kEfaStatusOk = 0;
inline constexpr std::uint64_t kEfaStatusTimeout = 1;
inline constexpr std::uint64_t kEfaStatusRemoteError = 2;
inline constexpr std::uint64_t kEfaStatusInvalid = 3;

// Names of the device wrappers the EFA lowering calls. The library is linked
// as Triton extern bitcode and inlined, so these are the only symbols the
// generated kernel refers to.
inline constexpr const char *kEfaLibraryName = "aboleth_efa";
inline constexpr const char *kEfaSymbolBeginBatch = "aboleth_efa_begin_batch";
inline constexpr const char *kEfaSymbolPostWrite = "aboleth_efa_post_write";
inline constexpr const char *kEfaSymbolPostWriteImm =
    "aboleth_efa_post_write_imm";
inline constexpr const char *kEfaSymbolPostInline = "aboleth_efa_post_inline";
inline constexpr const char *kEfaSymbolFlush = "aboleth_efa_flush";
inline constexpr const char *kEfaSymbolCompletionAdvance =
    "aboleth_efa_completion_advance";
inline constexpr const char *kEfaSymbolReceiveAdvance =
    "aboleth_efa_receive_advance";

} // namespace aboleth::abi
