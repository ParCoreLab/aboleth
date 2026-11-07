#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aboleth::abi {

inline constexpr std::uint32_t kRdmaEndpointVersion = 3;
inline constexpr std::size_t kRdmaAtomicResultOffset =
    4 * sizeof(std::uint64_t);

// mlx5 send-WQE geometry, shared by the lowering that builds descriptors and
// by the verifiers and transforms that have to refuse a payload the wire
// cannot carry.
//
// A WQE occupies whole 16-byte segments, and the control segment's ds field
// counts them in six bits: at most 63. An inline write spends two segments on
// control and remote address, so the inline-data segment gets the remaining
// 61 — its 4-byte header plus the payload, rounded up to a whole segment.
// Solving 2 + ceil((4 + bytes) / 16) <= 63 gives 972 bytes; one byte more
// wraps the ds field and the responder reads a WQE that was never written.
inline constexpr std::uint64_t kWqeSegmentBytes = 16;
inline constexpr std::uint64_t kWqeMaxDataSegments = 63;
inline constexpr std::uint64_t kInlineSegmentHeaderBytes = 4;
inline constexpr std::uint64_t kMaxInlinePayloadBytes = 972;

// The send queue's slots and its producer index are counted in 64-byte WQEBBs,
// so an inline write costs more than one slot as soon as its payload leaves
// the first: 32 bytes of control and remote-address segments, the 4-byte
// inline header, and 28 bytes of payload exactly fill one.
inline constexpr std::uint64_t kWqeBlockBytes = 64;
inline constexpr std::uint64_t kSingleBlockInlinePayloadBytes = 28;

inline constexpr std::uint64_t inline_wqe_data_segments(
    std::uint64_t payload_bytes) {
    return 2 + (kInlineSegmentHeaderBytes + payload_bytes +
                kWqeSegmentBytes - 1) /
                   kWqeSegmentBytes;
}

// Producer-index slots an inline write occupies.
inline constexpr std::uint64_t inline_wqe_blocks(std::uint64_t payload_bytes) {
    return (inline_wqe_data_segments(payload_bytes) * kWqeSegmentBytes +
            kWqeBlockBytes - 1) /
           kWqeBlockBytes;
}

static_assert(inline_wqe_blocks(kSingleBlockInlinePayloadBytes) == 1 &&
              inline_wqe_blocks(kSingleBlockInlinePayloadBytes + 1) == 2);
static_assert(inline_wqe_data_segments(kMaxInlinePayloadBytes) ==
              kWqeMaxDataSegments);
static_assert(inline_wqe_data_segments(kMaxInlinePayloadBytes + 1) >
              kWqeMaxDataSegments);
static_assert(2 * kWqeSegmentBytes + kInlineSegmentHeaderBytes +
                  kSingleBlockInlinePayloadBytes ==
              kWqeBlockBytes);

struct alignas(8) RdmaEndpointDescriptor {
    std::uint64_t send_queue;
    std::uint64_t send_dbrec;
    std::uint64_t blueflame;
    std::uint64_t completion_queue;
    std::uint64_t completion_dbrec;
    std::uint64_t producer_index_address;
    std::uint64_t completion_consumer_address;
    std::uint64_t completion_timeout_cycles;

    std::uint32_t send_stride;
    std::uint32_t send_wqe_count;
    std::uint32_t qpn;
    std::uint32_t completion_entry_size;
    std::uint32_t completion_entry_count;
    std::uint32_t abi_version;
    std::uint32_t atomic_result_lkey;
    std::uint32_t reserved1;
};

static_assert(std::is_standard_layout_v<RdmaEndpointDescriptor>);
static_assert(std::is_trivially_copyable_v<RdmaEndpointDescriptor>);
static_assert(sizeof(RdmaEndpointDescriptor) == 96);
static_assert(alignof(RdmaEndpointDescriptor) == 8);

static_assert(offsetof(RdmaEndpointDescriptor, send_queue) == 0);
static_assert(offsetof(RdmaEndpointDescriptor, send_dbrec) == 8);
static_assert(offsetof(RdmaEndpointDescriptor, blueflame) == 16);
static_assert(offsetof(RdmaEndpointDescriptor, completion_queue) == 24);
static_assert(offsetof(RdmaEndpointDescriptor, completion_dbrec) == 32);
static_assert(offsetof(RdmaEndpointDescriptor, producer_index_address) == 40);
static_assert(offsetof(RdmaEndpointDescriptor, completion_consumer_address) == 48);
static_assert(offsetof(RdmaEndpointDescriptor, completion_timeout_cycles) == 56);
static_assert(offsetof(RdmaEndpointDescriptor, send_stride) == 64);
static_assert(offsetof(RdmaEndpointDescriptor, send_wqe_count) == 68);
static_assert(offsetof(RdmaEndpointDescriptor, qpn) == 72);
static_assert(offsetof(RdmaEndpointDescriptor, completion_entry_size) == 76);
static_assert(offsetof(RdmaEndpointDescriptor, completion_entry_count) == 80);
static_assert(offsetof(RdmaEndpointDescriptor, abi_version) == 84);
static_assert(offsetof(RdmaEndpointDescriptor, atomic_result_lkey) == 88);
static_assert(offsetof(RdmaEndpointDescriptor, reserved1) == 92);

inline constexpr const char *kRdmaDeviceLibraryName =
    "aboleth_rdma";
inline constexpr const char *kRdmaSymbolReserveBatch =
    "aboleth_rdma_reserve_batch";
inline constexpr const char *kRdmaSymbolReserve =
    "aboleth_rdma_reserve";
inline constexpr const char *kRdmaSymbolReservePlanned =
    "aboleth_rdma_reserve_planned";
inline constexpr const char *kRdmaSymbolPostNop =
    "aboleth_rdma_post_nop";
inline constexpr const char *kRdmaSymbolPostWrite =
    "aboleth_rdma_post_write";
inline constexpr const char *kRdmaSymbolPostInlineU64 =
    "aboleth_rdma_post_inline_u64";
inline constexpr const char *kRdmaSymbolPostAtomicSignal =
    "aboleth_rdma_post_atomic_signal";
inline constexpr const char *kRdmaSymbolPublish =
    "aboleth_rdma_publish";
inline constexpr const char *kRdmaSymbolPublishHead =
    "aboleth_rdma_publish_head";
inline constexpr const char *kRdmaSymbolPoll =
    "aboleth_rdma_poll";
inline constexpr const char *kRdmaSymbolDrain =
    "aboleth_rdma_drain";

}  // namespace aboleth::abi
