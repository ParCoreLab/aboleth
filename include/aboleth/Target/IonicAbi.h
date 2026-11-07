#pragma once

#include <cstddef>
#include <cstdint>

namespace aboleth::abi {
// Ionic v2 opcodes with the v1 64-byte descriptor layout. Layout reference:
// ROCm/rocm-systems@2b22ab0195cc1461cd9abf3b969e9dd7c10af350,
// projects/rocshmem/src/gda/ionic/ionic_fw.h (Linux-OpenIB dual license).
// This header defines Aboleth's own binding and wire encoder.
inline constexpr std::uint32_t kIonicEndpointVersion = 1;
inline constexpr std::uint8_t kIonicWrite = 0x21;
// Installed Ionic 25.12.15.001 and the upstream reference encode FENCE as bit 0.
inline constexpr std::uint16_t kIonicFence = 1u << 0;
inline constexpr std::uint16_t kIonicInline = 1u << 2;
inline constexpr std::uint16_t kIonicSignaled = 1u << 3;
inline constexpr std::uint16_t kIonicColor = 1u << 4;
inline constexpr std::uint32_t kIonicMsnMask = 0xffffff;

struct IonicWqe {
  std::uint64_t index; // little endian
  std::uint8_t opcode, sge_count;
  std::uint16_t flags; // all following integer fields are big endian
  std::uint32_t immediate;
  std::uint32_t remote_high, remote_low, remote_key, length;
  std::uint8_t payload[32];
};
struct IonicCqe {
  std::uint32_t reserved, msn;
  std::uint64_t reserved2, index_timestamp;
  std::uint32_t status, qid_type_flags;
};
// Private per-endpoint GPU state. One lock serializes producers and CQ readers;
// it is held from reserve through commit, including capacity backpressure.
struct alignas(64) IonicQueueState {
  std::uint64_t producer, published, retired, status;
  std::uint32_t lock;
};
struct alignas(16) IonicEndpointDescriptor {
  std::uint64_t send_queue, completion_queue, send_doorbell, completion_doorbell;
  std::uint64_t send_dbval, completion_dbval, state, timeout_ticks;
  std::uint32_t send_mask, completion_mask, send_stride, completion_stride;
  std::uint32_t version, scratch_key;
  std::uint64_t scratch_address;
};
static_assert(sizeof(IonicWqe) == 64 && offsetof(IonicWqe, flags) == 10);
static_assert(offsetof(IonicWqe, payload) == 32);
static_assert(sizeof(IonicCqe) == 32 && offsetof(IonicCqe, msn) == 4);
static_assert(offsetof(IonicCqe, qid_type_flags) == 28);
static_assert(sizeof(IonicQueueState) == 64);
static_assert(sizeof(IonicEndpointDescriptor) == 96);
} // namespace aboleth::abi
