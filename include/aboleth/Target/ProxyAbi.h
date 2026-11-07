#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aboleth::abi {

inline constexpr std::uint32_t kProxyEndpointVersion = 8;
inline constexpr std::uint32_t kProxySlotCount = 64;
inline constexpr std::uint32_t kProxyMaximumCommands = 256;

inline constexpr std::uint32_t kProxyCommandNop = 0;
inline constexpr std::uint32_t kProxyCommandWrite = 1;
inline constexpr std::uint32_t kProxyCommandWeakSignal = 2;
inline constexpr std::uint32_t kProxyCommandStrongSignal = 3;

struct alignas(8) ProxyCommand {
    std::uint64_t source_address;
    std::uint64_t destination_address;
    std::uint32_t bytes;
    std::uint32_t source_key;
    std::uint32_t destination_key;
    std::uint32_t opcode;
};

struct alignas(8) ProxyBatchSlot {
    std::uint64_t ready_sequence;
    std::uint32_t command_count;
    std::uint32_t reserved;
    ProxyCommand commands[kProxyMaximumCommands];
};

struct alignas(8) ProxyQueueState {
    std::uint64_t completed_sequence;
    std::uint64_t status;
    ProxyBatchSlot slots[kProxySlotCount];
};

struct alignas(8) ProxyEndpointDescriptor {
    std::uint64_t queue_state;
    std::uint64_t producer_index_address;
    std::uint64_t completion_timeout_cycles;
    std::uint32_t slot_count;
    std::uint32_t slot_stride;
    std::uint32_t maximum_commands;
    std::uint32_t abi_version;
};

static_assert(std::is_standard_layout_v<ProxyCommand>);
static_assert(std::is_trivially_copyable_v<ProxyCommand>);
static_assert(sizeof(ProxyCommand) == 32);
static_assert(alignof(ProxyCommand) == 8);
static_assert(offsetof(ProxyCommand, source_address) == 0);
static_assert(offsetof(ProxyCommand, destination_address) == 8);
static_assert(offsetof(ProxyCommand, bytes) == 16);
static_assert(offsetof(ProxyCommand, source_key) == 20);
static_assert(offsetof(ProxyCommand, destination_key) == 24);
static_assert(offsetof(ProxyCommand, opcode) == 28);

static_assert(std::is_standard_layout_v<ProxyBatchSlot>);
static_assert(std::is_trivially_copyable_v<ProxyBatchSlot>);
static_assert(sizeof(ProxyBatchSlot) == 8208);
static_assert(alignof(ProxyBatchSlot) == 8);
static_assert(offsetof(ProxyBatchSlot, ready_sequence) == 0);
static_assert(offsetof(ProxyBatchSlot, command_count) == 8);
static_assert(offsetof(ProxyBatchSlot, commands) == 16);

static_assert(std::is_standard_layout_v<ProxyQueueState>);
static_assert(std::is_trivially_copyable_v<ProxyQueueState>);
static_assert(sizeof(ProxyQueueState) == 525328);
static_assert(alignof(ProxyQueueState) == 8);
static_assert(offsetof(ProxyQueueState, completed_sequence) == 0);
static_assert(offsetof(ProxyQueueState, status) == 8);
static_assert(offsetof(ProxyQueueState, slots) == 16);

static_assert(std::is_standard_layout_v<ProxyEndpointDescriptor>);
static_assert(std::is_trivially_copyable_v<ProxyEndpointDescriptor>);
static_assert(sizeof(ProxyEndpointDescriptor) == 40);
static_assert(alignof(ProxyEndpointDescriptor) == 8);
static_assert(offsetof(ProxyEndpointDescriptor, queue_state) == 0);
static_assert(
    offsetof(ProxyEndpointDescriptor, producer_index_address) == 8);
static_assert(
    offsetof(ProxyEndpointDescriptor, completion_timeout_cycles) == 16);
static_assert(offsetof(ProxyEndpointDescriptor, slot_count) == 24);
static_assert(offsetof(ProxyEndpointDescriptor, slot_stride) == 28);
static_assert(offsetof(ProxyEndpointDescriptor, maximum_commands) == 32);
static_assert(offsetof(ProxyEndpointDescriptor, abi_version) == 36);

}  // namespace aboleth::abi
