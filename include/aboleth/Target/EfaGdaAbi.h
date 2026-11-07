#pragma once

#include <cstdint>
#include <type_traits>

#include "aboleth/Target/ProxyAbi.h"

namespace aboleth::abi {

inline constexpr std::uint32_t
    kEfaGdaEndpointVersion = 1;

struct alignas(8) EfaGdaEndpointDescriptor {
    std::uint64_t qp;
    std::uint64_t cq;
    std::uint64_t completion_timeout_cycles;
    std::uint32_t local_window_lkey;
    std::uint32_t local_control_lkey;
    std::uint32_t local_batch_lkey;
    std::uint32_t remote_window_rkey;
    std::uint32_t remote_barrier_rkey;
    std::uint16_t peer_ah;
    std::uint16_t peer_qpn;
    std::uint32_t peer_qkey;
    std::uint32_t abi_version;
    std::uint32_t reserved;
};

struct alignas(8) EfaGdaBatch {
    std::uint32_t command_count;
    std::uint32_t reserved;
    ProxyCommand commands[
        kProxyMaximumCommands];
};

static_assert(
    std::is_standard_layout_v<
        EfaGdaEndpointDescriptor>);
static_assert(
    std::is_trivially_copyable_v<
        EfaGdaEndpointDescriptor>);
static_assert(
    std::is_standard_layout_v<EfaGdaBatch>);
static_assert(
    std::is_trivially_copyable_v<EfaGdaBatch>);

}  // namespace aboleth::abi
