#pragma once

// Aboleth Tier 1: one-sided transport from arbitrary CUDA code.
//
//   DeviceContext ctx{world_endpoints, world_doorbell_locks};
//   aboleth::put(ctx, peer, remote_row, local_row, bytes);
//   aboleth::put_inline(ctx, peer, remote_flag, header_word);
//   aboleth::signal(ctx, peer, remote_flag, epoch);
//   aboleth::quiet(ctx, peer);
//   aboleth::wait_signal(ctx, local_flag, epoch);
//
// Addresses come from the rank's peer-region table (world.exchange_regions),
// never from a symmetric heap: a RemoteRegion is one peer's address plus the
// rkey it exported, a LocalRegion is this rank's address plus its lkey.
//
// put, put_inline, and signal post and doorbell without waiting; quiet is the
// only blocking call and retires everything posted to that peer so far. A
// client that never quiets will eventually block in put, because the send queue
// cannot be reclaimed until a completion is consumed.
//
// This facade delegates to the header-only aboleth::cuda::rdma primitives
// rather than to the extern "C" entry points of aboleth_rdma_<arch>.bc, so
// including this header is sufficient — no bitcode or archive to link. The
// prebuilt device library is compiled from those same primitives, so a client
// that prefers to link it gets identical code either way; the library exists for
// Triton (extern_libs) and for separate compilation, not as a different
// implementation.

#include <cstdint>

#include "aboleth/Target/RdmaAbi.h"
#include "aboleth/Target/RdmaDevice.cuh"

namespace aboleth {

namespace detail {
namespace rdma = ::aboleth::cuda::rdma;
} // namespace detail

struct LocalRegion {
    std::uint64_t address;
    std::uint32_t key;
};

struct RemoteRegion {
    std::uint64_t address;
    std::uint32_t key;
};

// Kernel-argument view of one rank's endpoints: the per-peer table of endpoint
// descriptor addresses (aboleth.EndpointTable.device_handles) and, when several
// CTAs share a queue, the per-peer doorbell locks.
struct DeviceContext {
    const std::uint64_t* endpoints;
    std::uint32_t* doorbell_locks;

    __device__ __forceinline__
    const abi::RdmaEndpointDescriptor& endpoint(int peer) const {
        return *reinterpret_cast<const abi::RdmaEndpointDescriptor*>(
            endpoints[peer]);
    }

    __device__ __forceinline__
    std::uint32_t* doorbell_lock(int peer) const {
        return doorbell_locks == nullptr ? nullptr : doorbell_locks + peer;
    }
};

__device__ __forceinline__ void put(
    const DeviceContext& context, int peer,
    const RemoteRegion& destination, const LocalRegion& source,
    std::uint32_t bytes) {
    const auto& endpoint = context.endpoint(peer);
    const auto base = detail::rdma::reserve(endpoint);
    detail::rdma::write_rdma(
        endpoint, base,
        reinterpret_cast<const void*>(source.address), source.key,
        destination.address, destination.key, bytes,
        /*request_completion=*/false);
    detail::rdma::publish(
        endpoint, context.doorbell_lock(peer), base, base + 1);
}

// Eight bytes carried in the descriptor itself, straight out of registers.
__device__ __forceinline__ void put_inline(
    const DeviceContext& context, int peer,
    const RemoteRegion& destination, std::uint64_t value) {
    const auto& endpoint = context.endpoint(peer);
    const auto base = detail::rdma::reserve(endpoint);
    detail::rdma::write_inline<std::uint64_t>(
        endpoint, base, destination.address, destination.key, value,
        /*request_completion=*/false);
    detail::rdma::publish(
        endpoint, context.doorbell_lock(peer), base, base + 1);
}

// Publication: a responder compare-and-swap of the epoch, ordered behind every
// payload write posted before it (its read cannot pass relaxed-ordering
// writes), so a peer that observes the epoch observes the payload. The cell
// must hold epoch - 1.
__device__ __forceinline__ void signal(
    const DeviceContext& context, int peer,
    const RemoteRegion& flag, std::uint64_t epoch) {
    const auto& endpoint = context.endpoint(peer);
    const auto base = detail::rdma::reserve(endpoint);
    detail::rdma::write_atomic_signal(
        endpoint, base, flag.address, flag.key, epoch,
        /*request_completion=*/false);
    detail::rdma::publish(
        endpoint, context.doorbell_lock(peer), base, base + 1);
}

// Sender-side completion up to here, which is also what reclaims the queue.
__device__ __forceinline__ void quiet(
    const DeviceContext& context, int peer) {
    detail::rdma::drain(
        context.endpoint(peer), context.doorbell_lock(peer));
}

// Epochs are monotonic, so a later epoch satisfies an earlier wait.
__device__ __forceinline__ void wait_signal(
    const DeviceContext&, const std::uint64_t* flag, std::uint64_t epoch) {
    while (detail::rdma::load_acquire(flag) < epoch)
        __nanosleep(64);
}

} // namespace aboleth
