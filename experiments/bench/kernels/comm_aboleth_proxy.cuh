#pragma once

// Tier-1 proxy arm of the comm policy: the same region/flag protocol as
// comm_aboleth.cuh, mapped onto the host-proxy device primitives. One
// semantic difference is inherent and recorded in the methodology block:
// proxy_detail::signal waits for host-proxy completion before returning,
// while the rdma facade's signal is fire-and-forget.

#include <cstdint>

#include <aboleth/Target/ProxyDevice.cuh>
#include <aboleth/Target/ProxyAbi.h>

struct AbolethProxyComm {
    struct Local {
        std::uint64_t address;
        std::uint32_t key;
    };
    struct Remote {
        std::uint64_t address;
        std::uint32_t key;
    };
    using Flag = const std::uint64_t*;
    struct Ctx {
        const std::uint64_t* endpoints;  // ProxyEndpointDescriptor* per peer
    };
    static constexpr bool kHasPutInline = true;
    static constexpr bool kFlagsInMemory = true;
    static void advance(Ctx&, std::uint64_t) {}

    // Host-side helpers for the launchers and the shared C++ driver; the
    // proxy path has no doorbell locks.
    static Ctx make_ctx(const std::uint64_t* endpoints, std::uint32_t*) {
        return Ctx{endpoints};
    }
    static Remote make_remote(std::uint64_t address, std::uint32_t key) {
        return {address, key};
    }
    static Local make_local(std::uint64_t address, std::uint32_t key) {
        return {address, key};
    }
    static Flag make_flag(std::uint64_t pointer) {
        return reinterpret_cast<Flag>(pointer);
    }

    static __device__ __forceinline__
    const aboleth::abi::ProxyEndpointDescriptor& endpoint(
        Ctx& ctx, int peer) {
        // The Tier-1 handle table carries the transport kind in the low
        // bits (aboleth/device.cuh); mask it off before dereferencing.
        return *reinterpret_cast<const aboleth::abi::ProxyEndpointDescriptor*>(
            ctx.endpoints[peer] & ~0x7ull);
    }
    static __device__ __forceinline__ Local offset(
        Local region, std::uint64_t bytes) {
        return {region.address + bytes, region.key};
    }
    static __device__ __forceinline__ Remote offset(
        Remote region, std::uint64_t bytes) {
        return {region.address + bytes, region.key};
    }
    static __device__ __forceinline__ void put(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes) {
        aboleth::cuda::proxy_detail::write(
            endpoint(ctx, peer),
            reinterpret_cast<const void*>(source.address), source.key,
            destination.address, destination.key, bytes);
    }
    static __device__ __forceinline__ void put_inline8(
        Ctx& ctx, int peer, Remote destination, std::uint64_t value) {
        aboleth::cuda::proxy_detail::inline_write(
            endpoint(ctx, peer), destination.address, destination.key, value,
            sizeof(value), aboleth::abi::kProxyCommandWeakSignal);
    }
    static __device__ __forceinline__ void signal(
        Ctx& ctx, int peer, Remote flag, std::uint64_t epoch) {
        aboleth::cuda::proxy_detail::signal(
            endpoint(ctx, peer), flag.address, flag.key, epoch);
    }
    static __device__ __forceinline__ void quiet(Ctx& ctx, int peer) {
        aboleth::cuda::proxy_detail::drain(endpoint(ctx, peer));
    }
    static __device__ __forceinline__ void wait_signal(
        Ctx&, Flag flag, std::uint64_t epoch) {
        while (aboleth::cuda::proxy_detail::load_acquire(flag) < epoch)
            __nanosleep(64);
    }
};
