#pragma once

// Tier-1 rdma arm of the comm policy (see bench_kernels.cuh for the
// contract): a direct mapping onto the device-library facade. This is the
// arm the suite exists to measure; the adapter adds nothing beyond region
// arithmetic.

#include <cstdint>

#include <aboleth/device.cuh>

struct AbolethComm {
    using Local = aboleth::LocalRegion;
    using Remote = aboleth::RemoteRegion;
    using Flag = const std::uint64_t*;
    struct Ctx {
        aboleth::DeviceContext dev;
    };
    static constexpr bool kHasPutInline = true;
    static constexpr bool kFlagsInMemory = true;
    static void advance(Ctx&, std::uint64_t) {}

    // Host-side helpers for the launchers and the shared C++ driver.
    static Ctx make_ctx(
        const std::uint64_t* endpoints, std::uint32_t* doorbell_locks) {
        return Ctx{aboleth::DeviceContext{endpoints, doorbell_locks}};
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
        aboleth::put(ctx.dev, peer, destination, source, bytes);
    }
    static __device__ __forceinline__ void put_inline8(
        Ctx& ctx, int peer, Remote destination, std::uint64_t value) {
        aboleth::put_inline(ctx.dev, peer, destination, value);
    }
    static __device__ __forceinline__ void signal(
        Ctx& ctx, int peer, Remote flag, std::uint64_t epoch) {
        aboleth::signal(ctx.dev, peer, flag, epoch);
    }
    static __device__ __forceinline__ void quiet(Ctx& ctx, int peer) {
        aboleth::quiet(ctx.dev, peer);
    }
    static __device__ __forceinline__ void wait_signal(
        Ctx& ctx, Flag flag, std::uint64_t epoch) {
        aboleth::wait_signal(ctx.dev, flag, epoch);
    }

#ifdef ABOLETH_TIER1_GENERATED
    // The generated library's composite: N descriptor-driven puts in one
    // batched submission — one reservation, one doorbell. Only the generated
    // header carries it; against the hand-written facade this block simply
    // does not exist and the kernels fall back to the per-put loop.
    static constexpr bool kHasPutMany = true;
    using PutDescT = aboleth::PutDesc;
    static __device__ __forceinline__ void make_desc(
        PutDescT& desc, Remote destination, Local source,
        std::uint32_t bytes) {
        desc = {destination.address, source.address, bytes,
                destination.key, source.key, 0};
    }
    static __device__ __forceinline__ void put_many(
        Ctx& ctx, int peer, const PutDescT* descriptors,
        std::uint32_t count) {
        aboleth::put_many(ctx.dev, peer, descriptors, count);
    }

    // Warp-collective put: on p2p all 32 lanes copy cooperatively; on rdma
    // one elected lane posts the WQE. The calling warp must be complete.
    static constexpr bool kHasPutWarp = true;
    static __device__ __forceinline__ void put_warp(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes) {
        aboleth::put_warp(ctx.dev, peer, destination, source, bytes);
    }

#ifdef ABOLETH_TIER1_PUT_SIGNAL
    // Fused publication: put(s) and the fenced epoch share one batch — one
    // reservation, one doorbell where put-then-signal pays two.
    static constexpr bool kHasPutSignal = true;
    static __device__ __forceinline__ void put_signal(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes, Remote flag, std::uint64_t epoch) {
        aboleth::put_signal(ctx.dev, peer, destination, source, bytes,
                            flag, epoch);
    }
    static __device__ __forceinline__ void put_many_signal(
        Ctx& ctx, int peer, const PutDescT* descriptors,
        std::uint32_t count, Remote flag, std::uint64_t epoch) {
        aboleth::put_many_signal(ctx.dev, peer, descriptors, count,
                                 flag, epoch);
    }
#endif
#ifdef ABOLETH_TIER1_PUT_MANY_FLUSH
    // The window and its retirement in one batch: N puts + a signalled NOP
    // + a coalesced poll — one doorbell, one CQE, quiet's contract.
    static constexpr bool kHasPutManyFlush = true;
    static __device__ __forceinline__ void put_many_flush(
        Ctx& ctx, int peer, const PutDescT* descriptors,
        std::uint32_t count) {
        aboleth::put_many_flush(ctx.dev, peer, descriptors, count);
    }
#endif
#endif
};
