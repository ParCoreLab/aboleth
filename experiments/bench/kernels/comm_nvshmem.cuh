#pragma once

// NVSHMEM arm of the comm policy. Addresses are symmetric-heap VAs, so a
// Remote is the same address the local rank holds and the peer index does
// the addressing; keys do not exist. signal is issued as fence + signal_op
// to match the strong-ordered signal of the Tier-1 facade (and the existing
// benchmark's use_fence=1 default); put_inline8 maps to the scalar
// nvshmem_uint64_p, NVSHMEM's register-payload put. Every signal is that
// same fenced signal wherever it sits: dropping the fence where nothing
// precedes it is Aboleth's publication planning, not done by hand here.
// nvshmem_quiet completes every outstanding operation of the PE, so the
// all-to-all issues it once per rank per round (kQuietIsGlobal).
//
// Native fast paths (the library at its best):
// put_signal is nvshmem_putmem_signal_nbi — payload and epoch in one call,
// the signal delivered after that put's data — and put_warp is
// nvshmemx_putmem_nbi_warp (a 32-lane cooperative copy over P2P, one
// posting over IBGDA/IBRC), and put_signal_warp is
// nvshmemx_putmem_signal_nbi_warp. NVSHMEM has no batched put, so there is no
// put_many; put_many_signal exists only because the fused ping-pong path
// names it and is N-1 puts + fence + one put-with-signal.

#include <cstdint>

#include <nvshmem.h>
#include <nvshmemx.h>

struct NvshmemComm {
    struct Local {
        std::uint64_t address;
    };
    struct Remote {
        std::uint64_t address;
    };
    using Flag = std::uint64_t*;
    struct Ctx {
        // Put-with-signal orders only its own payload (OpenSHMEM 1.5
        // signaling semantics), so a fused round of N > 1 puts fences the
        // earlier ones first. The host sets it from the round shape.
        bool fence_prior;
    };
    static constexpr bool kHasPutInline = true;
    static constexpr bool kFlagsInMemory = true;
    static constexpr bool kQuietIsGlobal = true;
    // A put waits for room and reclaims the queue itself (--stream).
    static constexpr bool kSelfReclaim = true;
    static void advance(Ctx&, std::uint64_t) {}
    static void set_round_shape(Ctx& ctx, std::uint32_t puts,
                                std::uint32_t) {
        ctx.fence_prior = puts > 1;
    }

    static Ctx make_ctx(const std::uint64_t*, std::uint32_t*) {
        return Ctx{false};
    }
    static Remote make_remote(std::uint64_t address, std::uint32_t) {
        return {address};
    }
    static Local make_local(std::uint64_t address, std::uint32_t) {
        return {address};
    }
    static Flag make_flag(std::uint64_t pointer) {
        return reinterpret_cast<Flag>(pointer);
    }

    static __device__ __forceinline__ Local offset(
        Local region, std::uint64_t bytes) {
        return {region.address + bytes};
    }
    static __device__ __forceinline__ Remote offset(
        Remote region, std::uint64_t bytes) {
        return {region.address + bytes};
    }
    static __device__ __forceinline__ void put(
        Ctx&, int peer, Remote destination, Local source,
        std::uint32_t bytes) {
        nvshmem_putmem_nbi(reinterpret_cast<void*>(destination.address),
                           reinterpret_cast<const void*>(source.address),
                           bytes, peer);
    }
    static __device__ __forceinline__ void put_inline8(
        Ctx&, int peer, Remote destination, std::uint64_t value) {
        nvshmem_uint64_p(reinterpret_cast<std::uint64_t*>(
                             destination.address),
                         value, peer);
    }
    static __device__ __forceinline__ void signal(
        Ctx&, int peer, Remote flag, std::uint64_t epoch) {
        nvshmem_fence();
        nvshmemx_signal_op(reinterpret_cast<std::uint64_t*>(flag.address),
                           epoch, NVSHMEM_SIGNAL_SET, peer);
    }
    static __device__ __forceinline__ void quiet(Ctx&, int) {
        nvshmem_quiet();
    }
    static __device__ __forceinline__ void wait_signal(
        Ctx&, Flag flag, std::uint64_t epoch) {
        nvshmem_signal_wait_until(flag, NVSHMEM_CMP_GE, epoch);
    }

    // Warp-collective put: every lane of a complete warp enters. The
    // NVSHMEM warp call ends in a warp sync; the explicit __syncwarp keeps
    // that ordering visible here, so lane 0's fenced signal (cumulative
    // over the warp's stores) publishes the whole slice.
    static constexpr bool kHasPutWarp = true;
    static __device__ __forceinline__ void put_warp(
        Ctx&, int peer, Remote destination, Local source,
        std::uint32_t bytes) {
        nvshmemx_putmem_nbi_warp(
            reinterpret_cast<void*>(destination.address),
            reinterpret_cast<const void*>(source.address), bytes, peer);
        __syncwarp();
    }

    // Fused publication: one put-with-signal carries the payload and SETs
    // the epoch, so no separate fence + signal_op is posted.
    static constexpr bool kHasPutSignal = true;
    struct PutDescT {
        std::uint64_t destination;
        std::uint64_t source;
        std::uint32_t bytes;
    };
    static __device__ __forceinline__ void make_desc(
        PutDescT& desc, Remote destination, Local source,
        std::uint32_t bytes) {
        desc = {destination.address, source.address, bytes};
    }
    static __device__ __forceinline__ void put_signal(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes, Remote flag, std::uint64_t epoch) {
        if (ctx.fence_prior)
            nvshmem_fence();
        nvshmem_putmem_signal_nbi(
            reinterpret_cast<void*>(destination.address),
            reinterpret_cast<const void*>(source.address), bytes,
            reinterpret_cast<std::uint64_t*>(flag.address), epoch,
            NVSHMEM_SIGNAL_SET, peer);
    }
    // Warp-collective fused publication: every lane of a complete warp
    // enters; the earlier puts of a multi-put round are fenced first.
    static constexpr bool kHasPutSignalWarp = true;
    static __device__ __forceinline__ void put_signal_warp(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes, Remote flag, std::uint64_t epoch) {
        if (ctx.fence_prior) {
            if (threadIdx.x % 32 == 0)
                nvshmem_fence();
            __syncwarp();
        }
        nvshmemx_putmem_signal_nbi_warp(
            reinterpret_cast<void*>(destination.address),
            reinterpret_cast<const void*>(source.address), bytes,
            reinterpret_cast<std::uint64_t*>(flag.address), epoch,
            NVSHMEM_SIGNAL_SET, peer);
        __syncwarp();
    }
    static __device__ __forceinline__ void put_many_signal(
        Ctx&, int peer, const PutDescT* descriptors, std::uint32_t count,
        Remote flag, std::uint64_t epoch) {
        for (std::uint32_t index = 0; index + 1 < count; ++index)
            nvshmem_putmem_nbi(
                reinterpret_cast<void*>(descriptors[index].destination),
                reinterpret_cast<const void*>(descriptors[index].source),
                descriptors[index].bytes, peer);
        if (count > 1)
            nvshmem_fence();
        const PutDescT& last = descriptors[count - 1];
        nvshmem_putmem_signal_nbi(
            reinterpret_cast<void*>(last.destination),
            reinterpret_cast<const void*>(last.source), last.bytes,
            reinterpret_cast<std::uint64_t*>(flag.address), epoch,
            NVSHMEM_SIGNAL_SET, peer);
    }
};
