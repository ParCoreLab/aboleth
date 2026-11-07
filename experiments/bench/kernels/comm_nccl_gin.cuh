#pragma once

// NCCL device-API (GIN) arm of the comm policy. Addresses are byte offsets
// into one symmetric registered window; signals are indexed cumulative
// counters (slot = sender rank), so kFlagsInMemory is false and the host
// advances epoch_base between launches instead of zeroing flag memory.
// signal() is a bare GIN signal carrying a StrongSignalAdd (it settles the
// puts before it). Every signal is the same ordered signal wherever it
// sits: choosing a weaker one from what precedes it is what Aboleth's
// publication planning does, so the library arm does not do it by hand.
// put_inline has no coherent mapping
// (an inline value write would not bump a signal counter), so the arm
// refuses that op. A fresh ncclGin handle is constructed per operation —
// it is a light view over the devComm; recorded as a methodology caveat.
//
// Counter accounting: every signal and every put_signal adds exactly 1 to
// the sender's slot. A round therefore advances a receiver's slot by the
// number of signals the peer publishes per round toward it (wait_stride:
// N under `--op put+signal --puts N` for the ping-pong follower, else 1),
// and the wait target for round `epoch` is (epoch_base + epoch) *
// wait_stride, epoch_base counting rounds of earlier launches.
//
// Native fast paths: put_signal is ONE GIN put carrying the StrongSignalAdd
// (the strong signal settles this and every earlier put on the context to
// that peer); put_warp and put_signal_warp are the same puts entered by a
// complete warp (ncclCoopWarp: the warp syncs, lane 0 posts, the warp
// syncs). put_many_flush is the hand-batched window: every put but the last
// skips its doorbell (ncclGinOptFlagsAggregateRequests). Batching doorbells
// is what Aboleth's coalescing does automatically, so it runs only under
// --batch-window on, as a labelled arm next to Tier 1's own composite.
//
// LSA lanes (`lsa` set, transports gdaki-lsa / proxy-lsa): peers in this
// rank's LSA team — load/store accessible over NVLink, decided per peer at
// runtime from the devComm's lsaRank/lsaSize — bypass GIN entirely: puts
// are NCCL's own LSA copy (ncclLsaReduceSumLsaCopy over one source and one
// destination) through ncclGetPeerPointer, signals are release stores of the
// epoch into the peer's flag word, waits are acquire spins on the local
// word, and quiet is a system-scope fence. LSA flags are memory words in
// the GIN epoch numbering (epoch_base + epoch, monotonic across launches,
// SET semantics like the other memory-flag arms); a peer is LSA from both
// sides or neither, so one peer never mixes the two namespaces.

#include <cstdint>

#include <nccl.h>
#include <nccl_device.h>

struct NcclGinComm {
    struct Local {
        std::uint64_t offset;
    };
    struct Remote {
        std::uint64_t offset;
    };
    using Flag = std::uint32_t;  // signal slot index = sender rank
    struct Ctx {
        ncclDevComm dev;
        ncclWindow_t window;
        std::uint32_t context_count;
        int rank;
        std::uint64_t epoch_base;    // rounds consumed by earlier launches
        std::uint32_t wait_stride;   // signals per round the peer sends me
        std::uint64_t flags_offset;  // window offset of the flag words
        bool lsa;                    // same-node peers ride LSA, not GIN
        std::uint32_t issuers;       // threads of a CTA posting at once
    };
    static constexpr bool kHasPutInline = false;
    static constexpr bool kFlagsInMemory = false;
    // DOCA's put waits for a free slot itself (--stream).
    static constexpr bool kSelfReclaim = true;

    static void advance(Ctx& ctx, std::uint64_t rounds) {
        ctx.epoch_base += rounds;
    }
    static void set_round_shape(Ctx& ctx, std::uint32_t,
                                std::uint32_t inbound_signals) {
        ctx.wait_stride = inbound_signals ? inbound_signals : 1;
    }
    static void set_issuers(Ctx& ctx, std::uint32_t issuers) {
        ctx.issuers = issuers;
    }
    static Remote make_remote(std::uint64_t offset, std::uint32_t) {
        return {offset};
    }
    static Local make_local(std::uint64_t offset, std::uint32_t) {
        return {offset};
    }
    // The shared loops hand out 8-byte flag slots; the signal slot index is
    // the slot number.
    static Flag make_flag(std::uint64_t value) {
        return static_cast<Flag>(value / 8);
    }

    static __device__ __forceinline__ Local offset(
        Local region, std::uint64_t bytes) {
        return {region.offset + bytes};
    }
    static __device__ __forceinline__ Remote offset(
        Remote region, std::uint64_t bytes) {
        return {region.offset + bytes};
    }

    // A context is exclusive to its CTA's one issuing thread, shared by
    // the CTA's threads when several post at once (--issuers), or shared
    // GPU-wide when CTAs outnumber contexts.
    static __device__ __forceinline__ ncclGin gin(Ctx& ctx) {
        const int context = static_cast<int>(blockIdx.x) %
                            static_cast<int>(ctx.context_count);
        const ncclGinResourceSharingMode sharing =
            ctx.context_count < gridDim.x ? NCCL_GIN_RESOURCE_SHARING_GPU
            : ctx.issuers > 1             ? NCCL_GIN_RESOURCE_SHARING_CTA
                                          : NCCL_GIN_RESOURCE_SHARING_THREAD;
        return ncclGin{ctx.dev, context, sharing};
    }

    // ---- LSA lane --------------------------------------------------------

    // The LSA team is the contiguous world-rank block [rank - lsaRank,
    // rank - lsaRank + lsaSize) (ncclTeamLsa: stride 1).
    static __device__ __forceinline__ bool is_lsa(const Ctx& ctx, int peer) {
        const int first = ctx.dev.rank - ctx.dev.lsaRank;
        return ctx.lsa && static_cast<unsigned>(peer - first) <
                              static_cast<unsigned>(ctx.dev.lsaSize);
    }
    // NCCL's own unrolled, alignment-aware LSA copy: a one-source sum is a
    // copy. `Coop` is the participating thread set (a thread or a warp).
    // The API takes device lambdas (nvcc --extended-lambda).
    template <class Coop>
    static __device__ __forceinline__ void lsa_copy(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes, Coop coop) {
        auto* to = static_cast<std::uint8_t*>(
            ncclGetPeerPointer(ctx.window, destination.offset, peer));
        const auto* from = static_cast<const std::uint8_t*>(
            ncclGetLocalPointer(ctx.window, source.offset));
        ncclLsaReduceSumLsaCopy<std::uint8_t>(
            coop,
            [=] __device__(int) { return const_cast<std::uint8_t*>(from); },
            1, [=] __device__(int) { return to; }, 1, bytes);
    }
    static __device__ __forceinline__ void store_release_sys(
        std::uint64_t* word, std::uint64_t value) {
        asm volatile("st.release.sys.u64 [%0], %1;"
                     :: "l"(word), "l"(value) : "memory");
    }
    static __device__ __forceinline__ std::uint64_t load_acquire_sys(
        const std::uint64_t* word) {
        std::uint64_t value;
        asm volatile("ld.acquire.sys.u64 %0, [%1];"
                     : "=l"(value) : "l"(word) : "memory");
        return value;
    }
    static __device__ __forceinline__ void lsa_signal(
        Ctx& ctx, int peer, Remote flag, std::uint64_t epoch) {
        store_release_sys(static_cast<std::uint64_t*>(ncclGetPeerPointer(
                              ctx.window, flag.offset, peer)),
                          ctx.epoch_base + epoch);
    }

    // ---- policy ----------------------------------------------------------

    static __device__ __forceinline__ void put(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes) {
        put_flagged(ctx, peer, destination, source, bytes,
                    ncclGinOptFlagsDefault);
    }
    static __device__ __forceinline__ void put_flagged(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes, std::uint32_t flags) {
        if (is_lsa(ctx, peer)) {
            lsa_copy(ctx, peer, destination, source, bytes,
                     ncclCoopThread{});
            return;
        }
        gin(ctx).put(ncclTeamWorld(ctx.dev), peer, ctx.window,
                     destination.offset, ctx.window, source.offset, bytes,
                     ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
                     ncclGin_None{}, cuda::thread_scope_thread,
                     cuda::thread_scope_device, flags);
    }
    static __device__ __forceinline__ void put_inline8(
        Ctx&, int, Remote, std::uint64_t) {}
    static __device__ __forceinline__ void signal(
        Ctx& ctx, int peer, Remote flag, std::uint64_t epoch) {
        if (is_lsa(ctx, peer)) {
            lsa_signal(ctx, peer, flag, epoch);
            return;
        }
        gin(ctx).signal(ncclTeamWorld(ctx.dev), peer,
                        ncclGin_StrongSignalAdd{
                            static_cast<ncclGinSignal_t>(ctx.rank), 1});
    }

    static __device__ __forceinline__ void quiet(Ctx& ctx, int peer) {
        if (is_lsa(ctx, peer)) {
            // The copies are retired stores; a system-scope fence makes
            // them visible to the peer GPU before anything that follows.
            __threadfence_system();
            return;
        }
        gin(ctx).flush(ncclCoopThread{});
    }
    static __device__ __forceinline__ void wait_signal(
        Ctx& ctx, Flag slot, std::uint64_t epoch) {
        const int sender = static_cast<int>(slot);
        if (is_lsa(ctx, sender)) {
            const auto* word = static_cast<const std::uint64_t*>(
                ncclGetLocalPointer(ctx.window,
                                    ctx.flags_offset + 8ull * slot));
            while (load_acquire_sys(word) < ctx.epoch_base + epoch)
                ;
            return;
        }
        // Signals are context-local state (GetSignalPtr dereferences the
        // context handle): a signal from rank `slot` lands in the table of
        // the context the SENDER issued on, which by symmetric context
        // numbering is my index in the sender's CTA order — not this CTA's
        // outgoing context.
        const int incoming_block = ctx.rank - (ctx.rank > sender);
        const int incoming_context =
            incoming_block % static_cast<int>(ctx.context_count);
        ncclGin incoming{ctx.dev, incoming_context,
                         NCCL_GIN_RESOURCE_SHARING_GPU};
        incoming.waitSignal(ncclCoopThread{},
                            static_cast<ncclGinSignal_t>(slot),
                            (ctx.epoch_base + epoch) * ctx.wait_stride);
    }

    // Warp-collective put: a complete warp enters. GIN: ncclCoopWarp makes
    // the warp one participant set (sync, lane 0 posts the WQE, sync). LSA:
    // all 32 lanes copy, then the warp syncs so lane 0's release-store
    // signal (cumulative) or fence publishes every lane's stores.
    static constexpr bool kHasPutWarp = true;
    static __device__ __forceinline__ void put_warp(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes) {
        if (is_lsa(ctx, peer)) {
            lsa_copy(ctx, peer, destination, source, bytes, ncclCoopWarp{});
            __syncwarp();
            return;
        }
        gin(ctx).put(ncclTeamWorld(ctx.dev), peer, ctx.window,
                     destination.offset, ctx.window, source.offset, bytes,
                     ncclGin_None{}, ncclGin_None{}, ncclCoopWarp{},
                     ncclGin_None{}, cuda::thread_scope_thread,
                     cuda::thread_scope_device, ncclGinOptFlagsDefault);
    }

    // Fused publication: the payload put carries the strong signal action
    // itself — one WQE where put-then-signal pays two. The flag scratch
    // word is not written (waits read the counter); LSA peers copy and
    // release-store.
    static constexpr bool kHasPutSignal = true;
    struct PutDescT {
        std::uint64_t destination;
        std::uint64_t source;
        std::uint32_t bytes;
    };
    static __device__ __forceinline__ void make_desc(
        PutDescT& desc, Remote destination, Local source,
        std::uint32_t bytes) {
        desc = {destination.offset, source.offset, bytes};
    }
    static __device__ __forceinline__ void put_signal(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes, Remote flag, std::uint64_t epoch) {
        if (is_lsa(ctx, peer)) {
            lsa_copy(ctx, peer, destination, source, bytes,
                     ncclCoopThread{});
            lsa_signal(ctx, peer, flag, epoch);
            return;
        }
        gin_put_signal(ctx, peer, destination, source, bytes,
                       ncclCoopThread{});
    }
    template <class Coop>
    static __device__ __forceinline__ void gin_put_signal(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes, Coop coop) {
        gin(ctx).put(ncclTeamWorld(ctx.dev), peer, ctx.window,
                     destination.offset, ctx.window, source.offset, bytes,
                     ncclGin_StrongSignalAdd{
                         static_cast<ncclGinSignal_t>(ctx.rank), 1},
                     ncclGin_None{}, coop);
    }
    // Warp-collective fused publication: the warp copies (LSA) or posts
    // (GIN) the slice, and the signal rides the same call.
    static constexpr bool kHasPutSignalWarp = true;
    static __device__ __forceinline__ void put_signal_warp(
        Ctx& ctx, int peer, Remote destination, Local source,
        std::uint32_t bytes, Remote flag, std::uint64_t epoch) {
        if (is_lsa(ctx, peer)) {
            lsa_copy(ctx, peer, destination, source, bytes, ncclCoopWarp{});
            __syncwarp();
            if (threadIdx.x % 32 == 0)
                lsa_signal(ctx, peer, flag, epoch);
            __syncwarp();
            return;
        }
        gin_put_signal(ctx, peer, destination, source, bytes,
                       ncclCoopWarp{});
    }
    // The hand-batched window (--batch-window on only): every put but the
    // last skips the doorbell (ncclGinOptFlagsAggregateRequests), the last
    // rings it for all of them, and the flush retires the window.
    static constexpr bool kHasPutManyFlush = true;
    static __device__ __forceinline__ void put_many_flush(
        Ctx& ctx, int peer, const PutDescT* descriptors,
        std::uint32_t count) {
        for (std::uint32_t index = 0; index < count; ++index)
            put_flagged(ctx, peer, Remote{descriptors[index].destination},
                        Local{descriptors[index].source},
                        descriptors[index].bytes,
                        index + 1 < count ? ncclGinOptFlagsAggregateRequests
                                          : ncclGinOptFlagsDefault);
        quiet(ctx, peer);
    }
    // Only the put_many path names this (the GIN arm has no put_many, so
    // it is unreachable today): N-1 puts and a signal-carrying last put on
    // the same context — the strong signal covers the earlier ones.
    static __device__ __forceinline__ void put_many_signal(
        Ctx& ctx, int peer, const PutDescT* descriptors,
        std::uint32_t count, Remote flag, std::uint64_t epoch) {
        for (std::uint32_t index = 0; index + 1 < count; ++index)
            put(ctx, peer, Remote{descriptors[index].destination},
                Local{descriptors[index].source}, descriptors[index].bytes);
        const PutDescT& last = descriptors[count - 1];
        put_signal(ctx, peer, Remote{last.destination}, Local{last.source},
                   last.bytes, flag, epoch);
    }
};
