#pragma once

// A doorbell agent is the CPU half of a staged GPU-initiated datapath: the
// GPU builds hardware work-queue entries itself and advances a host-pinned
// producer cell; a pinned agent thread watches that cell and performs the one
// action the platform will not accept from the GPU. What that action is
// belongs to the transport — for EFA the send queue is a push-mode window in
// the adapter's BAR, so the hook copies the newly staged 64-byte entries into
// it and writes the doorbell; a fetch-mode transport such as mlx5 would leave
// its queue in GPU memory untouched and only ring.
//
// The worker skeleton mirrors ProxyWorker in proxy/aboleth_proxy.cu:
// registration under a mutex with an atomic generation the loop observes, so
// endpoints can come and go while the thread runs, and removal blocks until
// the loop can no longer touch the departing endpoint's memory.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <immintrin.h>

#include "../aboleth_runtime_common.h"

namespace aboleth::runtime {

class DoorbellAgent {
public:
    // Submit hook: move staged entries in [from, to) to the adapter.
    // Producer counters carry doorbell semantics — a running count truncated
    // to 32 bits — so all arithmetic is wraparound-safe deltas.
    using Submit =
        std::function<void(std::uint32_t from, std::uint32_t to)>;

    // Placement hook, run once on the agent thread before it starts
    // polling. Empty keeps the historical EFA placement
    // (pin_efa_proxy_worker's per-core formula); mlx5 agents pass
    // pin_numa_local to claim the GPU's NUMA node at cpuset granularity.
    using PinFn = std::function<void(int gpu_index, int shard)>;

    explicit DoorbellAgent(
        int gpu_index, int shard = 0, PinFn pin = {})
        : gpu_index_(gpu_index),
          shard_(shard),
          pin_(std::move(pin)) {
        thread_ = std::thread([this] { run(); });
    }

    ~DoorbellAgent() {
        stop_.store(true, std::memory_order_release);
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    DoorbellAgent(const DoorbellAgent&) = delete;
    DoorbellAgent& operator=(const DoorbellAgent&) = delete;

    // `cell` must stay valid until remove() returns.
    std::uint64_t add(
        const volatile std::uint32_t* cell,
        Submit submit) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::uint64_t token = next_token_++;
        entries_.push_back(
            {token, cell, std::move(submit), 0});
        if (std::getenv("ABOLETH_DOORBELL_DEBUG") !=
            nullptr) {
            std::fprintf(
                stderr,
                "doorbell-agent gpu=%d add token=%llu cell=%p "
                "value=%u\n",
                gpu_index_,
                static_cast<unsigned long long>(token),
                static_cast<const void*>(
                    const_cast<const std::uint32_t*>(cell)),
                *cell);
            std::fflush(stderr);
        }
        publish_locked();
        return token;
    }

    // Blocks until the worker loop has picked up the removal, after which
    // the endpoint's staging memory and queues may be freed.
    void remove(std::uint64_t token) {
        std::uint64_t target = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (std::size_t index = 0;
                 index < entries_.size();
                 ++index) {
                if (entries_[index].token == token) {
                    entries_.erase(
                        entries_.begin() + index);
                    break;
                }
            }
            target = publish_locked();
        }
        while (observed_.load(std::memory_order_acquire) <
               target) {
            _mm_pause();
        }
    }

private:
    struct Entry {
        std::uint64_t token;
        const volatile std::uint32_t* cell;
        Submit submit;
        std::uint32_t submitted;
    };

    std::uint64_t publish_locked() {
        return generation_.fetch_add(
                   1, std::memory_order_release) +
            1;
    }

    void run() {
        if (pin_) {
            pin_(gpu_index_, shard_);
        } else {
            pin_efa_proxy_worker(gpu_index_, shard_);
        }
        std::vector<Entry> active;
        std::uint64_t seen = 0;
        // ABOLETH_DOORBELL_WATCHDOG: report agent liveness while no
        // submissions arrive. During a fleet-wide stall this line is the
        // discriminator between "GPU stopped staging work" (counts frozen,
        // agent healthy — the wedge is device-side) and a dead agent
        // thread (no line at all).
        const bool watchdog =
            std::getenv("ABOLETH_DOORBELL_WATCHDOG") != nullptr;
        auto last_progress = std::chrono::steady_clock::now();
        auto last_report = last_progress;
        while (!stop_.load(std::memory_order_acquire)) {
            const std::uint64_t current =
                generation_.load(std::memory_order_acquire);
            if (current != seen) {
                std::lock_guard<std::mutex> lock(mutex_);
                // Keep each entry's submitted count across refreshes; a new
                // snapshot must not re-submit old work.
                std::vector<Entry> refreshed = entries_;
                for (auto& entry : refreshed) {
                    for (const auto& previous : active) {
                        if (previous.token == entry.token) {
                            entry.submitted =
                                previous.submitted;
                        }
                    }
                }
                active = std::move(refreshed);
                seen = current;
                observed_.store(
                    seen, std::memory_order_release);
            }
            bool progressed = false;
            for (auto& entry : active) {
                const std::uint32_t staged = *entry.cell;
                if (staged != entry.submitted) {
                    entry.submit(entry.submitted, staged);
                    entry.submitted = staged;
                    progressed = true;
                }
            }
            if (!progressed) {
                _mm_pause();
                if (watchdog) {
                    const auto now =
                        std::chrono::steady_clock::now();
                    if (now - last_progress >=
                            std::chrono::seconds(5) &&
                        now - last_report >=
                            std::chrono::seconds(5)) {
                        std::uint64_t forwarded = 0;
                        for (const auto& entry : active) {
                            forwarded += entry.submitted;
                        }
                        std::fprintf(
                            stderr,
                            "[aboleth-doorbell] gpu %d shard %d "
                            "alive, no new submissions for 5s: "
                            "%zu entries, forwarded-sum %llu\n",
                            gpu_index_, shard_, active.size(),
                            static_cast<unsigned long long>(
                                forwarded));
                        last_report = now;
                    }
                }
            } else if (watchdog) {
                last_progress =
                    std::chrono::steady_clock::now();
            }
        }
    }

    const int gpu_index_;
    const int shard_;
    const PinFn pin_;
    std::mutex mutex_;
    std::vector<Entry> entries_;
    std::uint64_t next_token_{1};
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<std::uint64_t> observed_{0};
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

// ABOLETH_DOORBELL_AGENTS: agent threads per (owner, GPU), 1 by default.
// Endpoints are dealt to them round-robin in creation order, so each thread
// sweeps and rings a disjoint set of queues.
inline int doorbell_agents_per_gpu() {
    const char* configured = std::getenv("ABOLETH_DOORBELL_AGENTS");
    if (configured == nullptr || configured[0] == '\0') {
        return 1;
    }
    const int agents = std::atoi(configured);
    if (agents < 1 || agents > 64) {
        throw_error("ABOLETH_DOORBELL_AGENTS must be 1..64");
    }
    return agents;
}

// Doorbell agents live ABOLETH_DOORBELL_AGENTS per (owner, GPU), shared by
// the staged endpoints on that GPU and torn down with the last of them. The
// owner pointer scopes agents to a runtime instance; any stable per-runtime
// address works. The first acquisition on a key decides the placement hook —
// later callers on the same key share the running thread as-is.
inline std::shared_ptr<DoorbellAgent> acquire_doorbell_agent(
    const void* owner,
    int gpu_index,
    DoorbellAgent::PinFn pin = {}) {
    static std::mutex registry_mutex;
    static std::map<
        std::tuple<const void*, int, int>,
        std::weak_ptr<DoorbellAgent>>
        registry;
    static std::map<std::pair<const void*, int>, int> dealt;
    std::lock_guard<std::mutex> lock(registry_mutex);
    const int shard =
        dealt[std::make_pair(owner, gpu_index)]++ % doorbell_agents_per_gpu();
    const auto key = std::make_tuple(owner, gpu_index, shard);
    auto existing = registry[key].lock();
    if (existing != nullptr) {
        return existing;
    }
    auto agent = std::make_shared<DoorbellAgent>(
        gpu_index, shard, std::move(pin));
    registry[key] = agent;
    return agent;
}

// The EFA submit hook: the send queue is a push-mode window in the adapter's
// BAR — the device never fetches descriptors from memory — so newly staged
// 64-byte slots are copied in at the same indices and the doorbell carries
// the producer count. A fetch-mode transport (mlx5) would ring without
// copying. Both EFA control planes (raw efadv and FI_EFA_GDA_OPS) produce
// the same queue shape, so they share this hook.
inline DoorbellAgent::Submit make_efa_llq_submit(
    std::uint8_t* adapter_queue,
    volatile std::uint32_t* adapter_doorbell,
    const std::uint8_t* staging_ring,
    std::uint32_t entry_bytes,
    std::uint32_t slot_mask) {
    const bool debug =
        std::getenv("ABOLETH_DOORBELL_DEBUG") != nullptr;
    return [adapter_queue, adapter_doorbell, staging_ring,
            entry_bytes, slot_mask, debug](
               std::uint32_t from, std::uint32_t to) {
        if (debug) {
            std::fprintf(
                stderr,
                "doorbell-agent submit [%u, %u) ring=%p\n",
                from,
                to,
                static_cast<const void*>(staging_ring));
            std::fflush(stderr);
        }
        for (std::uint32_t producer = from; producer != to;
             ++producer) {
            const std::uint32_t slot = producer & slot_mask;
            volatile std::uint64_t* destination =
                reinterpret_cast<volatile std::uint64_t*>(
                    adapter_queue +
                    static_cast<std::size_t>(slot) *
                        entry_bytes);
            const std::uint64_t* source =
                reinterpret_cast<const std::uint64_t*>(
                    staging_ring +
                    static_cast<std::size_t>(slot) *
                        entry_bytes);
            // One 64-byte descriptor per slot, as eight 64-bit
            // stores; entry_bytes is validated as 64 by both
            // control planes before an endpoint reaches here.
            for (int word = 0; word < 8; ++word) {
                destination[word] = source[word];
            }
        }
        __sync_synchronize();
        *adapter_doorbell = to;
        __sync_synchronize();
    };
}

// QPs created from one verbs context can be assigned the same BlueFlame
// register, and endpoints on different GPUs poll from different agent
// threads, so the MMIO store takes a per-register lock — the same
// serialization libmlx5 applies to shared registers. A spinlock: a
// contended std::mutex parks the losing agent in the kernel for
// microseconds. Resolved once at hook creation.
class UarSpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            _mm_pause();
        }
    }
    void unlock() { flag_.clear(std::memory_order_release); }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

inline UarSpinLock& uar_register_lock(
    const volatile void* uar_register) {
    static std::mutex map_mutex;
    static std::map<
        const volatile void*,
        std::unique_ptr<UarSpinLock>>
        locks;
    std::lock_guard<std::mutex> lock(map_mutex);
    auto& slot = locks[uar_register];
    if (slot == nullptr) {
        slot = std::make_unique<UarSpinLock>();
    }
    return *slot;
}

// The mlx5 submit hook: the NIC fetches WQEs from memory on its own, so the
// agent forwards only the doorbell word the kernel already release-published
// into the endpoint's staged cell. The DBREC — written before that release
// store, and therefore visible by the time the agent wakes — carries the
// authoritative count, which is what makes coalescing to the latest observed
// value safe. The word is rebuilt from the observed low half and the
// endpoint's constant byte-swapped QPN half instead of re-read from the
// cell, so a racing publisher cannot hand the agent a torn value.
inline DoorbellAgent::Submit make_mlx5_uar_submit(
    volatile std::uint64_t* uar_register,
    std::uint32_t qpn) {
    const std::uint32_t qpn_be =
        __builtin_bswap32(qpn << 8);
    UarSpinLock& lock = uar_register_lock(uar_register);
    // ABOLETH_DOORBELL_LOCK=none: the doorbell is one aligned 8-byte store
    // (no BlueFlame copy), atomic on x86 even into a shared register, so
    // agents sharing a UAR need no lock.
    const char* lock_mode = std::getenv("ABOLETH_DOORBELL_LOCK");
    const bool locked =
        lock_mode == nullptr || std::strcmp(lock_mode, "none") != 0;
    const bool debug =
        std::getenv("ABOLETH_DOORBELL_DEBUG") != nullptr;
    return [uar_register, qpn_be, &lock, debug, locked](
               std::uint32_t /*from*/, std::uint32_t to) {
        if (debug) {
            std::fprintf(
                stderr,
                "doorbell-agent ring uar=%p index=%08x\n",
                static_cast<const void*>(
                    const_cast<const std::uint64_t*>(
                        uar_register)),
                to);
            std::fflush(stderr);
        }
        const std::uint64_t doorbell =
            (static_cast<std::uint64_t>(qpn_be) << 32) |
            to;
        if (locked) {
            lock.lock();
        }
        __sync_synchronize();
        *uar_register = doorbell;
        __sync_synchronize();
        if (locked) {
            lock.unlock();
        }
    };
}

} // namespace aboleth::runtime
