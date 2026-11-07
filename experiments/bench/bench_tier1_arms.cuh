#pragma once

// Shared host driver for the Tier-1 CUDA benchmark arms: option parsing,
// the ladder/bandwidth/alltoall run loops, and reporting — everything
// except transport bring-up, which is the only thing an arm binary adds.
// The loops are templated over the same comm policy the kernels use, so
// every arm runs literally the same benchmark.
//
// Measurement is the device %globaltimer everywhere, identical to the
// Python-hosted arms: latency samples each round trip on the leader,
// bandwidth takes bytes over max(end) - min(start) of the per-CTA spans
// the windowed kernel records, alltoall samples each round on rank 0.
// Every row carries its raw samples, and every record the kernel's
// resources, the library's device-memory footprint and the versions.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>

#include "bench_common.h"
#include "bench_kernels.cuh"

namespace bench {

#define BENCH_CUDA_CHECK(call)                                             \
    do {                                                                   \
        const cudaError_t status_ = (call);                                \
        if (status_ != cudaSuccess) {                                      \
            std::fprintf(stderr, "bench: %s failed: %s\n", #call,          \
                         cudaGetErrorString(status_));                     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

struct Options {
    std::string bench{"ladder"};      // ladder | alltoall
    std::string mode{"latency"};      // latency | bandwidth
    std::string transport;            // arm-specific; default in the binary
    std::string op{"put"};            // put | put+signal | put_inline
    std::uint64_t min_bytes{0};
    std::uint64_t max_bytes{0};
    std::uint64_t step_factor{2};
    std::uint64_t step_bytes{0};
    std::uint32_t warmup{50};
    std::uint32_t iters{50};
    std::uint32_t puts{1};
    std::uint32_t window{32};
    std::uint32_t ctas{1};            // windowed-bandwidth grid size
    std::uint32_t issuers{1};         // windowed bandwidth: threads per CTA
    bool stream{false};               // windowed bandwidth: retire at the end
    std::string put_issue{"thread"};  // thread | warp  -> kernels' kWarpC
    std::string fuse_signal{"off"};   // off | on       -> kernels' kFuseC
    std::uint32_t gin_contexts{0};    // NCCL device API cap; 0: one per peer
    std::uint32_t graph_rounds{0};    // NCCL host: rounds per CUDA graph
    bool batch_window{false};         // windows as one hand-batched call
    bool collective{false};           // the library's own all-to-all
    bool smoke{false};
    std::vector<std::string> clamped;  // every "# clamp:" note, recorded
};

inline std::uint64_t parse_size_arg(const char* text) {
    char* end = nullptr;
    double value = std::strtod(text, &end);
    switch (*end) {
        case 'K': case 'k': value *= 1024.0; break;
        case 'M': case 'm': value *= 1024.0 * 1024.0; break;
        case 'G': case 'g': value *= 1024.0 * 1024.0 * 1024.0; break;
        case '\0': break;
        default:
            std::fprintf(stderr, "bench: bad size %s\n", text);
            std::exit(2);
    }
    return static_cast<std::uint64_t>(value);
}

inline Options parse_options(int argc, char** argv,
                             const char* default_transport) {
    Options options;
    options.transport = default_transport;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        auto value = [&]() -> const char* {
            if (index + 1 >= argc) {
                std::fprintf(stderr, "bench: %s needs a value\n",
                             flag.c_str());
                std::exit(2);
            }
            return argv[++index];
        };
        if (flag == "--bench") options.bench = value();
        else if (flag == "--mode") options.mode = value();
        else if (flag == "--transport") options.transport = value();
        else if (flag == "--op") options.op = value();
        else if (flag == "-b" || flag == "--minbytes")
            options.min_bytes = parse_size_arg(value());
        else if (flag == "-e" || flag == "--maxbytes")
            options.max_bytes = parse_size_arg(value());
        else if (flag == "-f" || flag == "--stepfactor")
            options.step_factor = std::strtoull(value(), nullptr, 10);
        else if (flag == "-i" || flag == "--stepbytes")
            options.step_bytes = parse_size_arg(value());
        else if (flag == "-w" || flag == "--warmup")
            options.warmup = std::strtoul(value(), nullptr, 10);
        else if (flag == "-n" || flag == "--iters")
            options.iters = std::strtoul(value(), nullptr, 10);
        else if (flag == "--puts")
            options.puts = std::strtoul(value(), nullptr, 10);
        else if (flag == "--window")
            options.window = std::strtoul(value(), nullptr, 10);
        else if (flag == "--ctas")
            options.ctas = std::strtoul(value(), nullptr, 10);
        else if (flag == "--issuers")
            options.issuers = std::strtoul(value(), nullptr, 10);
        else if (flag == "--stream")
            options.stream = std::string(value()) == "on";
        else if (flag == "--put-issue") options.put_issue = value();
        else if (flag == "--fuse-signal") options.fuse_signal = value();
        else if (flag == "--gin-contexts")
            options.gin_contexts = std::strtoul(value(), nullptr, 10);
        else if (flag == "--graph-rounds")
            options.graph_rounds = std::strtoul(value(), nullptr, 10);
        else if (flag == "--batch-window")
            options.batch_window = std::string(value()) == "on";
        else if (flag == "--collective")
            options.collective = std::string(value()) == "on";
        else if (flag == "--smoke") options.smoke = true;
        else {
            std::fprintf(stderr, "bench: unknown flag %s\n", flag.c_str());
            std::exit(2);
        }
    }
    if (options.put_issue != "thread" && options.put_issue != "warp") {
        std::fprintf(stderr, "bench: --put-issue thread|warp\n");
        std::exit(2);
    }
    if (options.fuse_signal != "off" && options.fuse_signal != "on") {
        std::fprintf(stderr, "bench: --fuse-signal off|on\n");
        std::exit(2);
    }
    if (options.ctas == 0 || options.window == 0 || options.puts == 0) {
        std::fprintf(stderr, "bench: --ctas, --window and --puts are >= 1\n");
        std::exit(2);
    }
    if (options.issuers != 1 &&
        (options.issuers == 0 || options.issuers % 32 ||
         options.issuers > 1024 || options.window % options.issuers)) {
        std::fprintf(stderr, "bench: --issuers is 1, or a multiple of 32 "
                             "(at most 1024) that divides --window\n");
        std::exit(2);
    }
    const bool latency =
        options.bench == "ladder" && options.mode == "latency";
    if (options.min_bytes == 0)
        options.min_bytes = options.bench == "alltoall" ? 1024
                            : latency ? 8 : 4096;
    if (options.max_bytes == 0)
        options.max_bytes = options.bench == "alltoall" ? (64ull << 20)
                            : latency ? (64ull << 10) : (256ull << 20);
    if (options.smoke) {
        options.warmup = 2;
        options.iters = 5;
    }
    return options;
}

inline std::vector<std::uint64_t> size_ladder(const Options& options) {
    std::vector<std::uint64_t> sizes;
    for (std::uint64_t size = options.min_bytes; size <= options.max_bytes;
         size = options.step_factor > 1 ? size * options.step_factor
                                        : size + options.step_bytes) {
        sizes.push_back(size);
        if (options.step_factor <= 1 && options.step_bytes == 0)
            break;
    }
    if (options.smoke && sizes.size() > 2)
        sizes = {sizes.front(), sizes.back()};
    return sizes;
}

inline int env_int(const char* name, int fallback = -1) {
    const char* text = std::getenv(name);
    if (text == nullptr) {
        if (fallback >= 0)
            return fallback;
        std::fprintf(stderr,
                     "bench: %s is unset — launch through experiments/bench "
                     "or scripts/multinode-launch.sh\n", name);
        std::exit(2);
    }
    return std::atoi(text);
}

// Print a clamp loudly and keep it for the record's clamped[].
inline void note_clamp(Options& options, const std::string& note) {
    std::printf("# clamp: %s\n", note.c_str());
    options.clamped.push_back(note);
}

// Shrink max_bytes until the arena fits free GPU memory, loudly.
inline void clamp_options(Options& options, int world) {
    std::size_t free_bytes = 0, total_bytes = 0;
    BENCH_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::uint64_t budget =
        static_cast<std::uint64_t>(free_bytes * 0.8);
    auto need = [&](std::uint64_t size) {
        if (options.bench == "alltoall")
            return 2ull * world * size + (1ull << 20);
        if (options.mode == "bandwidth")
            return static_cast<std::uint64_t>(options.window) * size +
                   (1ull << 20);
        const std::uint64_t bytes = size * options.puts;
        return (bytes < 8 ? 8 : bytes) + (1ull << 20);
    };
    while (options.max_bytes > options.min_bytes &&
           need(options.max_bytes) > budget) {
        char note[160];
        std::snprintf(
            note, sizeof(note),
            "size %llu dropped: needs %llu B, budget %llu B",
            static_cast<unsigned long long>(options.max_bytes),
            static_cast<unsigned long long>(need(options.max_bytes)),
            static_cast<unsigned long long>(budget));
        note_clamp(options, note);
        options.max_bytes /=
            options.step_factor > 1 ? options.step_factor : 2;
    }
}

// Arena sizing after the clamp; stride is the alltoall per-peer stride.
inline std::uint64_t plan_data_bytes(const Options& options, int world,
                                     std::uint64_t* stride) {
    const auto sizes = size_ladder(options);
    const std::uint64_t top = sizes.back();
    *stride = 0;
    if (options.bench == "alltoall") {
        *stride = top;
        return 2ull * world * top;
    }
    if (options.mode == "bandwidth")
        return static_cast<std::uint64_t>(options.window) * top;
    const std::uint64_t bytes = top * options.puts;
    return bytes < 8 ? 8 : bytes;
}

// Free device memory now (cudaMemGetInfo). The arms bracket library
// bring-up with it: free after cudaSetDevice + cudaFree(0), and free right
// after the library is initialized and connected, before the benchmark
// arena is allocated — the difference is what the library itself holds.
inline std::uint64_t device_free_bytes() {
    std::size_t free_bytes = 0, total_bytes = 0;
    BENCH_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// The record's "memory" object; `extra` adds arm-specific members.
inline std::string memory_json(std::uint64_t free_before,
                               std::uint64_t free_after,
                               const std::string& extra = "") {
    char text[256];
    std::snprintf(
        text, sizeof(text),
        "{\"free_after\": %llu, \"free_before\": %llu, "
        "\"library_bytes\": %lld",
        static_cast<unsigned long long>(free_after),
        static_cast<unsigned long long>(free_before),
        static_cast<long long>(free_before) -
            static_cast<long long>(free_after));
    std::string json = text;
    if (!extra.empty())
        json += ", " + extra;
    return json + "}";
}

// Everything the run loops need from a bring-up; "addresses" are whatever
// the arm's make_remote/make_local consume (VAs for aboleth, symmetric VAs
// for nvshmem, window offsets for nccl-gin). `arena` is always the real
// device VA the fills/checks use; `local_data_base` and `flag_base` are the
// comm-address forms of the arena and flag area (identical to the VAs for
// memory-addressed arms, 0-based window offsets for nccl-gin).
template <class Comm>
struct Arm {
    typename Comm::Ctx ctx{};
    std::uint8_t* arena{nullptr};
    std::uint64_t data_bytes{0};
    std::uint64_t flags_offset{0};
    std::uint64_t local_data_base{0};
    std::uint64_t flag_base{0};
    int rank{0};
    int world{0};
    std::vector<std::uint64_t> peer_data;
    std::vector<std::uint64_t> peer_flags;
    std::vector<std::uint32_t> peer_rkey;
    std::uint32_t lkey{0};
    const char* name{"?"};
    const char* submit_mode{"n/a"};
    // Largest bandwidth grid the arm can drive (0 = no cap).
    std::uint32_t max_ctas{0};
    // Record fragments the bring-up fills: the "memory" object, extra
    // "versions" members (",\"nccl\": ..." style, comma-led) and an
    // arm-specific "arm_config" object ("" = omitted).
    std::string memory;
    std::string versions;
    std::string config;
};

// Reset the flag state between launches: memory-backed flags are zeroed;
// counter-backed flags (nccl-gin) instead advance the policy's epoch base by
// the rounds the previous launch consumed.
template <class Comm>
inline void reset_flags(Arm<Comm>& arm, std::uint64_t rounds_consumed) {
    if (Comm::kFlagsInMemory)
        BENCH_CUDA_CHECK(cudaMemset(arm.arena + arm.flags_offset, 0,
                                    8ull * arm.world));
    else
        Comm::advance(arm.ctx, rounds_consumed);
}

// Optional policy hook: `set_round_shape(Ctx&, puts, inbound_signals)`
// tells the policy how many payload puts a round carries and how many
// signals the peer publishes toward this rank per round (counter-backed
// waits scale by it; NVSHMEM fences a fused multi-put round).
namespace detail {
template <class Comm, class = void>
struct has_round_shape : std::false_type {};
template <class Comm>
struct has_round_shape<
    Comm, std::void_t<decltype(Comm::set_round_shape(
              std::declval<typename Comm::Ctx&>(), 0u, 0u))>>
    : std::true_type {};
}  // namespace detail

template <class Comm>
inline void set_round_shape(typename Comm::Ctx& ctx, std::uint32_t puts,
                            std::uint32_t inbound_signals) {
    if constexpr (detail::has_round_shape<Comm>::value)
        Comm::set_round_shape(ctx, puts, inbound_signals);
}

// Optional policy hook: `set_issuers(Ctx&, issuers)` tells the policy how
// many threads of a CTA post on its queue at once (NCCL GIN picks its
// resource-sharing scope from it).
namespace detail {
template <class Comm, class = void>
struct has_issuers : std::false_type {};
template <class Comm>
struct has_issuers<
    Comm, std::void_t<decltype(Comm::set_issuers(
              std::declval<typename Comm::Ctx&>(), 0u))>>
    : std::true_type {};
}  // namespace detail

template <class Comm>
inline void set_issuers(typename Comm::Ctx& ctx, std::uint32_t issuers) {
    if constexpr (detail::has_issuers<Comm>::value)
        Comm::set_issuers(ctx, issuers);
}

struct Row {
    std::string op;
    std::uint64_t size;
    std::uint64_t count;
    double p50, p95, p99, mean;
    double algbw, busbw;
    std::uint64_t wrong;
    // Raw data: the per-iteration device samples behind the percentiles.
    std::vector<std::uint64_t> samples_ns{};
    // Bandwidth rows: the device span and the shape that moved the bytes.
    bool bandwidth{false};
    std::uint64_t span_ns{0};
    std::uint32_t ctas{0}, window{0}, iters{0};
    // Alltoall rows: every rank's mean round.
    std::vector<double> rank_mean_us{};
};

struct Stats {
    double p50, p95, p99, mean;
};

inline Stats stats_us(const std::vector<double>& samples) {
    double sum = 0.0;
    for (const double value : samples)
        sum += value;
    return {quantile_inclusive(samples, 0.50),
            quantile_inclusive(samples, 0.95),
            quantile_inclusive(samples, 0.99),
            samples.empty() ? 0.0 : sum / samples.size()};
}

inline std::uint64_t share_max(Rendezvous& rendezvous, std::uint64_t mine,
                               int world) {
    const auto gathered = rendezvous.allgather(&mine, sizeof(mine));
    std::uint64_t highest = 0;
    for (int peer = 0; peer < world; ++peer) {
        std::uint64_t value = 0;
        std::memcpy(&value, gathered.data() + peer * sizeof(value),
                    sizeof(value));
        highest = std::max(highest, value);
    }
    return highest;
}

// Every rank's value, in rank order.
inline std::vector<double> share_all(Rendezvous& rendezvous, double mine,
                                     int world) {
    const auto gathered = rendezvous.allgather(&mine, sizeof(mine));
    std::vector<double> values(static_cast<std::size_t>(world), 0.0);
    std::memcpy(values.data(), gathered.data(), sizeof(double) * world);
    return values;
}

inline int op_code(const std::string& op) {
    if (op == "put") return kOpPut;
    if (op == "put+signal") return kOpPutSignal;
    if (op == "put_inline") return kOpPutInline;
    std::fprintf(stderr, "bench: unknown --op %s\n", op.c_str());
    std::exit(2);
}

inline std::uint8_t route_salt(int sender, int receiver) {
    return static_cast<std::uint8_t>((sender * 29 + receiver * 13) & 0xFF);
}

// The timed launch moves a differently salted pattern into zeroed
// receivers, so the after-timing check sees only what the timed rounds
// delivered (not the correctness pass's bytes left in place).
constexpr std::uint8_t kTimedSalt = 0xA5;

inline std::vector<std::uint64_t> copy_samples(const std::uint64_t* device,
                                               std::uint32_t count) {
    std::vector<std::uint64_t> raw(count);
    if (count)
        BENCH_CUDA_CHECK(cudaMemcpy(raw.data(), device,
                                    sizeof(std::uint64_t) * count,
                                    cudaMemcpyDeviceToHost));
    return raw;
}

inline std::vector<double> to_us(const std::vector<std::uint64_t>& raw) {
    std::vector<double> us(raw.begin(), raw.end());
    for (double& value : us)
        value /= 1000.0;
    return us;
}

// The inbound 8-byte flag word for `peer` (memory-backed flags).
template <class Comm>
inline std::uint64_t read_flag(const Arm<Comm>& arm, int peer) {
    std::uint64_t flag = 0;
    BENCH_CUDA_CHECK(cudaMemcpy(
        &flag, arm.arena + arm.flags_offset + 8ull * peer, sizeof(flag),
        cudaMemcpyDeviceToHost));
    return flag;
}

inline void report_wrong(const char* when, std::uint64_t size,
                         std::uint64_t wrong) {
    if (wrong)
        std::fprintf(stderr, "FAIL: size %llu: %llu wrong (%s)\n",
                     static_cast<unsigned long long>(size),
                     static_cast<unsigned long long>(wrong), when);
}

// ---- issue shape -> kernel instantiation ----------------------------------

// --put-issue / --fuse-signal map onto the kernels' kWarpC / kFuseC exactly
// as the Python Aboleth arm bakes them (ladder.py render_prelude).
struct Issue {
    bool warp;
    bool fuse;
};

inline Issue issue_of(const Options& options) {
    return {options.put_issue == "warp", options.fuse_signal == "on"};
}

// Refuse issue shapes the policy or the kernels cannot run, with the same
// rules as ladder.py / alltoall.py. Returns an exit code (0 = runnable).
template <class Comm>
int check_issue(const Options& options, const char* binary) {
    const Issue issue = issue_of(options);
    auto refuse = [&](const char* why) {
        std::fprintf(stderr, "%s: %s\n", binary, why);
        return 2;
    };
    if (issue.warp && !bench_detail::has_put_warp<Comm>::value)
        return refuse("--put-issue warp: this arm's comm policy has no "
                      "warp-cooperative put (put_warp)");
    if (issue.fuse && !bench_detail::has_put_signal<Comm>::value)
        return refuse("--fuse-signal on: this arm's comm policy has no "
                      "fused put_signal");
    if (issue.warp && issue.fuse &&
        !bench_detail::has_put_signal_warp<Comm>::value)
        return refuse("--fuse-signal on with --put-issue warp: this arm's "
                      "comm policy has no warp-collective put_signal");
    const bool latency =
        options.bench == "ladder" && options.mode == "latency";
    if (latency && issue.warp && options.op != "put")
        return refuse("--put-issue warp supports --op put only");
    if (latency && issue.fuse && options.op != "put")
        return refuse("--fuse-signal on supports --op put only");
    if (!latency && options.bench == "ladder" && issue.fuse)
        std::printf("# note: --fuse-signal has no effect on the windowed "
                    "bandwidth kernel (it posts no signals)\n");
    if (options.batch_window &&
        (!bench_detail::has_put_many_flush<Comm>::value || issue.warp ||
         latency || options.bench != "ladder"))
        return refuse("--batch-window on: lane-issued windowed bandwidth "
                      "only, on a policy with a hand-batched window "
                      "(put_many_flush)");
    if (options.issuers > 1 &&
        (latency || options.bench != "ladder" || issue.warp ||
         options.batch_window))
        return refuse("--issuers > 1: lane-issued per-put windowed "
                      "bandwidth only");
    if (options.stream &&
        (!bench_detail::self_reclaim<Comm>::value || latency ||
         options.bench != "ladder" || options.batch_window))
        return refuse("--stream on: windowed bandwidth on a policy that "
                      "reclaims its own queue (no per-window completion)");
    return 0;
}

// One struct per benchmark kernel naming the instantiation an issue shape
// selects; the runtime arguments override the shape constants (0 / -1).
template <class Comm, bool kWarp, bool kFuse>
struct PingpongKernel {
    static auto get() {
        return &bench_pingpong<Comm, 0, 0, -1, kWarp, kFuse>;
    }
    static constexpr const char* kName = "bench_pingpong";
};
template <class Comm, bool kWarp, bool kFuse>
struct WindowedKernel {
    // The windowed kernel posts no signals: kFuse selects nothing here.
    static auto get() { return &bench_windowed_bw<Comm, 0, kWarp, false>; }
    static constexpr const char* kName = "bench_windowed_bw";
};
template <class Comm, bool kWarp, bool kFuse>
struct AlltoallKernel {
    static auto get() { return &bench_alltoall<Comm, 0, kFuse, kWarp>; }
    static constexpr const char* kName = "bench_alltoall";
};

// Only the shapes the policy supports are instantiated; check_issue has
// already refused the rest, so the fall-through is unreachable.
template <class Comm, template <class, bool, bool> class Kernel>
auto select_kernel(const Issue& issue)
    -> decltype(Kernel<Comm, false, false>::get()) {
    if (issue.warp && issue.fuse) {
        if constexpr (bench_detail::has_put_signal_warp<Comm>::value)
            return Kernel<Comm, true, true>::get();
    } else if (issue.warp && !issue.fuse) {
        if constexpr (bench_detail::has_put_warp<Comm>::value)
            return Kernel<Comm, true, false>::get();
    } else if (issue.fuse && !issue.warp) {
        if constexpr (bench_detail::has_put_signal<Comm>::value)
            return Kernel<Comm, false, true>::get();
    } else if (!issue.warp && !issue.fuse) {
        return Kernel<Comm, false, false>::get();
    }
    std::fprintf(stderr, "bench: unsupported issue shape reached %s\n",
                 Kernel<Comm, false, false>::kName);
    std::exit(2);
}

// Registers, local (spill) bytes and static shared bytes of the exact
// instantiation launched, plus its launch shape.
struct KernelInfo {
    const char* name{nullptr};
    int registers{0};
    int local_bytes{0};
    int shared_bytes{0};
    unsigned threads{0};
    unsigned ctas{0};
};

template <class Function>
KernelInfo kernel_info(Function* kernel, const char* name, unsigned threads,
                       unsigned ctas) {
    cudaFuncAttributes attributes{};
    BENCH_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    return {name, attributes.numRegs,
            static_cast<int>(attributes.localSizeBytes),
            static_cast<int>(attributes.sharedSizeBytes), threads, ctas};
}

// ---- run loops --------------------------------------------------------------

template <class Comm>
std::vector<Row> run_ladder_latency(const Options& options, Arm<Comm>& arm,
                                    Rendezvous& rendezvous,
                                    KernelInfo& info) {
    const int rank = arm.rank, peer = 1 - rank;
    const bool leader = rank == 0;
    const int op = op_code(options.op);
    auto sizes = size_ladder(options);
    if (op == kOpPutInline)
        sizes = {8};
    const std::uint32_t puts = op == kOpPutInline ? 1 : options.puts;
    // The follower receives one signal per put under put+signal; the
    // leader receives one echo per round.
    set_round_shape<Comm>(arm.ctx, puts,
                          !leader && op == kOpPutSignal ? puts : 1);
    const auto kernel = select_kernel<Comm, PingpongKernel>(issue_of(options));
    info = kernel_info(kernel, "bench_pingpong", 32, 1);
    std::uint64_t* samples = nullptr;
    BENCH_CUDA_CHECK(cudaMalloc(
        &samples, sizeof(std::uint64_t) * std::max(options.iters, 1u)));
    std::vector<std::uint8_t> host(arm.data_bytes);
    std::vector<Row> rows;

    const auto dst = Comm::make_remote(arm.peer_data[peer],
                                       arm.peer_rkey[peer]);
    const auto src = Comm::make_local(arm.local_data_base, arm.lkey);
    const auto peer_flag = Comm::make_remote(
        arm.peer_flags[peer] + 8ull * rank, arm.peer_rkey[peer]);
    const auto my_flag = Comm::make_flag(arm.flag_base + 8ull * peer);
    auto launch = [&](std::uint64_t size, std::uint32_t warmup,
                      std::uint32_t iters) {
        kernel<<<1, 32>>>(arm.ctx, peer, leader, dst, src, peer_flag,
                          my_flag, static_cast<std::uint32_t>(size), puts,
                          op, warmup, iters, samples, nullptr, 0, nullptr);
        BENCH_CUDA_CHECK(cudaGetLastError());
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
    };
    // The follower's payload against the pattern (put_inline: the flag).
    auto payload_wrong = [&](std::uint64_t payload,
                             std::uint64_t inline_flag,
                             std::uint8_t salt) -> std::uint64_t {
        if (leader)
            return 0;
        if (op == kOpPutInline)
            return read_flag(arm, peer) != inline_flag;
        BENCH_CUDA_CHECK(cudaMemcpy(host.data(), arm.arena, payload,
                                    cudaMemcpyDeviceToHost));
        return pattern_check_host(host.data(), payload, salt);
    };
    auto stage = [&](std::uint64_t payload, std::uint8_t salt) {
        if (leader) {
            pattern_fill_host(host.data(), payload, salt);
            BENCH_CUDA_CHECK(cudaMemcpy(arm.arena, host.data(), payload,
                                        cudaMemcpyHostToDevice));
        } else {
            BENCH_CUDA_CHECK(cudaMemset(arm.arena, 0, payload));
        }
    };

    const std::uint64_t rounds =
        static_cast<std::uint64_t>(options.warmup) + options.iters;
    std::uint64_t consumed = 0;
    for (const std::uint64_t size : sizes) {
        const std::uint64_t payload = size * puts;
        // Correctness before timing: the leader sends the pattern once.
        stage(payload, 0);
        reset_flags(arm, consumed);
        rendezvous.barrier();
        launch(size, 0, 1);
        rendezvous.barrier();
        std::uint64_t wrong = share_max(
            rendezvous, payload_wrong(payload, 1, 0), arm.world);
        report_wrong("before timing", size, wrong);

        stage(payload, kTimedSalt);
        reset_flags(arm, 1);
        rendezvous.barrier();
        launch(size, options.warmup, options.iters);
        rendezvous.barrier();
        consumed = rounds;

        // Correctness after timing: each side's inbound flag holds the
        // last round's epoch (memory-backed flags; under put_inline the
        // follower's flag IS the payload, checked below) and the
        // follower's zeroed payload holds the timed rounds' pattern.
        std::uint64_t late = 0;
        if (Comm::kFlagsInMemory && rounds &&
            (leader || op != kOpPutInline))
            late += read_flag(arm, peer) != rounds;
        late += payload_wrong(payload, rounds, kTimedSalt);
        late = share_max(rendezvous, late, arm.world);
        report_wrong("after timing", size, late);
        wrong += late;

        if (leader) {
            Row row{options.op, size, puts, 0, 0, 0, 0, 0, 0, wrong};
            row.samples_ns = copy_samples(samples, options.iters);
            const Stats stats = stats_us(to_us(row.samples_ns));
            const double gbps =
                stats.mean > 0.0 ? payload / (stats.mean * 1e-6) / 1e9 : 0.0;
            row.p50 = stats.p50;
            row.p95 = stats.p95;
            row.p99 = stats.p99;
            row.mean = stats.mean;
            row.algbw = row.busbw = gbps;
            rows.push_back(std::move(row));
        } else {
            rows.push_back(
                {options.op, size, puts, 0, 0, 0, 0, 0, 0, wrong});
        }
    }
    cudaFree(samples);
    return rows;
}

template <class Comm>
std::vector<Row> run_ladder_bandwidth(const Options& options, Arm<Comm>& arm,
                                      Rendezvous& rendezvous,
                                      std::uint32_t window,
                                      KernelInfo& info) {
    const int rank = arm.rank, peer = 1 - rank;
    const bool leader = rank == 0;
    const auto sizes = size_ladder(options);
    const std::uint32_t ctas = options.ctas;
    set_round_shape<Comm>(arm.ctx, 1, 1);
    // A hand-batched window (--batch-window on) is one library call that
    // rings one doorbell and retires itself; by default each put is its
    // own call, as written against the plain API.
    auto kernel = select_kernel<Comm, WindowedKernel>(issue_of(options));
    if constexpr (bench_detail::has_put_many_flush<Comm>::value)
        if (options.batch_window)
            kernel = &bench_windowed_bw<Comm, 0, false, true>;
    // Several issuers: the CTA is exactly that many threads, and a policy
    // that shares queues by thread scope learns it (set_issuers).
    const std::uint32_t issuers = options.issuers;
    if (issuers > 1 && window % issuers) {
        std::fprintf(stderr, "bench: --issuers %u does not divide the "
                             "arm's window %u\n", issuers, window);
        std::exit(2);
    }
    const unsigned threads = issuers > 1 ? issuers : 32;
    set_issuers<Comm>(arm.ctx, issuers);
    info = kernel_info(kernel, "bench_windowed_bw", threads, ctas);
    std::uint64_t* samples = nullptr;
    std::uint64_t* spans = nullptr;
    // A rank-wide quiet retires windows at a grid barrier (bench_windowed_bw).
    unsigned* barrier_state = nullptr;
    BENCH_CUDA_CHECK(cudaMalloc(
        &samples, sizeof(std::uint64_t) * std::max(options.iters, 1u)));
    BENCH_CUDA_CHECK(cudaMalloc(&spans, sizeof(std::uint64_t) * 2 * ctas));
    BENCH_CUDA_CHECK(cudaMalloc(&barrier_state, sizeof(unsigned)));
    // Descriptor scratch for windows that carry their own retirement.
    void* descs = nullptr;
    if constexpr (bench_detail::has_put_many_flush<Comm>::value)
        BENCH_CUDA_CHECK(cudaMalloc(
            &descs, sizeof(typename Comm::PutDescT) * ctas * window));
    std::vector<Row> rows;

    const auto dst = Comm::make_remote(arm.peer_data[peer],
                                       arm.peer_rkey[peer]);
    const auto src = Comm::make_local(arm.local_data_base, arm.lkey);
    auto launch = [&](std::uint64_t size, std::uint32_t grid,
                      std::uint32_t iters, std::uint64_t* samples_ns,
                      std::uint64_t* spans_ns) {
        // Zeroed per launch: the barrier's epochs assume one grid size.
        BENCH_CUDA_CHECK(cudaMemset(barrier_state, 0, sizeof(unsigned)));
        kernel<<<grid, threads>>>(arm.ctx, peer, dst, src,
                                  static_cast<std::uint32_t>(size), window,
                                  iters, arm.data_bytes, samples_ns, descs,
                                  spans_ns, issuers, barrier_state,
                                  options.stream);
        BENCH_CUDA_CHECK(cudaGetLastError());
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
    };

    for (const std::uint64_t size : sizes) {
        const std::uint64_t checked = size * window;
        std::vector<std::uint8_t> host(checked);
        auto receiver_wrong = [&](std::uint8_t salt) -> std::uint64_t {
            if (leader)
                return 0;
            BENCH_CUDA_CHECK(cudaMemcpy(host.data(), arm.arena, checked,
                                        cudaMemcpyDeviceToHost));
            return pattern_check_host(host.data(), checked, salt);
        };
        auto stage = [&](std::uint8_t salt) {
            if (leader) {
                pattern_fill_host(host.data(), checked, salt);
                BENCH_CUDA_CHECK(cudaMemcpy(arm.arena, host.data(), checked,
                                            cudaMemcpyHostToDevice));
            } else {
                BENCH_CUDA_CHECK(cudaMemset(arm.arena, 0, checked));
            }
        };
        // Correctness: one single-CTA iteration covers [0, window*size)
        // once; quiet means the data is resident, so a barrier suffices.
        stage(0);
        rendezvous.barrier();
        if (leader)
            launch(size, 1, 1, nullptr, nullptr);
        rendezvous.barrier();
        std::uint64_t wrong =
            share_max(rendezvous, receiver_wrong(0), arm.world);
        report_wrong("before timing", size, wrong);

        stage(kTimedSalt);
        rendezvous.barrier();
        if (leader) {
            launch(size, ctas, options.warmup, nullptr, nullptr);
            BENCH_CUDA_CHECK(
                cudaMemset(spans, 0, sizeof(std::uint64_t) * 2 * ctas));
            launch(size, ctas, options.iters, samples, spans);
        }
        rendezvous.barrier();
        // Correctness after timing: source and destination offsets are
        // equal and the first window covers [0, window*size), so the
        // zeroed receiver holds the timed launch's pattern.
        const std::uint64_t late =
            share_max(rendezvous, receiver_wrong(kTimedSalt), arm.world);
        report_wrong("after timing", size, late);
        wrong += late;

        Row row{"put", size, window, 0, 0, 0, 0, 0, 0, wrong};
        row.bandwidth = true;
        row.ctas = ctas;
        row.window = window;
        row.iters = options.iters;
        if (leader) {
            row.samples_ns = copy_samples(samples, options.iters);
            const auto marks = copy_samples(spans, 2 * ctas);
            std::uint64_t first = marks[0], last = marks[1];
            for (std::uint32_t cta = 1; cta < ctas; ++cta) {
                first = std::min(first, marks[2 * cta]);
                last = std::max(last, marks[2 * cta + 1]);
            }
            row.span_ns = last > first ? last - first : 0;
            const Stats stats = stats_us(to_us(row.samples_ns));
            row.p50 = stats.p50;
            row.p95 = stats.p95;
            row.p99 = stats.p99;
            row.mean = stats.mean;
            const double moved = static_cast<double>(ctas) * window *
                                 options.iters * size;
            // bytes per ns == decimal GB/s
            row.algbw = row.busbw =
                row.span_ns ? moved / static_cast<double>(row.span_ns) : 0.0;
        }
        rows.push_back(std::move(row));
    }
    cudaFree(samples);
    cudaFree(spans);
    cudaFree(barrier_state);
    cudaFree(descs);
    return rows;
}

template <class Comm>
std::vector<Row> run_alltoall(const Options& options, Arm<Comm>& arm,
                              Rendezvous& rendezvous, std::uint64_t stride,
                              KernelInfo& info) {
    const int rank = arm.rank, world = arm.world;
    const std::uint64_t recv_offset =
        static_cast<std::uint64_t>(world) * stride;
    const auto sizes = size_ladder(options);
    set_round_shape<Comm>(arm.ctx, 1, 1);
    const auto kernel = select_kernel<Comm, AlltoallKernel>(issue_of(options));
    info = kernel_info(kernel, "bench_alltoall", 32,
                       static_cast<unsigned>(world - 1));
    std::uint64_t* samples = nullptr;
    unsigned* barrier_state = nullptr;
    BENCH_CUDA_CHECK(cudaMalloc(
        &samples, sizeof(std::uint64_t) * std::max(options.iters, 1u)));
    BENCH_CUDA_CHECK(cudaMalloc(&barrier_state, sizeof(unsigned)));
    BENCH_CUDA_CHECK(cudaMemset(barrier_state, 0, sizeof(unsigned)));

    std::vector<typename Comm::Remote> dsts, peer_flags;
    std::vector<typename Comm::Local> srcs;
    std::vector<typename Comm::Flag> my_flags;
    for (int peer = 0; peer < world; ++peer) {
        if (peer == rank)
            continue;
        dsts.push_back(Comm::make_remote(
            arm.peer_data[peer] + recv_offset +
                static_cast<std::uint64_t>(rank) * stride,
            arm.peer_rkey[peer]));
        srcs.push_back(Comm::make_local(
            arm.local_data_base + static_cast<std::uint64_t>(peer) * stride,
            arm.lkey));
        peer_flags.push_back(Comm::make_remote(
            arm.peer_flags[peer] + 8ull * rank, arm.peer_rkey[peer]));
        my_flags.push_back(Comm::make_flag(arm.flag_base + 8ull * peer));
    }
    auto upload = [](const auto& table) {
        void* device = nullptr;
        const std::size_t bytes = table.size() * sizeof(table[0]);
        BENCH_CUDA_CHECK(cudaMalloc(&device, bytes));
        BENCH_CUDA_CHECK(cudaMemcpy(device, table.data(), bytes,
                                    cudaMemcpyHostToDevice));
        return device;
    };
    void* dsts_device = upload(dsts);
    void* srcs_device = upload(srcs);
    void* peer_flags_device = upload(peer_flags);
    void* my_flags_device = upload(my_flags);
    auto launch = [&](std::uint64_t size, std::uint32_t warmup,
                      std::uint32_t iters) {
        kernel<<<world - 1, 32>>>(
            arm.ctx, rank,
            static_cast<const typename Comm::Remote*>(dsts_device),
            static_cast<const typename Comm::Local*>(srcs_device),
            static_cast<const typename Comm::Remote*>(peer_flags_device),
            static_cast<const typename Comm::Flag*>(my_flags_device),
            static_cast<std::uint32_t>(size), warmup, iters, samples,
            barrier_state);
        BENCH_CUDA_CHECK(cudaGetLastError());
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
    };

    std::vector<Row> rows;
    std::vector<std::uint8_t> host(stride);
    // Every inbound slice against its sender's salted pattern.
    auto inbound_wrong = [&](std::uint64_t size,
                             std::uint8_t salt) -> std::uint64_t {
        std::uint64_t wrong = 0;
        for (int peer = 0; peer < world; ++peer) {
            if (peer == rank)
                continue;
            BENCH_CUDA_CHECK(cudaMemcpy(
                host.data(), arm.arena + recv_offset + peer * stride, size,
                cudaMemcpyDeviceToHost));
            wrong += pattern_check_host(host.data(), size,
                                        route_salt(peer, rank) ^ salt);
        }
        return wrong;
    };
    auto stage = [&](std::uint64_t size, std::uint8_t salt) {
        for (int peer = 0; peer < world; ++peer) {
            if (peer == rank)
                continue;
            pattern_fill_host(host.data(), size,
                              route_salt(rank, peer) ^ salt);
            BENCH_CUDA_CHECK(cudaMemcpy(arm.arena + peer * stride,
                                        host.data(), size,
                                        cudaMemcpyHostToDevice));
            BENCH_CUDA_CHECK(cudaMemset(
                arm.arena + recv_offset + peer * stride, 0, size));
        }
    };
    const std::uint64_t rounds =
        static_cast<std::uint64_t>(options.warmup) + options.iters;
    std::uint64_t consumed = 0;
    for (const std::uint64_t size : sizes) {
        stage(size, 0);
        reset_flags(arm, consumed);
        rendezvous.barrier();
        launch(size, 0, 1);
        rendezvous.barrier();
        std::uint64_t wrong =
            share_max(rendezvous, inbound_wrong(size, 0), world);
        report_wrong("before timing", size, wrong);

        stage(size, kTimedSalt);
        reset_flags(arm, 1);
        rendezvous.barrier();
        launch(size, options.warmup, options.iters);
        rendezvous.barrier();
        consumed = rounds;

        // Correctness after timing: every inbound flag carries the last
        // round's epoch (memory-backed flags) and every zeroed slice holds
        // its sender's timed pattern.
        std::uint64_t late = inbound_wrong(size, kTimedSalt);
        if (Comm::kFlagsInMemory && rounds)
            for (int peer = 0; peer < world; ++peer)
                if (peer != rank)
                    late += read_flag(arm, peer) != rounds;
        late = share_max(rendezvous, late, world);
        report_wrong("after timing", size, late);
        wrong += late;

        Row row{"put", size, static_cast<std::uint64_t>(world - 1),
                0, 0, 0, 0, 0, 0, wrong};
        row.samples_ns = copy_samples(samples, options.iters);
        const Stats stats = stats_us(to_us(row.samples_ns));
        // Conservative: bandwidth from the slowest rank's mean round.
        row.rank_mean_us = share_all(rendezvous, stats.mean, world);
        const double slowest_us =
            *std::max_element(row.rank_mean_us.begin(),
                              row.rank_mean_us.end());
        if (rank == 0) {
            const double algbw =
                slowest_us > 0.0 ? static_cast<double>(world) * size /
                                       (slowest_us * 1e-6) / 1e9
                                 : 0.0;
            row.p50 = stats.p50;
            row.p95 = stats.p95;
            row.p99 = stats.p99;
            row.mean = stats.mean;
            row.algbw = algbw;
            row.busbw = algbw * (world - 1) / world;
        }
        rows.push_back(std::move(row));
    }
    cudaFree(dsts_device);
    cudaFree(srcs_device);
    cudaFree(peer_flags_device);
    cudaFree(my_flags_device);
    cudaFree(samples);
    cudaFree(barrier_state);
    return rows;
}

// ---- reporting --------------------------------------------------------------

namespace json {

inline void text(std::string& out, const std::string& value) {
    out += '"';
    for (const char c : value) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            out += escaped;
        } else {
            out += c;
        }
    }
    out += '"';
}

inline void integer(std::string& out, std::uint64_t value) {
    char digits[24];
    std::snprintf(digits, sizeof(digits), "%llu",
                  static_cast<unsigned long long>(value));
    out += digits;
}

// JSON has no NaN/Inf: non-finite values become null.
inline void real(std::string& out, double value, int decimals) {
    if (!std::isfinite(value)) {
        out += "null";
        return;
    }
    char digits[48];
    std::snprintf(digits, sizeof(digits), "%.*f", decimals, value);
    out += digits;
}

inline void key(std::string& out, const char* name, bool first = false) {
    if (!first)
        out += ", ";
    out += '"';
    out += name;
    out += "\": ";
}

}  // namespace json

inline std::string row_json(const Row& row) {
    std::string out;
    out.reserve(256 + 12 * row.samples_ns.size() +
                12 * row.rank_mean_us.size());
    out += '{';
    json::key(out, "algbw_gbps", true);
    json::real(out, row.algbw, 6);
    json::key(out, "busbw_gbps");
    json::real(out, row.busbw, 6);
    json::key(out, "count");
    json::integer(out, row.count);
    if (row.bandwidth) {
        json::key(out, "ctas");
        json::integer(out, row.ctas);
        json::key(out, "iters");
        json::integer(out, row.iters);
    }
    json::key(out, "mean_us");
    json::real(out, row.mean, 4);
    json::key(out, "op");
    json::text(out, row.op);
    json::key(out, "p50_us");
    json::real(out, row.p50, 4);
    json::key(out, "p95_us");
    json::real(out, row.p95, 4);
    json::key(out, "p99_us");
    json::real(out, row.p99, 4);
    if (!row.rank_mean_us.empty()) {
        json::key(out, "rank_mean_us");
        out += '[';
        for (std::size_t index = 0; index < row.rank_mean_us.size();
             ++index) {
            if (index)
                out += ", ";
            json::real(out, row.rank_mean_us[index], 4);
        }
        out += ']';
    }
    json::key(out, "samples_ns");
    out += '[';
    for (std::size_t index = 0; index < row.samples_ns.size(); ++index) {
        if (index)
            out += ", ";
        json::integer(out, row.samples_ns[index]);
    }
    out += ']';
    json::key(out, "size");
    json::integer(out, row.size);
    if (row.bandwidth) {
        json::key(out, "span_ns");
        json::integer(out, row.span_ns);
        json::key(out, "window");
        json::integer(out, row.window);
    }
    json::key(out, "wrong");
    json::integer(out, row.wrong);
    out += '}';
    return out;
}

// What the record says about the run beyond its rows.
struct RecordMeta {
    const char* arm{"?"};
    const char* submit_mode{"n/a"};
    int world{0};
    KernelInfo kernel{};   // name == nullptr -> "kernel": null
    std::string memory;    // JSON object; "" -> null
    std::string versions;  // extra comma-led members of "versions"
    std::string config;    // "arm_config" object; "" -> omitted
    const char* timer{nullptr};
};

inline std::string cuda_versions_json() {
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    char text[96];
    std::snprintf(text, sizeof(text),
                  "\"cuda_driver\": %d, \"cuda_runtime\": %d", driver,
                  runtime);
    return text;
}

inline void report(const Options& options, const RecordMeta& meta,
                   const std::vector<Row>& rows) {
    const bool alltoall = options.bench == "alltoall";
    const std::string mode = alltoall ? "alltoall" : options.mode;
    const unsigned ctas =
        meta.kernel.name ? meta.kernel.ctas
                         : static_cast<unsigned>(alltoall ? meta.world - 1
                                                          : 1);
    std::printf(
        "# suite=tier1 benchmark=%s mode=%s arm=%s transport=%s "
        "submit_mode=%s world=%d ctas=%u put_issue=%s fuse_signal=%s\n",
        options.bench.c_str(), mode.c_str(), meta.arm,
        options.transport.c_str(), meta.submit_mode, meta.world, ctas,
        options.put_issue.c_str(), options.fuse_signal.c_str());
    if (meta.kernel.name)
        std::printf("# kernel=%s registers=%d local_bytes=%d "
                    "shared_bytes=%d threads=%u ctas=%u\n",
                    meta.kernel.name, meta.kernel.registers,
                    meta.kernel.local_bytes, meta.kernel.shared_bytes,
                    meta.kernel.threads, meta.kernel.ctas);
    std::printf("# %9s  %9s  %9s  %9s  %9s  %9s  %9s  %9s  %9s\n", "size",
                "count", "p50_us", "p95_us", "p99_us", "mean_us",
                "algbw_GBs", "busbw_GBs", "wrong");
    std::size_t reserve = 4096;
    for (const Row& row : rows) {
        std::printf(
            "  %9llu  %9llu  %9.2f  %9.2f  %9.2f  %9.2f  %9.2f  %9.2f  "
            "%9llu\n",
            static_cast<unsigned long long>(row.size),
            static_cast<unsigned long long>(row.count), row.p50, row.p95,
            row.p99, row.mean, row.algbw, row.busbw,
            static_cast<unsigned long long>(row.wrong));
        reserve += 512 + 12 * row.samples_ns.size();
    }

    // One sorted single-line JSON record, built in one reserved string.
    std::string out;
    out.reserve(reserve);
    out += '{';
    json::key(out, "arm", true);
    json::text(out, meta.arm);
    if (!meta.config.empty()) {
        json::key(out, "arm_config");
        out += meta.config;
    }
    json::key(out, "benchmark");
    json::text(out, options.bench);
    json::key(out, "clamped");
    out += '[';
    for (std::size_t index = 0; index < options.clamped.size(); ++index) {
        if (index)
            out += ", ";
        json::text(out, options.clamped[index]);
    }
    out += ']';
    json::key(out, "ctas");
    json::integer(out, ctas);
    json::key(out, "flags");
    out += '{';
    json::key(out, "ctas", true);
    json::integer(out, options.ctas);
    json::key(out, "fuse_signal");
    json::text(out, options.fuse_signal);
    json::key(out, "issuers");
    json::integer(out, options.issuers);
    json::key(out, "stream");
    out += options.stream ? "true" : "false";
    json::key(out, "iters");
    json::integer(out, options.iters);
    json::key(out, "op");
    json::text(out, options.op);
    json::key(out, "put_issue");
    json::text(out, options.put_issue);
    json::key(out, "puts");
    json::integer(out, options.puts);
    json::key(out, "smoke");
    out += options.smoke ? "true" : "false";
    json::key(out, "warmup");
    json::integer(out, options.warmup);
    json::key(out, "window");
    json::integer(out, options.window);
    out += '}';
    json::key(out, "fuse_signal");
    json::text(out, options.fuse_signal);
    json::key(out, "kernel");
    if (meta.kernel.name) {
        out += '{';
        json::key(out, "ctas", true);
        json::integer(out, meta.kernel.ctas);
        json::key(out, "local_bytes");
        json::integer(out, static_cast<std::uint64_t>(
                               std::max(meta.kernel.local_bytes, 0)));
        json::key(out, "name");
        json::text(out, meta.kernel.name);
        json::key(out, "registers");
        json::integer(out, static_cast<std::uint64_t>(
                               std::max(meta.kernel.registers, 0)));
        json::key(out, "shared_bytes");
        json::integer(out, static_cast<std::uint64_t>(
                               std::max(meta.kernel.shared_bytes, 0)));
        json::key(out, "threads_per_cta");
        json::integer(out, meta.kernel.threads);
        out += '}';
    } else {
        out += "null";
    }
    // The library environment as the binary left it after its own
    // setenv calls: what the library actually ran with.
    json::key(out, "library_env");
    out += '{';
    {
        std::vector<std::string> entries;
        for (char** entry = environ; *entry; ++entry) {
            const std::string text(*entry);
            for (const char* prefix : {"NCCL_", "NVSHMEM_", "DOCA_"})
                if (text.rfind(prefix, 0) == 0)
                    entries.push_back(text);
        }
        std::sort(entries.begin(), entries.end());
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const auto split = entries[index].find('=');
            json::key(out, entries[index].substr(0, split).c_str(),
                      index == 0);
            json::text(out, entries[index].substr(split + 1));
        }
    }
    out += '}';
    json::key(out, "memory");
    out += meta.memory.empty() ? "null" : meta.memory;
    json::key(out, "methodology");
    out += '{';
    json::key(out, "bandwidth_definition", true);
    json::text(out, "bytes moved by all CTAs / (max(end) - min(start)) of "
                    "the per-CTA device %globaltimer spans; samples_ns are "
                    "CTA 0's per-window (window + quiet) times");
    json::key(out, "latency_definition");
    json::text(out, "full round trip: payload put(s)+signal, signal ack "
                    "back; samples_ns per round on the leader");
    json::key(out, "percentile_method");
    json::text(out, "statistics.quantiles inclusive");
    json::key(out, "timer");
    json::text(out, meta.timer ? meta.timer
                               : "device %globaltimer (every arm, every "
                                 "mode)");
    json::key(out, "wrong_definition");
    json::text(out, "pattern mismatches after the correctness launch and "
                    "after the timed launch (re-salted pattern into zeroed "
                    "receivers), plus memory-backed inbound flags != "
                    "warmup+iters");
    out += '}';
    json::key(out, "mode");
    json::text(out, mode);
    json::key(out, "put_issue");
    json::text(out, options.put_issue);
    json::key(out, "issuers");
    json::integer(out, options.issuers);
    json::key(out, "stream");
    out += options.stream ? "true" : "false";
    json::key(out, "results");
    out += '[';
    for (std::size_t index = 0; index < rows.size(); ++index) {
        if (index)
            out += ", ";
        out += row_json(rows[index]);
    }
    out += ']';
    json::key(out, "skipped");
    out += "false";
    json::key(out, "submit_mode");
    json::text(out, meta.submit_mode);
    json::key(out, "suite");
    json::text(out, "tier1");
    json::key(out, "transport");
    json::text(out, options.transport);
    json::key(out, "versions");
    out += '{';
    out += cuda_versions_json();
    out += meta.versions;
    out += '}';
    json::key(out, "world_size");
    json::integer(out, static_cast<std::uint64_t>(meta.world));
    out += "}\n";
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
}

// Dispatch, report, and turn per-size wrong counts into the exit code. The
// binary has already run check_issue<Comm> before bring-up.
// An arm may replace the all-to-all with its library's own collective
// (same sizes, clock, checks and record; its own kernel).
template <class Comm>
using AlltoallRunner = std::vector<Row> (*)(const Options&, Arm<Comm>&,
                                            Rendezvous&, std::uint64_t,
                                            KernelInfo&);

template <class Comm>
int run_arm(Options options, Arm<Comm>& arm, Rendezvous& rendezvous,
            std::uint64_t stride, std::uint32_t window,
            AlltoallRunner<Comm> collective = nullptr) {
    if (options.bench == "ladder" && options.mode == "bandwidth" &&
        arm.max_ctas && options.ctas > arm.max_ctas) {
        note_clamp(options, "ctas " + std::to_string(options.ctas) +
                                " -> " + std::to_string(arm.max_ctas) +
                                ": the arm's quiet is single-consumer per "
                                "endpoint");
        options.ctas = arm.max_ctas;
    }
    KernelInfo info;
    std::vector<Row> rows;
    if (options.bench == "alltoall" && options.collective && collective)
        rows = collective(options, arm, rendezvous, stride, info);
    else if (options.bench == "alltoall")
        rows = run_alltoall<Comm>(options, arm, rendezvous, stride, info);
    else if (options.mode == "bandwidth")
        rows = run_ladder_bandwidth<Comm>(options, arm, rendezvous, window,
                                          info);
    else
        rows = run_ladder_latency<Comm>(options, arm, rendezvous, info);
    std::uint64_t failures = 0;
    for (const Row& row : rows)
        failures += row.wrong;
    if (arm.rank == 0) {
        RecordMeta meta;
        meta.arm = arm.name;
        meta.submit_mode = arm.submit_mode;
        meta.world = arm.world;
        meta.kernel = info;
        meta.memory = arm.memory;
        meta.versions = arm.versions;
        meta.config = arm.config;
        report(options, meta, rows);
    }
    rendezvous.barrier();
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

}  // namespace bench
