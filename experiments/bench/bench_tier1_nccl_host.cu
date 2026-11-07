// Base-NCCL reference arm: the alltoall through NCCL's HOST API (grouped
// ncclSend/ncclRecv per peer, one group per round) with no transport or
// NIC forcing — NCCL's own topology-driven best configuration (P2P/SHM for
// same-node peers, per-GPU NIC selection, GIN not required). This is what
// a stock NCCL user gets for the same exchange, so it bounds the
// comparison from the "just use NCCL collectives" side. Per-round time is
// host-measured around the group launch and stream sync: the launch and
// group overhead ARE the cost of the host API and are deliberately
// included. With --graph-rounds K the rounds are captured K to a CUDA
// graph, NCCL's way of removing that overhead, and each sample is one
// graph launch and sync over K. Rank model and rendezvous follow
// bench_tier1_nccl.cu.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#include "bench_tier1_arms.cuh"

#define NCCL_CHECK(expression)                                             \
    do {                                                                   \
        const ncclResult_t nccl_result = (expression);                     \
        if (nccl_result != ncclSuccess) {                                  \
            std::fprintf(stderr, "bench_tier1_nccl_host: %s failed: %s\n", \
                         #expression, ncclGetErrorString(nccl_result));    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

namespace {

// One full exchange: every peer's slice sent and received in one group.
void alltoall_round(ncclComm_t comm, cudaStream_t stream,
                    std::uint8_t* arena, std::uint64_t recv_offset,
                    std::uint64_t stride, std::uint64_t size, int rank,
                    int world) {
    NCCL_CHECK(ncclGroupStart());
    for (int peer = 0; peer < world; ++peer) {
        if (peer == rank)
            continue;
        NCCL_CHECK(ncclSend(arena + peer * stride, size, ncclChar, peer,
                            comm, stream));
        NCCL_CHECK(ncclRecv(arena + recv_offset + peer * stride, size,
                            ncclChar, peer, comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());
}

}  // namespace

int main(int argc, char** argv) {
    bench::Options options = bench::parse_options(argc, argv, "host");
    const int rank = bench::env_int("RANK");
    const int world = bench::env_int("WORLD_SIZE");
    const int gpu = bench::env_int("LOCAL_RANK", bench::env_int("RANK"));
    const char* master = std::getenv("MASTER_ADDR");
    const int port = bench::env_int("MASTER_PORT");
    if (master == nullptr) {
        std::fprintf(stderr, "bench_tier1_nccl_host: MASTER_ADDR unset\n");
        return 2;
    }
    if (options.bench != "alltoall") {
        std::fprintf(stderr,
                     "bench_tier1_nccl_host: --bench alltoall only (the "
                     "host API has no one-sided ladder analogue)\n");
        return 2;
    }
    if (options.transport != "host") {
        std::fprintf(stderr,
                     "bench_tier1_nccl_host: --transport host only\n");
        return 2;
    }

    if (options.put_issue != "thread" || options.fuse_signal != "off") {
        std::fprintf(stderr,
                     "bench_tier1_nccl_host: --put-issue/--fuse-signal are "
                     "device-API knobs; the host API has neither\n");
        return 2;
    }

    BENCH_CUDA_CHECK(cudaSetDevice(gpu));
    BENCH_CUDA_CHECK(cudaFree(nullptr));
    const std::uint64_t free_before = bench::device_free_bytes();
    bench::Rendezvous rendezvous(rank, world, master, port);

    ncclUniqueId unique_id{};
    if (rank == 0)
        NCCL_CHECK(ncclGetUniqueId(&unique_id));
    const auto ids = rendezvous.allgather(&unique_id, sizeof(unique_id));
    std::memcpy(&unique_id, ids.data(), sizeof(unique_id));
    ncclComm_t comm = nullptr;
    NCCL_CHECK(ncclCommInitRank(&comm, world, unique_id, rank));
    // Library footprint: the initialized communicator, before the arena.
    const std::uint64_t free_after = bench::device_free_bytes();
    cudaStream_t stream = nullptr;
    BENCH_CUDA_CHECK(cudaStreamCreate(&stream));

    bench::clamp_options(options, world);
    std::uint64_t stride = 0;
    const std::uint64_t data_bytes =
        bench::plan_data_bytes(options, world, &stride);
    const std::uint64_t recv_offset =
        static_cast<std::uint64_t>(world) * stride;
    void* arena_pointer = nullptr;
    NCCL_CHECK(ncclMemAlloc(&arena_pointer, data_bytes));
    auto* arena = static_cast<std::uint8_t*>(arena_pointer);
    BENCH_CUDA_CHECK(cudaMemset(arena, 0, data_bytes));

    const auto sizes = bench::size_ladder(options);
    std::vector<bench::Row> rows;
    std::vector<std::uint8_t> host(stride);
    auto stage = [&](std::uint64_t size, std::uint8_t salt) {
        for (int peer = 0; peer < world; ++peer) {
            if (peer == rank)
                continue;
            bench::pattern_fill_host(host.data(), size,
                                     bench::route_salt(rank, peer) ^ salt);
            BENCH_CUDA_CHECK(cudaMemcpy(arena + peer * stride, host.data(),
                                        size, cudaMemcpyHostToDevice));
            BENCH_CUDA_CHECK(cudaMemset(
                arena + recv_offset + peer * stride, 0, size));
        }
    };
    const std::uint32_t chunk = options.graph_rounds;
    for (const std::uint64_t size : sizes) {
        // Correctness before timing: sender-salted slices catch
        // misrouting, not just corruption (same gate as every arm).
        stage(size, 0);
        rendezvous.barrier();
        alltoall_round(comm, stream, arena, recv_offset, stride, size,
                       rank, world);
        BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        rendezvous.barrier();
        std::uint64_t wrong = 0;
        for (int peer = 0; peer < world; ++peer) {
            if (peer == rank)
                continue;
            BENCH_CUDA_CHECK(cudaMemcpy(
                host.data(), arena + recv_offset + peer * stride, size,
                cudaMemcpyDeviceToHost));
            wrong += bench::pattern_check_host(
                host.data(), size, bench::route_salt(peer, rank));
        }
        wrong = bench::share_max(rendezvous, wrong, world);
        if (wrong)
            std::fprintf(stderr, "FAIL: size %llu: %llu wrong\n",
                         static_cast<unsigned long long>(size),
                         static_cast<unsigned long long>(wrong));

        stage(size, bench::kTimedSalt);
        rendezvous.barrier();
        std::vector<double> samples_us;
        if (chunk) {
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            BENCH_CUDA_CHECK(cudaStreamBeginCapture(
                stream, cudaStreamCaptureModeGlobal));
            for (std::uint32_t round = 0; round < chunk; ++round)
                alltoall_round(comm, stream, arena, recv_offset, stride,
                               size, rank, world);
            BENCH_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            BENCH_CUDA_CHECK(
                cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            const std::uint32_t warm = (options.warmup + chunk - 1) / chunk;
            for (std::uint32_t launch = 0; launch < warm; ++launch)
                BENCH_CUDA_CHECK(cudaGraphLaunch(exec, stream));
            BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::uint32_t launches =
                std::max(1u, options.iters / chunk);
            samples_us.assign(launches, 0.0);
            for (std::uint32_t launch = 0; launch < launches; ++launch) {
                const auto start = std::chrono::steady_clock::now();
                BENCH_CUDA_CHECK(cudaGraphLaunch(exec, stream));
                BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
                samples_us[launch] =
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count() /
                    chunk;
            }
            BENCH_CUDA_CHECK(cudaGraphExecDestroy(exec));
            BENCH_CUDA_CHECK(cudaGraphDestroy(graph));
        } else {
            for (std::uint32_t round = 0; round < options.warmup; ++round)
                alltoall_round(comm, stream, arena, recv_offset, stride,
                               size, rank, world);
            BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
            samples_us.assign(options.iters, 0.0);
            for (std::uint32_t round = 0; round < options.iters; ++round) {
                const auto start = std::chrono::steady_clock::now();
                alltoall_round(comm, stream, arena, recv_offset, stride,
                               size, rank, world);
                BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
                samples_us[round] =
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count();
            }
        }
        rendezvous.barrier();
        // Correctness after timing: every zeroed slice holds its sender's
        // timed pattern.
        std::uint64_t late = 0;
        for (int peer = 0; peer < world; ++peer) {
            if (peer == rank)
                continue;
            BENCH_CUDA_CHECK(cudaMemcpy(
                host.data(), arena + recv_offset + peer * stride, size,
                cudaMemcpyDeviceToHost));
            late += bench::pattern_check_host(
                host.data(), size,
                bench::route_salt(peer, rank) ^ bench::kTimedSalt);
        }
        late = bench::share_max(rendezvous, late, world);
        bench::report_wrong("after timing", size, late);
        wrong += late;

        const bench::Stats stats = bench::stats_us(samples_us);
        bench::Row row{"sendrecv", size,
                       static_cast<std::uint64_t>(world - 1),
                       0, 0, 0, 0, 0, 0, wrong};
        // Conservative: bandwidth from the slowest rank's mean round.
        row.rank_mean_us = bench::share_all(rendezvous, stats.mean, world);
        const double slowest_us = *std::max_element(
            row.rank_mean_us.begin(), row.rank_mean_us.end());
        if (rank == 0) {
            const double algbw =
                slowest_us > 0.0 ? static_cast<double>(world) * size /
                                       (slowest_us * 1e-6) / 1e9
                                 : 0.0;
            row.samples_ns.reserve(samples_us.size());
            for (const double sample : samples_us)
                row.samples_ns.push_back(
                    static_cast<std::uint64_t>(sample * 1000.0 + 0.5));
            row.p50 = stats.p50;
            row.p95 = stats.p95;
            row.p99 = stats.p99;
            row.mean = stats.mean;
            row.algbw = algbw;
            row.busbw = algbw * (world - 1) / world;
        }
        rows.push_back(std::move(row));
    }

    std::uint64_t failures = 0;
    for (const bench::Row& row : rows)
        failures += row.wrong;
    if (rank == 0) {
        bench::RecordMeta meta;
        meta.arm = "nccl-host";
        meta.submit_mode = "host";
        meta.world = world;
        meta.memory = bench::memory_json(free_before, free_after);
        int nccl_code = 0;
        NCCL_CHECK(ncclGetVersion(&nccl_code));
        char versions[96];
        std::snprintf(versions, sizeof(versions),
                      ", \"nccl\": \"%d.%d.%d\", \"nccl_code\": %d",
                      nccl_code / 10000, nccl_code % 10000 / 100,
                      nccl_code % 100, nccl_code);
        meta.versions = versions;
        // The host API's launch and group overhead is its cost, so this arm
        // alone is host-timed (and has no benchmark kernel of ours).
        static std::string timer;
        timer = chunk ? "host steady_clock around one CUDA-graph launch of " +
                            std::to_string(chunk) +
                            " captured rounds and its stream sync, over " +
                            std::to_string(chunk) + " (a sample is a mean)"
                      : std::string("host steady_clock per round around the "
                                    "grouped send/recv launch and stream "
                                    "sync");
        meta.timer = timer.c_str();
        bench::report(options, meta, rows);
    }
    rendezvous.barrier();
    NCCL_CHECK(ncclMemFree(arena_pointer));
    BENCH_CUDA_CHECK(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
