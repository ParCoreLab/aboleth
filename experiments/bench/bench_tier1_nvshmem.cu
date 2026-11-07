// NVSHMEM arm of the Tier-1 benchmark suite: the shared driver loops from
// bench_tier1_arms.cuh instantiated with the NVSHMEM comm adapter. The
// environment block sets NVSHMEM's transport (ibgda|ibrc) and lets the
// ambient environment win elsewhere. Rank model and rendezvous
// are identical to bench_tier1_aboleth: env ranks + bench::Rendezvous, with
// the NVSHMEM unique id broadcast over the rendezvous instead of fork().
// --put-issue warp / --fuse-signal on select NVSHMEM's native warp put and
// put-with-signal (comm_nvshmem.cuh); the record carries the library's
// device footprint, versions and the kernel's resources.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "bench_tier1_arms.cuh"
#include "comm_nvshmem.cuh"

namespace {

void configure_nvshmem(const std::string& transport,
                       std::uint64_t symmetric_bytes, bool nic_only,
                       std::uint32_t ctas) {
    // Third setenv argument: 1 = force, 0 = the ambient environment wins.
    // The 2-rank ladder is a wire microbenchmark: same-node P2P is
    // disabled and mlx5_0 is the near HCA for the single rank on GPU 0.
    // Multi-rank alltoall compares each library's BEST configuration:
    // NVSHMEM keeps P2P for same-node peers and its GPU-affinity NIC
    // selection — pinning HCA_LIST would bottleneck 4 local ranks onto
    // one NIC and misattribute the ceiling to the library.
    if (nic_only) {
        setenv("NVSHMEM_DISABLE_P2P", "1", 1);
        setenv("NVSHMEM_ENABLE_NIC_PE_MAPPING", "1", 1);
        setenv("NVSHMEM_HCA_LIST", "mlx5_0:1", 0);
    }
    // Library defaults otherwise: tuning knobs are set by the tuned arms
    // (matrix.sh), never here.
    setenv("NVSHMEM_DISABLE_NCCL", "1", 1);
    setenv("NVSHMEM_IB_DISABLE_DMABUF", "1", 0);
    setenv("NVSHMEM_IB_GID_INDEX", "3", 0);
    const std::string heap = std::to_string(symmetric_bytes);
    setenv("NVSHMEM_SYMMETRIC_SIZE", heap.c_str(), 0);
    if (transport == "ibgda") {
        setenv("NVSHMEM_REMOTE_TRANSPORT", "none", 1);
        setenv("NVSHMEM_IB_ENABLE_IBGDA", "1", 1);
        // Resource parity, not tuning: one RC QP per CTA, each CTA on its
        // own, as the other arms get an endpoint or a GIN context per CTA.
        // One CTA gets one.
        const std::string qps = std::to_string(std::max(ctas, 1u));
        setenv("NVSHMEM_IBGDA_NUM_RC_PER_PE", qps.c_str(), 0);
        if (ctas > 1)
            setenv("NVSHMEM_IBGDA_RC_MAP_BY", "cta", 0);
    } else {
        setenv("NVSHMEM_REMOTE_TRANSPORT", "ibrc", 1);
        setenv("NVSHMEM_IB_ENABLE_IBGDA", "0", 1);
        unsetenv("NVSHMEM_IBGDA_NIC_HANDLER");
        unsetenv("NVSHMEM_IBGDA_NUM_RC_PER_PE");
        unsetenv("NVSHMEM_IBGDA_RC_MAP_BY");
        unsetenv("NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH");
    }
}

// NVSHMEM's own all-to-all (--collective on): every round is one
// nvshmemx_alltoallmem_block over the world team, which returns once every
// slice has arrived and the sources may be reused. One CTA of
// kCollectiveThreads threads, launched collectively; thread 0 keeps the
// round time. NVSHMEM also copies the rank's own slice, which the other
// arms skip. The collective compiles to ~250 registers unbounded, more
// than an SM holds at 1024 threads; the launch bound caps it at 128.
// NVSHMEM 3.8's collective calls a 254-register function, which ptxas
// refuses under that cap: the nvshmem38 arm is this file built against
// NVSHMEM 3.8.0 with -DBENCH_NVSHMEM_COLLECTIVE_THREADS=256, installed as
// bench_tier1_nvshmem38.
#ifndef BENCH_NVSHMEM_COLLECTIVE_THREADS
#define BENCH_NVSHMEM_COLLECTIVE_THREADS 512
#endif
constexpr unsigned kCollectiveThreads = BENCH_NVSHMEM_COLLECTIVE_THREADS;

__global__ void __launch_bounds__(kCollectiveThreads)
nvshmem_alltoall_collective(
    std::uint8_t* receive, const std::uint8_t* send, std::uint64_t bytes,
    std::uint32_t warmup, std::uint32_t iters, std::uint64_t* samples_ns) {
    for (std::uint32_t round = 0; round < warmup + iters; ++round) {
        std::uint64_t start = 0;
        if (threadIdx.x == 0)
            start = bench::globaltimer();
        nvshmemx_alltoallmem_block(NVSHMEM_TEAM_WORLD, receive, send, bytes);
        if (threadIdx.x == 0 && round >= warmup)
            samples_ns[round - warmup] = bench::globaltimer() - start;
        __syncthreads();
    }
}

// Same sizes, salted checks and record as bench::run_alltoall; slices are
// contiguous (slice p at p * size), the layout the collective takes.
std::vector<bench::Row> run_alltoall_collective(
    const bench::Options& options, bench::Arm<NvshmemComm>& arm,
    bench::Rendezvous& rendezvous, std::uint64_t stride,
    bench::KernelInfo& info) {
    const int rank = arm.rank, world = arm.world;
    std::uint8_t* send = arm.arena;
    std::uint8_t* receive =
        arm.arena + static_cast<std::uint64_t>(world) * stride;
    info = bench::kernel_info(&nvshmem_alltoall_collective,
                              "nvshmem_alltoall_collective",
                              kCollectiveThreads, 1);
    std::uint64_t* samples = nullptr;
    BENCH_CUDA_CHECK(cudaMalloc(
        &samples, sizeof(std::uint64_t) * std::max(options.iters, 1u)));
    std::vector<std::uint8_t> host(stride);
    auto stage = [&](std::uint64_t size, std::uint8_t salt) {
        for (int peer = 0; peer < world; ++peer) {
            bench::pattern_fill_host(host.data(), size,
                                     bench::route_salt(rank, peer) ^ salt);
            BENCH_CUDA_CHECK(cudaMemcpy(send + peer * size, host.data(), size,
                                        cudaMemcpyHostToDevice));
        }
        BENCH_CUDA_CHECK(cudaMemset(receive, 0, world * size));
    };
    auto inbound_wrong = [&](std::uint64_t size, std::uint8_t salt) {
        std::uint64_t wrong = 0;
        for (int peer = 0; peer < world; ++peer) {
            if (peer == rank)
                continue;
            BENCH_CUDA_CHECK(cudaMemcpy(host.data(), receive + peer * size,
                                        size, cudaMemcpyDeviceToHost));
            wrong += bench::pattern_check_host(
                host.data(), size, bench::route_salt(peer, rank) ^ salt);
        }
        return wrong;
    };
    auto launch = [&](std::uint64_t size, std::uint32_t warmup,
                      std::uint32_t iters) {
        std::uint8_t* receive_arg = receive;
        const std::uint8_t* send_arg = send;
        std::uint64_t bytes = size;
        std::uint64_t* samples_arg = samples;
        void* args[] = {&receive_arg, &send_arg, &bytes, &warmup, &iters,
                        &samples_arg};
        if (nvshmemx_collective_launch(
                reinterpret_cast<const void*>(&nvshmem_alltoall_collective),
                dim3(1), dim3(kCollectiveThreads), args, 0, 0) != 0) {
            std::fprintf(stderr, "bench_tier1_nvshmem: collective launch "
                                 "failed\n");
            std::exit(EXIT_FAILURE);
        }
        BENCH_CUDA_CHECK(cudaGetLastError());
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
    };
    std::vector<bench::Row> rows;
    for (const std::uint64_t size : bench::size_ladder(options)) {
        stage(size, 0);
        rendezvous.barrier();
        launch(size, 0, 1);
        rendezvous.barrier();
        std::uint64_t wrong =
            bench::share_max(rendezvous, inbound_wrong(size, 0), world);
        bench::report_wrong("before timing", size, wrong);

        stage(size, bench::kTimedSalt);
        rendezvous.barrier();
        launch(size, options.warmup, options.iters);
        rendezvous.barrier();
        std::uint64_t late = bench::share_max(
            rendezvous, inbound_wrong(size, bench::kTimedSalt), world);
        bench::report_wrong("after timing", size, late);
        wrong += late;

        bench::Row row{"put", size, static_cast<std::uint64_t>(world - 1),
                       0, 0, 0, 0, 0, 0, wrong};
        row.samples_ns = bench::copy_samples(samples, options.iters);
        const bench::Stats stats = bench::stats_us(bench::to_us(row.samples_ns));
        row.rank_mean_us = bench::share_all(rendezvous, stats.mean, world);
        const double slowest_us = *std::max_element(
            row.rank_mean_us.begin(), row.rank_mean_us.end());
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
    cudaFree(samples);
    return rows;
}

}  // namespace

int main(int argc, char** argv) {
    bench::Options options = bench::parse_options(argc, argv, "ibgda");
    const int rank = bench::env_int("RANK");
    const int world = bench::env_int("WORLD_SIZE");
    const int gpu = bench::env_int("LOCAL_RANK", bench::env_int("RANK"));
    const char* master = std::getenv("MASTER_ADDR");
    const int port = bench::env_int("MASTER_PORT");
    if (master == nullptr) {
        std::fprintf(stderr, "bench_tier1_nvshmem: MASTER_ADDR is unset\n");
        return 2;
    }
    if (options.transport != "ibgda" && options.transport != "ibrc") {
        std::fprintf(stderr,
                     "bench_tier1_nvshmem: --transport ibgda|ibrc only\n");
        return 2;
    }
    if (options.bench == "ladder" && world != 2) {
        std::fprintf(stderr,
                     "bench_tier1_nvshmem: ladder needs exactly 2 ranks\n");
        return 2;
    }
    if (const int refused =
            bench::check_issue<NvshmemComm>(options, "bench_tier1_nvshmem"))
        return refused;
    if (options.collective &&
        (options.bench != "alltoall" || options.put_issue != "thread" ||
         options.fuse_signal != "off")) {
        std::fprintf(stderr,
                     "bench_tier1_nvshmem: --collective on is the library's "
                     "own all-to-all (no issue-shape knobs)\n");
        return 2;
    }
    BENCH_CUDA_CHECK(cudaSetDevice(gpu));
    BENCH_CUDA_CHECK(cudaFree(nullptr));
    const std::uint64_t free_before = bench::device_free_bytes();
    bench::Rendezvous rendezvous(rank, world, master, port);

    bench::clamp_options(options, world);
    std::uint64_t stride = 0;
    bench::Arm<NvshmemComm> arm;
    arm.rank = rank;
    arm.world = world;
    arm.name = "nvshmem";
    arm.submit_mode = options.transport == "ibgda" ? "ibgda" : "ibrc";
    arm.data_bytes = bench::plan_data_bytes(options, world, &stride);
    arm.flags_offset = (arm.data_bytes + 7) / 8 * 8;
    const std::uint64_t arena_bytes = arm.flags_offset + 8ull * world;

    const std::uint64_t heap_bytes = arena_bytes * 2 + (64ull << 20);
    configure_nvshmem(options.transport, heap_bytes,
                      options.bench == "ladder",
                      options.bench == "ladder" && options.mode == "bandwidth"
                          ? options.ctas
                          : 1u);
    nvshmemx_uniqueid_t unique_id = NVSHMEMX_UNIQUEID_INITIALIZER;
    if (rank == 0)
        nvshmemx_get_uniqueid(&unique_id);
    const auto ids = rendezvous.allgather(&unique_id, sizeof(unique_id));
    std::memcpy(&unique_id, ids.data(), sizeof(unique_id));
    nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_uniqueid_args(rank, world, &unique_id, &attributes);
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attributes) != 0) {
        std::fprintf(stderr, "bench_tier1_nvshmem: nvshmem init failed\n");
        return EXIT_FAILURE;
    }
    // Library footprint: everything nvshmem_init reserved and connected
    // (the symmetric heap reservation included), measured before the
    // benchmark's own nvshmem_malloc — the arena lives inside that heap.
    const std::uint64_t free_after = bench::device_free_bytes();
    const char* heap = std::getenv("NVSHMEM_SYMMETRIC_SIZE");
    std::string heap_json = "\"nvshmem_symmetric_size\": ";
    if (heap != nullptr) {
        heap_json += '"';
        heap_json += heap;
        heap_json += '"';
    } else {
        heap_json += "null";
    }
    arm.memory = bench::memory_json(free_before, free_after, heap_json);
    // nvshmem_info_get_version reports the OpenSHMEM spec level; the
    // library's own release is the vendor version.
    int spec_major = 0, spec_minor = 0;
    int major = 0, minor = 0, patch = 0;
    nvshmem_info_get_version(&spec_major, &spec_minor);
    nvshmemx_vendor_get_version_info(&major, &minor, &patch);
    char versions[128];
    std::snprintf(versions, sizeof(versions),
                  ", \"nvshmem\": \"%d.%d.%d\", \"openshmem_spec\": "
                  "\"%d.%d\"",
                  major, minor, patch, spec_major, spec_minor);
    arm.versions = versions;

    arm.arena = static_cast<std::uint8_t*>(nvshmem_malloc(arena_bytes));
    if (arm.arena == nullptr) {
        std::fprintf(stderr,
                     "bench_tier1_nvshmem: nvshmem_malloc(%llu) failed — "
                     "raise NVSHMEM_SYMMETRIC_SIZE\n",
                     static_cast<unsigned long long>(arena_bytes));
        return EXIT_FAILURE;
    }
    BENCH_CUDA_CHECK(cudaMemset(arm.arena, 0, arena_bytes));
    // Symmetric heap: every peer's view of the arena is this rank's own VA.
    arm.local_data_base = reinterpret_cast<std::uint64_t>(arm.arena);
    arm.flag_base =
        reinterpret_cast<std::uint64_t>(arm.arena) + arm.flags_offset;
    arm.peer_data.assign(world, reinterpret_cast<std::uint64_t>(arm.arena));
    arm.peer_flags.assign(
        world, reinterpret_cast<std::uint64_t>(arm.arena) + arm.flags_offset);
    arm.peer_rkey.assign(world, 0);
    arm.ctx = NvshmemComm::make_ctx(nullptr, nullptr);
    nvshmem_barrier_all();

    if (options.collective) {
        arm.name = "nvshmem-collective";
        arm.config = "{\"collective\": \"nvshmemx_alltoallmem_block\", "
                     "\"threads\": " +
                     std::to_string(kCollectiveThreads) + ", \"ctas\": 1}";
    }
    const int code = bench::run_arm<NvshmemComm>(
        options, arm, rendezvous, stride, options.window,
        &run_alltoall_collective);
    nvshmem_barrier_all();
    nvshmem_free(arm.arena);
    nvshmem_finalize();
    return code;
}
