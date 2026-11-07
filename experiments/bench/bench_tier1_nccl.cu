// NCCL device-API (GIN) arm of the Tier-1 benchmark suite: the shared
// driver loops instantiated with the GIN comm adapter. On a comm without
// device-API/GIN support this arm SELF-SKIPS: it prints "SKIP: <reason>" and exits 0, which
// experiments/bench/common.py maps to a skipped record (run-gated.sh treats
// a wrapped exit 77 as failure, so 77 is deliberately not used).
//
// Transports: gdaki | proxy drive every peer through GIN (the pure-GIN
// regression shape); gdaki-lsa | proxy-lsa route peers in this rank's LSA
// team (same node, load/store accessible over NVLink) through the device
// API's LSA pointers and keep GIN for the rest — the mixed NVLink+RDMA
// shape Aboleth and NVSHMEM run. The
// per-peer choice is made on the device from the devComm's LSA team.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <nccl.h>
#include <nccl_device.h>

#include "bench_tier1_arms.cuh"
#include "comm_nccl_gin.cuh"

namespace {

#define NCCL_CHECK(call)                                                   \
    do {                                                                   \
        const ncclResult_t status_ = (call);                               \
        if (status_ != ncclSuccess) {                                      \
            std::fprintf(stderr, "bench_tier1_nccl: %s failed: %s\n",      \
                         #call, ncclGetErrorString(status_));              \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

void configure_nccl(const std::string& gin_backend, bool nic_only) {
    // The 2-rank ladder is a wire microbenchmark: peer traffic must
    // traverse the NIC, and with one rank per node on GPU 0, mlx5_0 is the
    // near HCA. Multi-rank
    // alltoall instead compares each library's BEST configuration: NCCL
    // keeps its topology-driven P2P/SHM transports and per-GPU NIC
    // selection — forcing either would bottleneck 4 local ranks onto one
    // NIC and misattribute the ceiling to the library.
    if (nic_only) {
        setenv("NCCL_P2P_DISABLE", "1", 1);
        setenv("NCCL_SHM_DISABLE", "1", 1);
        if (std::getenv("NCCL_IB_HCA") == nullptr)
            setenv("NCCL_IB_HCA", "mlx5_0:1", 1);
    }
    setenv("NCCL_NET", "IB", 1);
    setenv("NCCL_IB_DISABLE", "0", 1);
    if (std::getenv("NCCL_IB_GID_INDEX") == nullptr)
        setenv("NCCL_IB_GID_INDEX", "3", 1);
    // The installed rdma-core does not expose the optional OOO receive
    // capability queried by NCCL 2.30. GIN itself does not require it.
    setenv("NCCL_IB_OOO_RQ", "0", 1);
    setenv("NCCL_GIN_TYPE", gin_backend == "gdaki" ? "3" : "2", 1);
}

const char* gin_type_name(ncclGinType_t type) {
    switch (type) {
        case NCCL_GIN_TYPE_PROXY: return "proxy";
        case NCCL_GIN_TYPE_GDAKI: return "gdaki";
        default: return "none";
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Options options = bench::parse_options(argc, argv, "gdaki");
    const int rank = bench::env_int("RANK");
    const int world = bench::env_int("WORLD_SIZE");
    const int gpu = bench::env_int("LOCAL_RANK", bench::env_int("RANK"));
    const char* master = std::getenv("MASTER_ADDR");
    const int port = bench::env_int("MASTER_PORT");
    if (master == nullptr) {
        std::fprintf(stderr, "bench_tier1_nccl: MASTER_ADDR is unset\n");
        return 2;
    }
    const std::string suffix = "-lsa";
    const bool lsa = options.transport.size() > suffix.size() &&
                     options.transport.compare(
                         options.transport.size() - suffix.size(),
                         suffix.size(), suffix) == 0;
    const std::string gin_backend =
        lsa ? options.transport.substr(0, options.transport.size() -
                                              suffix.size())
            : options.transport;
    if (gin_backend != "gdaki" && gin_backend != "proxy") {
        std::fprintf(stderr,
                     "bench_tier1_nccl: --transport "
                     "gdaki|proxy|gdaki-lsa|proxy-lsa only\n");
        return 2;
    }
    if (options.op == "put_inline") {
        std::fprintf(stderr,
                     "bench_tier1_nccl: --op put_inline has no coherent GIN "
                     "mapping (signals are counters, not flag words)\n");
        return 2;
    }
    if (options.bench == "ladder" && world != 2) {
        std::fprintf(stderr,
                     "bench_tier1_nccl: ladder needs exactly 2 ranks\n");
        return 2;
    }
    if (const int refused =
            bench::check_issue<NcclGinComm>(options, "bench_tier1_nccl"))
        return refused;
    configure_nccl(gin_backend, options.bench == "ladder");
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

    ncclCommProperties properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    const ncclResult_t query =
        ncclCommQueryProperties(comm, &properties);
    const ncclGinType_t requested = gin_backend == "gdaki"
                                        ? NCCL_GIN_TYPE_GDAKI
                                        : NCCL_GIN_TYPE_PROXY;
    if (query != ncclSuccess || !properties.deviceApiSupport ||
        properties.ginType == NCCL_GIN_TYPE_NONE ||
        properties.ginType != requested) {
        if (rank == 0)
            std::printf(
                "SKIP: NCCL device API/GIN unavailable or mismatched "
                "(query=%d deviceApi=%d ginType=%s requested=%s)\n",
                static_cast<int>(query),
                query == ncclSuccess ? properties.deviceApiSupport : -1,
                query == ncclSuccess ? gin_type_name(properties.ginType)
                                     : "unknown",
                gin_backend.c_str());
        rendezvous.barrier();
        ncclCommDestroy(comm);
        return 0;
    }

    // One GIN context per outgoing CTA: world-1 alltoall lanes, or the
    // bandwidth grid when it is wider (a hint; the devComm says what it
    // granted, and the policy shares contexts when it granted fewer).
    const bool bandwidth =
        options.bench == "ladder" && options.mode == "bandwidth";
    int wanted_contexts = std::max(
        world > 1 ? world - 1 : 1,
        bandwidth ? static_cast<int>(options.ctas) : 1);
    // Every context is fully connected, so one per peer costs peers^2 QPs
    // per rank, which ncclDevCommCreate refuses at world 128; --gin-contexts
    // caps it, and the CTAs share the contexts round-robin.
    if (options.gin_contexts)
        wanted_contexts = std::min(wanted_contexts,
                                   static_cast<int>(options.gin_contexts));
    ncclDevCommRequirements requirements =
        NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    requirements.ginContextCount = wanted_contexts;
    requirements.ginSignalCount = world;
    requirements.ginExclusiveContexts = true;
    requirements.ginStrongSignalsRequired = true;
    requirements.ginVaSignalsRequired = false;
    ncclDevComm dev_comm{};
    NCCL_CHECK(ncclDevCommCreate(comm, &requirements, &dev_comm));
    if (dev_comm.ginContextCount == 0) {
        if (rank == 0)
            std::printf("SKIP: device communicator has no GIN contexts\n");
        rendezvous.barrier();
        ncclDevCommDestroy(comm, &dev_comm);
        ncclCommDestroy(comm);
        return 0;
    }
    // Library footprint: the communicator and the device communicator
    // (GIN contexts, signals, connections), before the benchmark window.
    const std::uint64_t free_after = bench::device_free_bytes();

    bench::clamp_options(options, world);
    std::uint64_t stride = 0;
    bench::Arm<NcclGinComm> arm;
    arm.rank = rank;
    arm.world = world;
    arm.name = "nccl-device";
    static const std::string submit_mode =
        std::string(gin_type_name(properties.ginType)) + (lsa ? "+lsa" : "");
    arm.submit_mode = submit_mode.c_str();
    arm.data_bytes = bench::plan_data_bytes(options, world, &stride);
    arm.flags_offset = (arm.data_bytes + 7) / 8 * 8;
    const std::uint64_t arena_bytes = arm.flags_offset + 8ull * world;

    void* arena = nullptr;
    NCCL_CHECK(ncclMemAlloc(&arena, arena_bytes));
    BENCH_CUDA_CHECK(cudaMemset(arena, 0, arena_bytes));
    ncclWindow_t window{};
    NCCL_CHECK(ncclCommWindowRegister(comm, arena, arena_bytes, &window,
                                      NCCL_WIN_COLL_SYMMETRIC));

    // LSA team: the contiguous world ranks [rank - lsaRank, + lsaSize).
    // Under the ladder's NIC forcing (P2P disabled) it is normally {self},
    // so every peer stays on GIN.
    const int lsa_peers = lsa ? dev_comm.lsaSize - 1 : 0;
    arm.memory = bench::memory_json(free_before, free_after);
    int nccl_code = 0;
    NCCL_CHECK(ncclGetVersion(&nccl_code));
    char text[256];
    std::snprintf(text, sizeof(text),
                  ", \"nccl\": \"%d.%d.%d\", \"nccl_code\": %d",
                  nccl_code / 10000, nccl_code % 10000 / 100,
                  nccl_code % 100, nccl_code);
    arm.versions = text;
    std::snprintf(text, sizeof(text),
                  "{\"gin_contexts\": %u, \"gin_contexts_requested\": %d, "
                  "\"gin_type\": \"%s\", \"lsa\": %s, \"lsa_peers\": %d, "
                  "\"lsa_rank\": %d, \"lsa_size\": %d}",
                  static_cast<unsigned>(dev_comm.ginContextCount),
                  wanted_contexts, gin_type_name(properties.ginType),
                  lsa ? "true" : "false", lsa_peers, dev_comm.lsaRank,
                  dev_comm.lsaSize);
    arm.config = text;
    // stdout is reported only at the end; say per rank that bring-up is
    // done before the first device wait, so a hang is attributable.
    std::fprintf(stderr,
                 "bench_tier1_nccl: rank %d up (gin=%s contexts=%u lsa=%d "
                 "lsa_size=%d lsa_peers=%d)\n",
                 rank, gin_type_name(properties.ginType),
                 static_cast<unsigned>(dev_comm.ginContextCount),
                 lsa ? 1 : 0, dev_comm.lsaSize, lsa_peers);

    arm.arena = static_cast<std::uint8_t*>(arena);
    // Window-offset addressing: the arena starts at offset 0 of its window
    // on every rank.
    arm.local_data_base = 0;
    arm.flag_base = 0;
    arm.peer_data.assign(world, 0);
    arm.peer_flags.assign(world, arm.flags_offset);
    arm.peer_rkey.assign(world, 0);
    arm.ctx = NcclGinComm::Ctx{
        dev_comm, window,
        static_cast<std::uint32_t>(dev_comm.ginContextCount), rank,
        /*epoch_base=*/0, /*wait_stride=*/1, arm.flags_offset, lsa,
        /*issuers=*/1};

    const int code = bench::run_arm<NcclGinComm>(options, arm, rendezvous,
                                                 stride, options.window);
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    NCCL_CHECK(ncclMemFree(arena));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    ncclCommDestroy(comm);
    return code;
}
