// Tier-1 C++-hosted benchmark arm: an nccl-tests-style binary whose host
// side drives the runtime C ABI (include/aboleth/runtime_c.h) and whose
// benchmark loops are the shared driver in bench_tier1_arms.cuh — the same
// loops every other arm runs, instantiated with the Tier-1 device-library
// adapters.
//
// Rank model: one process per rank. RANK / WORLD_SIZE / MASTER_ADDR /
// MASTER_PORT / LOCAL_RANK come from the environment (exactly what
// scripts/multinode-launch.sh exports; the experiments/bench drivers spawn
// local ranks the same way). Rendezvous is bench::Rendezvous over
// MASTER_PORT.
//
// No hand-rolled QP bring-up — endpoint creation, connection exchange
// (128-byte AbolethConnectionInfo PODs), and registration all go through
// the C ABI, which is the point of this arm: aboleth::put from ordinary
// CUDA with an ordinary host program.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aboleth/Target/ProxyAbi.h"
#include "aboleth/Target/RdmaAbi.h"
#include "aboleth/runtime_c.h"
#include "bench_tier1_arms.cuh"
#include "comm_aboleth.cuh"
#include "comm_aboleth_proxy.cuh"

namespace {

#define ABI_CHECK(condition)                                               \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::fprintf(stderr, "bench_tier1_aboleth: %s failed: %s\n",   \
                         #condition, aboleth_runtime_last_error());        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

struct Fabric {
    void* runtime{nullptr};
    void* registration{nullptr};
    std::vector<void*> endpoints;
    std::vector<void*> descriptors;
    std::uint64_t* handles_device{nullptr};
    std::uint8_t* arena{nullptr};
    std::uint32_t capacity{0};
    std::uint32_t submit_mode{3};
    int world{0};
};

const char* submit_name(std::uint32_t mode) {
    switch (mode) {
        case 0: return "direct";
        case 1: return "staged";
        case 2: return "proxy";
        default: return "unknown";
    }
}

// What every rank shares at bootstrap, after its per-peer connection infos:
// one MR covers data and flags, so a single rkey suffices.
struct RankShareTail {
    std::uint64_t data_address;
    std::uint64_t flags_address;
    std::uint32_t rkey;
    std::uint32_t pad;
};

template <class Comm>
Fabric bring_up(const bench::Options& options, bench::Arm<Comm>& arm,
                int gpu, bench::Rendezvous& rendezvous) {
    Fabric fabric;
    fabric.world = arm.world;
    const std::uint64_t arena_bytes = arm.flags_offset + 8ull * arm.world;

    // Registration needs the arena, so it is allocated first and the
    // library footprint is bracketed after it: runtime, registration,
    // endpoint descriptors and queues, up to connected endpoints.
    BENCH_CUDA_CHECK(cudaMalloc(&fabric.arena, arena_bytes));
    BENCH_CUDA_CHECK(cudaMemset(fabric.arena, 0, arena_bytes));
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());
    const std::uint64_t free_before = bench::device_free_bytes();
    fabric.runtime =
        aboleth_runtime_create_for_device(gpu, std::getenv("ABOLETH_HCA"));
    ABI_CHECK(fabric.runtime != nullptr);
    fabric.registration = aboleth_runtime_register_buffer(
        fabric.runtime, gpu, fabric.arena, arena_bytes, 0);
    ABI_CHECK(fabric.registration != nullptr);

    const bool proxy = options.transport == "proxy";
    const std::uint32_t transport =
        proxy ? kAbolethTransportProxy : kAbolethTransportDirect;
    const std::size_t descriptor_bytes =
        proxy ? sizeof(aboleth::abi::ProxyEndpointDescriptor)
              : sizeof(aboleth::abi::RdmaEndpointDescriptor);
    fabric.endpoints.assign(arm.world, nullptr);
    fabric.descriptors.assign(arm.world, nullptr);
    std::vector<AbolethConnectionInfo> mine(arm.world);
    std::memset(mine.data(), 0, mine.size() * sizeof(AbolethConnectionInfo));
    for (int peer = 0; peer < arm.world; ++peer) {
        if (peer == arm.rank)
            continue;
        BENCH_CUDA_CHECK(
            cudaMalloc(&fabric.descriptors[peer], descriptor_bytes));
        fabric.endpoints[peer] = aboleth_runtime_create_local_endpoint(
            fabric.runtime, transport, gpu, fabric.descriptors[peer],
            descriptor_bytes, nullptr);
        ABI_CHECK(fabric.endpoints[peer] != nullptr);
        ABI_CHECK(aboleth_runtime_local_endpoint_connection_info(
                      fabric.endpoints[peer], &mine[peer]) == 0);
    }

    const std::size_t share_bytes =
        arm.world * sizeof(AbolethConnectionInfo) + sizeof(RankShareTail);
    std::vector<std::uint8_t> share(share_bytes);
    std::memcpy(share.data(), mine.data(),
                arm.world * sizeof(AbolethConnectionInfo));
    RankShareTail tail{};
    tail.data_address = reinterpret_cast<std::uint64_t>(fabric.arena);
    tail.flags_address =
        reinterpret_cast<std::uint64_t>(fabric.arena) + arm.flags_offset;
    tail.rkey = aboleth_runtime_buffer_rkey(fabric.registration);
    std::memcpy(share.data() + arm.world * sizeof(AbolethConnectionInfo),
                &tail, sizeof(tail));
    const auto matrix = rendezvous.allgather(share.data(), share_bytes);

    std::vector<std::uint64_t> handles(arm.world, 0);
    arm.peer_data.assign(arm.world, 0);
    arm.peer_flags.assign(arm.world, 0);
    arm.peer_rkey.assign(arm.world, 0);
    for (int peer = 0; peer < arm.world; ++peer) {
        const std::uint8_t* record = matrix.data() + peer * share_bytes;
        RankShareTail peer_tail{};
        std::memcpy(&peer_tail,
                    record + arm.world * sizeof(AbolethConnectionInfo),
                    sizeof(peer_tail));
        arm.peer_data[peer] = peer_tail.data_address;
        arm.peer_flags[peer] = peer_tail.flags_address;
        arm.peer_rkey[peer] = peer_tail.rkey;
        if (peer == arm.rank)
            continue;
        AbolethConnectionInfo info_for_me{};
        std::memcpy(&info_for_me,
                    record + arm.rank * sizeof(AbolethConnectionInfo),
                    sizeof(info_for_me));
        ABI_CHECK(aboleth_runtime_connect_local_endpoint(
                      fabric.endpoints[peer], &info_for_me) == 0);
        handles[peer] =
            reinterpret_cast<std::uint64_t>(fabric.descriptors[peer]);
        if (fabric.capacity == 0)
            aboleth_runtime_local_endpoint_capacity(fabric.endpoints[peer],
                                                    &fabric.capacity);
        aboleth_runtime_local_endpoint_submit_mode(fabric.endpoints[peer],
                                                   &fabric.submit_mode);
    }
    BENCH_CUDA_CHECK(
        cudaMalloc(&fabric.handles_device,
                   arm.world * sizeof(std::uint64_t)));
    BENCH_CUDA_CHECK(cudaMemcpy(fabric.handles_device, handles.data(),
                                arm.world * sizeof(std::uint64_t),
                                cudaMemcpyHostToDevice));
    arm.arena = fabric.arena;
    arm.local_data_base = reinterpret_cast<std::uint64_t>(fabric.arena);
    arm.flag_base =
        reinterpret_cast<std::uint64_t>(fabric.arena) + arm.flags_offset;
    arm.lkey = aboleth_runtime_buffer_lkey(fabric.registration);
    arm.ctx = Comm::make_ctx(fabric.handles_device, nullptr);
    arm.submit_mode = submit_name(fabric.submit_mode);
    arm.memory = bench::memory_json(free_before, bench::device_free_bytes(),
                                    "\"measured_after_arena\": true");
    // The Tier-1 quiet is a single-consumer CQ poll per endpoint (README).
    arm.max_ctas = 1;
    return fabric;
}

void tear_down(Fabric& fabric) {
    for (int peer = 0; peer < fabric.world; ++peer) {
        if (fabric.endpoints[peer] != nullptr)
            aboleth_runtime_destroy_local_endpoint(fabric.endpoints[peer]);
        if (fabric.descriptors[peer] != nullptr)
            cudaFree(fabric.descriptors[peer]);
    }
    if (fabric.registration != nullptr)
        aboleth_runtime_unregister_buffer(fabric.registration);
    cudaFree(fabric.handles_device);
    cudaFree(fabric.arena);
    if (fabric.runtime != nullptr)
        aboleth_runtime_destroy(fabric.runtime);
}

template <class Comm>
int run(bench::Options options, int rank, int world, int gpu,
        bench::Rendezvous& rendezvous) {
    if (const int refused =
            bench::check_issue<Comm>(options, "bench_tier1_aboleth"))
        return refused;
    bench::clamp_options(options, world);
    std::uint64_t stride = 0;
    bench::Arm<Comm> arm;
    arm.rank = rank;
    arm.world = world;
    arm.name = "aboleth-cpp";
    arm.data_bytes = bench::plan_data_bytes(options, world, &stride);
    arm.flags_offset = (arm.data_bytes + 7) / 8 * 8;

    Fabric fabric = bring_up<Comm>(options, arm, gpu, rendezvous);
    std::uint32_t window = options.window;
    if (fabric.capacity != 0 && window > fabric.capacity / 2) {
        window = fabric.capacity / 2;
        bench::note_clamp(options,
                          "window reduced to " + std::to_string(window) +
                              " (queue capacity " +
                              std::to_string(fabric.capacity) + ")");
    }
    const int code =
        bench::run_arm<Comm>(options, arm, rendezvous, stride, window);
    tear_down(fabric);
    return code;
}

}  // namespace

int main(int argc, char** argv) {
    const bench::Options options =
        bench::parse_options(argc, argv, "rdma");
    const int rank = bench::env_int("RANK");
    const int world = bench::env_int("WORLD_SIZE");
    const int gpu = bench::env_int("LOCAL_RANK", bench::env_int("RANK"));
    const char* master = std::getenv("MASTER_ADDR");
    const int port = bench::env_int("MASTER_PORT");
    if (master == nullptr) {
        std::fprintf(stderr, "bench_tier1_aboleth: MASTER_ADDR is unset\n");
        return 2;
    }
    if (options.bench == "ladder" && world != 2) {
        std::fprintf(stderr,
                     "bench_tier1_aboleth: ladder needs exactly 2 ranks\n");
        return 2;
    }
    if (options.transport != "rdma" && options.transport != "proxy") {
        std::fprintf(stderr,
                     "bench_tier1_aboleth: --transport rdma|proxy only "
                     "(the C ABI has no P2P transport)\n");
        return 2;
    }
    BENCH_CUDA_CHECK(cudaSetDevice(gpu));
    bench::Rendezvous rendezvous(rank, world, master, port);
    if (options.transport == "proxy")
        return run<AbolethProxyComm>(options, rank, world, gpu, rendezvous);
    return run<AbolethComm>(options, rank, world, gpu, rendezvous);
}
