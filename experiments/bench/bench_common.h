#pragma once

// Shared pieces of the Tier-1 benchmark arms (workstream 1). Workstream 2
// grows this into the common home for the CUDA-side benchmarks; until then
// only the bench_tier1_* binaries and experiments/bench/ include it, and the
// existing copies of these helpers stay where they are.

#include <cstddef>
#include <cstdint>

namespace bench {

// Benchmark op selectors, shared by every arm's CLI and kernels.
inline constexpr int kOpPut = 0;        // N puts, one trailing signal
inline constexpr int kOpPutSignal = 1;  // a signal after every put
inline constexpr int kOpPutInline = 2;  // one 8-byte inline write that IS the flag

#if defined(__CUDACC__)
#define BENCH_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define BENCH_HOST_DEVICE inline
#endif

// The tree's payload pattern: byte i of a filled region holds (i*17+3)&0xff.
BENCH_HOST_DEVICE std::uint8_t pattern_byte(std::uint64_t index) {
    return static_cast<std::uint8_t>((index * 17 + 3) & 0xff);
}

#if defined(__CUDACC__)

__device__ __forceinline__ std::uint64_t globaltimer() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// Sense-free grid barrier over a zero-initialized device counter: each
// round's release target is the next multiple of the CTA count, so
// consecutive rounds cannot race the counter.
__device__ __forceinline__ void grid_epoch_barrier(
    unsigned* state, unsigned ctas) {
    const unsigned arrived = atomicAdd(state, 1u) + 1u;
    const unsigned target = ((arrived - 1u) / ctas + 1u) * ctas;
    while (*reinterpret_cast<volatile unsigned*>(state) < target)
        __nanosleep(64);
}

// DeepEP-style cache disturbance: every participating thread strides a dummy
// buffer. Callers run this outside the timed region.
__device__ __forceinline__ void l2_flush(
    std::uint8_t* buffer, std::uint32_t bytes) {
    auto* words = reinterpret_cast<std::uint32_t*>(buffer);
    const std::uint32_t count = bytes / sizeof(std::uint32_t);
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += gridDim.x * blockDim.x)
        words[index] = 0;
}

#endif  // __CUDACC__

#if !defined(BENCH_COMMON_NO_HOST_HELPERS)

// salt makes alltoall slices sender-distinct so misrouting is caught, not
// just corruption; 0 keeps the tree's plain pattern.
inline void pattern_fill_host(std::uint8_t* destination, std::size_t bytes,
                              std::uint8_t salt = 0) {
    for (std::size_t index = 0; index < bytes; ++index)
        destination[index] = pattern_byte(index) ^ salt;
}

// Number of mismatching bytes against the (salted) pattern.
inline std::size_t pattern_check_host(
    const std::uint8_t* observed, std::size_t bytes, std::uint8_t salt = 0) {
    std::size_t wrong = 0;
    for (std::size_t index = 0; index < bytes; ++index)
        wrong += observed[index] != (pattern_byte(index) ^ salt);
    return wrong;
}

#endif  // BENCH_COMMON_NO_HOST_HELPERS

}  // namespace bench

// Host-side rendezvous and stats for the CUDA benchmark arms. Kept in the
// header (inline) so the arms stay single-TU; POSIX-only, like the rest of
// the experiments tree. Not guarded by __CUDA_ARCH__: nvcc parses the whole
// TU in both passes and needs the declarations visible in each.
#if !defined(BENCH_COMMON_NO_HOST_HELPERS)

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

namespace detail {

inline void io_all(int descriptor, void* data, std::size_t bytes, bool write) {
    auto* cursor = static_cast<std::uint8_t*>(data);
    while (bytes) {
        const auto moved = write ? ::send(descriptor, cursor, bytes, 0)
                                 : ::recv(descriptor, cursor, bytes, 0);
        if (moved <= 0) {
            if (moved < 0 && errno == EINTR)
                continue;
            throw std::runtime_error("bench: rendezvous socket closed early");
        }
        cursor += moved;
        bytes -= static_cast<std::size_t>(moved);
    }
}

}  // namespace detail

// Rank rendezvous over persistent TCP connections: rank 0 accepts world-1
// peers once, then any number of fixed-size allgathers run over the same
// sockets (a one-byte allgather is a barrier). Persistent connections avoid
// the rebind race a listen-per-collective design would have. Every rank
// must call each collective in the same order with the same length.
class Rendezvous {
  public:
    Rendezvous(int rank, int world, const char* master, int port)
        : rank_(rank), world_(world) {
        if (world < 2)
            return;
        if (rank == 0) {
            const int server = ::socket(AF_INET, SOCK_STREAM, 0);
            const int enable = 1;
            ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &enable,
                         sizeof(enable));
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = INADDR_ANY;
            address.sin_port = htons(static_cast<std::uint16_t>(port));
            if (::bind(server, reinterpret_cast<sockaddr*>(&address),
                       sizeof(address)) != 0 ||
                ::listen(server, world) != 0)
                throw std::runtime_error(
                    "bench: rank 0 cannot listen for peers");
            connections_.assign(static_cast<std::size_t>(world), -1);
            for (int accepted = 1; accepted < world; ++accepted) {
                const int connection = ::accept(server, nullptr, nullptr);
                std::int32_t peer = -1;
                detail::io_all(connection, &peer, sizeof(peer), false);
                connections_[static_cast<std::size_t>(peer)] = connection;
            }
            ::close(server);
            return;
        }
        addrinfo hints{}, *resolved = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        const std::string service = std::to_string(port);
        if (::getaddrinfo(master, service.c_str(), &hints, &resolved) != 0)
            throw std::runtime_error(
                "bench: cannot resolve the master address");
        int connection = -1;
        for (int attempt = 0; attempt < 1200 && connection < 0; ++attempt) {
            connection = ::socket(AF_INET, SOCK_STREAM, 0);
            if (::connect(connection, resolved->ai_addr,
                          resolved->ai_addrlen) != 0) {
                ::close(connection);
                connection = -1;
                ::usleep(100'000);
            }
        }
        ::freeaddrinfo(resolved);
        if (connection < 0)
            throw std::runtime_error(
                "bench: cannot reach the rank 0 rendezvous");
        const std::int32_t me = rank;
        detail::io_all(connection, const_cast<std::int32_t*>(&me),
                       sizeof(me), true);
        connections_.assign(1, connection);
    }

    ~Rendezvous() {
        for (const int connection : connections_)
            if (connection >= 0)
                ::close(connection);
    }

    std::vector<std::uint8_t> allgather(const void* blob, std::size_t len) {
        std::vector<std::uint8_t> matrix(
            static_cast<std::size_t>(world_) * len);
        std::memcpy(matrix.data() + rank_ * len, blob, len);
        if (rank_ == 0) {
            for (int peer = 1; peer < world_; ++peer)
                detail::io_all(connections_[static_cast<std::size_t>(peer)],
                               matrix.data() + peer * len, len, false);
            for (int peer = 1; peer < world_; ++peer)
                detail::io_all(connections_[static_cast<std::size_t>(peer)],
                               matrix.data(), matrix.size(), true);
        } else {
            detail::io_all(connections_[0], const_cast<void*>(blob), len,
                           true);
            detail::io_all(connections_[0], matrix.data(), matrix.size(),
                           false);
        }
        return matrix;
    }

    void barrier() {
        const std::uint8_t token = 0;
        allgather(&token, sizeof(token));
    }

  private:
    int rank_;
    int world_;
    std::vector<int> connections_;
};

// Linear-interpolation quantile matching Python's statistics.quantiles
// (method="inclusive"), so every arm reports the same statistic.
inline double quantile_inclusive(std::vector<double> values, double q) {
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double position = q * (static_cast<double>(values.size()) - 1.0);
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    return values[low] + (values[high] - values[low]) * (position - low);
}

}  // namespace bench

#endif  // host helpers
