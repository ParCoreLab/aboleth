#pragma once

// Registering memory with whichever provider the runtime opened.
//
// Two layers live here. register_memory() is the verbs primitive: it tries a
// direct registration, falls back to dmabuf when the pointer is device memory
// the driver will not pin directly, and negotiates relaxed ordering. Above it,
// register_buffer() is what callers of the C ABI get — one record per
// registered region, keyed for both providers, holding the runtime alive.
//
// Internal to the runtime; see aboleth_runtime_internal.h.

#include "../aboleth_runtime_internal.h"

#if ABOLETH_HAS_EFA
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>

namespace aboleth::runtime {

// Registers `bytes` at `address`, falling back to a dmabuf registration when
// the direct one fails (device pointers on drivers without peer mapping).
// Relaxed ordering is requested unless disabled, and required when
// ABOLETH_REQUIRE_RELAXED_ORDERING is set. `relaxed`, when non-null, reports
// whether the registration that succeeded carries relaxed ordering. Returns
// nullptr on failure.
ibv_mr* register_memory(
    ibv_pd* pd,
    void* address,
    std::size_t bytes,
    int access,
    bool allow_relaxed = true,
    bool* relaxed = nullptr);

// One registered region: an mlx5 memory region or an EFA one, reduced to the
// pair of keys the ABI carries. `generation` lets a caller notice that a
// window was re-registered underneath it.
struct RegisteredBufferContext {
    std::shared_ptr<SharedRuntimeContext> runtime;
    ibv_mr* mr{};
#if ABOLETH_HAS_EFA
    fid_mr* efa_mr{};
#endif
    std::uint32_t local_key{};
    std::uint32_t remote_key{};
    void* address{};
    std::size_t bytes{};
    int gpu_index{};
    std::uint64_t generation{};
    // A verbs registration made without relaxed ordering. Publication may then
    // rely on a fenced write. False whenever that is unknown (EFA).
    bool strictly_ordered{};

    // Out of line: unregistering an EFA region reaches into EfaFabricContext,
    // which this header only sees as a declaration.
    ~RegisteredBufferContext();
};

std::shared_ptr<RegisteredBufferContext> register_buffer(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    void* address,
    std::size_t bytes,
    std::uint64_t generation,
    bool allow_relaxed = true);

// What aboleth_runtime_register_buffer() hands back across the C ABI.
struct BufferHandle {
    std::shared_ptr<RegisteredBufferContext> buffer;
};

} // namespace aboleth::runtime
