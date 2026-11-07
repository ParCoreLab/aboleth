#pragma once

// Bringing up the shared runtime context.
//
// One SharedRuntimeContext per opened device carries everything endpoints,
// registered buffers and client integrations share. Creating it is the only step
// that decides which provider is in play, so the proxy provider selection is
// resolved here before any queue exists.
//
// Internal to the runtime; see aboleth_runtime_internal.h.

#include "../aboleth_runtime_internal.h"

#include <memory>

namespace aboleth::runtime {

// Opens the device closest to `gpu_index` (or the one `requested_name` /
// ABOLETH_HCA names), queries its port, and allocates the protection domain.
// An EFA-OFI runtime stops after the fabric context: it has no verbs objects.
std::shared_ptr<SharedRuntimeContext> create_shared_runtime(
    int gpu_index = -1,
    const char* requested_name = nullptr);

// What aboleth_runtime_create() hands back across the C ABI.
struct RuntimeHandle {
    std::shared_ptr<SharedRuntimeContext> runtime;
};

} // namespace aboleth::runtime
