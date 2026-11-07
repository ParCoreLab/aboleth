#pragma once

// Choosing which RDMA device and port this runtime will open.
//
// Everything here answers a question before any queue exists: which mlx5
// device sits closest to the GPU on the PCI topology, which port of it is
// usable, and which GID entry of that port addresses the fabric. The answers
// are environment- and sysfs-driven, which is why they are separable from the
// endpoint code that consumes them.
//
// Internal to the runtime; see aboleth_runtime_internal.h.

#include "../aboleth_runtime_internal.h"

#include <cstdint>
#include <string>

namespace aboleth::runtime {

// The GID entry to address the fabric with. ABOLETH_GID_INDEX wins outright;
// InfiniBand ports always use entry 0; RoCE prefers a v2 entry and otherwise
// falls back to the first populated one.
std::uint32_t select_gid_index(
    ibv_context* context,
    const PortAddress& port,
    std::uint32_t gid_table_length);

// The active port named by ABOLETH_PORT, with its LID, MTU and GID resolved.
// Throws when the port is not active.
PortAddress query_port_address(
    ibv_context* context);

// The mlx5 device sharing the deepest PCI path with `gpu_index`, unless
// ABOLETH_HCA or `requested_name` names one. `selected_name` receives the
// device name that was chosen.
ibv_device* select_runtime_device(
    ibv_device** devices,
    int device_count,
    int gpu_index,
    const char* requested_name,
    std::string& selected_name);

} // namespace aboleth::runtime
