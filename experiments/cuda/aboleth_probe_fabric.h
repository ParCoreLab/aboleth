#pragma once

// Where a direct-mlx5 probe gets its RDMA device and its local RC address.
//
// The probes in this directory open their own queue pairs rather than going
// through the runtime, but the questions that come before a queue pair — which
// HCA belongs to this GPU, which GID entry addresses the fabric, whether a path
// is carried by LID or by GRH, how large a packet the port will take — have one
// answer per machine, and it is the runtime's. Answering them independently is
// how these probes came to hardcode a RoCE GID index and an is_global address
// vector, which an InfiniBand fabric answers with transport-retry-exceeded.
//
// So they are asked of the runtime's exports (aboleth/runtime_c.h), which is
// also what brings ABOLETH_HCA, ABOLETH_GID_INDEX, ABOLETH_PORT and
// ABOLETH_PATH_MTU to the probes. Nothing else here is shared: each probe's
// queues, kernels and measurements remain its own.

#include "aboleth/runtime_c.h"

#include <arpa/inet.h>
#include <infiniband/verbs.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace aboleth::probe {

// The local end of an RC path. Every probe here connects a pair of queue pairs
// on one port of one device, so a single one of these describes both ends.
struct FabricAddress {
    char device_name[64]{};
    int port{};
    int gid_index{};
    ibv_gid gid{};
    std::uint16_t lid{};
    std::uint8_t link_layer{};
    ibv_mtu path_mtu{};

    // InfiniBand names its peer by LID and sends no GRH; everything else (RoCE)
    // addresses by GRH. This is the is_global decision, and the one these
    // probes used to get wrong.
    bool addresses_by_lid() const {
        return link_layer == IBV_LINK_LAYER_INFINIBAND;
    }
};

// The port the runtime would open, which is the port a probe opens.
// aboleth_runtime_query_port_address() below refuses any other, so this cannot
// drift away from the runtime's own choice without saying so.
inline int probe_port() {
    const char* configured = std::getenv("ABOLETH_PORT");
    if (configured == nullptr || configured[0] == '\0') {
        return 1;
    }
    return std::atoi(configured);
}

// The HCA for CUDA ordinal `gpu_index` and the local address of its active
// port. Exits on failure: a probe whose fabric will not come up has nothing
// left to measure, and the runtime's message says why.
inline FabricAddress resolve_fabric_address(int gpu_index) {
    FabricAddress address{};
    if (aboleth_runtime_select_device(
            gpu_index,
            address.device_name,
            sizeof(address.device_name)) != 0) {
        std::fprintf(
            stderr,
            "RDMA device selection for GPU %d failed: %s\n",
            gpu_index,
            aboleth_runtime_last_error());
        std::exit(EXIT_FAILURE);
    }
    address.port = probe_port();

    // The GID index reported here is aboleth_runtime_select_gid_index()'s
    // answer — the same helper computes both — so asking for it separately
    // would only reopen the device to recompute a value already in hand.
    std::uint32_t path_mtu = 0;
    if (aboleth_runtime_query_port_address(
            address.device_name,
            address.port,
            &address.lid,
            &address.link_layer,
            address.gid.raw,
            &address.gid_index,
            &path_mtu) != 0) {
        std::fprintf(
            stderr,
            "RDMA port query on %s port %d failed: %s\n",
            address.device_name,
            address.port,
            aboleth_runtime_last_error());
        std::exit(EXIT_FAILURE);
    }
    address.path_mtu = static_cast<ibv_mtu>(path_mtu);
    return address;
}

// The selected device in a list the probe already holds open. ibv_device
// pointers only come from a list, and the probes keep theirs for the lifetime
// of the run, so selection by name is how the export's answer reaches them.
inline ibv_device* find_fabric_device(
    ibv_device** devices,
    int device_count,
    const FabricAddress& address) {
    for (int index = 0; index < device_count; ++index) {
        if (std::strcmp(
                ibv_get_device_name(devices[index]),
                address.device_name) == 0) {
            return devices[index];
        }
    }
    std::fprintf(
        stderr,
        "%s was not found\n",
        address.device_name);
    std::exit(EXIT_FAILURE);
}

// The path and address-vector half of an RTR transition, filled the way the
// runtime fills it. Both ends of a probe's pair sit on this port, so `address`
// is at once the local and the remote end of the path.
inline void apply_address_vector(
    ibv_qp_attr& attr,
    const FabricAddress& address) {
    attr.path_mtu = address.path_mtu;
    attr.ah_attr.port_num = static_cast<std::uint8_t>(address.port);
    if (address.addresses_by_lid()) {
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = address.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
    } else {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = address.gid;
        attr.ah_attr.grh.sgid_index =
            static_cast<std::uint8_t>(address.gid_index);
        attr.ah_attr.grh.hop_limit = 64;
    }
}

// One line naming the device, port and addressing mode a probe brought up, so
// a run is self-describing when ABOLETH_HCA or ABOLETH_GID_INDEX moved it.
inline void report_fabric_address(const FabricAddress& address) {
    char gid_text[64]{};
    inet_ntop(
        AF_INET6,
        address.gid.raw,
        gid_text,
        sizeof(gid_text));
    std::printf(
        "Fabric: %s port %d, %s addressing, LID 0x%04x, GID index %d (%s), MTU %d\n",
        address.device_name,
        address.port,
        address.addresses_by_lid() ? "LID" : "GRH",
        address.lid,
        address.gid_index,
        gid_text,
        static_cast<int>(address.path_mtu));
}

}  // namespace aboleth::probe
