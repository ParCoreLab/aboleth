// Choosing which RDMA device, port and GID this runtime will address the
// fabric through.
//
// Everything here answers a question before any queue exists: which mlx5
// device sits closest to the GPU on the PCI topology, which port of it is
// usable, and which GID entry of that port the link layer wants. The answers
// come from the environment and from sysfs, never from an open queue pair,
// which is what makes them separable from the endpoint code that consumes
// them — and what makes them worth exporting on their own (the C surface at
// the end of this file).
//
// Internal declarations are in aboleth_device_select.h; the exported entry
// points are in aboleth/runtime_c.h.

#include "aboleth_device_select.h"

#include "../aboleth_runtime_common.h"
#include "../aboleth_runtime_internal.h"
#include "aboleth_connection.h"

#include "aboleth/runtime_c.h"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include "../dl/shims.h"

namespace dl = aboleth::dl;

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <string>

// aboleth::runtime is where the shared internals live (see
// aboleth_runtime_internal.h). The anonymous-namespace helpers and the
// extern "C" arm below reach them unqualified, as the runtime translation
// unit this code came from does.
using namespace aboleth::runtime;

namespace {

// The port an RDMA device is opened on unless ABOLETH_PORT says otherwise.
constexpr int kDefaultPort = 1;

bool gid_is_unspecified(const ibv_gid& gid) {
    return std::all_of(
        std::begin(gid.raw),
        std::end(gid.raw),
        [](std::uint8_t byte) {
            return byte == 0;
        });
}

}  // namespace

namespace aboleth::runtime {

std::uint32_t select_gid_index(
    ibv_context* context,
    const PortAddress& port,
    std::uint32_t gid_table_length) {
    const char* configured =
        std::getenv("ABOLETH_GID_INDEX");
    if (configured != nullptr &&
        configured[0] != '\0') {
        return static_cast<std::uint32_t>(
            configured_index(
                "ABOLETH_GID_INDEX",
                0));
    }
    if (port.link_layer ==
        IBV_LINK_LAYER_INFINIBAND) {
        return 0;
    }

    std::uint32_t fallback =
        std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t index = 0;
         index < gid_table_length;
         ++index) {
        ibv_gid_entry entry{};
        if (dl::verbs().query_gid_ex(
                context,
                port.port_num,
                index,
                &entry,
                0) != 0 ||
            gid_is_unspecified(entry.gid)) {
            continue;
        }
        if (fallback ==
            std::numeric_limits<std::uint32_t>::max()) {
            fallback = index;
        }
        if (entry.gid_type ==
            IBV_GID_TYPE_ROCE_V2) {
            return index;
        }
    }
    if (fallback ==
        std::numeric_limits<std::uint32_t>::max()) {
        throw_error(
            "no populated GID was found on the active RDMA port");
    }
    return fallback;
}

PortAddress query_port_address(
    ibv_context* context) {
    PortAddress address{};
    address.port_num =
        static_cast<std::uint8_t>(
            configured_index(
                "ABOLETH_PORT",
                kDefaultPort));
    ibv_port_attr attributes{};
    verbs_check(
        dl::verbs().query_port(
            context,
            address.port_num,
            &attributes),
        "ibv_query_port");
    if (attributes.state != IBV_PORT_ACTIVE) {
        throw_error("RDMA port is not active");
    }
    address.link_layer = attributes.link_layer;
    address.lid = attributes.lid;
    address.active_mtu = attributes.active_mtu;
    address.gid_index = select_gid_index(
        context,
        address,
        attributes.gid_tbl_len);
    verbs_check(
        dl::verbs().ibv_query_gid(
            context,
            address.port_num,
            address.gid_index,
            &address.gid),
        "ibv_query_gid");
    return address;
}

// The direct datapath maps mlx5 queues, so selection admits only devices the
// mlx5 driver claims. The name prefix is no proxy for that: udev renames HCAs
// (an mlx5 port may present as ibp26s0), and foreign verbs devices such as
// EFA's must stay ineligible however they are named. Asking the driver keeps
// both properties. Without libmlx5 no device qualifies, which is also the
// truth of the matter.
bool device_is_mlx5(ibv_device* device) {
    return dl::mlx5dv_available() &&
        dl::mlx5dv().mlx5dv_is_supported(device);
}

ibv_device* select_runtime_device(
    ibv_device** devices,
    int device_count,
    int gpu_index,
    const char* requested_name,
    std::string& selected_name) {
    const char* environment_name =
        std::getenv("ABOLETH_HCA");
    const char* explicit_name =
        requested_name != nullptr &&
                requested_name[0] != '\0'
            ? requested_name
            : environment_name;
    if (explicit_name != nullptr &&
        explicit_name[0] != '\0') {
        for (int index = 0;
             index < device_count;
             ++index) {
            const char* name =
                dl::verbs().ibv_get_device_name(devices[index]);
            if (std::strcmp(name, explicit_name) == 0) {
                selected_name = name;
                return devices[index];
            }
        }
        throw_error(
            std::string("requested RDMA device was not found: ") +
            explicit_name);
    }

    if (gpu_index >= 0) {
        char pci_bus_id[32]{};
        cuda_check(
            cudaDeviceGetPCIBusId(
                pci_bus_id,
                sizeof(pci_bus_id),
                gpu_index),
            "cudaDeviceGetPCIBusId");
        std::string normalized_pci_bus_id =
            pci_bus_id;
        const std::size_t domain_separator =
            normalized_pci_bus_id.find(':');
        if (domain_separator > 4 &&
            domain_separator !=
                std::string::npos) {
            normalized_pci_bus_id =
                normalized_pci_bus_id.substr(
                    domain_separator - 4);
        }
        std::transform(
            normalized_pci_bus_id.begin(),
            normalized_pci_bus_id.end(),
            normalized_pci_bus_id.begin(),
            [](unsigned char character) {
                return static_cast<char>(
                    std::tolower(character));
            });
        const auto gpu_path =
            canonical_device_path(
                std::filesystem::path(
                    "/sys/bus/pci/devices") /
                normalized_pci_bus_id);
        if (gpu_path.empty()) {
            throw_error(
                "could not resolve the CUDA device PCI path");
        }

        ibv_device* selected = nullptr;
        std::size_t best_depth = 0;
        for (int index = 0;
             index < device_count;
             ++index) {
            const char* name =
                dl::verbs().ibv_get_device_name(devices[index]);
            if (!device_is_mlx5(devices[index])) {
                continue;
            }
            const auto hca_path =
                canonical_device_path(
                    std::filesystem::path(
                        "/sys/class/infiniband") /
                    name /
                    "device");
            if (hca_path.empty()) {
                continue;
            }
            const std::size_t depth =
                common_path_depth(
                    gpu_path,
                    hca_path);
            if (selected == nullptr ||
                depth > best_depth) {
                selected = devices[index];
                selected_name = name;
                best_depth = depth;
            }
        }
        if (selected != nullptr) {
            return selected;
        }
        throw_error(
            "no mlx5-capable device with a resolvable PCI path was found");
    }

    // Without a GPU to match against, prefer the conventional first port and
    // otherwise settle for any device the mlx5 driver claims.
    for (int index = 0;
         index < device_count;
         ++index) {
        const char* name =
            dl::verbs().ibv_get_device_name(devices[index]);
        if (std::strcmp(name, "mlx5_0") == 0) {
            selected_name = name;
            return devices[index];
        }
    }
    for (int index = 0;
         index < device_count;
         ++index) {
        if (device_is_mlx5(devices[index])) {
            selected_name =
                dl::verbs().ibv_get_device_name(devices[index]);
            return devices[index];
        }
    }
    throw_error("no mlx5-capable device was found");
}

}  // namespace aboleth::runtime

// The C surface below is new in this file rather than moved into it: it is the
// same three answers the runtime computes for itself, made reachable without
// creating a runtime. Every entry point is a wrapper over the helpers above —
// the selection, link-layer and MTU rules have exactly one implementation, and
// it is the one a live runtime uses.

namespace {

// The device list, released however the query leaves. ibv_device pointers are
// only valid while the list is alive, which is what makes this worth a type.
class ScopedDeviceList {
public:
    ScopedDeviceList() {
        devices_ =
            dl::verbs().ibv_get_device_list(
                &count_);
        if (devices_ == nullptr) {
            throw_error(
                "ibv_get_device_list failed");
        }
    }

    ~ScopedDeviceList() {
        dl::verbs().ibv_free_device_list(
            devices_);
    }

    ScopedDeviceList(
        const ScopedDeviceList&) = delete;
    ScopedDeviceList& operator=(
        const ScopedDeviceList&) = delete;

    // Selects into `name` and returns the device, exactly as the runtime's own
    // bring-up does.
    ibv_device* select(
        int gpu_index,
        const char* requested_name,
        std::string& name) const {
        return select_runtime_device(
            devices_,
            count_,
            gpu_index,
            requested_name,
            name);
    }

private:
    ibv_device** devices_{};
    int count_{};
};

// A device selected and opened for the duration of one query.
class ScopedDevice {
public:
    explicit ScopedDevice(
        const char* requested_name) {
        std::string name;
        context_ =
            dl::verbs().ibv_open_device(
                devices_.select(
                    /*gpu_index=*/-1,
                    requested_name,
                    name));
        if (context_ == nullptr) {
            throw_error(
                "ibv_open_device failed");
        }
    }

    ~ScopedDevice() {
        if (context_ != nullptr) {
            dl::verbs().ibv_close_device(
                context_);
        }
    }

    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(
        const ScopedDevice&) = delete;

    ibv_context* context() const {
        return context_;
    }

private:
    ScopedDeviceList devices_;
    ibv_context* context_{};
};

// `port` is the caller's way of naming a port explicitly; 0 means "whichever
// one the runtime would open", which is ABOLETH_PORT or kDefaultPort.
int requested_port(int port) {
    return port > 0
        ? port
        : configured_index(
              "ABOLETH_PORT",
              kDefaultPort);
}

}  // namespace

extern "C" {

int aboleth_runtime_select_device(
    int gpu_index,
    char* name_out,
    std::size_t name_capacity) {
    last_error.clear();
    try {
        if (name_out == nullptr ||
            name_capacity == 0) {
            throw_error(
                "null device-name buffer");
        }
        std::string name;
        ScopedDeviceList().select(
            gpu_index,
            nullptr,
            name);
        if (name.size() + 1 > name_capacity) {
            throw_error(
                "device-name buffer is too small");
        }
        std::memcpy(
            name_out,
            name.c_str(),
            name.size() + 1);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_select_gid_index(
    const char* device_name,
    int port) {
    last_error.clear();
    try {
        ScopedDevice device(device_name);
        // query_port_address() is the runtime's own answer — including the
        // active-port refusal — and it reads ABOLETH_PORT rather than taking
        // a port, so a mismatch is reported, exactly as in
        // aboleth_runtime_query_port_address().
        const PortAddress address =
            query_port_address(
                device.context());
        if (address.port_num !=
            requested_port(port)) {
            throw_error(
                "requested port is not the port ABOLETH_PORT selects");
        }
        return static_cast<int>(
            address.gid_index);
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_query_port_address(
    const char* device_name,
    int port,
    std::uint16_t* lid_out,
    std::uint8_t* link_layer_out,
    std::uint8_t* gid_out16,
    int* gid_index_out,
    std::uint32_t* path_mtu_out) {
    last_error.clear();
    try {
        ScopedDevice device(device_name);
        // query_port_address() is the runtime's own answer, and it reads
        // ABOLETH_PORT rather than taking a port. Asking it about a different
        // port would mean a second implementation of the same rules, so a
        // mismatch is reported instead.
        const PortAddress address =
            query_port_address(
                device.context());
        if (address.port_num !=
            requested_port(port)) {
            throw_error(
                "requested port is not the port ABOLETH_PORT selects");
        }
        if (lid_out != nullptr) {
            *lid_out = address.lid;
        }
        if (link_layer_out != nullptr) {
            *link_layer_out =
                address.link_layer;
        }
        if (gid_out16 != nullptr) {
            std::memcpy(
                gid_out16,
                address.gid.raw,
                sizeof(address.gid.raw));
        }
        if (gid_index_out != nullptr) {
            *gid_index_out =
                static_cast<int>(
                    address.gid_index);
        }
        if (path_mtu_out != nullptr) {
            *path_mtu_out =
                static_cast<std::uint32_t>(
                    select_path_mtu(
                        address.active_mtu));
        }
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

}  // extern "C"
