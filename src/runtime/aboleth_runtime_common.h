#pragma once

// Host-runtime helpers shared by the provider-neutral proxy runtime and the
// GPU-initiated EFA runtime.
//
// Both need to answer the same two questions before they touch any provider
// API: which EFA device is closest to a given GPU, and how fast is that GPU's
// clock. Those answers depend only on sysfs topology and the CUDA runtime, so
// they live here rather than inside either provider's translation unit.

#include <cuda_runtime.h>

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aboleth::runtime {

[[noreturn]] inline void throw_error(const std::string& message) {
    throw std::runtime_error(message);
}

inline void cuda_check(
    cudaError_t status,
    const char* operation) {
    if (status != cudaSuccess) {
        throw_error(
            std::string(operation) + ": " +
            cudaGetErrorString(status));
    }
}

inline std::uint64_t cuda_clock_rate_khz(int device) {
    int clock_rate = 0;
    cuda_check(
        cudaDeviceGetAttribute(
            &clock_rate,
            cudaDevAttrClockRate,
            device),
        "cudaDeviceGetAttribute(clock rate)");
    return static_cast<std::uint64_t>(
        clock_rate);
}

inline std::size_t common_path_depth(
    const std::filesystem::path& first,
    const std::filesystem::path& second) {
    auto first_part = first.begin();
    auto second_part = second.begin();
    std::size_t depth = 0;
    while (first_part != first.end() &&
           second_part != second.end() &&
           *first_part == *second_part) {
        ++depth;
        ++first_part;
        ++second_part;
    }
    return depth;
}

inline std::filesystem::path canonical_device_path(
    const std::filesystem::path& path) {
    std::error_code error;
    auto canonical =
        std::filesystem::canonical(path, error);
    if (error) {
        return {};
    }
    return canonical;
}

inline std::string normalized_gpu_pci_bus_id(
    int gpu_index,
    const char* operation) {
    char pci_bus_id[32]{};
    cuda_check(
        cudaDeviceGetPCIBusId(
            pci_bus_id,
            sizeof(pci_bus_id),
            gpu_index),
        operation);
    std::string normalized = pci_bus_id;
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char character) {
            return static_cast<char>(
                std::tolower(character));
        });
    return normalized;
}

// libfabric names an EFA domain "<device>-rdm"; sysfs and ibverbs know it by
// the bare device name.
inline std::string efa_device_base_name(
    const std::string& name) {
    if (name.size() > 4 &&
        name.compare(
            name.size() - 4,
            4,
            "-rdm") == 0) {
        return name.substr(0, name.size() - 4);
    }
    return name;
}

// The EFA device sharing the deepest PCI path with the GPU, unless one was
// named explicitly.
inline std::string select_efa_device_name(
    int gpu_index,
    const char* requested_name) {
    const char* environment_name =
        std::getenv("ABOLETH_EFA_DEVICE");
    const char* legacy_name =
        std::getenv("ABOLETH_HCA");
    const char* explicit_name =
        requested_name != nullptr &&
                requested_name[0] != '\0'
            ? requested_name
            : (environment_name != nullptr &&
                       environment_name[0] != '\0'
                   ? environment_name
                   : legacy_name);
    if (explicit_name != nullptr &&
        explicit_name[0] != '\0') {
        const std::string base =
            efa_device_base_name(explicit_name);
        if (!std::filesystem::exists(
                std::filesystem::path(
                    "/sys/class/infiniband") /
                base)) {
            throw_error(
                "requested EFA device was not found: " +
                base);
        }
        return base;
    }

    std::filesystem::path gpu_path;
    if (gpu_index >= 0) {
        gpu_path = canonical_device_path(
            std::filesystem::path(
                "/sys/bus/pci/devices") /
            normalized_gpu_pci_bus_id(
                gpu_index,
                "cudaDeviceGetPCIBusId(EFA)"));
    }

    std::string selected;
    std::size_t best_depth = 0;
    const std::filesystem::path root(
        "/sys/class/infiniband");
    for (const auto& entry :
         std::filesystem::directory_iterator(root)) {
        const std::string name =
            entry.path().filename().string();
        if (name.rfind("rdmap", 0) != 0) {
            continue;
        }
        if (gpu_path.empty()) {
            return name;
        }
        const auto device_path =
            canonical_device_path(
                entry.path() / "device");
        if (device_path.empty()) {
            continue;
        }
        const std::size_t depth =
            common_path_depth(
                gpu_path,
                device_path);
        if (selected.empty() ||
            depth > best_depth) {
            selected = name;
            best_depth = depth;
        }
    }
    if (selected.empty()) {
        throw_error(
            "no EFA device with a resolvable PCI path was found");
    }
    return selected;
}

// The GPU-local EFA devices ranked by PCI affinity, for striping work across
// rails.
inline std::vector<std::string>
select_efa_device_names(
    int gpu_index,
    std::size_t maximum_count) {
    if (maximum_count == 0) {
        return {};
    }
    const char* explicit_device =
        std::getenv("ABOLETH_EFA_DEVICE");
    if (explicit_device != nullptr &&
        explicit_device[0] != '\0') {
        return {
            efa_device_base_name(
                explicit_device)};
    }
    const auto gpu_path =
        canonical_device_path(
            std::filesystem::path(
                "/sys/bus/pci/devices") /
            normalized_gpu_pci_bus_id(
                gpu_index,
                "cudaDeviceGetPCIBusId(EFA rails)"));
    std::vector<
        std::pair<std::size_t, std::string>>
        candidates;
    for (const auto& entry :
         std::filesystem::directory_iterator(
             "/sys/class/infiniband")) {
        const std::string name =
            entry.path().filename().string();
        if (name.rfind("rdmap", 0) != 0) {
            continue;
        }
        const auto device_path =
            canonical_device_path(
                entry.path() / "device");
        if (device_path.empty()) {
            continue;
        }
        candidates.emplace_back(
            common_path_depth(
                gpu_path,
                device_path),
            name);
    }
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const auto& first,
           const auto& second) {
            if (first.first != second.first) {
                return first.first >
                    second.first;
            }
            return first.second <
                second.second;
        });
    std::vector<std::string> selected;
    for (const auto& candidate :
         candidates) {
        selected.push_back(candidate.second);
        if (selected.size() ==
            maximum_count) {
            break;
        }
    }
    if (selected.empty()) {
        throw_error(
            "no GPU-local EFA rails were found");
    }
    return selected;
}

// EFA generations 1 and 2 (1d0f:efa0 / 1d0f:efa1 — the p4 and p5 families)
// drop GPU MMIO stores to the adapter's BAR in the Nitro fabric, so
// GPU-submitted posting cannot work there.
inline bool gpu_posting_unsupported(
    const std::string& device_name) {
    std::ifstream stream(
        "/sys/class/infiniband/" + device_name +
        "/device/device");
    std::string pci_device_id;
    if (!(stream >> pci_device_id)) {
        return false;
    }
    return pci_device_id == "0xefa0" ||
        pci_device_id == "0xefa1";
}

// ABOLETH_EFA_SUBMIT selects who moves descriptors into the adapter: "gpu"
// stores them from the kernel (needs a platform that routes GPU MMIO into
// the BAR), "host" stages them for a doorbell agent. Unset picks by device
// generation.
inline bool resolve_staged_submit(
    const std::string& device_name) {
    const char* configured =
        std::getenv("ABOLETH_EFA_SUBMIT");
    const std::string submit =
        configured == nullptr ? "" : configured;
    if (!submit.empty() && submit != "gpu" &&
        submit != "host") {
        throw_error(
            "ABOLETH_EFA_SUBMIT must be gpu or host");
    }
    const bool unsupported =
        gpu_posting_unsupported(device_name);
    if (submit == "host" ||
        (submit.empty() && unsupported)) {
        return true;
    }
    if (unsupported) {
        std::fprintf(
            stderr,
            "aboleth: %s is an EFA gen-1/2 device; this platform "
            "drops GPU stores to the adapter BAR, so GPU-submitted "
            "EFA cannot complete transfers. Unset "
            "ABOLETH_EFA_SUBMIT for staged submission, use "
            "Transport.PROXY with ABOLETH_PROXY_PROVIDER=efa, or a "
            "p5en/p6-class instance. See docs/EFA_ABI.md.\n",
            device_name.c_str());
    }
    return false;
}

// Keep a progress thread on a core of the NUMA node its GPU hangs off.
inline void pin_efa_proxy_worker(
    int gpu_index,
    int worker_shard) {
    char pci_bus_id[32]{};
    if (cudaDeviceGetPCIBusId(
            pci_bus_id,
            sizeof(pci_bus_id),
            gpu_index) != cudaSuccess) {
        return;
    }
    std::string normalized_pci_bus_id =
        pci_bus_id;
    std::transform(
        normalized_pci_bus_id.begin(),
        normalized_pci_bus_id.end(),
        normalized_pci_bus_id.begin(),
        [](unsigned char character) {
            return static_cast<char>(
                std::tolower(character));
        });
    std::ifstream numa_file(
        std::filesystem::path(
            "/sys/bus/pci/devices") /
        normalized_pci_bus_id /
        "numa_node");
    int numa_node = -1;
    numa_file >> numa_node;
    if (numa_node < 0) {
        return;
    }
    const int local_gpu =
        gpu_index % 4;
    const int cpu =
        numa_node * 48 +
        8 +
        local_gpu * 10 +
        (worker_shard % 9);
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu, &affinity);
    pthread_setaffinity_np(
        pthread_self(),
        sizeof(affinity),
        &affinity);
}

// Keep an agent thread on the NUMA node its GPU hangs off without claiming
// a specific core: the target cpuset is the node's cpulist intersected with
// the affinity the launcher granted (srun/MPI cgroups included), which is
// how NCCL places its proxy threads. An empty intersection or any sysfs
// surprise keeps the inherited affinity rather than fighting the launcher.
inline void pin_numa_local(
    int gpu_index,
    int worker_shard) {
    char pci_bus_id[32]{};
    if (cudaDeviceGetPCIBusId(
            pci_bus_id,
            sizeof(pci_bus_id),
            gpu_index) != cudaSuccess) {
        return;
    }
    std::string normalized_pci_bus_id =
        pci_bus_id;
    std::transform(
        normalized_pci_bus_id.begin(),
        normalized_pci_bus_id.end(),
        normalized_pci_bus_id.begin(),
        [](unsigned char character) {
            return static_cast<char>(
                std::tolower(character));
        });
    std::ifstream numa_file(
        std::filesystem::path(
            "/sys/bus/pci/devices") /
        normalized_pci_bus_id /
        "numa_node");
    int numa_node = -1;
    numa_file >> numa_node;
    if (numa_node < 0) {
        return;
    }
    std::ifstream cpulist_file(
        "/sys/devices/system/node/node" +
        std::to_string(numa_node) +
        "/cpulist");
    std::string cpulist;
    if (!(cpulist_file >> cpulist)) {
        return;
    }
    cpu_set_t node_set;
    CPU_ZERO(&node_set);
    std::size_t start = 0;
    while (start < cpulist.size()) {
        std::size_t end =
            cpulist.find(',', start);
        if (end == std::string::npos) {
            end = cpulist.size();
        }
        const std::string range =
            cpulist.substr(start, end - start);
        int low = -1;
        int high = -1;
        if (std::sscanf(
                range.c_str(),
                "%d-%d",
                &low,
                &high) == 1) {
            high = low;
        }
        for (int cpu = low;
             low >= 0 && cpu <= high &&
             cpu < CPU_SETSIZE;
             ++cpu) {
            CPU_SET(cpu, &node_set);
        }
        start = end + 1;
    }
    cpu_set_t inherited;
    CPU_ZERO(&inherited);
    if (sched_getaffinity(
            0,
            sizeof(inherited),
            &inherited) != 0) {
        return;
    }
    cpu_set_t target;
    CPU_ZERO(&target);
    CPU_AND(&target, &node_set, &inherited);
    if (CPU_COUNT(&target) == 0) {
        if (std::getenv("ABOLETH_DOORBELL_PLACEMENT_LOG") != nullptr) {
            std::fprintf(
                stderr,
                "[aboleth-doorbell] gpu %d shard %d numa %d: no inherited "
                "CPU on the node (%d inherited), unpinned\n",
                gpu_index, worker_shard, numa_node, CPU_COUNT(&inherited));
            std::fflush(stderr);
        }
        return;
    }
    pthread_setaffinity_np(
        pthread_self(),
        sizeof(target),
        &target);
    if (std::getenv("ABOLETH_DOORBELL_PLACEMENT_LOG") != nullptr) {
        std::string listed;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &target)) {
                listed += std::to_string(cpu) + " ";
            }
        }
        std::fprintf(
            stderr,
            "[aboleth-doorbell] gpu %d shard %d numa %d inherited %d cpus, "
            "pinned to: %s\n",
            gpu_index, worker_shard, numa_node, CPU_COUNT(&inherited),
            listed.c_str());
        std::fflush(stderr);
    }
}

} // namespace aboleth::runtime
