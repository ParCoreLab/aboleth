// Bringing the shared runtime context up, and the C surface that owns it.
//
// One SharedRuntimeContext per opened device carries everything endpoints,
// registered buffers and client integrations share: the ibv_context, the
// protection domain, the port address, the host-mapping registry. Creating it
// is the only step that decides which provider is in play, so the proxy
// provider is resolved here before any queue exists.
//
// The runtime handle's lifetime and accessors are the outermost layer of the
// C ABI, so they live with it — including the thread_local behind
// aboleth_runtime_last_error(), which every other translation unit writes to
// and exactly one has to define.

#include "aboleth_runtime.h"

#include "../aboleth_runtime_common.h"
#include "../aboleth_runtime_internal.h"
#include "../efa/aboleth_efa_internal.h"
#include "../proxy/aboleth_proxy.h"
#include "aboleth_connection.h"
#include "aboleth_device_select.h"

#include "aboleth/runtime_c.h"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include "../dl/shims.h"

namespace dl = aboleth::dl;

#include <exception>
#include <memory>
#include <string>

// aboleth::runtime is where the shared internals live (see
// aboleth_runtime_internal.h). The extern "C" arm below reaches them
// unqualified, as the runtime translation unit this code came from does.
using namespace aboleth::runtime;

namespace aboleth::runtime {

thread_local std::string last_error;

std::shared_ptr<SharedRuntimeContext>
create_shared_runtime(
    int gpu_index,
    const char* requested_name) {
    auto runtime =
        std::make_shared<SharedRuntimeContext>();
    runtime->gpu_index = gpu_index;
    runtime->proxy_provider =
        configured_proxy_provider();
#if ABOLETH_HAS_EFA
    if (runtime->proxy_provider ==
        ProxyProviderKind::EfaOfi) {
        runtime->efa =
            create_efa_fabric_context(
                gpu_index,
                requested_name);
        runtime->device_name =
            runtime->efa->device_name;
        runtime->path_mtu = IBV_MTU_4096;
        return runtime;
    }
#endif
    int device_count = 0;
    runtime->devices =
        dl::verbs().ibv_get_device_list(&device_count);
    if (runtime->devices == nullptr) {
        throw_error("ibv_get_device_list failed");
    }

    ibv_device* selected =
        select_runtime_device(
            runtime->devices,
            device_count,
            gpu_index,
            requested_name,
            runtime->device_name);

    runtime->verbs_context =
        dl::verbs().ibv_open_device(selected);
    if (runtime->verbs_context == nullptr) {
        throw_error("ibv_open_device failed");
    }
    ibv_device_attr device_attributes{};
    verbs_check(
        dl::verbs().ibv_query_device(
            runtime->verbs_context,
            &device_attributes),
        "ibv_query_device");
    if (device_attributes.atomic_cap ==
        IBV_ATOMIC_NONE) {
        throw_error(
            "selected RDMA device does not support 64-bit atomics");
    }
    runtime->port =
        query_port_address(
            runtime->verbs_context);
    runtime->path_mtu =
        select_path_mtu(
            runtime->port.active_mtu);

    runtime->pd =
        dl::verbs().ibv_alloc_pd(runtime->verbs_context);
    if (runtime->pd == nullptr) {
        throw_error("ibv_alloc_pd failed");
    }
    return runtime;
}

}  // namespace aboleth::runtime

extern "C" {

void* aboleth_runtime_create() {
    last_error.clear();
    try {
        auto handle =
            std::make_unique<RuntimeHandle>();
        handle->runtime =
            create_shared_runtime();
        return handle.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

void* aboleth_runtime_create_for_device(
    int gpu_index,
    const char* hca_name) {
    last_error.clear();
    try {
        if (gpu_index < 0) {
            throw_error(
                "runtime GPU index must be non-negative");
        }
        auto handle =
            std::make_unique<RuntimeHandle>();
        handle->runtime =
            create_shared_runtime(
                gpu_index,
                hca_name);
        return handle.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

const char* aboleth_runtime_hca_name(
    void* runtime_handle) {
    auto* handle =
        static_cast<RuntimeHandle*>(
            runtime_handle);
    if (handle == nullptr ||
        !handle->runtime) {
        return "";
    }
    return handle->runtime->
        device_name.c_str();
}

int aboleth_runtime_gpu_index(
    void* runtime_handle) {
    auto* handle =
        static_cast<RuntimeHandle*>(
            runtime_handle);
    if (handle == nullptr ||
        !handle->runtime) {
        return -1;
    }
    return handle->runtime->gpu_index;
}

int aboleth_runtime_port(
    void* runtime_handle) {
    auto* handle =
        static_cast<RuntimeHandle*>(
            runtime_handle);
    if (handle == nullptr ||
        !handle->runtime) {
        return -1;
    }
    return handle->runtime->port.port_num;
}

int aboleth_runtime_gid_index(
    void* runtime_handle) {
    auto* handle =
        static_cast<RuntimeHandle*>(
            runtime_handle);
    if (handle == nullptr ||
        !handle->runtime) {
        return -1;
    }
    return static_cast<int>(
        handle->runtime->port.gid_index);
}

int aboleth_runtime_link_layer(
    void* runtime_handle) {
    auto* handle =
        static_cast<RuntimeHandle*>(
            runtime_handle);
    if (handle == nullptr ||
        !handle->runtime) {
        return -1;
    }
    return handle->runtime->port.link_layer;
}

int aboleth_runtime_lid(
    void* runtime_handle) {
    auto* handle =
        static_cast<RuntimeHandle*>(
            runtime_handle);
    if (handle == nullptr ||
        !handle->runtime) {
        return -1;
    }
    return handle->runtime->port.lid;
}

int aboleth_runtime_destroy(
    void* runtime_handle) {
    last_error.clear();
    try {
        delete static_cast<RuntimeHandle*>(
            runtime_handle);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

const char* aboleth_runtime_last_error() {
    return last_error.c_str();
}

}  // extern "C"
