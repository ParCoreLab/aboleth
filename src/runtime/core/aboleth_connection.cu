// Turning endpoints into connections, and the C surface that owns them.
//
// The queue-pair state machine is the same whoever owns the endpoints, so the
// INIT/RTR/RTS transitions, the packet sequence numbers and the path-MTU
// negotiation live here rather than in any one endpoint module. Above them sit
// the two handles the C ABI hands out: a local endpoint, which publishes a
// connection blob and consumes its peer's, and an owned pair, which is both
// halves of a path this process cross-connects to itself.
//
// Which transport a handle carries is a field and not a type - direct RC,
// proxied, or Transport.EFA - so the file dispatches on the backend and leaves
// the transport-specific halves where they belong: proxy/ for the proxied
// datapath, efa/ for the libfabric endpoints, core/aboleth_endpoint_rc.cu for
// the direct one.
//
// The internal declarations are in aboleth_connection.h.

#include "aboleth_connection.h"

#include "../aboleth_runtime_common.h"
#include "../aboleth_runtime_internal.h"
#include "../efa/aboleth_efa_internal.h"
#include "../proxy/aboleth_proxy.h"
#include "aboleth_endpoint_rc.h"
#include "aboleth_runtime.h"

#include "aboleth/Target/EfaAbi.h"
#include "aboleth/Target/ProxyAbi.h"
#include "aboleth/Target/RdmaAbi.h"
#include "aboleth/runtime_c.h"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#if ABOLETH_HAS_EFA
// The Transport.EFA and EFA-proxy arms are the only libfabric consumers here;
// everything else is verbs.
#include <rdma/fabric.h>
#endif

#include "../dl/shims.h"

namespace dl = aboleth::dl;

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <unistd.h>

// aboleth::runtime is where the shared internals live (see
// aboleth_runtime_internal.h). The anonymous-namespace halves of the endpoint
// handles and the extern "C" arm below reach them unqualified, as the runtime
// translation unit this code came from does.
using namespace aboleth::runtime;

namespace aboleth::runtime {

std::uint32_t allocate_psn() {
    static std::atomic<std::uint32_t> next{
        static_cast<std::uint32_t>(
            std::chrono::steady_clock::now()
                .time_since_epoch()
                .count()) ^
        static_cast<std::uint32_t>(getpid())};
    std::uint32_t value =
        next.fetch_add(
            0x101U,
            std::memory_order_relaxed) &
        0x00ffffffU;
    return value;
}

std::uint32_t log_rd_atomic_depth() {
    const char* configured =
        std::getenv("ABOLETH_RD_ATOMIC_DEPTH");
    if (configured == nullptr ||
        configured[0] == '\0') {
        return 0;
    }
    const long depth =
        std::strtol(configured, nullptr, 10);
    for (std::uint32_t log = 0; log <= 4; ++log) {
        if (depth == (1L << log)) {
            return log;
        }
    }
    throw_error(
        "ABOLETH_RD_ATOMIC_DEPTH must be 1, 2, 4, 8, or 16");
}

ibv_mtu select_path_mtu(ibv_mtu active_mtu) {
    const char* configured =
        std::getenv("ABOLETH_PATH_MTU");
    if (configured == nullptr ||
        configured[0] == '\0') {
        return active_mtu;
    }
    const long bytes =
        std::strtol(configured, nullptr, 10);
    ibv_mtu selected{};
    switch (bytes) {
        case 256:
            selected = IBV_MTU_256;
            break;
        case 512:
            selected = IBV_MTU_512;
            break;
        case 1024:
            selected = IBV_MTU_1024;
            break;
        case 2048:
            selected = IBV_MTU_2048;
            break;
        case 4096:
            selected = IBV_MTU_4096;
            break;
        default:
            throw_error(
                "ABOLETH_PATH_MTU must be 256, 512, 1024, 2048, or 4096");
    }
    if (selected > active_mtu) {
        throw_error(
            "ABOLETH_PATH_MTU exceeds the active port MTU");
    }
    return selected;
}

void qp_to_init(
    ibv_qp* qp,
    const PortAddress& local) {
    ibv_qp_attr attributes{};
    attributes.qp_state = IBV_QPS_INIT;
    attributes.port_num = local.port_num;
    attributes.pkey_index = 0;
    attributes.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_ATOMIC;
    const int mask =
        IBV_QP_STATE |
        IBV_QP_PKEY_INDEX |
        IBV_QP_PORT |
        IBV_QP_ACCESS_FLAGS;
    verbs_check(
        dl::verbs().ibv_modify_qp(qp, &attributes, mask),
        "ibv_modify_qp(INIT)");
}

void qp_to_rtr(
    ibv_qp* qp,
    std::uint32_t remote_qpn,
    std::uint32_t remote_psn,
    const PortAddress& local,
    const PortAddress& remote,
    ibv_mtu path_mtu) {
    ibv_qp_attr attributes{};
    attributes.qp_state = IBV_QPS_RTR;
    attributes.path_mtu = path_mtu;
    attributes.dest_qp_num = remote_qpn;
    attributes.rq_psn = remote_psn;
    attributes.max_dest_rd_atomic =
        static_cast<std::uint8_t>(1u << log_rd_atomic_depth());
    attributes.min_rnr_timer = 12;
    attributes.ah_attr.port_num = local.port_num;
    if (local.link_layer ==
            IBV_LINK_LAYER_INFINIBAND &&
        remote.link_layer ==
            IBV_LINK_LAYER_INFINIBAND) {
        attributes.ah_attr.is_global = 0;
        attributes.ah_attr.dlid = remote.lid;
        attributes.ah_attr.sl = 0;
        attributes.ah_attr.src_path_bits = 0;
    } else {
        attributes.ah_attr.is_global = 1;
        attributes.ah_attr.grh.dgid = remote.gid;
        attributes.ah_attr.grh.sgid_index =
            local.gid_index;
        attributes.ah_attr.grh.hop_limit = 64;
    }
    const int mask =
        IBV_QP_STATE |
        IBV_QP_AV |
        IBV_QP_PATH_MTU |
        IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN |
        IBV_QP_MAX_DEST_RD_ATOMIC |
        IBV_QP_MIN_RNR_TIMER;
    verbs_check(
        dl::verbs().ibv_modify_qp(qp, &attributes, mask),
        "ibv_modify_qp(RTR)");
}

void qp_to_rts(
    ibv_qp* qp,
    std::uint32_t psn) {
    ibv_qp_attr attributes{};
    attributes.qp_state = IBV_QPS_RTS;
    attributes.timeout = 14;
    attributes.retry_cnt = 7;
    attributes.rnr_retry = 7;
    attributes.sq_psn = psn;
    attributes.max_rd_atomic =
        static_cast<std::uint8_t>(1u << log_rd_atomic_depth());
    const int mask =
        IBV_QP_STATE |
        IBV_QP_TIMEOUT |
        IBV_QP_RETRY_CNT |
        IBV_QP_RNR_RETRY |
        IBV_QP_SQ_PSN |
        IBV_QP_MAX_QP_RD_ATOMIC;
    verbs_check(
        dl::verbs().ibv_modify_qp(qp, &attributes, mask),
        "ibv_modify_qp(RTS)");
}

std::uint32_t owned_qpn(
    const OwnedEndpointContext& endpoint) {
    return endpoint.devx != nullptr
        ? endpoint.devx->qpn
        : endpoint.sender->qp_num;
}

// RTR/RTS over either queue implementation: the classic ibv_qp, or the
// DEVX object a GPU-resident staged endpoint carries instead.
void owned_to_rtr(
    OwnedEndpointContext& endpoint,
    std::uint32_t remote_qpn,
    std::uint32_t remote_psn,
    const PortAddress& local,
    const PortAddress& remote,
    ibv_mtu path_mtu) {
    if (endpoint.devx != nullptr) {
        devx_qp_to_rtr(
            *endpoint.devx,
            endpoint.runtime->verbs_context,
            remote_qpn,
            remote_psn,
            remote.lid,
            path_mtu);
        return;
    }
    qp_to_rtr(
        endpoint.sender, remote_qpn, remote_psn, local, remote, path_mtu);
}

void owned_to_rts(
    OwnedEndpointContext& endpoint,
    std::uint32_t psn) {
    if (endpoint.devx != nullptr) {
        devx_qp_to_rts(*endpoint.devx, psn);
    } else {
        qp_to_rts(endpoint.sender, psn);
    }
    dump_queue_pair_context(
        endpoint.devx.get(), endpoint.sender, endpoint.gpu_index);
}

void connect_owned_endpoints(
    OwnedEndpointContext* first,
    OwnedEndpointContext* second) {
    const ibv_mtu path_mtu =
        std::min(
            first->runtime->path_mtu,
            second->runtime->path_mtu);
    owned_to_rtr(
        *first,
        owned_qpn(*second),
        second->psn,
        first->runtime->port,
        second->runtime->port,
        path_mtu);
    owned_to_rtr(
        *second,
        owned_qpn(*first),
        first->psn,
        second->runtime->port,
        first->runtime->port,
        path_mtu);
    owned_to_rts(
        *first,
        first->psn);
    owned_to_rts(
        *second,
        second->psn);
}

}  // namespace aboleth::runtime

namespace {

struct LocalEndpointHandle {
    std::shared_ptr<SharedRuntimeContext> runtime;
    OwnedEndpointBackend backend{
        OwnedEndpointBackend::Direct};
    std::unique_ptr<OwnedEndpointContext> direct;
    std::unique_ptr<ProxyEndpointContext> proxy;
#if ABOLETH_HAS_EFA
    std::unique_ptr<EfaAbiEndpointState> efa;
#endif
    bool connected{};
};

std::uint32_t endpoint_transport(
    const LocalEndpointHandle& endpoint) {
    switch (endpoint.backend) {
    case OwnedEndpointBackend::Direct:
        return kAbolethTransportDirect;
    case OwnedEndpointBackend::Efa:
        return kAbolethTransportEfa;
    default:
        return kAbolethTransportProxy;
    }
}

ibv_qp* endpoint_qp(
    LocalEndpointHandle& endpoint) {
    return endpoint.backend ==
            OwnedEndpointBackend::Direct
        ? endpoint.direct->sender
        : endpoint.proxy->sender;
}

std::uint32_t endpoint_psn(
    const LocalEndpointHandle& endpoint) {
    return endpoint.backend ==
            OwnedEndpointBackend::Direct
        ? endpoint.direct->psn
        : endpoint.proxy->psn;
}

AbolethConnectionInfo connection_info(
    const LocalEndpointHandle& endpoint) {
    AbolethConnectionInfo info{};
    info.abi_version =
        kAbolethConnectionInfoVersion;
    info.transport =
        endpoint_transport(endpoint);
#if ABOLETH_HAS_EFA
    if (endpoint.backend == OwnedEndpointBackend::Efa) {
        // Transport.EFA rides the same FI_ADDR_EFA address exchange the
        // EFA proxy uses.
        info.provider = kAbolethProxyProviderEfaOfi;
        info.address_format = FI_ADDR_EFA;
        info.address_length =
            endpoint.efa->context->address_length;
        std::memcpy(
            info.address,
            endpoint.efa->context->address.data(),
            info.address_length);
        return info;
    }
#endif
    info.provider =
        endpoint.backend ==
                OwnedEndpointBackend::Direct ||
            endpoint.proxy->provider ==
                ProxyProviderKind::VerbsRc
        ? kAbolethProxyProviderVerbsRc
        : kAbolethProxyProviderEfaOfi;
#if ABOLETH_HAS_EFA
    if (endpoint.backend ==
            OwnedEndpointBackend::Proxy &&
        endpoint.proxy->provider ==
            ProxyProviderKind::EfaOfi) {
        info.address_format = FI_ADDR_EFA;
        info.address_length =
            endpoint.proxy->
                efa_address_length;
        std::memcpy(
            info.address,
            endpoint.proxy->
                efa_address.data(),
            info.address_length);
        return info;
    }
#endif
    info.qpn = endpoint.backend ==
            OwnedEndpointBackend::Direct
        ? owned_qpn(*endpoint.direct)
        : endpoint.proxy->sender->qp_num;
    info.psn = endpoint_psn(endpoint);
    info.gid_index =
        endpoint.runtime->port.gid_index;
    info.path_mtu =
        static_cast<std::uint32_t>(
            endpoint.runtime->path_mtu);
    info.lid = endpoint.runtime->port.lid;
    info.port_num =
        endpoint.runtime->port.port_num;
    info.link_layer =
        endpoint.runtime->port.link_layer;
    std::memcpy(
        info.gid,
        endpoint.runtime->port.gid.raw,
        sizeof(info.gid));
    return info;
}

// The deferred-control surface, checked.
//
// AbolethEndpointConfig carries three kinds of field: what this runtime acts
// on (the queue depth and inline request an mlx5 endpoint is built with), what
// it can only agree with (the HCA, port and GID entry, all fixed when the
// runtime opened the device), and what it does not implement yet. The first is
// threaded into create_owned_endpoint; the other two are checked here, so a
// caller learns that a value did not take effect instead of assuming it did.
// docs/transport-control.md carries the reasoning and the deferrals.

[[noreturn]] void refuse_config(
    const char* field,
    const std::string& value,
    const std::string& reason) {
    throw_error(
        "AbolethEndpointConfig." + std::string(field) +
        "=" + value + " " + reason +
        " (see docs/transport-control.md)");
}

// The caller's struct read through its own struct_size: fields their header
// does not reach keep this runtime's defaults, which is what lets an older
// caller link against a newer runtime. A struct_size larger than this
// runtime's is the reverse case, and cannot be guessed at.
AbolethEndpointConfig checked_config(
    const AbolethEndpointConfig* config) {
    AbolethEndpointConfig value{};
    value.struct_size =
        sizeof(AbolethEndpointConfig);
    value.gid_index = -1;
    if (config == nullptr) {
        return value;
    }
    const std::size_t declared = config->struct_size;
    const std::size_t smallest =
        offsetof(AbolethEndpointConfig, qp_type);
    if (declared < smallest ||
        declared %
                alignof(AbolethEndpointConfig) !=
            0) {
        refuse_config(
            "struct_size",
            std::to_string(declared),
            "is not a whole prefix of the struct: set it to "
            "sizeof(AbolethEndpointConfig), which is " +
                std::to_string(
                    sizeof(AbolethEndpointConfig)) +
                " bytes here");
    }
    if (declared > sizeof(AbolethEndpointConfig)) {
        refuse_config(
            "struct_size",
            std::to_string(declared),
            "exceeds this runtime's " +
                std::to_string(
                    sizeof(AbolethEndpointConfig)) +
                " bytes: the caller's header is newer than the "
                "runtime it is calling");
    }
    // Defaults first, then the caller's bytes over the prefix they declared.
    std::memcpy(&value, config, declared);
    value.struct_size =
        sizeof(AbolethEndpointConfig);
    value.hca[sizeof(value.hca) - 1] = '\0';
    return value;
}

// Whether this endpoint's path is an ibverbs one, which is what makes the
// port and GID fields answerable at all.
bool verbs_addressed(
    const SharedRuntimeContext& runtime,
    std::uint32_t transport) {
    if (transport == kAbolethTransportEfa) {
        return false;
    }
    return transport == kAbolethTransportDirect ||
        runtime.proxy_provider ==
            ProxyProviderKind::VerbsRc;
}

// Everything the config asks for that this endpoint cannot become, refused by
// name. What survives is the two requests create_owned_endpoint honors.
OwnedEndpointRequest validated_request(
    const AbolethEndpointConfig& config,
    const SharedRuntimeContext& runtime,
    std::uint32_t transport) {
    if (config.qp_type != 0) {
        refuse_config(
            "qp_type",
            std::to_string(config.qp_type),
            "is not implemented: 0 (reliable connected) is the only "
            "queue-pair type this runtime creates");
    }
    if (config.qps_per_peer > 1) {
        refuse_config(
            "qps_per_peer",
            std::to_string(config.qps_per_peer),
            "is not implemented: one endpoint is one queue pair, and a "
            "per-peer mesh is built by the client integration rather than by "
            "this entry point");
    }
    if (config.sm_map_policy != 0) {
        refuse_config(
            "sm_map_policy",
            std::to_string(config.sm_map_policy),
            "is not implemented: there is no SM-to-queue-pair mapping to "
            "choose while an endpoint carries one queue pair");
    }
    if (config.hca[0] != '\0' &&
        runtime.device_name != config.hca) {
        refuse_config(
            "hca",
            std::string(config.hca),
            "is not the device this runtime opened (" +
                runtime.device_name +
                "): an endpoint hangs off the runtime's ibv_context, so "
                "name the HCA to aboleth_runtime_create_for_device() or "
                "ABOLETH_HCA instead");
    }
    if (config.port != 0) {
        if (!verbs_addressed(runtime, transport)) {
            refuse_config(
                "port",
                std::to_string(config.port),
                "has no meaning for this transport: only an ibverbs path "
                "has a port");
        }
        if (config.port !=
            static_cast<std::int32_t>(
                runtime.port.port_num)) {
            refuse_config(
                "port",
                std::to_string(config.port),
                "is not the port this runtime opened (" +
                    std::to_string(
                        runtime.port.port_num) +
                    "): the port is fixed when the device is opened, so "
                    "set ABOLETH_PORT before creating the runtime");
        }
    }
    if (config.gid_index != -1) {
        if (!verbs_addressed(runtime, transport)) {
            refuse_config(
                "gid_index",
                std::to_string(config.gid_index),
                "has no meaning for this transport: only an ibverbs path "
                "has a GID table");
        }
        if (config.gid_index !=
            static_cast<std::int32_t>(
                runtime.port.gid_index)) {
            refuse_config(
                "gid_index",
                std::to_string(config.gid_index),
                "is not the GID entry this runtime resolved (" +
                    std::to_string(
                        runtime.port.gid_index) +
                    "): the entry is fixed when the device is opened, so "
                    "set ABOLETH_GID_INDEX before creating the runtime");
        }
    }
    if (transport != kAbolethTransportDirect) {
        // The proxy's device-visible ring is kProxySlotCount slots of ABI and
        // its inline capacity is the proxy's own; Transport.EFA sizes its
        // queues from the libfabric endpoint. Neither can act on these.
        if (config.qp_depth != 0) {
            refuse_config(
                "qp_depth",
                std::to_string(config.qp_depth),
                "applies to the direct (mlx5) transport only: this "
                "endpoint's queue depth is fixed by its ABI");
        }
        if (config.max_inline != 0) {
            refuse_config(
                "max_inline",
                std::to_string(config.max_inline),
                "applies to the direct (mlx5) transport only: this "
                "endpoint's inline capacity is fixed by its ABI");
        }
    }
    OwnedEndpointRequest request{};
    request.send_wqe_requests = config.qp_depth;
    request.max_inline_bytes = config.max_inline;
    return request;
}

ibv_mtu checked_mtu(std::uint32_t value) {
    switch (value) {
        case IBV_MTU_256:
            return IBV_MTU_256;
        case IBV_MTU_512:
            return IBV_MTU_512;
        case IBV_MTU_1024:
            return IBV_MTU_1024;
        case IBV_MTU_2048:
            return IBV_MTU_2048;
        case IBV_MTU_4096:
            return IBV_MTU_4096;
        default:
            throw_error(
                "remote endpoint has an invalid path MTU");
    }
}

}  // namespace

namespace aboleth::runtime {

PortAddress port_address(
    const AbolethConnectionInfo& info) {
    PortAddress port{};
    port.port_num = info.port_num;
    port.link_layer = info.link_layer;
    port.lid = info.lid;
    port.gid_index = info.gid_index;
    std::memcpy(
        port.gid.raw,
        info.gid,
        sizeof(info.gid));
    port.active_mtu =
        checked_mtu(info.path_mtu);
    return port;
}

}  // namespace aboleth::runtime

namespace {

void connect_local_endpoint(
    LocalEndpointHandle& endpoint,
    const AbolethConnectionInfo& remote) {
    if (endpoint.connected) {
        throw_error(
            "local endpoint is already connected");
    }
    if (remote.abi_version !=
        kAbolethConnectionInfoVersion) {
        throw_error(
            "remote endpoint connection-info version mismatch");
    }
    if (remote.transport !=
        endpoint_transport(endpoint)) {
        throw_error(
            "remote endpoint transport does not match");
    }
#if ABOLETH_HAS_EFA
    if (endpoint.backend == OwnedEndpointBackend::Efa) {
        if (remote.provider !=
                kAbolethProxyProviderEfaOfi ||
            remote.address_format != FI_ADDR_EFA ||
            remote.address_length == 0 ||
            remote.address_length >
                kAbolethConnectionAddressBytes) {
            throw_error(
                "remote EFA connection info is incomplete");
        }
        connect_efa_abi_endpoint(
            *endpoint.efa,
            remote.address,
            remote.address_length);
        endpoint.connected = true;
        return;
    }
    if (endpoint.backend ==
            OwnedEndpointBackend::Proxy &&
        endpoint.proxy->provider ==
            ProxyProviderKind::EfaOfi) {
        if (remote.provider !=
                kAbolethProxyProviderEfaOfi ||
            remote.address_format !=
                FI_ADDR_EFA ||
            remote.address_length == 0 ||
            remote.address_length >
                kAbolethConnectionAddressBytes) {
            throw_error(
                "remote EFA proxy connection info is incomplete");
        }
        connect_efa_proxy_endpoint(
            endpoint.proxy.get(),
            remote.address,
            remote.address_length);
        endpoint.connected = true;
        return;
    }
#endif
    if (remote.provider !=
        kAbolethProxyProviderVerbsRc) {
        throw_error(
            "remote proxy provider does not match");
    }
    if (remote.qpn == 0 ||
        remote.port_num == 0) {
        throw_error(
            "remote endpoint connection info is incomplete");
    }
    const PortAddress remote_port =
        port_address(remote);
    if (remote_port.link_layer !=
        endpoint.runtime->port.link_layer) {
        throw_error(
            "local and remote RDMA link layers do not match");
    }
    const ibv_mtu path_mtu =
        std::min(
            endpoint.runtime->path_mtu,
            remote_port.active_mtu);
    if (endpoint.backend == OwnedEndpointBackend::Direct) {
        owned_to_rtr(
            *endpoint.direct,
            remote.qpn,
            remote.psn,
            endpoint.runtime->port,
            remote_port,
            path_mtu);
        owned_to_rts(
            *endpoint.direct,
            endpoint_psn(endpoint));
    } else {
        qp_to_rtr(
            endpoint_qp(endpoint),
            remote.qpn,
            remote.psn,
            endpoint.runtime->port,
            remote_port,
            path_mtu);
        qp_to_rts(
            endpoint_qp(endpoint),
            endpoint_psn(endpoint));
    }
    endpoint.connected = true;
}

struct OwnedEndpointPairContext {
    std::shared_ptr<SharedRuntimeContext> runtime;
    OwnedEndpointBackend backend{
        OwnedEndpointBackend::Direct};
    std::unique_ptr<OwnedEndpointContext> forward;
    std::unique_ptr<OwnedEndpointContext> reverse;
    std::unique_ptr<ProxyEndpointContext>
        proxy_forward;
    std::unique_ptr<ProxyEndpointContext>
        proxy_reverse;
#if ABOLETH_HAS_EFA
    std::unique_ptr<EfaAbiEndpointState> efa_forward;
    std::unique_ptr<EfaAbiEndpointState> efa_reverse;
#endif
};

}  // namespace

extern "C" {

#if !ABOLETH_HAS_EFA

// The EFA arm of this entry point moved to efa/aboleth_efa_endpoints.cu with
// the endpoints it drives; only the compiled-out reply stays here.
int aboleth_efa_gda_local_smoke(
    int left_gpu,
    int right_gpu,
    std::size_t bytes) {
    (void)left_gpu;
    (void)right_gpu;
    (void)bytes;
    last_error = kEfaUnavailable;
    return -1;
}

#endif  // !ABOLETH_HAS_EFA

void* aboleth_runtime_create_local_endpoint(
    void* runtime_handle,
    std::uint32_t transport,
    int gpu_index,
    void* descriptor_device,
    std::size_t descriptor_bytes,
    const AbolethEndpointConfig* config) {
    last_error.clear();
    try {
        // The config is the caller's own bytes, so its versioning is settled
        // before anything is opened or dereferenced.
        const AbolethEndpointConfig requested =
            checked_config(config);
        auto* runtime =
            static_cast<RuntimeHandle*>(
                runtime_handle);
        if (runtime == nullptr ||
            !runtime->runtime ||
            descriptor_device == nullptr) {
            throw_error(
                "null local-endpoint argument");
        }
        const OwnedEndpointRequest request =
            validated_request(
                requested,
                *runtime->runtime,
                transport);
        if (runtime->runtime->gpu_index >= 0 &&
            runtime->runtime->gpu_index !=
                gpu_index) {
            throw_error(
                "endpoint GPU does not match the rank-local runtime");
        }

        auto endpoint =
            std::make_unique<
                LocalEndpointHandle>();
        endpoint->runtime = runtime->runtime;
        if (transport ==
            kAbolethTransportDirect) {
            if (descriptor_bytes <
                sizeof(
                    aboleth::abi::
                        RdmaEndpointDescriptor)) {
                throw_error(
                    "direct endpoint descriptor buffer is too small");
            }
            endpoint->backend =
                OwnedEndpointBackend::Direct;
            aboleth::abi::RdmaEndpointDescriptor
                descriptor{};
            endpoint->direct =
                create_owned_endpoint(
                    runtime->runtime,
                    gpu_index,
                    descriptor,
                    /*dedicated_thread_domain=*/false,
                    request);
            cuda_check(
                cudaSetDevice(gpu_index),
                "cudaSetDevice(local direct endpoint)");
            cuda_check(
                cudaMemcpy(
                    descriptor_device,
                    &descriptor,
                    sizeof(descriptor),
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(local direct descriptor)");
        } else if (transport ==
                   kAbolethTransportProxy) {
            if (descriptor_bytes <
                sizeof(
                    aboleth::abi::
                        ProxyEndpointDescriptor)) {
                throw_error(
                    "proxy endpoint descriptor buffer is too small");
            }
            endpoint->backend =
                OwnedEndpointBackend::Proxy;
            aboleth::abi::ProxyEndpointDescriptor
                descriptor{};
            endpoint->proxy =
                create_proxy_endpoint(
                    runtime->runtime,
                    gpu_index,
                    descriptor);
            cuda_check(
                cudaSetDevice(gpu_index),
                "cudaSetDevice(local proxy endpoint)");
            cuda_check(
                cudaMemcpy(
                    descriptor_device,
                    &descriptor,
                    sizeof(descriptor),
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(local proxy descriptor)");
        } else if (transport == kAbolethTransportEfa) {
#if ABOLETH_HAS_EFA
            endpoint->backend =
                OwnedEndpointBackend::Efa;
            endpoint->efa = create_efa_abi_endpoint(
                runtime->runtime,
                gpu_index,
                descriptor_device,
                descriptor_bytes);
#else
            throw_error(kEfaUnavailable);
#endif
        } else {
            throw_error(
                "unsupported local endpoint transport");
        }
        return endpoint.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

int aboleth_runtime_local_endpoint_connection_info(
    void* endpoint_handle,
    AbolethConnectionInfo* info) {
    last_error.clear();
    try {
        auto* endpoint =
            static_cast<LocalEndpointHandle*>(
                endpoint_handle);
        if (endpoint == nullptr ||
            info == nullptr) {
            throw_error(
                "null local-endpoint connection-info argument");
        }
        *info = connection_info(*endpoint);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_connect_local_endpoint(
    void* endpoint_handle,
    const AbolethConnectionInfo* remote_info) {
    last_error.clear();
    try {
        auto* endpoint =
            static_cast<LocalEndpointHandle*>(
                endpoint_handle);
        if (endpoint == nullptr ||
            remote_info == nullptr) {
            throw_error(
                "null local-endpoint connect argument");
        }
        connect_local_endpoint(
            *endpoint,
            *remote_info);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_local_endpoint_state(
    void* endpoint_handle,
    std::uint64_t* producer,
    std::uint64_t* consumer,
    std::uint64_t* status) {
    last_error.clear();
    try {
        auto* endpoint =
            static_cast<LocalEndpointHandle*>(
                endpoint_handle);
        if (endpoint == nullptr ||
            producer == nullptr ||
            consumer == nullptr ||
            status == nullptr) {
            throw_error(
                "null local-endpoint state argument");
        }
#if ABOLETH_HAS_EFA
        if (endpoint->backend ==
            OwnedEndpointBackend::Efa) {
            cuda_check(
                cudaSetDevice(
                    endpoint->efa->gpu_index),
                "cudaSetDevice(local EFA state)");
            std::array<
                std::uint64_t,
                aboleth::abi::kEfaCompletionStateWords>
                completion{};
            cuda_check(
                cudaMemcpy(
                    completion.data(),
                    endpoint->efa->completion_state,
                    sizeof(completion),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(local EFA state)");
            *producer =
                completion[aboleth::abi::kEfaStateProducer];
            *consumer =
                completion[aboleth::abi::kEfaStateRetired];
            *status =
                completion[aboleth::abi::kEfaStateError];
            return 0;
        }
#endif
        if (endpoint->backend ==
            OwnedEndpointBackend::Direct) {
            cuda_check(
                cudaSetDevice(
                    endpoint->direct->gpu_index),
                "cudaSetDevice(local direct state)");
            std::array<
                std::uint64_t,
                kCompletionStateWords>
                completion{};
            cuda_check(
                cudaMemcpy(
                    producer,
                    endpoint->direct->
                        producer_index,
                    sizeof(*producer),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(local direct producer)");
            cuda_check(
                cudaMemcpy(
                    completion.data(),
                    endpoint->direct->
                        completion_consumer,
                    sizeof(completion),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(local direct completion)");
            *consumer = completion[0];
            *status = completion[3];
        } else {
            cuda_check(
                cudaSetDevice(
                    endpoint->proxy->gpu_index),
                "cudaSetDevice(local proxy state)");
            cuda_check(
                cudaMemcpy(
                    producer,
                    endpoint->proxy->
                        producer_index,
                    sizeof(*producer),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(local proxy producer)");
            *consumer = atomic_load_acquire(
                &endpoint->proxy->queue_host->
                    completed_sequence);
            *status = atomic_load_acquire(
                &endpoint->proxy->queue_host->
                    status);
        }
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_local_endpoint_capacity(
    void* endpoint_handle,
    std::uint32_t* capacity) {
    last_error.clear();
    try {
        auto* endpoint =
            static_cast<LocalEndpointHandle*>(
                endpoint_handle);
        if (endpoint == nullptr ||
            capacity == nullptr) {
            throw_error(
                "null local-endpoint capacity argument");
        }
        switch (endpoint->backend) {
        case OwnedEndpointBackend::Direct:
            *capacity =
                endpoint->direct->send_wqe_count;
            break;
#if ABOLETH_HAS_EFA
        case OwnedEndpointBackend::Efa:
            *capacity =
                endpoint->efa->descriptor.send_depth;
            break;
#endif
        default:
            *capacity = aboleth::abi::kProxySlotCount;
            break;
        }
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_local_endpoint_max_inline(
    void* endpoint_handle,
    std::uint32_t* max_inline_bytes) {
    last_error.clear();
    try {
        auto* endpoint =
            static_cast<LocalEndpointHandle*>(
                endpoint_handle);
        if (endpoint == nullptr ||
            max_inline_bytes == nullptr) {
            throw_error(
                "null local-endpoint max-inline argument");
        }
        switch (endpoint->backend) {
        case OwnedEndpointBackend::Direct:
            *max_inline_bytes =
                endpoint->direct->max_inline_bytes;
            break;
#if ABOLETH_HAS_EFA
        case OwnedEndpointBackend::Efa:
            *max_inline_bytes =
                endpoint->efa->descriptor
                    .max_inline_bytes;
            break;
#endif
        default:
            *max_inline_bytes = 0;
            break;
        }
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_local_endpoint_submit_mode(
    void* endpoint_handle,
    std::uint32_t* submit_mode) {
    last_error.clear();
    try {
        auto* endpoint =
            static_cast<LocalEndpointHandle*>(
                endpoint_handle);
        if (endpoint == nullptr ||
            submit_mode == nullptr) {
            throw_error(
                "null local-endpoint submit-mode argument");
        }
        switch (endpoint->backend) {
        case OwnedEndpointBackend::Direct:
            *submit_mode =
                endpoint->direct->staged_submit
                    ? 1u
                    : 0u;
            break;
#if ABOLETH_HAS_EFA
        case OwnedEndpointBackend::Efa:
            *submit_mode =
                endpoint->efa->staged ? 1u : 0u;
            break;
#endif
        default:
            *submit_mode = 2u;
            break;
        }
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_local_endpoint_gpu_resident_queues(
    void* endpoint_handle,
    std::uint32_t* resident) {
    last_error.clear();
    try {
        auto* endpoint =
            static_cast<LocalEndpointHandle*>(
                endpoint_handle);
        if (endpoint == nullptr ||
            resident == nullptr) {
            throw_error(
                "null local-endpoint queue-placement argument");
        }
        *resident =
            endpoint->backend == OwnedEndpointBackend::Direct &&
                    endpoint->direct->devx != nullptr
                ? 1u
                : 0u;
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_destroy_local_endpoint(
    void* endpoint_handle) {
    last_error.clear();
    try {
        delete static_cast<LocalEndpointHandle*>(
            endpoint_handle);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

void* aboleth_runtime_create_endpoint_pair(
    void* runtime_handle,
    int device0,
    void* descriptor0_device,
    std::size_t descriptor0_bytes,
    int device1,
    void* descriptor1_device,
    std::size_t descriptor1_bytes,
    void* expected0_host,
    void* expected1_host) {
    last_error.clear();
    try {
        auto* runtime =
            static_cast<RuntimeHandle*>(
                runtime_handle);
        if (runtime == nullptr ||
            descriptor0_device == nullptr ||
            descriptor1_device == nullptr) {
            throw_error("null endpoint-pair argument");
        }
        if (runtime->runtime->gpu_index >= 0) {
            throw_error(
                "rank-local runtime cannot create an endpoint pair");
        }
        if (descriptor0_bytes <
                sizeof(
                    aboleth::abi::
                        RdmaEndpointDescriptor) ||
            descriptor1_bytes <
                sizeof(
                    aboleth::abi::
                        RdmaEndpointDescriptor)) {
            throw_error(
                "endpoint descriptor buffer is too small");
        }

        auto pair =
            std::make_unique<
                OwnedEndpointPairContext>();
        pair->runtime = runtime->runtime;
        pair->backend =
            OwnedEndpointBackend::Direct;
        aboleth::abi::RdmaEndpointDescriptor
            descriptor0{};
        aboleth::abi::RdmaEndpointDescriptor
            descriptor1{};
        pair->forward = create_owned_endpoint(
            runtime->runtime,
            device0,
            descriptor0);
        cuda_check(
            cudaSetDevice(device0),
            "cudaSetDevice(endpoint 0)");
        cuda_check(
            cudaMemcpy(
                descriptor0_device,
                &descriptor0,
                sizeof(descriptor0),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(endpoint descriptor 0)");
        if (expected0_host != nullptr) {
            std::memcpy(
                expected0_host,
                &descriptor0,
                sizeof(descriptor0));
        }

        pair->reverse = create_owned_endpoint(
            runtime->runtime,
            device1,
            descriptor1);
        connect_owned_endpoints(
            pair->forward.get(),
            pair->reverse.get());
        cuda_check(
            cudaSetDevice(device1),
            "cudaSetDevice(endpoint 1)");
        cuda_check(
            cudaMemcpy(
                descriptor1_device,
                &descriptor1,
                sizeof(descriptor1),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(endpoint descriptor 1)");
        if (expected1_host != nullptr) {
            std::memcpy(
                expected1_host,
                &descriptor1,
                sizeof(descriptor1));
        }
        return pair.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

void* aboleth_runtime_create_proxy_endpoint_pair(
    void* runtime_handle,
    int device0,
    void* descriptor0_device,
    std::size_t descriptor0_bytes,
    int device1,
    void* descriptor1_device,
    std::size_t descriptor1_bytes) {
    last_error.clear();
    try {
        auto* runtime =
            static_cast<RuntimeHandle*>(
                runtime_handle);
        if (runtime == nullptr ||
            descriptor0_device == nullptr ||
            descriptor1_device == nullptr) {
            throw_error(
                "null proxy endpoint-pair argument");
        }
        if (runtime->runtime->gpu_index >= 0) {
            throw_error(
                "rank-local runtime cannot create a proxy endpoint pair");
        }
        if (descriptor0_bytes <
                sizeof(
                    aboleth::abi::
                        ProxyEndpointDescriptor) ||
            descriptor1_bytes <
                sizeof(
                    aboleth::abi::
                        ProxyEndpointDescriptor)) {
            throw_error(
                "proxy endpoint descriptor buffer is too small");
        }

        auto pair =
            std::make_unique<
                OwnedEndpointPairContext>();
        pair->runtime = runtime->runtime;
        pair->backend =
            OwnedEndpointBackend::Proxy;
        aboleth::abi::ProxyEndpointDescriptor
            descriptor0{};
        aboleth::abi::ProxyEndpointDescriptor
            descriptor1{};
        pair->proxy_forward =
            create_proxy_endpoint(
                runtime->runtime,
                device0,
                descriptor0);
        cuda_check(
            cudaSetDevice(device0),
            "cudaSetDevice(proxy endpoint 0)");
        cuda_check(
            cudaMemcpy(
                descriptor0_device,
                &descriptor0,
                sizeof(descriptor0),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(proxy endpoint descriptor 0)");

        pair->proxy_reverse =
            create_proxy_endpoint(
                runtime->runtime,
                device1,
                descriptor1);
        connect_proxy_endpoints(
            pair->proxy_forward.get(),
            pair->proxy_reverse.get());
        cuda_check(
            cudaSetDevice(device1),
            "cudaSetDevice(proxy endpoint 1)");
        cuda_check(
            cudaMemcpy(
                descriptor1_device,
                &descriptor1,
                sizeof(descriptor1),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(proxy endpoint descriptor 1)");
        return pair.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

void* aboleth_runtime_create_efa_endpoint_pair(
    void* runtime_handle,
    int device0,
    void* descriptor0_device,
    std::size_t descriptor0_bytes,
    int device1,
    void* descriptor1_device,
    std::size_t descriptor1_bytes) {
    last_error.clear();
    try {
        auto* runtime =
            static_cast<RuntimeHandle*>(runtime_handle);
        if (runtime == nullptr ||
            descriptor0_device == nullptr ||
            descriptor1_device == nullptr) {
            throw_error(
                "null EFA endpoint-pair argument");
        }
        if (runtime->runtime->gpu_index >= 0) {
            throw_error(
                "rank-local runtime cannot create an EFA endpoint pair");
        }
#if !ABOLETH_HAS_EFA
        (void)device0;
        (void)descriptor0_bytes;
        (void)device1;
        (void)descriptor1_bytes;
        throw_error(kEfaUnavailable);
#else
        auto pair = std::make_unique<
            OwnedEndpointPairContext>();
        pair->runtime = runtime->runtime;
        pair->backend = OwnedEndpointBackend::Efa;
        pair->efa_forward = create_efa_abi_endpoint(
            runtime->runtime,
            device0,
            descriptor0_device,
            descriptor0_bytes);
        pair->efa_reverse = create_efa_abi_endpoint(
            runtime->runtime,
            device1,
            descriptor1_device,
            descriptor1_bytes);
        connect_efa_abi_endpoint(
            *pair->efa_forward,
            pair->efa_reverse->context->address.data(),
            pair->efa_reverse->context->address_length);
        connect_efa_abi_endpoint(
            *pair->efa_reverse,
            pair->efa_forward->context->address.data(),
            pair->efa_forward->context->address_length);
        return pair.release();
#endif
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

int aboleth_runtime_destroy_endpoint_pair(
    void* pair_handle) {
    last_error.clear();
    try {
        delete static_cast<
            OwnedEndpointPairContext*>(
                pair_handle);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_endpoint_pair_state(
    void* pair_handle,
    std::uint64_t* forward_producer,
    std::uint64_t* forward_consumer,
    std::uint64_t* reverse_producer,
    std::uint64_t* reverse_consumer) {
    last_error.clear();
    try {
        auto* pair =
            static_cast<
                OwnedEndpointPairContext*>(
                pair_handle);
        if (pair == nullptr ||
            forward_producer == nullptr ||
            forward_consumer == nullptr ||
            reverse_producer == nullptr ||
            reverse_consumer == nullptr) {
            throw_error(
                "null endpoint-pair state argument");
        }
#if ABOLETH_HAS_EFA
        if (pair->backend == OwnedEndpointBackend::Efa) {
            std::uint64_t words[
                aboleth::abi::kEfaCompletionStateWords]{};
            cuda_check(
                cudaSetDevice(
                    pair->efa_forward->gpu_index),
                "cudaSetDevice(EFA state 0)");
            cuda_check(
                cudaMemcpy(
                    words,
                    pair->efa_forward->completion_state,
                    sizeof(words),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(EFA state 0)");
            *forward_producer =
                words[aboleth::abi::kEfaStateProducer];
            *forward_consumer =
                words[aboleth::abi::kEfaStateRetired];
            cuda_check(
                cudaSetDevice(
                    pair->efa_reverse->gpu_index),
                "cudaSetDevice(EFA state 1)");
            cuda_check(
                cudaMemcpy(
                    words,
                    pair->efa_reverse->completion_state,
                    sizeof(words),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(EFA state 1)");
            *reverse_producer =
                words[aboleth::abi::kEfaStateProducer];
            *reverse_consumer =
                words[aboleth::abi::kEfaStateRetired];
            return 0;
        }
#endif
        if (pair->backend ==
            OwnedEndpointBackend::Proxy) {
            cuda_check(
                cudaSetDevice(
                    pair->proxy_forward->
                        gpu_index),
                "cudaSetDevice(proxy state 0)");
            cuda_check(
                cudaMemcpy(
                    forward_producer,
                    pair->proxy_forward->
                        producer_index,
                    sizeof(std::uint64_t),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(proxy producer 0)");
            *forward_consumer =
                atomic_load_acquire(
                    &pair->proxy_forward->
                        queue_host->
                        completed_sequence);

            cuda_check(
                cudaSetDevice(
                    pair->proxy_reverse->
                        gpu_index),
                "cudaSetDevice(proxy state 1)");
            cuda_check(
                cudaMemcpy(
                    reverse_producer,
                    pair->proxy_reverse->
                        producer_index,
                    sizeof(std::uint64_t),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(proxy producer 1)");
            *reverse_consumer =
                atomic_load_acquire(
                    &pair->proxy_reverse->
                        queue_host->
                        completed_sequence);
            return 0;
        }

        cuda_check(
            cudaSetDevice(
                pair->forward->gpu_index),
            "cudaSetDevice(endpoint state 0)");
        cuda_check(
            cudaMemcpy(
                forward_producer,
                pair->forward->producer_index,
                sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(endpoint producer 0)");
        cuda_check(
            cudaMemcpy(
                forward_consumer,
                pair->forward->
                    completion_consumer,
                sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(endpoint consumer 0)");

        cuda_check(
            cudaSetDevice(
                pair->reverse->gpu_index),
            "cudaSetDevice(endpoint state 1)");
        cuda_check(
            cudaMemcpy(
                reverse_producer,
                pair->reverse->producer_index,
                sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(endpoint producer 1)");
        cuda_check(
            cudaMemcpy(
                reverse_consumer,
                pair->reverse->
                    completion_consumer,
                sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(endpoint consumer 1)");
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_endpoint_pair_status(
    void* pair_handle,
    std::uint64_t* forward_status,
    std::uint64_t* reverse_status) {
    last_error.clear();
    try {
        auto* pair =
            static_cast<
                OwnedEndpointPairContext*>(
                pair_handle);
        if (pair == nullptr ||
            forward_status == nullptr ||
            reverse_status == nullptr) {
            throw_error(
                "null endpoint-pair status argument");
        }
#if ABOLETH_HAS_EFA
        if (pair->backend == OwnedEndpointBackend::Efa) {
            std::uint64_t words[
                aboleth::abi::kEfaCompletionStateWords]{};
            cuda_check(
                cudaSetDevice(
                    pair->efa_forward->gpu_index),
                "cudaSetDevice(EFA status 0)");
            cuda_check(
                cudaMemcpy(
                    words,
                    pair->efa_forward->completion_state,
                    sizeof(words),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(EFA status 0)");
            *forward_status =
                words[aboleth::abi::kEfaStateError];
            cuda_check(
                cudaSetDevice(
                    pair->efa_reverse->gpu_index),
                "cudaSetDevice(EFA status 1)");
            cuda_check(
                cudaMemcpy(
                    words,
                    pair->efa_reverse->completion_state,
                    sizeof(words),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(EFA status 1)");
            *reverse_status =
                words[aboleth::abi::kEfaStateError];
            return 0;
        }
#endif
        if (pair->backend ==
            OwnedEndpointBackend::Proxy) {
            *forward_status =
                atomic_load_acquire(
                    &pair->proxy_forward->
                        queue_host->status);
            *reverse_status =
                atomic_load_acquire(
                    &pair->proxy_reverse->
                        queue_host->status);
            return 0;
        }
        cuda_check(
            cudaSetDevice(
                pair->forward->gpu_index),
            "cudaSetDevice(endpoint status 0)");
        cuda_check(
            cudaMemcpy(
                forward_status,
                pair->forward->
                    completion_consumer + 3,
                sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(endpoint status 0)");
        cuda_check(
            cudaSetDevice(
                pair->reverse->gpu_index),
            "cudaSetDevice(endpoint status 1)");
        cuda_check(
            cudaMemcpy(
                reverse_status,
                pair->reverse->
                    completion_consumer + 3,
                sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(endpoint status 1)");
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

std::uint64_t
aboleth_runtime_endpoint_pair_completion_state(
    void* pair_handle,
    int endpoint_index) {
    auto* pair =
        static_cast<OwnedEndpointPairContext*>(
            pair_handle);
    if (pair == nullptr) {
        return 0;
    }
#if ABOLETH_HAS_EFA
    if (pair->backend == OwnedEndpointBackend::Efa) {
        if (endpoint_index == 0) {
            return reinterpret_cast<std::uint64_t>(
                pair->efa_forward->completion_state);
        }
        if (endpoint_index == 1) {
            return reinterpret_cast<std::uint64_t>(
                pair->efa_reverse->completion_state);
        }
        return 0;
    }
#endif
    if (pair->backend ==
        OwnedEndpointBackend::Proxy) {
        if (endpoint_index == 0) {
            return reinterpret_cast<
                std::uint64_t>(
                pair->proxy_forward->
                    queue_device);
        }
        if (endpoint_index == 1) {
            return reinterpret_cast<
                std::uint64_t>(
                pair->proxy_reverse->
                    queue_device);
        }
        return 0;
    }
    if (endpoint_index == 0) {
        return reinterpret_cast<std::uint64_t>(
            pair->forward->completion_consumer);
    }
    if (endpoint_index == 1) {
        return reinterpret_cast<std::uint64_t>(
            pair->reverse->completion_consumer);
    }
    return 0;
}

int aboleth_runtime_endpoint_pair_completion_snapshot(
    void* pair_handle,
    int endpoint_index,
    std::uint64_t* words,
    std::size_t word_count) {
    last_error.clear();
    try {
        auto* pair =
            static_cast<OwnedEndpointPairContext*>(
                pair_handle);
        if (pair == nullptr ||
            words == nullptr ||
            word_count < kCompletionStateWords) {
            throw_error(
                "invalid completion snapshot argument");
        }
#if ABOLETH_HAS_EFA
        if (pair->backend == OwnedEndpointBackend::Efa) {
            EfaAbiEndpointState* endpoint = nullptr;
            if (endpoint_index == 0) {
                endpoint = pair->efa_forward.get();
            } else if (endpoint_index == 1) {
                endpoint = pair->efa_reverse.get();
            } else {
                throw_error("invalid EFA endpoint index");
            }
            cuda_check(
                cudaSetDevice(endpoint->gpu_index),
                "cudaSetDevice(EFA completion snapshot)");
            cuda_check(
                cudaMemcpy(
                    words,
                    endpoint->completion_state,
                    aboleth::abi::kEfaCompletionStateWords *
                        sizeof(std::uint64_t),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(EFA completion snapshot)");
            return 0;
        }
#endif
        if (pair->backend ==
            OwnedEndpointBackend::Proxy) {
            ProxyEndpointContext* endpoint =
                nullptr;
            if (endpoint_index == 0) {
                endpoint =
                    pair->proxy_forward.get();
            } else if (endpoint_index == 1) {
                endpoint =
                    pair->proxy_reverse.get();
            } else {
                throw_error(
                    "invalid proxy endpoint index");
            }
            const std::uint64_t completed =
                atomic_load_acquire(
                    &endpoint->queue_host->
                        completed_sequence);
            words[0] = completed;
            words[1] = completed;
            words[2] = completed;
            words[3] =
                atomic_load_acquire(
                    &endpoint->queue_host->
                        status);
            return 0;
        }
        OwnedEndpointContext* endpoint = nullptr;
        if (endpoint_index == 0) {
            endpoint = pair->forward.get();
        } else if (endpoint_index == 1) {
            endpoint = pair->reverse.get();
        } else {
            throw_error("invalid endpoint index");
        }
        cuda_check(
            cudaSetDevice(endpoint->gpu_index),
            "cudaSetDevice(completion snapshot)");
        cuda_check(
            cudaMemcpy(
                words,
                endpoint->completion_consumer,
                kCompletionStateWords *
                    sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(completion snapshot)");
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_endpoint_pair_max_inline(
    void* pair_handle,
    int endpoint_index,
    std::uint32_t* max_inline_bytes) {
    last_error.clear();
    try {
        auto* pair =
            static_cast<
                OwnedEndpointPairContext*>(
                pair_handle);
        if (pair == nullptr ||
            max_inline_bytes == nullptr) {
            throw_error(
                "null endpoint-pair inline-capacity argument");
        }
        if (pair->backend !=
            OwnedEndpointBackend::Direct) {
            throw_error(
                "only a direct endpoint reports a granted inline capacity");
        }
        OwnedEndpointContext* endpoint = nullptr;
        if (endpoint_index == 0) {
            endpoint = pair->forward.get();
        } else if (endpoint_index == 1) {
            endpoint = pair->reverse.get();
        } else {
            throw_error("invalid endpoint index");
        }
        *max_inline_bytes =
            endpoint->max_inline_bytes;
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int aboleth_runtime_endpoint_pair_send_queue_snapshot(
    void* pair_handle,
    int endpoint_index,
    void* output,
    std::size_t output_bytes,
    std::uint32_t* dbrec_head) {
    last_error.clear();
    try {
        auto* pair =
            static_cast<
                OwnedEndpointPairContext*>(
                pair_handle);
        if (pair == nullptr ||
            output == nullptr ||
            dbrec_head == nullptr) {
            throw_error(
                "null send-queue snapshot argument");
        }
        if (pair->backend !=
            OwnedEndpointBackend::Direct) {
            throw_error(
                "only a direct endpoint has an mlx5 send queue");
        }
        OwnedEndpointContext* endpoint = nullptr;
        if (endpoint_index == 0) {
            endpoint = pair->forward.get();
        } else if (endpoint_index == 1) {
            endpoint = pair->reverse.get();
        } else {
            throw_error("invalid endpoint index");
        }
        const std::size_t available =
            static_cast<std::size_t>(
                endpoint->send_stride) *
            endpoint->send_wqe_count;
        if (output_bytes > available) {
            throw_error(
                "snapshot exceeds send queue size");
        }
        cuda_check(
            cudaSetDevice(endpoint->gpu_index),
            "cudaSetDevice(snapshot)");
        cuda_check(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize(snapshot)");
        std::uint32_t raw_head = 0;
        if (endpoint->devx != nullptr) {
            // GPU-resident queues: the wire bytes live in device memory.
            cuda_check(
                cudaMemcpy(
                    output,
                    reinterpret_cast<const void*>(
                        endpoint->devx->send_buffer),
                    output_bytes,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(send queue snapshot)");
            cuda_check(
                cudaMemcpy(
                    &raw_head,
                    reinterpret_cast<const void*>(
                        endpoint->devx->dbr_buffer +
                        MLX5_SND_DBR * sizeof(std::uint32_t)),
                    sizeof(raw_head),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(dbrec snapshot)");
        } else {
            std::memcpy(
                output,
                endpoint->send_queue_host,
                output_bytes);
            std::memcpy(
                &raw_head,
                endpoint->send_dbrec_host,
                sizeof(raw_head));
        }
        *dbrec_head =
            __builtin_bswap32(raw_head);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

}  // extern "C"
