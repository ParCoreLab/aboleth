#pragma once

#include <cstddef>
#include <cstdint>

#include "aboleth/Target/ProxyAbi.h"
#include "aboleth/Target/RdmaAbi.h"

constexpr std::uint32_t kAbolethConnectionInfoVersion = 2;
constexpr std::uint32_t kAbolethTransportDirect = 1;
constexpr std::uint32_t kAbolethTransportProxy = 2;
// GPU-built EFA descriptors; submission mode follows ABOLETH_EFA_SUBMIT.
constexpr std::uint32_t kAbolethTransportEfa = 3;
constexpr std::uint32_t kAbolethTransportIonic = 4;
constexpr std::uint32_t kAbolethProxyProviderVerbsRc = 1;
constexpr std::uint32_t kAbolethProxyProviderEfaOfi = 2;
constexpr std::size_t kAbolethConnectionAddressBytes = 64;

// Provider capability bits reported by aboleth_runtime_provider_caps(). A bit
// is set only when the provider was compiled in, its library resolved at
// runtime, and the machine actually presents a device for it — so callers can
// pick a transport instead of discovering the gap as an exception.
constexpr std::uint32_t kAbolethProviderVerbs = 1u << 0;
constexpr std::uint32_t kAbolethProviderMlx5 = 1u << 1;
constexpr std::uint32_t kAbolethProviderEfa = 1u << 2;
constexpr std::uint32_t kAbolethProviderLibfabric = 1u << 3;
constexpr std::uint32_t kAbolethProviderCuda = 1u << 4;
constexpr std::uint32_t kAbolethProviderHip = 1u << 5;
constexpr std::uint32_t kAbolethProviderIonic = 1u << 6;

struct AbolethConnectionInfo {
    std::uint32_t abi_version;
    std::uint32_t transport;
    std::uint32_t qpn;
    std::uint32_t psn;
    std::uint32_t gid_index;
    std::uint32_t path_mtu;
    std::uint16_t lid;
    std::uint8_t port_num;
    std::uint8_t link_layer;
    std::uint8_t gid[16];
    std::uint32_t provider;
    std::uint32_t address_format;
    std::uint32_t address_length;
    std::uint32_t reserved0;
    std::uint8_t address[kAbolethConnectionAddressBytes];
    std::uint8_t reserved[4];
};
static_assert(
    sizeof(AbolethConnectionInfo) == 128,
    "Aboleth connection-info ABI must remain stable");

// What a caller may ask of one endpoint's queue pair, and what it may not.
//
// Everything here also has an environment variable behind it, because the
// environment is how a process that never sees this struct — a benchmark
// script, a framework's launcher — configures the same knobs. The struct is
// the programmatic path and wins wherever both are set; an unset field (0, or
// -1 for `gid_index`) falls back to the environment and then to the runtime's
// own default. Every field's accepted range is below; a value outside it is
// refused by name through aboleth_runtime_last_error() rather than ignored.
// The reasoning behind each refusal is in docs/transport-control.md.
struct AbolethEndpointConfig {
    // sizeof(AbolethEndpointConfig) as the caller's header declares it. A
    // smaller value is accepted — the fields past it take their defaults, so
    // an older caller keeps working against a newer runtime. A larger one is
    // refused: the runtime cannot know what the extra bytes mean.
    std::uint32_t struct_size;
    // 0 (reliable connected). No other queue-pair type exists yet; DC/DCT is
    // where the value would grow.
    std::uint32_t qp_type;
    // 0 for the runtime default, else the send-queue work-request count
    // requested of the adapter (ibv_qp_cap.max_send_wr). The provider rounds
    // its ring up to a power of two of 64-byte blocks, so the granted queue is
    // at least this deep: aboleth_runtime_local_endpoint_capacity() reports
    // what it became. Direct (mlx5) endpoints only.
    std::uint32_t qp_depth;
    // 0 for ABOLETH_RDMA_INLINE_BYTES, else the inline payload bytes requested
    // of the adapter. The adapter may grant fewer or refuse inline data
    // altogether, in which case the endpoint falls back to no inline capacity;
    // aboleth_runtime_local_endpoint_max_inline() reports what was granted.
    // Direct (mlx5) endpoints only.
    std::uint32_t max_inline;
    // 0 or 1. A per-peer queue-pair mesh is a runtime-level structure, not an
    // endpoint-level one; see docs/transport-control.md.
    std::uint32_t qps_per_peer;
    // 0. There is no SM-to-queue-pair mapping to choose while there is one
    // queue pair per endpoint.
    std::uint32_t sm_map_policy;
    // 0 for the port the runtime opened (ABOLETH_PORT), else that same port
    // stated explicitly. A different port is refused: the port is fixed when
    // the runtime opens the device, not when an endpoint is created.
    std::int32_t port;
    // -1 for the GID entry the runtime resolved (ABOLETH_GID_INDEX), else that
    // same index stated explicitly. A different index is refused, for the same
    // reason as `port`.
    std::int32_t gid_index;
    // Empty for the HCA the runtime opened (ABOLETH_HCA, or the topology
    // default), else that same device name, NUL-terminated. A different device
    // is refused: an endpoint hangs off the runtime's already-open
    // ibv_context, so pass the name to aboleth_runtime_create_for_device()
    // instead.
    char hca[64];
};
static_assert(
    sizeof(AbolethEndpointConfig) == 96,
    "AbolethEndpointConfig is versioned by struct_size, so its layout must "
    "only ever grow at the end");

extern "C" {

// The message behind the last failing call on this thread. Every entry point
// below reports failure as a null handle or a nonzero status and leaves the
// reason here.
const char* aboleth_runtime_last_error();

std::uint32_t aboleth_runtime_provider_caps();

void* aboleth_runtime_create();
void* aboleth_runtime_create_for_device(
    int gpu_index,
    const char* hca_name);
const char* aboleth_runtime_hca_name(void* runtime_handle);
int aboleth_runtime_gpu_index(void* runtime_handle);
int aboleth_runtime_port(void* runtime_handle);
int aboleth_runtime_gid_index(void* runtime_handle);
int aboleth_runtime_link_layer(void* runtime_handle);
int aboleth_runtime_lid(void* runtime_handle);
int aboleth_runtime_destroy(void* runtime_handle);

// Device, GID and port selection without a runtime.
//
// These answer the three questions a caller has to settle before it can open
// its own queue pair — which HCA belongs to a GPU, which GID entry addresses
// the fabric, and what the local end of an RC path looks like — using the same
// code a live runtime uses, so an external client and this runtime always
// agree. They open and close a short-lived ibv_context of their own; nothing
// is retained.

// The HCA the runtime would open for CUDA ordinal `gpu_index`: the mlx5 device
// sharing the deepest PCI path with it, unless ABOLETH_HCA names one. Pass a
// negative `gpu_index` for the topology-free default. `name_out` receives the
// NUL-terminated device name. Returns 0, or -1 with aboleth_runtime_last_error().
int aboleth_runtime_select_device(
    int gpu_index,
    char* name_out,
    std::size_t name_capacity);

// The GID index the runtime would address the fabric through on `device_name`.
// InfiniBand ports answer 0, RoCE ports prefer a v2 entry, and
// ABOLETH_GID_INDEX overrides both. `device_name` may be null or empty for the
// default device; `port` may be 0 for the port ABOLETH_PORT selects. Returns
// the index, or -1 with aboleth_runtime_last_error().
int aboleth_runtime_select_gid_index(
    const char* device_name,
    int port);

// The local address an RC connect has to publish. `link_layer_out` is an
// IBV_LINK_LAYER_* value: InfiniBand paths address by LID, everything else by
// GRH, so it is also the is_global decision. `path_mtu_out` is an ibv_mtu
// enumerator and already accounts for ABOLETH_PATH_MTU. Every out parameter is
// optional. `device_name` and `port` follow aboleth_runtime_select_gid_index().
// Returns 0, or -1 with aboleth_runtime_last_error().
int aboleth_runtime_query_port_address(
    const char* device_name,
    int port,
    std::uint16_t* lid_out,
    std::uint8_t* link_layer_out,
    std::uint8_t* gid_out16,
    int* gid_index_out,
    std::uint32_t* path_mtu_out);

// `config` may be null, which means every default. It is copied before this
// call returns; the caller keeps ownership.
void* aboleth_runtime_create_local_endpoint(
    void* runtime_handle,
    std::uint32_t transport,
    int gpu_index,
    void* descriptor_device,
    std::size_t descriptor_bytes,
    const AbolethEndpointConfig* config);
int aboleth_runtime_local_endpoint_connection_info(
    void* endpoint_handle,
    AbolethConnectionInfo* connection_info);
int aboleth_runtime_connect_local_endpoint(
    void* endpoint_handle,
    const AbolethConnectionInfo* remote_info);
int aboleth_runtime_local_endpoint_state(
    void* endpoint_handle,
    std::uint64_t* producer,
    std::uint64_t* consumer,
    std::uint64_t* status);
int aboleth_runtime_local_endpoint_capacity(
    void* endpoint_handle,
    std::uint32_t* capacity);
// Inline payload bytes the endpoint's queue pair granted at creation, which
// can exceed the conservative floor the compiler plans against. Zero when the
// transport has no runtime-granted inline form.
int aboleth_runtime_local_endpoint_max_inline(
    void* endpoint_handle,
    std::uint32_t* max_inline_bytes);
// Who performs the final submission action, resolved at bring-up: 0 = the
// GPU submits directly, 1 = staged (the GPU builds and publishes work and a
// host doorbell agent performs the one store the platform withholds from
// it), 2 = command proxy (the host builds and posts the work). See
// docs/RDMA_ABI.md and docs/EFA_ABI.md.
int aboleth_runtime_local_endpoint_submit_mode(
    void* endpoint_handle,
    std::uint32_t* submit_mode);
// 1 when the endpoint's send queue, doorbell records and completion queue all
// live in GPU memory (the DEVX staged path), so a GPU-scope release publishes
// a submission; 0 when any of them is host-resident or the backend is not
// mlx5.
int aboleth_runtime_local_endpoint_gpu_resident_queues(
    void* endpoint_handle,
    std::uint32_t* resident);
int aboleth_runtime_destroy_local_endpoint(
    void* endpoint_handle);

void* aboleth_runtime_register_buffer(
    void* runtime_handle,
    int gpu_index,
    void* address,
    std::size_t bytes,
    std::uint64_t generation);
int aboleth_runtime_unregister_buffer(void* buffer_handle);
std::uint64_t aboleth_runtime_buffer_address(void* buffer_handle);
std::uint64_t aboleth_runtime_buffer_length(void* buffer_handle);
std::uint32_t aboleth_runtime_buffer_lkey(void* buffer_handle);
std::uint32_t aboleth_runtime_buffer_rkey(void* buffer_handle);
int aboleth_runtime_buffer_device(void* buffer_handle);
std::uint64_t aboleth_runtime_buffer_generation(void* buffer_handle);
// 1 when the buffer's verbs registration carries no relaxed ordering, so a
// fenced write may publish into it; 0 when it does or when that is unknown.
int aboleth_runtime_buffer_strictly_ordered(void* buffer_handle);

// expected0_host/expected1_host, when non-null, receive the host-built
// descriptor bytes before the device copy — an independent record for
// conformance tests to compare a kernel's view against.
void* aboleth_runtime_create_endpoint_pair(
    void* runtime_handle,
    int device0,
    void* descriptor0_device,
    std::size_t descriptor0_bytes,
    int device1,
    void* descriptor1_device,
    std::size_t descriptor1_bytes,
    void* expected0_host,
    void* expected1_host);
// Two Transport.EFA endpoints connected both ways. Requires the runtime's
// EFA fabric to be efa-direct (ABOLETH_PROXY_PROVIDER=efa with
// ABOLETH_EFA_FABRIC=efa-direct).
void* aboleth_runtime_create_efa_endpoint_pair(
    void* runtime,
    int device0,
    void* descriptor0_device,
    std::size_t descriptor0_bytes,
    int device1,
    void* descriptor1_device,
    std::size_t descriptor1_bytes);

void* aboleth_runtime_create_proxy_endpoint_pair(
    void* runtime_handle,
    int device0,
    void* descriptor0_device,
    std::size_t descriptor0_bytes,
    int device1,
    void* descriptor1_device,
    std::size_t descriptor1_bytes);
int aboleth_runtime_destroy_endpoint_pair(void* pair_handle);
int aboleth_runtime_endpoint_pair_state(
    void* pair_handle,
    std::uint64_t* forward_producer,
    std::uint64_t* forward_consumer,
    std::uint64_t* reverse_producer,
    std::uint64_t* reverse_consumer);
int aboleth_runtime_endpoint_pair_status(
    void* pair_handle,
    std::uint64_t* forward_status,
    std::uint64_t* reverse_status);
std::uint64_t aboleth_runtime_endpoint_pair_completion_state(
    void* pair_handle,
    int endpoint_index);
int aboleth_runtime_endpoint_pair_completion_snapshot(
    void* pair_handle,
    int endpoint_index,
    std::uint64_t* words,
    std::size_t word_count);
// Inline payload bytes the queue pair behind `endpoint_index` was granted.
// Direct (mlx5) pairs only; proxy and EFA pairs report an error rather than
// a capacity.
int aboleth_runtime_endpoint_pair_max_inline(
    void* pair_handle,
    int endpoint_index,
    std::uint32_t* max_inline_bytes);
// The send queue of `endpoint_index` as the host sees it, plus the byte-swapped
// head of its doorbell record. This is the wire form a generated kernel wrote,
// which is what the differential tests read back and compare; direct endpoints
// only, since no other backend has an mlx5 send queue.
int aboleth_runtime_endpoint_pair_send_queue_snapshot(
    void* pair_handle,
    int endpoint_index,
    void* output,
    std::size_t output_bytes,
    std::uint32_t* dbrec_head);

}  // extern "C"
