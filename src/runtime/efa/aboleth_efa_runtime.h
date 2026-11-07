#pragma once

// Host control plane for the GPU-initiated EFA backend.
//
// The kernel posts its own SRD work requests through the efa-dp-direct device
// library, so the host's job ends once the queues exist, are reachable from the
// GPU, and are described by an EfaEndpointDescriptor. Everything here is
// control plane: nothing in this header runs on the data path.
//
// Declared separately from the implementation so the provider-neutral runtime
// can forward to it without pulling in efadv.

#include <cstddef>
#include <cstdint>

struct EfaRuntimeHandle;
struct EfaEndpointHandle;
struct EfaBufferHandle;

namespace aboleth::efa {

// Opens the EFA device closest to `gpu_index` on the PCI topology, unless
// `device_name` names one. Honors ABOLETH_EFA_DEVICE / ABOLETH_HCA.
EfaRuntimeHandle* runtime_create(
    int gpu_index,
    const char* device_name);
void runtime_destroy(EfaRuntimeHandle* runtime) noexcept;
const char* runtime_device_name(EfaRuntimeHandle* runtime);

// True when ABOLETH_EFA_SUBMIT (or the device generation) selected staged
// submission: the GPU builds descriptors into a host-pinned staging ring and
// the runtime's doorbell agent moves them into the adapter's queue. The
// device-side posting code is identical in both modes; only who performs the
// two adapter stores differs.
bool runtime_staged_submit(EfaRuntimeHandle* runtime);

// Creates an SRD queue pair with GPU-visible queues and writes the endpoint
// descriptor into `descriptor_device`. The endpoint cannot transfer until it
// is connected to a peer.
EfaEndpointHandle* endpoint_create(
    EfaRuntimeHandle* runtime,
    std::uint32_t send_depth,
    std::uint32_t recv_depth,
    void* descriptor_device,
    std::size_t descriptor_bytes);
void endpoint_destroy(EfaEndpointHandle* endpoint) noexcept;

// SRD addressing: a global identifier, a queue-pair number, and a queue key.
// There is no connection handshake — the peer is named per work request.
struct Address {
    std::uint8_t gid[16];
    std::uint32_t queue_pair_number;
    std::uint32_t queue_key;
};

Address endpoint_address(EfaEndpointHandle* endpoint);
void endpoint_connect(
    EfaEndpointHandle* endpoint,
    const Address& peer);

std::uint64_t endpoint_descriptor_address(
    EfaEndpointHandle* endpoint);
std::uint32_t endpoint_capacity(EfaEndpointHandle* endpoint);
void endpoint_state(
    EfaEndpointHandle* endpoint,
    std::uint64_t* producer,
    std::uint64_t* retired,
    std::uint64_t* status);

// Diagnostic host-posted datapath. Some platforms drop GPU stores into the
// adapter's BAR (EFA gen2 on p5 instances routes peer MMIO through the Nitro
// fabric, which discards them), leaving the doorbell and descriptor stores as
// the only pieces of the GPU path that cannot run. This posts one write
// descriptor from the host through the same send queue the GPU uses — same
// builder fields, same slots, same doorbell — so the rest of the datapath
// (GPU-memory completion queues included) can be exercised and measured.
// Not a transport: one descriptor per call, doorbell rung immediately.
std::uint64_t endpoint_host_post_write(
    EfaEndpointHandle* endpoint,
    std::uint64_t local_address,
    std::uint32_t local_key,
    std::uint64_t remote_address,
    std::uint32_t remote_key,
    std::uint32_t bytes,
    std::uint32_t request_id);

// GPU memory must be registered through dmabuf; EFA has no host fallback for
// device pointers.
EfaBufferHandle* register_buffer(
    EfaRuntimeHandle* runtime,
    void* address,
    std::size_t bytes);
void unregister_buffer(EfaBufferHandle* buffer) noexcept;
std::uint32_t buffer_local_key(EfaBufferHandle* buffer);
std::uint32_t buffer_remote_key(EfaBufferHandle* buffer);

// Packing used by the connection-info address blob:
// GID(16) || queue-pair number(4, little endian) || queue key(4).
inline constexpr std::size_t kAddressBytes = 24;
std::size_t pack_address(
    const Address& address,
    std::uint8_t* bytes,
    std::size_t capacity);
Address unpack_address(
    const std::uint8_t* bytes,
    std::size_t length);

const char* last_error();

} // namespace aboleth::efa
