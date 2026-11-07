// HIP control plane for coherent allocations, the host command proxy and
// GPU-initiated Ionic queues. Both network paths consume the shared Queue IR.
#include "../dl/verbs_shim.h"
#include "aboleth/runtime_c.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <hip/hip_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include "ionic_control.h"

namespace {
namespace dl = aboleth::dl;
namespace abi = aboleth::abi;
thread_local std::string error;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void hip(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
int setting(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    return fallback;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  require(end && !*end && parsed >= 0 && parsed <= 65535,
          "invalid HIP endpoint setting");
  return static_cast<int>(parsed);
}
template <class F, class T> T guarded(F function, T failure) {
  error.clear();
  try {
    return function();
  } catch (const std::exception &exception) {
    error = exception.what();
    return failure;
  }
}

void refuse_config(const char *field, const std::string &value,
                   const std::string &reason) {
  throw std::runtime_error("AbolethEndpointConfig." + std::string(field) +
      "=" + value + " " + reason + " (see docs/transport-control.md)");
}
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

struct Runtime {
  std::unique_ptr<aboleth::hip_ionic::DoorbellMapping> ionic_mapping;
  ibv_device **devices = nullptr;
  ibv_context *context = nullptr;
  ibv_pd *pd = nullptr;
  std::string name;
  int gpu = 0, port = 1, gid_index = 0;
  ibv_port_attr port_attributes{};
  ibv_gid gid{};
  ~Runtime() {
    ionic_mapping.reset();
    if (pd)
      dl::verbs().ibv_dealloc_pd(pd);
    if (context)
      dl::verbs().ibv_close_device(context);
    if (devices)
      dl::verbs().ibv_free_device_list(devices);
  }
};
using Owner = std::shared_ptr<Runtime>;
Owner &owner(void *handle) {
  require(handle, "closed HIP runtime");
  return *static_cast<Owner *>(handle);
}

Owner open_runtime(int gpu, const char *requested) {
  auto runtime = std::make_shared<Runtime>();
  runtime->gpu = std::max(0, gpu);
  hip(hipSetDevice(runtime->gpu));
  int count = 0;
  runtime->devices = dl::verbs().ibv_get_device_list(&count);
  require(runtime->devices && count, "no verbs device available for HIP transports");
  if (!requested || !*requested)
    requested = std::getenv("ABOLETH_HCA");
  char bus[32];
  hip(hipDeviceGetPCIBusId(bus, sizeof bus, runtime->gpu));
  const auto gpu_path =
      std::filesystem::canonical(std::string("/sys/bus/pci/devices/") + bus)
          .string();
  int best_depth = -1;
  ibv_device *selected = nullptr;
  for (int index = 0; index < count; ++index) {
    const char *name = dl::verbs().ibv_get_device_name(runtime->devices[index]);
    if (requested && *requested && std::strcmp(name, requested))
      continue;
    std::error_code ec;
    auto path =
        std::filesystem::canonical(
            std::string("/sys/class/infiniband/") + name + "/device", ec)
            .string();
    if (ec)
      continue;
    int depth = 0;
    while (depth < static_cast<int>(std::min(path.size(), gpu_path.size())) &&
           path[depth] == gpu_path[depth])
      ++depth;
    if (depth > best_depth) {
      best_depth = depth;
      selected = runtime->devices[index];
      runtime->name = name;
    }
  }
  require(selected, "requested verbs device not found");
  runtime->context = dl::verbs().ibv_open_device(selected);
  require(runtime->context, "ibv_open_device failed");
  runtime->port = setting("ABOLETH_PORT", 1);
  require(!dl::verbs().query_port(runtime->context, runtime->port,
                                  &runtime->port_attributes),
          "ibv_query_port failed");
  require(runtime->port_attributes.state == IBV_PORT_ACTIVE,
          "verbs port is not active");
  int default_gid = 0;
  if (runtime->port_attributes.link_layer != IBV_LINK_LAYER_INFINIBAND) {
    bool found = false;
    for (int index = 0; index < runtime->port_attributes.gid_tbl_len; ++index) {
      ibv_gid_entry entry{};
      if (!dl::verbs().query_gid_ex(runtime->context, runtime->port, index,
                                    &entry, 0) &&
          entry.gid_type == IBV_GID_TYPE_ROCE_V2) {
        if (!found) {
          default_gid = index;
          found = true;
        }
        // Ionic exposes an IPv6 link-local GID before its routed IPv4 GID.
        // Prefer the IPv4-mapped address used by this RoCE fabric.
        bool mapped = entry.gid.raw[10] == 0xff && entry.gid.raw[11] == 0xff;
        for (int byte = 0; byte < 10; ++byte)
          mapped &= entry.gid.raw[byte] == 0;
        if (mapped) {
          default_gid = index;
          break;
        }
      }
    }
    require(found, "no RoCEv2 GID available");
  }
  runtime->gid_index = setting("ABOLETH_GID_INDEX", default_gid);
  require(!dl::verbs().ibv_query_gid(runtime->context, runtime->port,
                                     runtime->gid_index, &runtime->gid),
          "ibv_query_gid failed");
  runtime->pd = dl::verbs().ibv_alloc_pd(runtime->context);
  require(runtime->pd, "ibv_alloc_pd failed");
  return runtime;
}

struct Buffer {
  Owner runtime;
  ibv_mr *mr = nullptr;
  void *address = nullptr;
  std::size_t bytes = 0;
  std::uint64_t generation = 0;
  int gpu = 0;
  // Unknown until the exact requested flags produced a successful MR.
  int access_flags = -1;
  ~Buffer() {
    if (mr)
      dl::verbs().ibv_dereg_mr(mr);
  }
};

struct Endpoint {
  std::unique_ptr<aboleth::hip_ionic::QueueResources> ionic;
  bool connected = false;
  Owner runtime;
  ibv_cq *cq = nullptr;
  ibv_qp *qp = nullptr;
  abi::ProxyQueueState *queue = nullptr;
  std::uint64_t *producer = nullptr;
  std::array<std::uint64_t, abi::kProxyMaximumCommands * 2> staging{};
  ibv_mr *staging_mr = nullptr;
  std::atomic<bool> stop{false};
  std::thread worker;
  int gpu = 0;
  std::uint32_t psn = 0;
  std::uint64_t next = 0;
  ~Endpoint() {
    stop.store(true);
    if (worker.joinable())
      worker.join();
    if (qp)
      dl::verbs().ibv_destroy_qp(qp);
    if (cq)
      dl::verbs().ibv_destroy_cq(cq);
    ionic.reset();
    if (staging_mr)
      dl::verbs().ibv_dereg_mr(staging_mr);
    // Free on the allocating device; destruction can occur on a Python GC
    // thread.
    hipSetDevice(gpu);
    if (producer)
      hipFree(producer);
    if (queue)
      hipHostFree(queue);
  }
  std::uint64_t post_and_complete(ibv_send_wr *requests, unsigned count) {
    if (!count)
      return 0;
    requests[count - 1].next = nullptr;
    requests[count - 1].send_flags |= IBV_SEND_SIGNALED;
    ibv_send_wr *bad = nullptr;
    int posted = ibv_post_send(qp, requests, &bad);
    if (posted) {
      fprintf(stderr,
              "aboleth HIP proxy post_send rc=%d (%s) errno=%d count=%u "
              "bad_index=%lld qp=%u batch=%llu\n",
              posted, std::strerror(posted), errno, count,
              bad ? static_cast<long long>(bad - requests) : -1ll,
              qp->qp_num, static_cast<unsigned long long>(next + 1));
      return 3;
    }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!stop.load(std::memory_order_relaxed)) {
      ibv_wc completion{};
      int result = ibv_poll_cq(cq, 1, &completion);
      if (result < 0) {
        fprintf(stderr, "aboleth HIP proxy poll_cq rc=%d errno=%d qp=%u\n",
                result, errno, qp->qp_num);
        return 2;
      }
      if (result == 1) {
        if (completion.status == IBV_WC_SUCCESS && completion.wr_id == next + 1)
          return 0;
        fprintf(stderr, "aboleth HIP proxy CQ status=%u vendor=%u wr_id=%llu\n",
                unsigned(completion.status), completion.vendor_err,
                static_cast<unsigned long long>(completion.wr_id));
        return 2;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        fprintf(stderr, "aboleth HIP proxy completion timeout qp=%u batch=%llu\n",
                qp->qp_num, static_cast<unsigned long long>(next + 1));
        return 1;
      }
      std::this_thread::yield();
    }
    return 1;
  }
  std::uint64_t post(const abi::ProxyBatchSlot &slot) {
    if (!slot.command_count || slot.command_count > abi::kProxyMaximumCommands)
      return 3;
    std::array<ibv_send_wr, abi::kProxyMaximumCommands> requests{};
    std::array<ibv_sge, abi::kProxyMaximumCommands> scatter{};
    unsigned count = 0;
    for (unsigned index = 0; index < slot.command_count; ++index) {
      const auto &command = slot.commands[index];
      if (command.opcode == abi::kProxyCommandNop)
        continue;
      const bool strong = command.opcode == abi::kProxyCommandStrongSignal;
      const bool signal =
          strong || command.opcode == abi::kProxyCommandWeakSignal;
      if (!command.bytes ||
          (!signal && command.opcode != abi::kProxyCommandWrite))
        return 3;
      if (strong) {
        // GPU-target CAS retries indefinitely on this Ionic firmware. A
        // strong publication instead retires every earlier payload write
        // before the host can post its epoch write. MR registrations never
        // enable relaxed ordering, and receiver flags/payload are fine-grained.
        auto status = post_and_complete(requests.data(), count);
        if (status)
          return status;
        count = 0;
      }
      auto &request = requests[count];
      auto &sge = scatter[count];
      request = {};
      sge = {};
      request.wr_id = next + 1;
      request.sg_list = &sge;
      request.num_sge = 1;
      request.opcode = IBV_WR_RDMA_WRITE;
      sge.length = command.bytes;
      if (signal) {
        if (command.bytes != 8)
          return 3;
        staging[count] = command.source_address;
        sge.addr = reinterpret_cast<std::uint64_t>(&staging[count]);
        sge.lkey = staging_mr->lkey;
      } else {
        sge.addr = command.source_address;
        sge.lkey = command.source_key;
      }
      request.wr.rdma.remote_addr = command.destination_address;
      request.wr.rdma.rkey = command.destination_key;
      if (count)
        requests[count - 1].next = &request;
      ++count;
    }
    return post_and_complete(requests.data(), count);
  }
  void progress() {
    while (!stop.load(std::memory_order_relaxed)) {
      auto &slot = queue->slots[next % abi::kProxySlotCount];
      if (__atomic_load_n(&slot.ready_sequence, __ATOMIC_ACQUIRE) != next + 1) {
        std::this_thread::yield();
        continue;
      }
      auto status = __atomic_load_n(&queue->status, __ATOMIC_RELAXED);
      if (!status)
        status = post(slot);
      if (status)
        __atomic_store_n(&queue->status, status, __ATOMIC_RELEASE);
      __atomic_store_n(&queue->completed_sequence, ++next, __ATOMIC_RELEASE);
    }
  }
};

std::unique_ptr<Endpoint> create_endpoint(Owner runtime, int gpu,
                                          void *descriptor, std::size_t bytes,
                                          const AbolethEndpointConfig *config) {
  require(bytes >= sizeof(abi::ProxyEndpointDescriptor),
          "proxy descriptor is too small");
  require(gpu == runtime->gpu, "endpoint GPU does not match HIP runtime");
  if (config) {
    require(config->struct_size == sizeof(*config),
            "unsupported endpoint config size");
    require(!config->qp_type && config->qps_per_peer <= 1 &&
                !config->sm_map_policy,
            "unsupported HIP proxy queue configuration");
    require(!config->qp_depth && !config->max_inline,
            "HIP proxy does not accept direct queue tuning");
    require(!config->port || config->port == runtime->port,
            "endpoint port differs from runtime");
    require(config->gid_index < 0 || config->gid_index == runtime->gid_index,
            "endpoint GID differs from runtime");
    require(!config->hca[0] || runtime->name == config->hca,
            "endpoint HCA differs from runtime");
  }
  auto endpoint = std::make_unique<Endpoint>();
  endpoint->runtime = runtime;
  endpoint->gpu = gpu;
  hip(hipSetDevice(gpu));
  hip(hipHostMalloc(reinterpret_cast<void **>(&endpoint->queue),
                    sizeof(*endpoint->queue),
                    hipHostMallocCoherent | hipHostMallocMapped));
  std::memset(endpoint->queue, 0, sizeof(*endpoint->queue));
  hip(hipExtMallocWithFlags(reinterpret_cast<void **>(&endpoint->producer), 8,
                            hipDeviceMallocFinegrained));
  hip(hipMemset(endpoint->producer, 0, 8));
  endpoint->cq =
      dl::verbs().ibv_create_cq(runtime->context, 512, nullptr, nullptr, 0);
  require(endpoint->cq, "ibv_create_cq failed");
  ibv_qp_init_attr init{};
  init.send_cq = endpoint->cq;
  init.recv_cq = endpoint->cq;
  init.qp_type = IBV_QPT_RC;
  init.cap.max_send_wr = 512;
  init.cap.max_recv_wr = 1;
  init.cap.max_send_sge = 1;
  init.cap.max_recv_sge = 1;
  endpoint->qp = dl::verbs().ibv_create_qp(runtime->pd, &init);
  require(endpoint->qp, "ibv_create_qp failed");
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = runtime->port;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_ATOMIC;
  require(!dl::verbs().ibv_modify_qp(endpoint->qp, &attr,
                                     IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                         IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
          "QP INIT failed");
  endpoint->staging_mr =
      dl::verbs().reg_mr(runtime->pd, endpoint->staging.data(),
                         sizeof(endpoint->staging), IBV_ACCESS_LOCAL_WRITE);
  require(endpoint->staging_mr, "proxy signal staging registration failed");
  endpoint->psn = (endpoint->qp->qp_num * 2654435761u) & 0xffffff;
  void *mapped = nullptr;
  hip(hipHostGetDevicePointer(&mapped, endpoint->queue, 0));
  abi::ProxyEndpointDescriptor value{
      reinterpret_cast<std::uint64_t>(mapped),
      reinterpret_cast<std::uint64_t>(endpoint->producer),
      3000000000ull,
      abi::kProxySlotCount,
      sizeof(abi::ProxyBatchSlot),
      abi::kProxyMaximumCommands,
      abi::kProxyEndpointVersion};
  hip(hipMemcpy(descriptor, &value, sizeof value, hipMemcpyHostToDevice));
  return endpoint;
}
std::unique_ptr<Endpoint> create_ionic_endpoint(Owner runtime, int gpu,
    void *descriptor, std::size_t bytes, const AbolethEndpointConfig &config) {
  require(gpu == runtime->gpu, "Ionic endpoint GPU differs from runtime");
  require(!config.qp_type && config.qps_per_peer <= 1 && !config.sm_map_policy,
          "unsupported Ionic queue configuration");
  require(!config.port || config.port == runtime->port, "Ionic port differs from runtime");
  require(config.gid_index < 0 || config.gid_index == runtime->gid_index,
          "Ionic GID differs from runtime");
  require(!config.hca[0] || runtime->name == config.hca, "Ionic HCA differs from runtime");
  require(!config.max_inline || config.max_inline == 16,
          "initial Ionic backend requires max_inline=16 (or default)");
  hip(hipSetDevice(gpu));
  auto ep = std::make_unique<Endpoint>();
  ep->runtime = runtime; ep->gpu = gpu;
  if (!runtime->ionic_mapping)
    runtime->ionic_mapping = std::make_unique<aboleth::hip_ionic::DoorbellMapping>(runtime->context, gpu);
  ep->ionic = std::make_unique<aboleth::hip_ionic::QueueResources>(gpu);
  ep->ionic->create(runtime->context, runtime->pd, *runtime->ionic_mapping,
      config.qp_depth ? config.qp_depth : 512, ep->cq, ep->qp, descriptor, bytes);
  ibv_qp_attr attr{}; attr.qp_state = IBV_QPS_INIT; attr.port_num = runtime->port;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  errno = 0;
  aboleth::hip_ionic::check_verbs(dl::verbs().ibv_modify_qp(ep->qp, &attr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
      "Ionic QP INIT failed");
  ep->psn = (ep->qp->qp_num * 2654435761u) & 0xffffff;
  return ep;
}
Endpoint &endpoint(void *handle) {
  require(handle, "closed HIP endpoint");
  return *static_cast<Endpoint *>(handle);
}
Buffer &buffer(void *handle) {
  require(handle, "closed HIP registration");
  return *static_cast<Buffer *>(handle);
}
} // namespace

extern "C" {
const char *aboleth_runtime_last_error() { return error.c_str(); }
std::uint32_t aboleth_runtime_provider_caps() {
  return guarded(
      [] {
        int count = 0;
        hip(hipGetDeviceCount(&count));
        return std::uint32_t(
            (count ? kAbolethProviderHip : 0) |
            (dl::verbs_available() ? kAbolethProviderVerbs : 0) |
            (aboleth::hip_ionic::available() ? kAbolethProviderIonic : 0));
      },
      std::uint32_t(0));
}
void *aboleth_runtime_create_for_device(int gpu, const char *hca) {
  return guarded([&]() -> void * { return new Owner(open_runtime(gpu, hca)); },
                 static_cast<void *>(nullptr));
}
void *aboleth_runtime_create() {
  return aboleth_runtime_create_for_device(0, nullptr);
}
const char *aboleth_runtime_hca_name(void *h) { return owner(h)->name.c_str(); }
int aboleth_runtime_gpu_index(void *h) { return owner(h)->gpu; }
int aboleth_runtime_port(void *h) { return owner(h)->port; }
int aboleth_runtime_gid_index(void *h) { return owner(h)->gid_index; }
int aboleth_runtime_link_layer(void *h) {
  return owner(h)->port_attributes.link_layer;
}
int aboleth_runtime_lid(void *h) { return owner(h)->port_attributes.lid; }
int aboleth_runtime_destroy(void *h) {
  return guarded(
      [&] {
        delete static_cast<Owner *>(h);
        return 0;
      },
      -1);
}
int aboleth_runtime_select_device(int gpu, char *name, std::size_t capacity) {
  return guarded(
      [&] {
        auto runtime = open_runtime(gpu, nullptr);
        require(name && capacity > runtime->name.size(),
                "device name output too small");
        std::memcpy(name, runtime->name.c_str(), runtime->name.size() + 1);
        return 0;
      },
      -1);
}
int aboleth_runtime_select_gid_index(const char *name, int port) {
  return guarded(
      [&] {
        auto runtime = open_runtime(0, name);
        require(!port || port == runtime->port, "port differs from runtime");
        return runtime->gid_index;
      },
      -1);
}
int aboleth_runtime_query_port_address(const char *name, int port,
                                       std::uint16_t *lid, std::uint8_t *layer,
                                       std::uint8_t *gid, int *index,
                                       std::uint32_t *mtu) {
  return guarded(
      [&] {
        auto runtime = open_runtime(0, name);
        require(!port || port == runtime->port, "port differs from runtime");
        if (lid)
          *lid = runtime->port_attributes.lid;
        if (layer)
          *layer = runtime->port_attributes.link_layer;
        if (gid)
          std::memcpy(gid, &runtime->gid, 16);
        if (index)
          *index = runtime->gid_index;
        if (mtu)
          *mtu = runtime->port_attributes.active_mtu;
        return 0;
      },
      -1);
}
void *aboleth_runtime_register_buffer(void *h, int gpu, void *address,
                                      std::size_t bytes,
                                      std::uint64_t generation) {
  return guarded(
      [&]() -> void * {
        auto registered = std::make_unique<Buffer>();
        registered->runtime = owner(h);
        require(gpu == registered->runtime->gpu && address && bytes,
                "invalid HIP buffer registration");
        registered->address = address;
        registered->bytes = bytes;
        registered->generation = generation;
        registered->gpu = gpu;
        // Peer-memory registration is qualified on Ionic for coarse, fine and
        // uncached device allocations. Never opt into relaxed ordering here.
        const int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                 IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
        registered->mr = dl::verbs().reg_mr(
            registered->runtime->pd, address, bytes, access_flags);
        require(registered->mr, "HIP peer-memory registration failed");
        registered->access_flags = access_flags;
        return registered.release();
      },
      static_cast<void *>(nullptr));
}
int aboleth_runtime_unregister_buffer(void *h) {
  return guarded(
      [&] {
        delete static_cast<Buffer *>(h);
        return 0;
      },
      -1);
}
std::uint64_t aboleth_runtime_buffer_address(void *h) {
  return reinterpret_cast<std::uint64_t>(buffer(h).address);
}
std::uint64_t aboleth_runtime_buffer_length(void *h) { return buffer(h).bytes; }
std::uint32_t aboleth_runtime_buffer_lkey(void *h) {
  return buffer(h).mr->lkey;
}
std::uint32_t aboleth_runtime_buffer_rkey(void *h) {
  return buffer(h).mr->rkey;
}
int aboleth_runtime_buffer_device(void *h) { return buffer(h).gpu; }
std::uint64_t aboleth_runtime_buffer_generation(void *h) {
  return buffer(h).generation;
}
int aboleth_runtime_buffer_strictly_ordered(void *h) {
  const auto &registered = buffer(h);
  // This reports the successful registration contract, not a guessed device
  // property. Unknown or relaxed registrations never qualify for fenced writes.
  return registered.mr && registered.access_flags >= 0 &&
         !(registered.access_flags & IBV_ACCESS_RELAXED_ORDERING);
}
void *
aboleth_runtime_create_local_endpoint(void *h, std::uint32_t transport, int gpu,
                                      void *descriptor, std::size_t bytes,
                                      const AbolethEndpointConfig *config) {
  return guarded(
      [&]() -> void * {
        const auto normalized = checked_config(config);
        require(h && descriptor && bytes, "null local-endpoint argument");
        if (transport == kAbolethTransportIonic)
          return create_ionic_endpoint(owner(h), gpu, descriptor, bytes, normalized).release();
        require(transport == kAbolethTransportProxy,
                "HIP supports the proxy transport; direct mlx5/EFA require "
                "their native backend");
        return create_endpoint(owner(h), gpu, descriptor, bytes, &normalized)
            .release();
      },
      static_cast<void *>(nullptr));
}
int aboleth_runtime_local_endpoint_connection_info(
    void *h, AbolethConnectionInfo *info) {
  return guarded(
      [&] {
        auto &ep = endpoint(h);
        require(info, "null connection output");
        *info = {};
        info->abi_version = kAbolethConnectionInfoVersion;
        info->transport = ep.ionic ? kAbolethTransportIonic : kAbolethTransportProxy;
        info->provider = kAbolethProxyProviderVerbsRc;
        info->qpn = ep.qp->qp_num;
        info->psn = ep.psn;
        info->port_num = ep.runtime->port;
        info->gid_index = ep.runtime->gid_index;
        info->lid = ep.runtime->port_attributes.lid;
        info->link_layer = ep.runtime->port_attributes.link_layer;
        info->path_mtu = ep.runtime->port_attributes.active_mtu;
        std::memcpy(info->gid, &ep.runtime->gid, 16);
        if (ep.ionic) {
          info->address_format = 0x494f4e31; // ION1 private scratch binding
          info->address_length = 16;
          auto address = reinterpret_cast<std::uint64_t>(ep.ionic->scratch);
          std::memcpy(info->address, &address, 8);
          std::memcpy(info->address + 8, &ep.ionic->scratch_mr->rkey, 4);
        }
        return 0;
      },
      -1);
}
int aboleth_runtime_connect_local_endpoint(
    void *h, const AbolethConnectionInfo *remote) {
  return guarded(
      [&] {
        auto &ep = endpoint(h);
        require(remote &&
                    remote->abi_version == kAbolethConnectionInfoVersion &&
                    remote->transport == (ep.ionic ? kAbolethTransportIonic : kAbolethTransportProxy) &&
                    remote->provider == kAbolethProxyProviderVerbsRc,
                "incompatible HIP endpoint peer");
        require(!ep.connected, "HIP endpoint already connected");
        if (ep.ionic) {
          require(remote->address_format == 0x494f4e31 && remote->address_length == 16,
                  "incompatible Ionic private scratch binding");
          std::uint64_t address = 0; std::uint32_t key = 0;
          std::memcpy(&address, remote->address, 8);
          std::memcpy(&key, remote->address + 8, 4);
          ep.ionic->connect_scratch(address, key);
        }
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = static_cast<ibv_mtu>(
            std::min(remote->path_mtu,
                     std::uint32_t(ep.runtime->port_attributes.active_mtu)));
        attr.dest_qp_num = remote->qpn;
        attr.rq_psn = remote->psn;
        attr.max_dest_rd_atomic = 16;
        attr.min_rnr_timer = 12;
        attr.ah_attr.port_num = ep.runtime->port;
        attr.ah_attr.dlid = remote->lid;
        attr.ah_attr.is_global =
            ep.runtime->port_attributes.link_layer != IBV_LINK_LAYER_INFINIBAND;
        if (attr.ah_attr.is_global) {
          std::memcpy(&attr.ah_attr.grh.dgid, remote->gid, 16);
          attr.ah_attr.grh.sgid_index = ep.runtime->gid_index;
          attr.ah_attr.grh.hop_limit = 64;
        }
        require(!dl::verbs().ibv_modify_qp(
                    ep.qp, &attr,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
                "QP RTR failed");
        attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 20;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = ep.psn;
        attr.max_rd_atomic = 16;
        require(!dl::verbs().ibv_modify_qp(
                    ep.qp, &attr,
                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC),
                "QP RTS failed");
        ep.connected = true;
        if (!ep.ionic) ep.worker = std::thread([&ep] { ep.progress(); });
        return 0;
      },
      -1);
}
int aboleth_runtime_local_endpoint_state(void *h, std::uint64_t *producer,
                                         std::uint64_t *consumer,
                                         std::uint64_t *status) {
  return guarded(
      [&] {
        auto &ep = endpoint(h);
        hip(hipSetDevice(ep.gpu));
        if (ep.ionic) {
          abi::IonicQueueState snapshot{};
          hip(hipMemcpy(&snapshot, ep.ionic->state, sizeof snapshot, hipMemcpyDeviceToHost));
          if (producer) *producer = snapshot.producer;
          if (consumer) *consumer = snapshot.retired;
          if (status) *status = snapshot.status;
          return 0;
        }
        if (producer)
          hip(hipMemcpy(producer, ep.producer, 8, hipMemcpyDeviceToHost));
        if (consumer)
          *consumer =
              __atomic_load_n(&ep.queue->completed_sequence, __ATOMIC_ACQUIRE);
        if (status)
          *status = __atomic_load_n(&ep.queue->status, __ATOMIC_ACQUIRE);
        return 0;
      },
      -1);
}
int aboleth_runtime_local_endpoint_capacity(void *handle, std::uint32_t *value) {
  return guarded([&] {
    require(value, "null endpoint capacity output");
    auto &ep = endpoint(handle);
    *value = ep.ionic ? ep.ionic->capacity : abi::kProxySlotCount;
    return 0;
  }, -1);
}
int aboleth_runtime_local_endpoint_max_inline(void *handle, std::uint32_t *value) {
  return guarded([&] {
    require(value, "null endpoint inline output");
    *value = endpoint(handle).ionic ? 16 : 0;
    return 0;
  }, -1);
}
int aboleth_runtime_local_endpoint_submit_mode(void *handle, std::uint32_t *value) {
  return guarded([&] {
    require(value, "null endpoint submit-mode output");
    *value = endpoint(handle).ionic ? 0 : 2;
    return 0;
  }, -1);
}
int aboleth_runtime_local_endpoint_gpu_resident_queues(void *handle,
                                                       std::uint32_t *value) {
  return guarded([&] {
    require(value, "null endpoint queue-placement output");
    *value = endpoint(handle).ionic ? 1 : 0;
    return 0;
  }, -1);
}
int aboleth_runtime_destroy_local_endpoint(void *h) {
  return guarded(
      [&] {
        delete static_cast<Endpoint *>(h);
        return 0;
      },
      -1);
}

// Pair helpers are deliberately rejected until their ownership/lifetime path
// is implemented. Rank-local endpoints already serve both local and remote
// bootstrap, so World needs no separate transport ABI.
void *aboleth_runtime_create_endpoint_pair(void *, int, void *, std::size_t,
                                           int, void *, std::size_t, void *,
                                           void *) {
  error = "direct mlx5 endpoint pairs are unavailable on HIP";
  return nullptr;
}
void *aboleth_runtime_create_efa_endpoint_pair(void *, int, void *, std::size_t,
                                               int, void *, std::size_t) {
  error = "EFA is unavailable on HIP";
  return nullptr;
}
void *aboleth_runtime_create_proxy_endpoint_pair(void *, int, void *,
                                                 std::size_t, int, void *,
                                                 std::size_t) {
  error = "use rank-local HIP proxy endpoints through World";
  return nullptr;
}
int aboleth_runtime_destroy_endpoint_pair(void *) {
  error = "HIP pair handle is unavailable";
  return -1;
}
int aboleth_runtime_endpoint_pair_state(void *, std::uint64_t *,
                                        std::uint64_t *, std::uint64_t *,
                                        std::uint64_t *) {
  return -1;
}
int aboleth_runtime_endpoint_pair_status(void *, std::uint64_t *,
                                         std::uint64_t *) {
  return -1;
}
std::uint64_t aboleth_runtime_endpoint_pair_completion_state(void *, int) {
  return 0;
}
int aboleth_runtime_endpoint_pair_completion_snapshot(void *, int,
                                                      std::uint64_t *,
                                                      std::size_t) {
  return -1;
}
int aboleth_runtime_endpoint_pair_max_inline(void *, int, std::uint32_t *) {
  return -1;
}
int aboleth_runtime_endpoint_pair_send_queue_snapshot(void *, int, void *,
                                                      std::size_t,
                                                      std::uint32_t *) {
  return -1;
}
}
