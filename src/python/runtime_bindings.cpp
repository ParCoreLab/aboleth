#include "aboleth/runtime_c.h"

// The compile-time registries the Python side plans against. These headers are
// free of LLVM, Triton, and CUDA dependencies by design, so binding them costs
// this module nothing at import time and saves Python from keeping its own
// copy of rows it does not own.
#include "aboleth/Target/TransportCaps.h"
#include "aboleth/Target/TransportPlan.h"

#include <pybind11/pybind11.h>
#if ABOLETH_HAS_HIP
#include <hip/hip_runtime.h>
#include <ATen/dlpack.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

#if ABOLETH_HAS_HIP
// The DLPack owner frees the allocation when the last tensor/storage view
// dies, with no Python callback and no borrowed allocation lifetime.
struct HipTensor {
  DLManagedTensor managed{};
  std::int64_t bytes = 0;
};

py::capsule hipAllocate(std::int64_t bytes, int device) {
  if (bytes <= 0) throw std::invalid_argument("HIP coherent allocation must be nonempty");
  auto tensor = std::make_unique<HipTensor>(); tensor->bytes = bytes;
  auto status = hipSetDevice(device);
  if (status == hipSuccess)
    status = hipExtMallocWithFlags(&tensor->managed.dl_tensor.data, bytes, hipDeviceMallocFinegrained);
  if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
  tensor->managed.dl_tensor.device = {kDLROCM, device};
  tensor->managed.dl_tensor.ndim = 1;
  tensor->managed.dl_tensor.dtype = {kDLUInt, 8, 1};
  tensor->managed.dl_tensor.shape = &tensor->bytes;
  tensor->managed.deleter = [](DLManagedTensor *managed) {
    int previous = 0; hipGetDevice(&previous);
    hipSetDevice(managed->dl_tensor.device.device_id); hipFree(managed->dl_tensor.data); hipSetDevice(previous);
    delete reinterpret_cast<HipTensor *>(managed);
  };
  auto *managed = &tensor.release()->managed;
  return py::capsule(managed, "dltensor", [](PyObject *capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
      auto *value = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
      value->deleter(value);
    }
  });
}
#endif

std::runtime_error lastError(const char* fallback) {
  const char* message = aboleth_runtime_last_error();
  if (message != nullptr && message[0] != '\0')
    return std::runtime_error(message);
  return std::runtime_error(fallback);
}

// One row of `aboleth::abi::kTransportCaps`, every column of it. Callers ask
// the registry what a transport can do rather than testing which transport it
// is, so the whole row travels and a new column reaches Python without a new
// binding.
py::dict transportCapsRow(const aboleth::abi::TransportCaps& caps) {
  py::dict row;
  row["name"] = caps.name;
  row["gpu_initiated"] = caps.gpu_initiated;
  row["load_store_addressable"] = caps.load_store_addressable;
  row["ordered_delivery"] = caps.ordered_delivery;
  row["native_atomics"] = caps.native_atomics;
  row["imm_data"] = caps.imm_data;
  row["queue_capacity_limited"] = caps.queue_capacity_limited;
  row["selective_completion"] = caps.selective_completion;
  row["max_inline_bytes"] = caps.max_inline_bytes;
  row["default_completion_interval"] = caps.default_completion_interval;
  row["completion_model"] =
      static_cast<std::uint32_t>(caps.completion_model);
  row["doorbell_model"] =
      static_cast<std::uint32_t>(caps.doorbell_model);
  row["path_mtu"] = caps.path_mtu;
  row["max_data_queues"] = caps.max_data_queues;
  return row;
}

// The registry itself, keyed by the name the MLIR attributes and the Python
// planners already use, in registry order.
py::dict transportCapsRegistry() {
  py::dict registry;
  for (std::size_t index = 0;
       index < aboleth::abi::kTransportCapsCount;
       ++index) {
    const aboleth::abi::TransportCaps& caps =
        aboleth::abi::kTransportCaps[index];
    registry[py::str(caps.name)] = transportCapsRow(caps);
  }
  return registry;
}

class NativeConnectionInfo {
public:
  NativeConnectionInfo() = default;

  explicit NativeConnectionInfo(
      const AbolethConnectionInfo& value)
      : value_(value) {}

  NativeConnectionInfo(std::uint32_t abiVersion,
                       std::uint32_t transport,
                       std::uint32_t qpn,
                       std::uint32_t psn,
                       std::uint32_t gidIndex,
                       std::uint32_t pathMtu,
                       std::uint16_t lid,
                       std::uint8_t portNum,
                       std::uint8_t linkLayer,
                       const py::bytes& gid,
                       std::uint32_t provider,
                       std::uint32_t addressFormat,
                       const py::bytes& address) {
    value_.abi_version = abiVersion;
    value_.transport = transport;
    value_.qpn = qpn;
    value_.psn = psn;
    value_.gid_index = gidIndex;
    value_.path_mtu = pathMtu;
    value_.lid = lid;
    value_.port_num = portNum;
    value_.link_layer = linkLayer;
    const std::string gidBytes = gid;
    if (gidBytes.size() != sizeof(value_.gid))
      throw std::invalid_argument(
          "connection-info GID must contain exactly 16 bytes");
    std::memcpy(value_.gid, gidBytes.data(), sizeof(value_.gid));
    value_.provider = provider;
    value_.address_format = addressFormat;
    const std::string addressBytes = address;
    if (addressBytes.size() > sizeof(value_.address))
      throw std::invalid_argument(
          "connection-info provider address exceeds 64 bytes");
    value_.address_length =
        static_cast<std::uint32_t>(addressBytes.size());
    std::memcpy(
        value_.address,
        addressBytes.data(),
        addressBytes.size());
    validate(value_);
  }

  // The wire form is the struct itself. It is fixed-width and padding-free by
  // static_assert, so a peer's bytes land in the fields the sender filled with
  // no format to keep in step on either side of the exchange.
  py::bytes toBytes() const {
    return py::bytes(
        reinterpret_cast<const char*>(&value_),
        sizeof(value_));
  }

  static NativeConnectionInfo fromBytes(const py::bytes& data) {
    const std::string encoded = data;
    if (encoded.size() != sizeof(AbolethConnectionInfo))
      throw std::invalid_argument(
          "invalid Aboleth endpoint connection-info size");
    AbolethConnectionInfo value{};
    std::memcpy(&value, encoded.data(), sizeof(value));
    validate(value);
    return NativeConnectionInfo(value);
  }

  std::uint32_t abiVersion() const { return value_.abi_version; }
  std::uint32_t transport() const { return value_.transport; }
  std::uint32_t qpn() const { return value_.qpn; }
  std::uint32_t psn() const { return value_.psn; }
  std::uint32_t gidIndex() const { return value_.gid_index; }
  std::uint32_t pathMtu() const { return value_.path_mtu; }
  std::uint16_t lid() const { return value_.lid; }
  std::uint8_t portNum() const { return value_.port_num; }
  std::uint8_t linkLayer() const { return value_.link_layer; }
  py::bytes gid() const {
    return py::bytes(
        reinterpret_cast<const char*>(value_.gid),
        sizeof(value_.gid));
  }
  std::uint32_t provider() const { return value_.provider; }
  std::uint32_t addressFormat() const {
    return value_.address_format;
  }
  py::bytes address() const {
    return py::bytes(
        reinterpret_cast<const char*>(value_.address),
        value_.address_length);
  }

  const AbolethConnectionInfo& value() const { return value_; }

private:
  // What a caller can get wrong about a connection info, checked wherever one
  // enters the process: from a peer's bytes or from Python's own fields.
  static void validate(const AbolethConnectionInfo& value) {
    if (value.address_length > sizeof(value.address))
      throw std::invalid_argument(
          "connection-info provider address length is invalid");
    if (value.provider == kAbolethProxyProviderEfaOfi &&
        value.address_length == 0)
      throw std::invalid_argument(
          "EFA connection info requires a provider address");
  }

  AbolethConnectionInfo value_{};
};

class NativeRegisteredBuffer {
public:
  explicit NativeRegisteredBuffer(void* handle) : handle_(handle) {}

  NativeRegisteredBuffer(const NativeRegisteredBuffer&) = delete;
  NativeRegisteredBuffer& operator=(const NativeRegisteredBuffer&) = delete;

  ~NativeRegisteredBuffer() { closeNoThrow(); }

  void close() {
    if (handle_ == nullptr)
      return;
    void* handle = handle_;
    handle_ = nullptr;
    if (aboleth_runtime_unregister_buffer(handle) != 0)
      throw lastError("failed to unregister buffer");
  }

  bool closed() const { return handle_ == nullptr; }

  std::uint64_t address() const {
    ensureOpen();
    return aboleth_runtime_buffer_address(handle_);
  }

  std::uint64_t length() const {
    ensureOpen();
    return aboleth_runtime_buffer_length(handle_);
  }

  std::uint32_t lkey() const {
    ensureOpen();
    return aboleth_runtime_buffer_lkey(handle_);
  }

  std::uint32_t rkey() const {
    ensureOpen();
    return aboleth_runtime_buffer_rkey(handle_);
  }

  int device() const {
    ensureOpen();
    return aboleth_runtime_buffer_device(handle_);
  }

  bool strictlyOrdered() const {
    ensureOpen();
    return aboleth_runtime_buffer_strictly_ordered(handle_) != 0;
  }

  std::uint64_t generation() const {
    ensureOpen();
    return aboleth_runtime_buffer_generation(handle_);
  }

private:
  void ensureOpen() const {
    if (handle_ == nullptr)
      throw std::runtime_error("native registered buffer is closed");
  }

  void closeNoThrow() noexcept {
    if (handle_ == nullptr)
      return;
    void* handle = handle_;
    handle_ = nullptr;
    aboleth_runtime_unregister_buffer(handle);
  }

  void* handle_{};
};

class NativeEndpointPair {
public:
  explicit NativeEndpointPair(void* handle) : handle_(handle) {}

  NativeEndpointPair(const NativeEndpointPair&) = delete;
  NativeEndpointPair& operator=(const NativeEndpointPair&) = delete;

  ~NativeEndpointPair() { closeNoThrow(); }

  py::tuple queueState() const {
    ensureOpen();
    std::array<std::uint64_t, 4> values{};
    if (aboleth_runtime_endpoint_pair_state(
            handle_, &values[0], &values[1], &values[2], &values[3]) != 0)
      throw lastError("failed to read endpoint-pair queue state");
    return py::make_tuple(values[0], values[1], values[2], values[3]);
  }

  py::tuple status() const {
    ensureOpen();
    std::array<std::uint64_t, 2> values{};
    if (aboleth_runtime_endpoint_pair_status(
            handle_, &values[0], &values[1]) != 0)
      throw lastError("failed to read endpoint-pair status");
    return py::make_tuple(values[0], values[1]);
  }

  py::tuple completionStateAddresses() const {
    ensureOpen();
    return py::make_tuple(
        aboleth_runtime_endpoint_pair_completion_state(handle_, 0),
        aboleth_runtime_endpoint_pair_completion_state(handle_, 1));
  }

  py::tuple completionState(int endpointIndex) const {
    ensureOpen();
    std::array<std::uint64_t, 4> values{};
    if (aboleth_runtime_endpoint_pair_completion_snapshot(
            handle_, endpointIndex, values.data(), values.size()) != 0)
      throw lastError("failed to snapshot endpoint completion state");
    return py::make_tuple(values[0], values[1], values[2], values[3]);
  }

  void close() {
    if (handle_ == nullptr)
      return;
    void* handle = handle_;
    handle_ = nullptr;
    if (aboleth_runtime_destroy_endpoint_pair(handle) != 0)
      throw lastError("failed to destroy endpoint pair");
  }

  bool closed() const { return handle_ == nullptr; }

private:
  void ensureOpen() const {
    if (handle_ == nullptr)
      throw std::runtime_error("native endpoint pair is closed");
  }

  void closeNoThrow() noexcept {
    if (handle_ == nullptr)
      return;
    void* handle = handle_;
    handle_ = nullptr;
    aboleth_runtime_destroy_endpoint_pair(handle);
  }

  void* handle_{};
};

class NativeLocalEndpoint {
public:
  explicit NativeLocalEndpoint(void* handle) : handle_(handle) {}

  NativeLocalEndpoint(const NativeLocalEndpoint&) = delete;
  NativeLocalEndpoint& operator=(const NativeLocalEndpoint&) = delete;

  ~NativeLocalEndpoint() { closeNoThrow(); }

  NativeConnectionInfo connectionInfo() const {
    ensureOpen();
    AbolethConnectionInfo info{};
    if (aboleth_runtime_local_endpoint_connection_info(
            handle_, &info) != 0)
      throw lastError(
          "failed to read local endpoint connection info");
    return NativeConnectionInfo(info);
  }

  void connect(const NativeConnectionInfo& remote) {
    ensureOpen();
    if (aboleth_runtime_connect_local_endpoint(
            handle_, &remote.value()) != 0)
      throw lastError("failed to connect local endpoint");
  }

  py::tuple state() const {
    ensureOpen();
    std::array<std::uint64_t, 3> values{};
    if (aboleth_runtime_local_endpoint_state(
            handle_, &values[0], &values[1], &values[2]) != 0)
      throw lastError("failed to read local endpoint state");
    return py::make_tuple(values[0], values[1], values[2]);
  }

  std::uint32_t capacity() const {
    ensureOpen();
    std::uint32_t value{};
    if (aboleth_runtime_local_endpoint_capacity(
            handle_, &value) != 0)
      throw lastError("failed to read local endpoint capacity");
    return value;
  }

  std::uint32_t maxInlineBytes() const {
    ensureOpen();
    std::uint32_t value{};
    if (aboleth_runtime_local_endpoint_max_inline(
            handle_, &value) != 0)
      throw lastError("failed to read local endpoint max inline");
    return value;
  }

  bool gpuResidentQueues() const {
    ensureOpen();
    std::uint32_t value{};
    if (aboleth_runtime_local_endpoint_gpu_resident_queues(
            handle_, &value) != 0)
      throw lastError("failed to read local endpoint queue placement");
    return value != 0;
  }

  std::string submitMode() const {
    ensureOpen();
    std::uint32_t value{};
    if (aboleth_runtime_local_endpoint_submit_mode(
            handle_, &value) != 0)
      throw lastError("failed to read local endpoint submit mode");
    switch (value) {
    case 0:
      return "direct";
    case 1:
      return "staged";
    default:
      return "proxy";
    }
  }

  void close() {
    if (handle_ == nullptr)
      return;
    void* handle = handle_;
    handle_ = nullptr;
    if (aboleth_runtime_destroy_local_endpoint(handle) != 0)
      throw lastError("failed to destroy local endpoint");
  }

  bool closed() const { return handle_ == nullptr; }

private:
  void ensureOpen() const {
    if (handle_ == nullptr)
      throw std::runtime_error("native local endpoint is closed");
  }

  void closeNoThrow() noexcept {
    if (handle_ == nullptr)
      return;
    void* handle = handle_;
    handle_ = nullptr;
    aboleth_runtime_destroy_local_endpoint(handle);
  }

  void* handle_{};
};

class NativeRuntime {
public:
  explicit NativeRuntime(int gpuIndex = -1,
                         const std::string& hcaName = {}) {
    handle_ = gpuIndex >= 0
                  ? aboleth_runtime_create_for_device(
                        gpuIndex,
                        hcaName.empty() ? nullptr : hcaName.c_str())
                  : aboleth_runtime_create();
    if (handle_ == nullptr)
      throw lastError("failed to create Aboleth runtime");
  }

  NativeRuntime(const NativeRuntime&) = delete;
  NativeRuntime& operator=(const NativeRuntime&) = delete;

  ~NativeRuntime() { closeNoThrow(); }

  std::string hcaName() const {
    ensureOpen();
    return aboleth_runtime_hca_name(handle_);
  }

  int gpuIndex() const {
    ensureOpen();
    return aboleth_runtime_gpu_index(handle_);
  }

  int port() const {
    ensureOpen();
    return aboleth_runtime_port(handle_);
  }

  int gidIndex() const {
    ensureOpen();
    return aboleth_runtime_gid_index(handle_);
  }

  int linkLayer() const {
    ensureOpen();
    return aboleth_runtime_link_layer(handle_);
  }

  int lid() const {
    ensureOpen();
    return aboleth_runtime_lid(handle_);
  }

  std::shared_ptr<NativeRegisteredBuffer>
  registerBuffer(int device, std::uintptr_t address, std::size_t bytes,
                 std::uint64_t generation) {
    ensureOpen();
    void* buffer = aboleth_runtime_register_buffer(
        handle_, device, reinterpret_cast<void*>(address), bytes, generation);
    if (buffer == nullptr)
      throw lastError("failed to register GPU buffer");
    return std::make_shared<NativeRegisteredBuffer>(buffer);
  }

  // The honorable subset of AbolethEndpointConfig. The fields this runtime
  // only refuses — qp_type, qps_per_peer, sm_map_policy — have no Python
  // spelling until there is something for them to select; see
  // docs/transport-control.md.
  std::shared_ptr<NativeLocalEndpoint>
  createLocalEndpoint(std::uint32_t transport, int device,
                      std::uintptr_t descriptor,
                      std::size_t descriptorBytes,
                      std::uint32_t qpDepth, std::uint32_t maxInline,
                      const std::string& hca, std::int32_t port,
                      std::int32_t gidIndex) {
    ensureOpen();
    AbolethEndpointConfig config{};
    config.struct_size = sizeof(AbolethEndpointConfig);
    config.qp_depth = qpDepth;
    config.max_inline = maxInline;
    config.port = port;
    config.gid_index = gidIndex;
    if (hca.size() >= sizeof(config.hca))
      throw std::invalid_argument(
          "endpoint hca name exceeds 63 characters");
    std::memcpy(config.hca, hca.data(), hca.size());
    void* endpoint = aboleth_runtime_create_local_endpoint(
        handle_, transport, device,
        reinterpret_cast<void*>(descriptor), descriptorBytes, &config);
    if (endpoint == nullptr)
      throw lastError("failed to create local endpoint");
    return std::make_shared<NativeLocalEndpoint>(endpoint);
  }

  std::shared_ptr<NativeEndpointPair>
  createEndpointPair(int device0, std::uintptr_t descriptor0,
                     std::size_t descriptor0Bytes, int device1,
                     std::uintptr_t descriptor1,
                     std::size_t descriptor1Bytes) {
    ensureOpen();
    void* pair = aboleth_runtime_create_endpoint_pair(
        handle_, device0, reinterpret_cast<void*>(descriptor0),
        descriptor0Bytes, device1, reinterpret_cast<void*>(descriptor1),
        descriptor1Bytes, nullptr, nullptr);
    if (pair == nullptr)
      throw lastError("failed to create endpoint pair");
    return std::make_shared<NativeEndpointPair>(pair);
  }

  std::shared_ptr<NativeEndpointPair>
  createEfaEndpointPair(int device0, std::uintptr_t descriptor0,
                        std::size_t descriptor0Bytes, int device1,
                        std::uintptr_t descriptor1,
                        std::size_t descriptor1Bytes) {
    ensureOpen();
    void* pair = aboleth_runtime_create_efa_endpoint_pair(
        handle_, device0, reinterpret_cast<void*>(descriptor0),
        descriptor0Bytes, device1, reinterpret_cast<void*>(descriptor1),
        descriptor1Bytes);
    if (pair == nullptr)
      throw lastError("failed to create EFA endpoint pair");
    return std::make_shared<NativeEndpointPair>(pair);
  }

  std::shared_ptr<NativeEndpointPair>
  createProxyEndpointPair(int device0, std::uintptr_t descriptor0,
                          std::size_t descriptor0Bytes, int device1,
                          std::uintptr_t descriptor1,
                          std::size_t descriptor1Bytes) {
    ensureOpen();
    void* pair = aboleth_runtime_create_proxy_endpoint_pair(
        handle_, device0, reinterpret_cast<void*>(descriptor0),
        descriptor0Bytes, device1, reinterpret_cast<void*>(descriptor1),
        descriptor1Bytes);
    if (pair == nullptr)
      throw lastError("failed to create proxy endpoint pair");
    return std::make_shared<NativeEndpointPair>(pair);
  }

  void close() {
    if (handle_ == nullptr)
      return;
    void* handle = handle_;
    handle_ = nullptr;
    if (aboleth_runtime_destroy(handle) != 0)
      throw lastError("failed to destroy Aboleth runtime");
  }

  bool closed() const { return handle_ == nullptr; }

private:
  void ensureOpen() const {
    if (handle_ == nullptr)
      throw std::runtime_error("native runtime is closed");
  }

  void closeNoThrow() noexcept {
    if (handle_ == nullptr)
      return;
    void* handle = handle_;
    handle_ = nullptr;
    aboleth_runtime_destroy(handle);
  }

  void* handle_{};
};

} // namespace

PYBIND11_MODULE(_runtime, module) {
#if ABOLETH_HAS_HIP
  module.def("hip_allocate", &hipAllocate,
             "Fine-grained HIP storage as an owning DLPack capsule.");
#endif
  module.doc() = "Native ownership layer for Aboleth endpoints";
  module.attr("TRANSPORT_DIRECT") = kAbolethTransportDirect;
  module.attr("TRANSPORT_PROXY") = kAbolethTransportProxy;
  module.attr("TRANSPORT_EFA") = kAbolethTransportEfa;
  module.attr("TRANSPORT_IONIC") = kAbolethTransportIonic;
  module.attr("CONNECTION_INFO_BYTES") =
      static_cast<std::uint32_t>(sizeof(AbolethConnectionInfo));
  // What a caller building an AbolethEndpointConfig by hand — through ctypes,
  // or from another language — has to put in its `struct_size`.
  module.attr("ENDPOINT_CONFIG_BYTES") =
      static_cast<std::uint32_t>(sizeof(AbolethEndpointConfig));

  module.attr("PROVIDER_VERBS") = kAbolethProviderVerbs;
  module.attr("PROVIDER_MLX5") = kAbolethProviderMlx5;
  module.attr("PROVIDER_EFA") = kAbolethProviderEfa;
  module.attr("PROVIDER_LIBFABRIC") = kAbolethProviderLibfabric;
  module.attr("PROVIDER_CUDA") = kAbolethProviderCuda;
  module.def(
      "provider_caps", &aboleth_runtime_provider_caps,
      "Bitmask of transport providers this build can actually use here.");

  // Target/TransportCaps.h. `TRANSPORT_CAPS` is the registry, one entry per
  // row, so a planner derives the facts it needs from the columns instead of
  // carrying a list of transport names that the next row invalidates.
  module.attr("TRANSPORT_CAPS") = transportCapsRegistry();
  module.attr("COMPLETION_MODEL_NONE") = static_cast<std::uint32_t>(
      aboleth::abi::TransportCompletionModel::None);
  module.attr("COMPLETION_MODEL_ORDERED_COUNTER") =
      static_cast<std::uint32_t>(
          aboleth::abi::TransportCompletionModel::OrderedCounter);
  module.attr("COMPLETION_MODEL_UNORDERED_COUNT") =
      static_cast<std::uint32_t>(
          aboleth::abi::TransportCompletionModel::UnorderedCount);
  module.attr("DOORBELL_MODEL_NONE") = static_cast<std::uint32_t>(
      aboleth::abi::TransportDoorbellModel::None);
  module.attr("DOORBELL_MODEL_BLUE_FLAME") = static_cast<std::uint32_t>(
      aboleth::abi::TransportDoorbellModel::BlueFlame);
  module.attr("DOORBELL_MODEL_HOST_RING") = static_cast<std::uint32_t>(
      aboleth::abi::TransportDoorbellModel::HostRing);
  module.attr("DOORBELL_MODEL_DOORBELL") = static_cast<std::uint32_t>(
      aboleth::abi::TransportDoorbellModel::Doorbell);

  // Target/TransportPlan.h: the phase-plan ABI a producer of rows has to
  // agree with, including the struct's own size.
  module.attr("PHASE_PLAN_ABI_VERSION") =
      aboleth::abi::kTransportPhasePlanVersion;
  module.attr("PHASE_PLAN_BYTE_SIZE") = static_cast<std::uint32_t>(
      sizeof(aboleth::abi::TransportPhasePlan));
  module.attr("TRANSPORT_NO_PHASE") = aboleth::abi::kTransportNoPhase;
  module.attr("PLAN_FIRST_USE") = aboleth::abi::kTransportPlanFirstUse;
  module.attr("PLAN_REUSE_SAFE") = aboleth::abi::kTransportPlanReuseSafe;
  module.attr("PLAN_QP_EXCLUSIVE") =
      aboleth::abi::kTransportPlanQpExclusive;
  module.attr("PLAN_DRAIN_AT_EPILOGUE") =
      aboleth::abi::kTransportPlanDrainAtEpilogue;
  module.attr("PLAN_GROUPED_COMPLETION") =
      aboleth::abi::kTransportPlanGroupedCompletion;
  module.attr("PLAN_COMPLETION_LEADER") =
      aboleth::abi::kTransportPlanCompletionLeader;
  module.attr("TOPOLOGY_ALL_NIC") = static_cast<std::uint32_t>(
      aboleth::abi::TransportTopologyKind::AllNIC);
  module.attr("TOPOLOGY_MIXED") = static_cast<std::uint32_t>(
      aboleth::abi::TransportTopologyKind::Mixed);
  module.attr("TOPOLOGY_ALL_P2P") = static_cast<std::uint32_t>(
      aboleth::abi::TransportTopologyKind::AllP2P);
  module.attr("QP0_ROLE_DATA") = static_cast<std::uint32_t>(
      aboleth::abi::TransportQp0Role::Data);
  module.attr("QP0_ROLE_CONTROL") = static_cast<std::uint32_t>(
      aboleth::abi::TransportQp0Role::Control);
  module.attr("PUBLICATION_SAME_QUEUE_STRONG") = static_cast<std::uint32_t>(
      aboleth::abi::TransportPublicationMode::SameQueueStrong);
  module.attr("PUBLICATION_P2P_RELEASE") = static_cast<std::uint32_t>(
      aboleth::abi::TransportPublicationMode::P2PRelease);
  module.attr("PUBLICATION_GIN_SIGNAL") = static_cast<std::uint32_t>(
      aboleth::abi::TransportPublicationMode::GinSignal);
  module.attr("PUBLICATION_FENCE_THEN_SIGNAL") = static_cast<std::uint32_t>(
      aboleth::abi::TransportPublicationMode::FenceThenSignal);
  module.attr("PUBLICATION_IMM_COUNTING") = static_cast<std::uint32_t>(
      aboleth::abi::TransportPublicationMode::ImmCounting);
  module.attr("PUBLICATION_SAME_QUEUE_FENCED_WRITE") =
      static_cast<std::uint32_t>(
          aboleth::abi::TransportPublicationMode::SameQueueFencedWrite);
  module.attr("BARRIER_NONE") = static_cast<std::uint32_t>(
      aboleth::abi::TransportBarrierKind::None);
  module.attr("BARRIER_TRANSPORT") = static_cast<std::uint32_t>(
      aboleth::abi::TransportBarrierKind::Transport);
  module.attr("BARRIER_NATIVE_P2P") = static_cast<std::uint32_t>(
      aboleth::abi::TransportBarrierKind::NativeP2P);

  py::class_<NativeConnectionInfo>(
      module, "NativeConnectionInfo")
      .def(py::init<std::uint32_t, std::uint32_t, std::uint32_t,
                    std::uint32_t, std::uint32_t, std::uint32_t,
                    std::uint16_t, std::uint8_t, std::uint8_t,
                    const py::bytes&, std::uint32_t,
                    std::uint32_t, const py::bytes&>(),
           py::arg("abi_version"), py::arg("transport"),
           py::arg("qpn"), py::arg("psn"),
           py::arg("gid_index"), py::arg("path_mtu"),
           py::arg("lid"), py::arg("port_num"),
           py::arg("link_layer"), py::arg("gid"),
           py::arg("provider"), py::arg("address_format"),
           py::arg("address"))
      .def_property_readonly(
          "abi_version", &NativeConnectionInfo::abiVersion)
      .def_property_readonly(
          "transport", &NativeConnectionInfo::transport)
      .def_property_readonly("qpn", &NativeConnectionInfo::qpn)
      .def_property_readonly("psn", &NativeConnectionInfo::psn)
      .def_property_readonly(
          "gid_index", &NativeConnectionInfo::gidIndex)
      .def_property_readonly(
          "path_mtu", &NativeConnectionInfo::pathMtu)
      .def_property_readonly("lid", &NativeConnectionInfo::lid)
      .def_property_readonly(
          "port_num", &NativeConnectionInfo::portNum)
      .def_property_readonly(
          "link_layer", &NativeConnectionInfo::linkLayer)
      .def_property_readonly("gid", &NativeConnectionInfo::gid)
      .def_property_readonly(
          "provider", &NativeConnectionInfo::provider)
      .def_property_readonly(
          "address_format", &NativeConnectionInfo::addressFormat)
      .def_property_readonly(
          "address", &NativeConnectionInfo::address)
      .def("to_bytes", &NativeConnectionInfo::toBytes)
      .def_static(
          "from_bytes", &NativeConnectionInfo::fromBytes,
          py::arg("data"));

  py::class_<NativeRegisteredBuffer,
             std::shared_ptr<NativeRegisteredBuffer>>(
      module, "NativeRegisteredBuffer")
      .def("close", &NativeRegisteredBuffer::close)
      .def_property_readonly("closed", &NativeRegisteredBuffer::closed)
      .def_property_readonly("address", &NativeRegisteredBuffer::address)
      .def_property_readonly("length", &NativeRegisteredBuffer::length)
      .def_property_readonly("lkey", &NativeRegisteredBuffer::lkey)
      .def_property_readonly("rkey", &NativeRegisteredBuffer::rkey)
      .def_property_readonly("device", &NativeRegisteredBuffer::device)
      .def_property_readonly("strictly_ordered",
                             &NativeRegisteredBuffer::strictlyOrdered)
      .def_property_readonly("generation",
                             &NativeRegisteredBuffer::generation);

  py::class_<NativeEndpointPair, std::shared_ptr<NativeEndpointPair>>(
      module, "NativeEndpointPair")
      .def("queue_state", &NativeEndpointPair::queueState)
      .def("status", &NativeEndpointPair::status)
      .def("completion_state_addresses",
           &NativeEndpointPair::completionStateAddresses)
      .def("completion_state", &NativeEndpointPair::completionState)
      .def("close", &NativeEndpointPair::close)
      .def_property_readonly("closed", &NativeEndpointPair::closed);

  py::class_<NativeLocalEndpoint,
             std::shared_ptr<NativeLocalEndpoint>>(
      module, "NativeLocalEndpoint")
      .def("connection_info",
           &NativeLocalEndpoint::connectionInfo)
      .def("connect", &NativeLocalEndpoint::connect)
      .def("state", &NativeLocalEndpoint::state)
      .def_property_readonly(
          "capacity", &NativeLocalEndpoint::capacity)
      .def_property_readonly(
          "max_inline_bytes", &NativeLocalEndpoint::maxInlineBytes)
      .def_property_readonly(
          "submit_mode", &NativeLocalEndpoint::submitMode)
      .def_property_readonly(
          "gpu_resident_queues", &NativeLocalEndpoint::gpuResidentQueues)
      .def("close", &NativeLocalEndpoint::close)
      .def_property_readonly(
          "closed", &NativeLocalEndpoint::closed);

  py::class_<NativeRuntime, std::shared_ptr<NativeRuntime>>(
      module, "NativeRuntime")
      .def(py::init<int, const std::string&>(),
           py::arg("gpu_index") = -1,
           py::arg("hca_name") = "")
      .def_property_readonly("hca_name", &NativeRuntime::hcaName)
      .def_property_readonly("gpu_index", &NativeRuntime::gpuIndex)
      .def_property_readonly("port", &NativeRuntime::port)
      .def_property_readonly("gid_index", &NativeRuntime::gidIndex)
      .def_property_readonly("link_layer", &NativeRuntime::linkLayer)
      .def_property_readonly("lid", &NativeRuntime::lid)
      .def("register_buffer", &NativeRuntime::registerBuffer,
           py::arg("device"), py::arg("address"), py::arg("bytes"),
           py::arg("generation"))
      .def("create_local_endpoint",
           &NativeRuntime::createLocalEndpoint,
           py::arg("transport"), py::arg("device"),
           py::arg("descriptor"),
           py::arg("descriptor_bytes"),
           py::arg("qp_depth") = 0,
           py::arg("max_inline") = 0,
           py::arg("hca") = "",
           py::arg("port") = 0,
           py::arg("gid_index") = -1)
      .def("create_endpoint_pair", &NativeRuntime::createEndpointPair,
           py::arg("device0"), py::arg("descriptor0"),
           py::arg("descriptor0_bytes"), py::arg("device1"),
           py::arg("descriptor1"), py::arg("descriptor1_bytes"))
      .def("create_proxy_endpoint_pair",
           &NativeRuntime::createProxyEndpointPair,
           py::arg("device0"), py::arg("descriptor0"),
           py::arg("descriptor0_bytes"), py::arg("device1"),
           py::arg("descriptor1"), py::arg("descriptor1_bytes"))
      .def("create_efa_endpoint_pair",
           &NativeRuntime::createEfaEndpointPair,
           py::arg("device0"), py::arg("descriptor0"),
           py::arg("descriptor0_bytes"), py::arg("device1"),
           py::arg("descriptor1"), py::arg("descriptor1_bytes"))
      .def("close", &NativeRuntime::close)
      .def_property_readonly("closed", &NativeRuntime::closed);
}
