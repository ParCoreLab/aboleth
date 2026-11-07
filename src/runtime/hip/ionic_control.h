#pragma once

#include "aboleth/Target/IonicAbi.h"
#include "vendor/ionic_dv.h"
#include <dlfcn.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace aboleth::hip_ionic {
inline void check(bool condition, const char *what) {
  if (!condition) throw std::runtime_error(what);
}
inline void check_errno(bool condition, const char *what) {
  if (condition) return;
  const int code = errno;
  throw std::runtime_error(std::string(what) + ": errno=" +
      std::to_string(code) + " (" + std::strerror(code) + ")");
}
inline void check_verbs(int code, const char *what) {
  if (!code) return;
  const int saved_errno = errno;
  throw std::runtime_error(std::string(what) + ": rc=" +
      std::to_string(code) + " (" + std::strerror(code < 0 ? -code : code) +
      "), errno=" + std::to_string(saved_errno));
}
inline void hip_check(hipError_t rc) {
  if (rc != hipSuccess) throw std::runtime_error(hipGetErrorString(rc));
}
template<class T> T symbol(void *library, const char *name) {
  auto function = reinterpret_cast<T>(dlsym(library, name));
  if (!function) throw std::runtime_error(std::string("Ionic symbol missing: ") + name);
  return function;
}
struct DirectVerbs {
  void *library;
  decltype(&ionic_dv_is_ionic_ctx) is_ionic;
  decltype(&ionic_dv_get_ctx) get_ctx;
  decltype(&ionic_dv_get_qp) get_qp;
  decltype(&ionic_dv_get_cq) get_cq;
  decltype(&ionic_dv_qp_get_udma_idx) udma_index;
  decltype(&ionic_dv_pd_set_sqcmb) sqcmb;
  decltype(&ionic_dv_pd_set_rqcmb) rqcmb;
  decltype(&ionic_dv_pd_set_udma_mask) udma_mask;
  decltype(&ionic_dv_create_cq_ex) create_cq;
  DirectVerbs() {
    library = dlopen("libionic.so", RTLD_NOW | RTLD_LOCAL);
    check(library, "Ionic direct verbs library unavailable");
#define IONIC_LOAD(field, name) field = symbol<decltype(field)>(library, name)
    IONIC_LOAD(is_ionic, "ionic_dv_is_ionic_ctx");
    IONIC_LOAD(get_ctx, "ionic_dv_get_ctx");
    IONIC_LOAD(get_qp, "ionic_dv_get_qp");
    IONIC_LOAD(get_cq, "ionic_dv_get_cq");
    IONIC_LOAD(udma_index, "ionic_dv_qp_get_udma_idx");
    IONIC_LOAD(sqcmb, "ionic_dv_pd_set_sqcmb");
    IONIC_LOAD(rqcmb, "ionic_dv_pd_set_rqcmb");
    IONIC_LOAD(udma_mask, "ionic_dv_pd_set_udma_mask");
    IONIC_LOAD(create_cq, "ionic_dv_create_cq_ex");
#undef IONIC_LOAD
    // The provider remains loaded for the process lifetime, like libibverbs.
  }
};
inline DirectVerbs &dv() { static DirectVerbs symbols; return symbols; }
inline bool available() noexcept {
  try {
    (void)dv();
    int count = 0;
    auto **devices = dl::verbs().ibv_get_device_list(&count);
    if (!devices) return false;
    bool found = false;
    for (int i = 0; i < count; ++i) {
      const char *name = dl::verbs().ibv_get_device_name(devices[i]);
      std::ifstream file(std::string("/sys/class/infiniband/") + name + "/device/vendor");
      unsigned vendor = 0; file >> std::hex >> vendor;
      found |= vendor == 0x1dd8;
    }
    dl::verbs().ibv_free_device_list(devices);
    return found;
  } catch (...) { return false; }
}


struct Hsa {
  void *library;
  decltype(&hsa_iterate_agents) agents;
  decltype(&hsa_agent_get_info) info;
  decltype(&hsa_amd_agent_iterate_memory_pools) pools;
  decltype(&hsa_amd_memory_pool_get_info) pool_info;
  decltype(&hsa_amd_memory_lock_to_pool) lock;
  decltype(&hsa_amd_memory_unlock) unlock;
  Hsa() {
    library = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_LOCAL);
    check(library, "HSA runtime unavailable for Ionic doorbell mapping");
#define HSA_LOAD(field, name) field = symbol<decltype(field)>(library, name)
    HSA_LOAD(agents, "hsa_iterate_agents");
    HSA_LOAD(info, "hsa_agent_get_info");
    HSA_LOAD(pools, "hsa_amd_agent_iterate_memory_pools");
    HSA_LOAD(pool_info, "hsa_amd_memory_pool_get_info");
    HSA_LOAD(lock, "hsa_amd_memory_lock_to_pool");
    HSA_LOAD(unlock, "hsa_amd_memory_unlock");
#undef HSA_LOAD
  }
};
inline Hsa &hsa() { static Hsa functions; return functions; }

// One mapping per runtime. Match the HIP device by PCI identity, rather than
// assuming HIP_VISIBLE_DEVICES preserves HSA's enumeration order.
struct DoorbellMapping {
  void *host = nullptr, *gpu = nullptr;
  std::uint64_t *send = nullptr, *completion = nullptr;
  struct Search { unsigned bdf, domain; hsa_agent_t gpu{}; hsa_amd_memory_pool_t pool{}; };
  static hsa_status_t pool(hsa_amd_memory_pool_t candidate, void *data) {
    auto &search = *static_cast<Search *>(data);
    hsa_amd_segment_t segment{};
    auto rc = hsa().pool_info(candidate, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (rc != HSA_STATUS_SUCCESS) return rc;
    if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t flags = 0;
    rc = hsa().pool_info(candidate, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (rc == HSA_STATUS_SUCCESS &&
        (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
        (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)) search.pool = candidate;
    return rc;
  }
  static hsa_status_t agent(hsa_agent_t candidate, void *data) {
    auto &search = *static_cast<Search *>(data); hsa_device_type_t type{};
    auto rc = hsa().info(candidate, HSA_AGENT_INFO_DEVICE, &type);
    if (rc != HSA_STATUS_SUCCESS) return rc;
    if (type == HSA_DEVICE_TYPE_CPU) return hsa().pools(candidate, pool, data);
    if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
    uint32_t bdf = 0, domain = 0;
    rc = hsa().info(candidate, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &bdf);
    if (rc != HSA_STATUS_SUCCESS) return rc;
    rc = hsa().info(candidate, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &domain);
    if (rc == HSA_STATUS_SUCCESS && bdf == search.bdf && domain == search.domain)
      search.gpu = candidate;
    return rc;
  }
  DoorbellMapping(ibv_context *context, int device) {
    check(dv().is_ionic(context), "direct Ionic requested on a different provider");
    ionic_dv_ctx info{};
    check(!dv().get_ctx(&info, context), "ionic_dv_get_ctx failed");
    char bus[32]; hip_check(hipDeviceGetPCIBusId(bus, sizeof bus, device));
    unsigned domain = 0, number = 0, slot = 0, function = 0;
    check(std::sscanf(bus, "%x:%x:%x.%x", &domain, &number, &slot, &function) == 4,
          "invalid HIP PCI identity");
    Search search{(number << 8) | (slot << 3) | function, domain};
    check(hsa().agents(agent, &search) == HSA_STATUS_SUCCESS &&
          search.gpu.handle && search.pool.handle, "HSA GPU/fine-grained CPU pool not found");
    check(info.db_page && info.db_ptr &&
          reinterpret_cast<uintptr_t>(info.db_ptr) >= reinterpret_cast<uintptr_t>(info.db_page),
          "invalid Ionic doorbell page");
    auto offset = reinterpret_cast<uintptr_t>(info.db_ptr) - reinterpret_cast<uintptr_t>(info.db_page);
    check(offset + (std::max(info.sq_qtype, info.cq_qtype) + 1u) * 8 <= 4096,
          "Ionic doorbell registers exceed mapped page");
    check(hsa().lock(info.db_page, 4096, &search.gpu, 1, search.pool, 0, &gpu) == HSA_STATUS_SUCCESS,
          "HSA Ionic MMIO mapping failed");
    host = info.db_page;
    auto *base = reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(gpu) + offset);
    send = base + info.sq_qtype; completion = base + info.cq_qtype;
  }
  ~DoorbellMapping() { if (host) hsa().unlock(host); }
};

struct QueueResources {
  ibv_pd *parent = nullptr;
  abi::IonicQueueState *state = nullptr;
  void *scratch = nullptr;
  ibv_mr *scratch_mr = nullptr;
  abi::IonicEndpointDescriptor binding{};
  void *device_binding = nullptr;
  uint32_t capacity = 0;
  static void *allocate(ibv_pd *, void *context, size_t size, size_t alignment, uint64_t) {
    // Provider requires GPU queue storage. Never throw through a C callback.
    if (!context || hipSetDevice(*static_cast<int *>(context)) != hipSuccess) return nullptr;
    if (alignment > 4096) return nullptr;
    void *pointer = nullptr;
    if (hipExtMallocWithFlags(&pointer, size, hipDeviceMallocUncached) != hipSuccess) return nullptr;
    if (hipMemset(pointer, 0, size) != hipSuccess || hipStreamSynchronize(nullptr) != hipSuccess) {
      hipFree(pointer); return nullptr;
    }
    return pointer;
  }
  static void release(ibv_pd *, void *context, void *pointer, uint64_t) {
    if (context && hipSetDevice(*static_cast<int *>(context)) == hipSuccess) hipFree(pointer);
  }
  int device;
  explicit QueueResources(int gpu) : device(gpu) {}
  ~QueueResources() {
    hipSetDevice(device);
    if (parent) dl::verbs().ibv_dealloc_pd(parent);
    if (scratch_mr) dl::verbs().ibv_dereg_mr(scratch_mr);
    if (scratch) hipFree(scratch);
    if (state) hipFree(state);
  }
  // QP and CQ are owned by the enclosing endpoint and must be destroyed first.
  void create(ibv_context *context, ibv_pd *pd, DoorbellMapping &mapping,
              uint32_t depth, ibv_cq *&cq, ibv_qp *&qp,
              void *descriptor, size_t descriptor_bytes) {
    check(descriptor_bytes >= sizeof(abi::IonicEndpointDescriptor), "Ionic descriptor too small");
    check(depth >= 2 && depth <= 32768 && !(depth & (depth - 1)), "Ionic depth must be a power of two in [2,32768]");
    ibv_parent_domain_init_attr attr{};
    attr.pd = pd; attr.comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS | IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT;
    attr.alloc = allocate; attr.free = release; attr.pd_context = &device;
    parent = ibv_alloc_parent_domain(context, &attr);
    check(parent, "Ionic GPU parent domain creation failed");
    check(!dv().sqcmb(parent, false, false, false) &&
          !dv().rqcmb(parent, false, false, false) && !dv().udma_mask(parent, 1),
          "Ionic GPU queue mode configuration failed");
    ibv_cq_init_attr_ex ci{}; ci.cqe = depth * 2;
    ci.comp_mask = IBV_CQ_INIT_ATTR_MASK_PD; ci.parent_domain = parent;
    ionic_cq_init_attr_ex ii{IONIC_CQ_INIT_ATTR_MASK_FLAGS, IONIC_CQ_INIT_ATTR_CCQE};
    errno = 0;
    auto *extended = dv().create_cq(context, &ci, &ii);
    check_errno(extended, "Ionic continuous CQ creation failed");
    cq = ibv_cq_ex_to_cq(extended);
    ibv_qp_init_attr_ex qi{}; qi.qp_type = IBV_QPT_RC;
    qi.pd = parent; qi.comp_mask = IBV_QP_INIT_ATTR_PD;
    qi.send_cq = cq; qi.recv_cq = cq; qi.cap.max_send_wr = depth;
    qi.cap.max_send_sge = 1; qi.cap.max_recv_sge = 1; qi.cap.max_inline_data = 16;
    auto *extended_context = verbs_get_ctx_op(context, create_qp_ex);
    check(extended_context && extended_context->create_qp_ex,
          "Ionic provider has no extended QP creation entry");
    errno = 0;
    qp = extended_context->create_qp_ex(context, &qi);
    check_errno(qp, "Ionic GPU QP creation failed");
    ionic_dv_qp sq{}; ionic_dv_cq ccq{};
    check(!dv().get_qp(&sq, qp) && !dv().get_cq(&ccq, cq, dv().udma_index(qp)),
          "Ionic GPU queue export failed");
    check(!ccq.q.mask && sq.sq.mask && sq.sq.stride_log2 == 6 && ccq.q.stride_log2 == 5,
          "Ionic backend requires 64-byte WQEs and a continuous 32-byte CQE");
    check((sq.sq.mask & (sq.sq.mask + 1u)) == 0 &&
          sq.sq.size >= size_t(sq.sq.mask + 1u) * 64, "invalid Ionic send ring export");
    capacity = sq.sq.mask; // leave one entry empty; disambiguates wrap
    hip_check(hipExtMallocWithFlags(reinterpret_cast<void **>(&state), sizeof(*state), hipDeviceMallocUncached));
    hip_check(hipMemset(state, 0, sizeof(*state)));
    hip_check(hipExtMallocWithFlags(&scratch, 16, hipDeviceMallocUncached));
    hip_check(hipMemset(scratch, 0, 16));
    scratch_mr = dl::verbs().reg_mr(pd, scratch, 16,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    check(scratch_mr, "Ionic private scratch registration failed");
    binding = {
      reinterpret_cast<uint64_t>(sq.sq.ptr), reinterpret_cast<uint64_t>(ccq.q.ptr),
      reinterpret_cast<uint64_t>(mapping.send), reinterpret_cast<uint64_t>(mapping.completion),
      sq.sq.db_val, ccq.q.db_val, reinterpret_cast<uint64_t>(state), 3000000000ull,
      sq.sq.mask, ccq.q.mask, 64, 32, abi::kIonicEndpointVersion, 0, 0};
    device_binding = descriptor;
    hip_check(hipMemcpy(descriptor, &binding, sizeof binding, hipMemcpyHostToDevice));
  }
  void connect_scratch(uint64_t address, uint32_t key) {
    check(address && !(address & 7), "invalid Ionic private peer scratch");
    binding.scratch_address = address; binding.scratch_key = key;
    hip_check(hipMemcpy(device_binding, &binding, sizeof binding, hipMemcpyHostToDevice));
  }
};
} // namespace aboleth::hip_ionic
