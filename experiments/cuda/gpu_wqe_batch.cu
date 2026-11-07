#include "aboleth_probe_fabric.h"

#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// The GPU this probe runs its loopback on; the HCA follows from it.
constexpr int kGpuIndex = 0;
constexpr std::uint32_t kDefaultWqeCount = 32;
constexpr std::uint32_t kMaximumWqeCount = 256;
constexpr std::size_t kTransferBytes = 256;
constexpr std::uint32_t kCompletionTimeoutSeconds = 5;

enum class SubmissionMode : std::uint32_t {
    Batch = 0,
    PerWqe = 1,
};

enum class KernelStatus : std::uint32_t {
    Success = 0,
    CompletionTimeout = 1,
    CompletionError = 2,
    UnexpectedWqeCounter = 3,
};

struct BenchmarkResult {
    std::uint64_t build_cycles;
    std::uint64_t submit_cycles;
    std::uint64_t completion_cycles;
    std::uint64_t total_cycles;
    std::uint64_t poll_iterations;
    std::uint32_t status;
    std::uint32_t completion_opcode;
    std::uint32_t completion_owner;
    std::uint32_t completion_wqe_counter;
    std::uint32_t completion_syndrome;
    std::uint32_t completion_vendor_syndrome;
    std::uint32_t doorbell_writes;
};

[[noreturn]] void fail(const char* what) {
    std::perror(what);
    std::exit(EXIT_FAILURE);
}

[[noreturn]] void fail_message(const char* what) {
    std::fprintf(stderr, "%s\n", what);
    std::exit(EXIT_FAILURE);
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

struct RegisteredMapping {
    void* host_base{};
    std::size_t bytes{};
    void* device_base{};
    unsigned int flags{};
};

class MappingRegistry {
public:
    void* map(void* pointer, std::size_t bytes, unsigned int flags) {
        const auto page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        const auto start = reinterpret_cast<std::uintptr_t>(pointer);
        const auto base = start & ~(page_size - 1);
        const auto end = (start + bytes + page_size - 1) & ~(page_size - 1);

        for (const auto& mapping : mappings_) {
            const auto mapping_base =
                reinterpret_cast<std::uintptr_t>(mapping.host_base);
            const auto mapping_end = mapping_base + mapping.bytes;
            const bool overlaps = base < mapping_end && end > mapping_base;
            if (!overlaps) {
                continue;
            }
            if (flags == mapping.flags &&
                base >= mapping_base &&
                end <= mapping_end) {
                const auto offset = start - mapping_base;
                return reinterpret_cast<void*>(
                    reinterpret_cast<std::uintptr_t>(mapping.device_base) +
                    offset);
            }
            fail_message("Overlapping CUDA host registrations are incompatible");
        }

        RegisteredMapping mapping{
            reinterpret_cast<void*>(base),
            end - base,
            nullptr,
            flags,
        };
        check_cuda(
            cudaHostRegister(mapping.host_base, mapping.bytes, flags),
            "cudaHostRegister");
        check_cuda(
            cudaHostGetDevicePointer(
                &mapping.device_base,
                mapping.host_base,
                0),
            "cudaHostGetDevicePointer");
        mappings_.push_back(mapping);

        const auto offset = start - base;
        return reinterpret_cast<void*>(
            reinterpret_cast<std::uintptr_t>(mapping.device_base) + offset);
    }

    void unmap_all() {
        for (auto mapping = mappings_.rbegin();
             mapping != mappings_.rend();
             ++mapping) {
            check_cuda(
                cudaHostUnregister(mapping->host_base),
                "cudaHostUnregister");
        }
        mappings_.clear();
    }

private:
    std::vector<RegisteredMapping> mappings_;
};

ibv_qp* create_rc_qp(ibv_pd* pd, ibv_cq* cq) {
    ibv_qp_init_attr attr{};
    attr.send_cq = cq;
    attr.recv_cq = cq;
    attr.cap.max_send_wr = kMaximumWqeCount;
    attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_RC;

    ibv_qp* qp = ibv_create_qp(pd, &attr);
    if (qp == nullptr) {
        fail("ibv_create_qp");
    }
    return qp;
}

void qp_to_init(
    ibv_qp* qp,
    const aboleth::probe::FabricAddress& fabric) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = static_cast<std::uint8_t>(fabric.port);
    attr.pkey_index = 0;
    attr.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_ATOMIC;

    const int mask =
        IBV_QP_STATE |
        IBV_QP_PKEY_INDEX |
        IBV_QP_PORT |
        IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(qp, &attr, mask) != 0) {
        fail("ibv_modify_qp(INIT)");
    }
}

void qp_to_rtr(
    ibv_qp* qp,
    std::uint32_t remote_qpn,
    std::uint32_t remote_psn,
    const aboleth::probe::FabricAddress& fabric) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    // Both queue pairs live on this port, so the local address is also the
    // remote one.
    aboleth::probe::apply_address_vector(attr, fabric);

    const int mask =
        IBV_QP_STATE |
        IBV_QP_AV |
        IBV_QP_PATH_MTU |
        IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN |
        IBV_QP_MAX_DEST_RD_ATOMIC |
        IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(qp, &attr, mask) != 0) {
        fail("ibv_modify_qp(RTR)");
    }
}

void qp_to_rts(ibv_qp* qp, std::uint32_t local_psn) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn;
    attr.max_rd_atomic = 1;

    const int mask =
        IBV_QP_STATE |
        IBV_QP_TIMEOUT |
        IBV_QP_RETRY_CNT |
        IBV_QP_RNR_RETRY |
        IBV_QP_SQ_PSN |
        IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(qp, &attr, mask) != 0) {
        fail("ibv_modify_qp(RTS)");
    }
}

struct alignas(16) WqeCtrl {
    std::uint32_t opmod_idx_opcode;
    std::uint32_t qpn_ds;
    std::uint32_t signature_dci_fm;
    std::uint32_t imm;
};

struct alignas(16) WqeRemoteAddress {
    std::uint64_t address;
    std::uint32_t rkey;
    std::uint32_t reserved;
};

struct alignas(16) WqeData {
    std::uint32_t byte_count;
    std::uint32_t lkey;
    std::uint64_t address;
};

static_assert(sizeof(WqeCtrl) == 16);
static_assert(sizeof(WqeRemoteAddress) == 16);
static_assert(sizeof(WqeData) == 16);

__device__ __forceinline__ std::uint32_t to_be32(std::uint32_t value) {
    std::uint32_t result;
    asm(
        "{\n\t"
        ".reg .b32 ignored;\n\t"
        "prmt.b32 %0, %1, ignored, 0x0123;\n\t"
        "}"
        : "=r"(result)
        : "r"(value));
    return result;
}

__device__ __forceinline__ std::uint16_t from_be16(std::uint16_t value) {
    return static_cast<std::uint16_t>(
        (value >> 8) |
        (value << 8));
}

__device__ __forceinline__ std::uint64_t to_be64(std::uint64_t value) {
    std::uint64_t result;
    asm(
        "{\n\t"
        ".reg .b32 low;\n\t"
        ".reg .b32 high;\n\t"
        ".reg .b32 new_low;\n\t"
        ".reg .b32 new_high;\n\t"
        ".reg .b32 ignored;\n\t"
        "mov.b64 {low,high}, %1;\n\t"
        "prmt.b32 new_high, low, ignored, 0x0123;\n\t"
        "prmt.b32 new_low, high, ignored, 0x0123;\n\t"
        "mov.b64 %0, {new_low,new_high};\n\t"
        "}"
        : "=l"(result)
        : "l"(value));
    return result;
}

__device__ __forceinline__ void store_no_allocate(
    int4* pointer,
    const int4& value) {
    asm volatile(
        "st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(pointer),
          "r"(value.x),
          "r"(value.y),
          "r"(value.z),
          "r"(value.w)
        : "memory");
}

__device__ __forceinline__ void store_release(
    std::uint32_t* pointer,
    std::uint32_t value) {
    asm volatile(
        "st.release.sys.global.L1::no_allocate.b32 [%0], %1;"
        :
        : "l"(pointer), "r"(value)
        : "memory");
}

__device__ __forceinline__ void store_release(
    std::uint64_t* pointer,
    std::uint64_t value) {
    asm volatile(
        "st.release.sys.global.L1::no_allocate.b64 [%0], %1;"
        :
        : "l"(pointer), "l"(value)
        : "memory");
}

__device__ __forceinline__ std::uint8_t load_no_allocate(
    const std::uint8_t* pointer) {
    std::uint16_t value;
    asm volatile(
        "ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];"
        : "=h"(value)
        : "l"(pointer)
        : "memory");
    return static_cast<std::uint8_t>(value);
}

__device__ __forceinline__ std::uint16_t load_no_allocate(
    const std::uint16_t* pointer) {
    std::uint16_t value;
    asm volatile(
        "ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];"
        : "=h"(value)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ void ring_doorbell(
    std::uint32_t* send_dbrec,
    std::uint64_t* blueflame,
    std::uint32_t qpn,
    std::uint32_t producer_index) {
    store_release(send_dbrec, to_be32(producer_index & 0xffff));
    const std::uint32_t doorbell_index = to_be32(producer_index << 8);
    const std::uint32_t doorbell_qpn = to_be32(qpn << 8);
    const std::uint64_t doorbell =
        (static_cast<std::uint64_t>(doorbell_qpn) << 32) |
        doorbell_index;
    store_release(blueflame, doorbell);
}

__global__ void run_batch(
    std::uint8_t* send_queue,
    std::uint32_t send_stride,
    std::uint32_t send_wqe_count,
    std::uint32_t* send_dbrec,
    std::uint64_t* blueflame,
    std::uint32_t qpn,
    const std::uint8_t* source,
    std::uint32_t source_lkey,
    std::uint8_t* destination,
    std::uint32_t destination_rkey,
    std::uint32_t bytes_per_wqe,
    std::uint32_t wqe_count,
    SubmissionMode mode,
    std::uint8_t* completion_queue,
    std::uint32_t completion_entry_size,
    std::uint32_t* completion_dbrec,
    std::uint64_t completion_timeout_cycles,
    BenchmarkResult* result) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    constexpr std::uint32_t mlx5_opcode_rdma_write = 0x08;
    constexpr std::uint8_t mlx5_wqe_ctrl_cq_update = 2 << 2;
    constexpr std::uint32_t data_segments = 3;
    constexpr std::uint8_t mlx5_cqe_request = 0;
    constexpr std::uint8_t mlx5_cqe_request_error = 13;
    constexpr std::uint8_t mlx5_cqe_invalid = 15;

    const std::uint64_t start_cycles = clock64();

    for (std::uint32_t index = 0; index < wqe_count; ++index) {
        auto* wqe =
            send_queue +
            (index % send_wqe_count) * send_stride;
        auto* ctrl_pointer = reinterpret_cast<WqeCtrl*>(wqe);
        auto* remote_pointer =
            reinterpret_cast<WqeRemoteAddress*>(
                wqe + sizeof(WqeCtrl));
        auto* data_pointer =
            reinterpret_cast<WqeData*>(
                wqe +
                sizeof(WqeCtrl) +
                sizeof(WqeRemoteAddress));

        WqeCtrl ctrl{};
        ctrl.opmod_idx_opcode =
            to_be32((index << 8) | mlx5_opcode_rdma_write);
        ctrl.qpn_ds = to_be32((qpn << 8) | data_segments);
        if (index + 1 == wqe_count) {
            ctrl.signature_dci_fm =
                static_cast<std::uint32_t>(
                    mlx5_wqe_ctrl_cq_update) << 24;
        }

        WqeRemoteAddress remote{};
        remote.address = to_be64(
            reinterpret_cast<std::uint64_t>(
                destination +
                static_cast<std::size_t>(index) * bytes_per_wqe));
        remote.rkey = to_be32(destination_rkey);

        WqeData data{};
        data.byte_count = to_be32(bytes_per_wqe);
        data.lkey = to_be32(source_lkey);
        data.address = to_be64(
            reinterpret_cast<std::uint64_t>(
                source +
                static_cast<std::size_t>(index) * bytes_per_wqe));

        store_no_allocate(
            reinterpret_cast<int4*>(ctrl_pointer),
            *reinterpret_cast<const int4*>(&ctrl));
        store_no_allocate(
            reinterpret_cast<int4*>(remote_pointer),
            *reinterpret_cast<const int4*>(&remote));
        store_no_allocate(
            reinterpret_cast<int4*>(data_pointer),
            *reinterpret_cast<const int4*>(&data));
    }

    __threadfence_system();
    const std::uint64_t build_end_cycles = clock64();

    if (mode == SubmissionMode::Batch) {
        ring_doorbell(
            send_dbrec,
            blueflame,
            qpn,
            wqe_count);
    } else {
        for (std::uint32_t index = 1;
             index <= wqe_count;
             ++index) {
            ring_doorbell(
                send_dbrec,
                blueflame,
                qpn,
                index);
        }
    }

    const std::uint64_t submit_end_cycles = clock64();

    const std::size_t cqe64_offset =
        completion_entry_size == 128 ? 64 : 0;
    const auto* cqe =
        completion_queue + cqe64_offset;
    const auto* wqe_counter_pointer =
        reinterpret_cast<const std::uint16_t*>(cqe + 60);
    const auto* op_own_pointer =
        reinterpret_cast<const std::uint8_t*>(cqe + 63);

    std::uint64_t poll_iterations = 0;
    std::uint8_t op_own = 0xff;
    std::uint8_t opcode = mlx5_cqe_invalid;
    bool completion_ready = false;
    while (clock64() - submit_end_cycles <
           completion_timeout_cycles) {
        op_own = load_no_allocate(op_own_pointer);
        opcode = op_own >> 4;
        const std::uint8_t owner = op_own & 1;
        const std::uint8_t expected_owner = 0;
        ++poll_iterations;
        if (opcode != mlx5_cqe_invalid &&
            owner == expected_owner) {
            completion_ready = true;
            break;
        }
        __nanosleep(64);
    }

    const std::uint64_t completion_end_cycles = clock64();

    result->build_cycles =
        build_end_cycles - start_cycles;
    result->submit_cycles =
        submit_end_cycles - build_end_cycles;
    result->completion_cycles =
        completion_end_cycles - submit_end_cycles;
    result->total_cycles =
        completion_end_cycles - start_cycles;
    result->poll_iterations = poll_iterations;
    result->completion_opcode = opcode;
    result->completion_owner = op_own & 1;
    result->doorbell_writes =
        mode == SubmissionMode::Batch ? 1 : wqe_count;

    if (!completion_ready) {
        result->status =
            static_cast<std::uint32_t>(
                KernelStatus::CompletionTimeout);
        return;
    }

    __threadfence_system();
    const std::uint16_t completed_wqe =
        from_be16(load_no_allocate(wqe_counter_pointer));
    result->completion_wqe_counter = completed_wqe;

    if (opcode == mlx5_cqe_request_error ||
        opcode != mlx5_cqe_request) {
        result->completion_vendor_syndrome =
            load_no_allocate(cqe + 54);
        result->completion_syndrome =
            load_no_allocate(cqe + 55);
        result->status =
            static_cast<std::uint32_t>(
                KernelStatus::CompletionError);
        return;
    }

    if (completed_wqe !=
        static_cast<std::uint16_t>(wqe_count - 1)) {
        result->status =
            static_cast<std::uint32_t>(
                KernelStatus::UnexpectedWqeCounter);
        return;
    }

    store_release(completion_dbrec, to_be32(1));
    result->status =
        static_cast<std::uint32_t>(KernelStatus::Success);
}

const char* mode_name(SubmissionMode mode) {
    return mode == SubmissionMode::Batch
        ? "batch"
        : "per-wqe";
}

const char* status_name(KernelStatus status) {
    switch (status) {
        case KernelStatus::Success:
            return "success";
        case KernelStatus::CompletionTimeout:
            return "completion-timeout";
        case KernelStatus::CompletionError:
            return "completion-error";
        case KernelStatus::UnexpectedWqeCounter:
            return "unexpected-wqe-counter";
    }
    return "unknown";
}

SubmissionMode parse_mode(int argc, char** argv) {
    if (argc < 2) {
        return SubmissionMode::Batch;
    }
    const std::string argument = argv[1];
    if (argument == "batch") {
        return SubmissionMode::Batch;
    }
    if (argument == "per-wqe") {
        return SubmissionMode::PerWqe;
    }
    fail_message("Usage: gpu_wqe_batch [batch|per-wqe] [wqe-count]");
}

std::uint32_t parse_wqe_count(int argc, char** argv) {
    if (argc < 3) {
        return kDefaultWqeCount;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(argv[2], &end, 10);
    if (end == argv[2] ||
        *end != '\0' ||
        parsed == 0 ||
        parsed > kMaximumWqeCount) {
        fail_message("WQE count must be between 1 and 256");
    }
    return static_cast<std::uint32_t>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
    const SubmissionMode mode = parse_mode(argc, argv);
    const std::uint32_t wqe_count =
        parse_wqe_count(argc, argv);
    const std::size_t total_bytes =
        static_cast<std::size_t>(wqe_count) *
        kTransferBytes;

    check_cuda(
        cudaSetDeviceFlags(cudaDeviceMapHost),
        "cudaSetDeviceFlags");
    check_cuda(cudaSetDevice(kGpuIndex), "cudaSetDevice");

    cudaDeviceProp device_properties{};
    check_cuda(
        cudaGetDeviceProperties(&device_properties, kGpuIndex),
        "cudaGetDeviceProperties");
#if CUDART_VERSION >= 13000
    int device_clock_khz = 0;
    check_cuda(
        cudaDeviceGetAttribute(
            &device_clock_khz,
            cudaDevAttrClockRate,
            kGpuIndex),
        "cudaDeviceGetAttribute(clockRate)");
#else
    const int device_clock_khz = device_properties.clockRate;
#endif
    const std::uint64_t timeout_cycles =
        static_cast<std::uint64_t>(
            device_clock_khz) *
        1000ULL *
        kCompletionTimeoutSeconds;

    const aboleth::probe::FabricAddress fabric =
        aboleth::probe::resolve_fabric_address(kGpuIndex);

    int device_count = 0;
    ibv_device** devices =
        ibv_get_device_list(&device_count);
    if (devices == nullptr) {
        fail("ibv_get_device_list");
    }
    ibv_device* selected =
        aboleth::probe::find_fabric_device(
            devices,
            device_count,
            fabric);

    ibv_context* context = ibv_open_device(selected);
    if (context == nullptr) {
        fail("ibv_open_device");
    }

    ibv_pd* pd = ibv_alloc_pd(context);
    if (pd == nullptr) {
        fail("ibv_alloc_pd");
    }
    ibv_cq* send_cq =
        ibv_create_cq(
            context,
            kMaximumWqeCount,
            nullptr,
            nullptr,
            0);
    ibv_cq* peer_cq =
        ibv_create_cq(context, 1, nullptr, nullptr, 0);
    if (send_cq == nullptr || peer_cq == nullptr) {
        fail("ibv_create_cq");
    }

    ibv_qp* sender = create_rc_qp(pd, send_cq);
    ibv_qp* peer = create_rc_qp(pd, peer_cq);
    qp_to_init(sender, fabric);
    qp_to_init(peer, fabric);

    constexpr std::uint32_t sender_psn = 0x123456;
    constexpr std::uint32_t peer_psn = 0x654321;
    qp_to_rtr(sender, peer->qp_num, peer_psn, fabric);
    qp_to_rtr(peer, sender->qp_num, sender_psn, fabric);
    qp_to_rts(sender, sender_psn);
    qp_to_rts(peer, peer_psn);

    auto* source = static_cast<std::uint8_t*>(nullptr);
    auto* destination = static_cast<std::uint8_t*>(nullptr);
    check_cuda(
        cudaMalloc(&source, total_bytes),
        "cudaMalloc(source)");
    check_cuda(
        cudaMalloc(&destination, total_bytes),
        "cudaMalloc(destination)");

    std::vector<std::uint8_t> expected(total_bytes);
    for (std::size_t index = 0;
         index < expected.size();
         ++index) {
        expected[index] =
            static_cast<std::uint8_t>(
                (index * 17 + 3) & 0xff);
    }
    check_cuda(
        cudaMemcpy(
            source,
            expected.data(),
            expected.size(),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(source)");
    check_cuda(
        cudaMemset(destination, 0, total_bytes),
        "cudaMemset(destination)");

    ibv_mr* source_mr =
        ibv_reg_mr(
            pd,
            source,
            total_bytes,
            IBV_ACCESS_LOCAL_WRITE);
    ibv_mr* destination_mr =
        ibv_reg_mr(
            pd,
            destination,
            total_bytes,
            IBV_ACCESS_LOCAL_WRITE |
                IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ);
    if (source_mr == nullptr ||
        destination_mr == nullptr) {
        fail("ibv_reg_mr(GPU memory)");
    }

    mlx5dv_qp direct_qp{};
    mlx5dv_cq direct_cq{};
    mlx5dv_obj direct_object{};
    direct_object.qp.in = sender;
    direct_object.qp.out = &direct_qp;
    direct_object.cq.in = send_cq;
    direct_object.cq.out = &direct_cq;
    if (mlx5dv_init_obj(
            &direct_object,
            MLX5DV_OBJ_QP | MLX5DV_OBJ_CQ) != 0) {
        fail("mlx5dv_init_obj(QP|CQ)");
    }
    if (direct_qp.sq.buf == nullptr ||
        direct_qp.dbrec == nullptr ||
        direct_qp.bf.reg == nullptr ||
        direct_cq.buf == nullptr ||
        direct_cq.dbrec == nullptr) {
        fail_message(
            "Direct-verbs resources are incomplete");
    }
    if (wqe_count > direct_qp.sq.wqe_cnt) {
        fail_message(
            "Requested WQEs exceed the hardware send queue");
    }
    if (direct_cq.cqe_size != 64 &&
        direct_cq.cqe_size != 128) {
        fail_message("Unsupported CQE size");
    }

    const std::size_t send_queue_bytes =
        static_cast<std::size_t>(
            direct_qp.sq.wqe_cnt) *
        direct_qp.sq.stride;
    const std::size_t completion_queue_bytes =
        static_cast<std::size_t>(
            direct_cq.cqe_cnt) *
        direct_cq.cqe_size;

    MappingRegistry mappings;
    auto* device_send_queue =
        static_cast<std::uint8_t*>(
            mappings.map(
                direct_qp.sq.buf,
                send_queue_bytes,
                cudaHostRegisterMapped));
    auto* device_completion_queue =
        static_cast<std::uint8_t*>(
            mappings.map(
                direct_cq.buf,
                completion_queue_bytes,
                cudaHostRegisterMapped));
    auto* device_send_dbrec =
        static_cast<std::uint32_t*>(
            mappings.map(
                direct_qp.dbrec,
                2 * sizeof(std::uint32_t),
                cudaHostRegisterMapped)) +
        MLX5_SND_DBR;
    auto* device_completion_dbrec =
        static_cast<std::uint32_t*>(
            mappings.map(
                direct_cq.dbrec,
                2 * sizeof(std::uint32_t),
                cudaHostRegisterMapped));
    auto* device_blueflame =
        static_cast<std::uint64_t*>(
            mappings.map(
                direct_qp.bf.reg,
                sizeof(std::uint64_t),
                cudaHostRegisterMapped |
                    cudaHostRegisterIoMemory));

    BenchmarkResult* device_result = nullptr;
    check_cuda(
        cudaMalloc(&device_result, sizeof(BenchmarkResult)),
        "cudaMalloc(result)");
    check_cuda(
        cudaMemset(
            device_result,
            0,
            sizeof(BenchmarkResult)),
        "cudaMemset(result)");

    const std::size_t cqe64_offset =
        direct_cq.cqe_size == 128 ? 64 : 0;
    const auto initial_op_own =
        *(static_cast<volatile std::uint8_t*>(
            direct_cq.buf) +
          cqe64_offset +
          63);

    std::printf(
        "Device: %s\n",
        device_properties.name);
    aboleth::probe::report_fabric_address(fabric);
    std::printf(
        "Mode: %s, WQEs: %u, bytes/WQE: %zu, doorbells: %u\n",
        mode_name(mode),
        wqe_count,
        kTransferBytes,
        mode == SubmissionMode::Batch ? 1 : wqe_count);
    std::printf(
        "QP %u: SQ entries=%u stride=%u; CQ entries=%u size=%u initial op_own=0x%02x\n",
        sender->qp_num,
        direct_qp.sq.wqe_cnt,
        direct_qp.sq.stride,
        direct_cq.cqe_cnt,
        direct_cq.cqe_size,
        initial_op_own);

    run_batch<<<1, 1>>>(
        device_send_queue,
        direct_qp.sq.stride,
        direct_qp.sq.wqe_cnt,
        device_send_dbrec,
        device_blueflame,
        sender->qp_num,
        source,
        source_mr->lkey,
        destination,
        destination_mr->rkey,
        static_cast<std::uint32_t>(kTransferBytes),
        wqe_count,
        mode,
        device_completion_queue,
        direct_cq.cqe_size,
        device_completion_dbrec,
        timeout_cycles,
        device_result);
    check_cuda(
        cudaGetLastError(),
        "run_batch launch");
    check_cuda(
        cudaDeviceSynchronize(),
        "run_batch execution");

    BenchmarkResult result{};
    check_cuda(
        cudaMemcpy(
            &result,
            device_result,
            sizeof(result),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(result)");

    std::vector<std::uint8_t> observed(total_bytes);
    check_cuda(
        cudaMemcpy(
            observed.data(),
            destination,
            observed.size(),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(destination)");
    const bool payload_matches =
        observed == expected;
    const auto kernel_status =
        static_cast<KernelStatus>(result.status);
    const bool success =
        kernel_status == KernelStatus::Success &&
        payload_matches;

    const double cycles_per_microsecond =
        static_cast<double>(
            device_clock_khz) /
        1000.0;
    std::printf(
        "Completion: status=%s opcode=%u owner=%u wqe_counter=%u syndrome=0x%x vendor=0x%x polls=%llu\n",
        status_name(kernel_status),
        result.completion_opcode,
        result.completion_owner,
        result.completion_wqe_counter,
        result.completion_syndrome,
        result.completion_vendor_syndrome,
        static_cast<unsigned long long>(
            result.poll_iterations));
    std::printf(
        "Cycles: build=%llu submit=%llu completion=%llu total=%llu\n",
        static_cast<unsigned long long>(
            result.build_cycles),
        static_cast<unsigned long long>(
            result.submit_cycles),
        static_cast<unsigned long long>(
            result.completion_cycles),
        static_cast<unsigned long long>(
            result.total_cycles));
    std::printf(
        "Approx us at reported SM clock: build=%.3f submit=%.3f completion=%.3f total=%.3f\n",
        result.build_cycles / cycles_per_microsecond,
        result.submit_cycles / cycles_per_microsecond,
        result.completion_cycles / cycles_per_microsecond,
        result.total_cycles / cycles_per_microsecond);
    std::printf(
        "Payload: %s\n",
        payload_matches ? "MATCH" : "MISMATCH");
    std::printf(
        "GPU-polled RDMA batch: %s\n",
        success ? "PASS" : "FAIL");

    cudaFree(device_result);
    mappings.unmap_all();
    ibv_dereg_mr(destination_mr);
    ibv_dereg_mr(source_mr);
    cudaFree(destination);
    cudaFree(source);
    ibv_destroy_qp(peer);
    ibv_destroy_qp(sender);
    ibv_destroy_cq(peer_cq);
    ibv_destroy_cq(send_cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(context);
    ibv_free_device_list(devices);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
