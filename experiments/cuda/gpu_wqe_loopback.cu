#include "aboleth_probe_fabric.h"

#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// The GPU this probe runs its loopback on; the HCA follows from it.
constexpr int kGpuIndex = 0;
constexpr std::size_t kTransferBytes = 256;

[[noreturn]] void fail(const char* what) {
    std::perror(what);
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
};

RegisteredMapping map_host_pages(void* pointer, std::size_t bytes, unsigned int flags) {
    const auto page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    const auto base = start & ~(page_size - 1);
    const auto end = (start + bytes + page_size - 1) & ~(page_size - 1);

    RegisteredMapping mapping{
        reinterpret_cast<void*>(base),
        end - base,
        nullptr,
    };
    check_cuda(cudaHostRegister(mapping.host_base, mapping.bytes, flags), "cudaHostRegister");
    check_cuda(
        cudaHostGetDevicePointer(&mapping.device_base, mapping.host_base, 0),
        "cudaHostGetDevicePointer");
    return mapping;
}

bool mapping_contains(
    const RegisteredMapping& mapping,
    const void* pointer,
    std::size_t bytes) {
    const auto base = reinterpret_cast<std::uintptr_t>(mapping.host_base);
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return start >= base && start + bytes <= base + mapping.bytes;
}

void* mapped_pointer(const RegisteredMapping& mapping, const void* pointer) {
    const auto offset =
        reinterpret_cast<std::uintptr_t>(pointer) -
        reinterpret_cast<std::uintptr_t>(mapping.host_base);
    return reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(mapping.device_base) + offset);
}

ibv_qp* create_rc_qp(ibv_pd* pd, ibv_cq* cq) {
    ibv_qp_init_attr attr{};
    attr.send_cq = cq;
    attr.recv_cq = cq;
    attr.cap.max_send_wr = 16;
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

__device__ __forceinline__ void store_no_allocate(int4* pointer, const int4& value) {
    asm volatile(
        "st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(pointer), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
        : "memory");
}

__device__ __forceinline__ void store_release(std::uint32_t* pointer, std::uint32_t value) {
    asm volatile(
        "st.release.sys.global.L1::no_allocate.b32 [%0], %1;"
        :
        : "l"(pointer), "r"(value)
        : "memory");
}

__device__ __forceinline__ void store_release(std::uint64_t* pointer, std::uint64_t value) {
    asm volatile(
        "st.release.sys.global.L1::no_allocate.b64 [%0], %1;"
        :
        : "l"(pointer), "l"(value)
        : "memory");
}

__global__ void post_rdma_write(
    std::uint8_t* send_queue,
    std::uint32_t send_stride,
    std::uint32_t send_wqe_count,
    std::uint32_t* send_dbrec,
    std::uint64_t* blueflame,
    std::uint32_t qpn,
    const void* source,
    std::uint32_t source_lkey,
    void* destination,
    std::uint32_t destination_rkey,
    std::uint32_t bytes) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    constexpr std::uint32_t producer_index = 0;
    constexpr std::uint32_t next_producer_index = 1;
    constexpr std::uint32_t mlx5_opcode_rdma_write = 0x08;
    constexpr std::uint8_t mlx5_wqe_ctrl_cq_update = 2 << 2;
    constexpr std::uint32_t data_segments = 3;

    auto* wqe = send_queue + (producer_index % send_wqe_count) * send_stride;
    auto* ctrl_pointer = reinterpret_cast<WqeCtrl*>(wqe);
    auto* remote_pointer = reinterpret_cast<WqeRemoteAddress*>(wqe + sizeof(WqeCtrl));
    auto* data_pointer =
        reinterpret_cast<WqeData*>(wqe + sizeof(WqeCtrl) + sizeof(WqeRemoteAddress));

    WqeCtrl ctrl{};
    ctrl.opmod_idx_opcode =
        to_be32((producer_index << 8) | mlx5_opcode_rdma_write);
    ctrl.qpn_ds = to_be32((qpn << 8) | data_segments);
    ctrl.signature_dci_fm =
        static_cast<std::uint32_t>(mlx5_wqe_ctrl_cq_update) << 24;

    WqeRemoteAddress remote{};
    remote.address = to_be64(reinterpret_cast<std::uint64_t>(destination));
    remote.rkey = to_be32(destination_rkey);

    WqeData data{};
    data.byte_count = to_be32(bytes);
    data.lkey = to_be32(source_lkey);
    data.address = to_be64(reinterpret_cast<std::uint64_t>(source));

    store_no_allocate(
        reinterpret_cast<int4*>(ctrl_pointer),
        *reinterpret_cast<const int4*>(&ctrl));
    store_no_allocate(
        reinterpret_cast<int4*>(remote_pointer),
        *reinterpret_cast<const int4*>(&remote));
    store_no_allocate(
        reinterpret_cast<int4*>(data_pointer),
        *reinterpret_cast<const int4*>(&data));

    __threadfence_system();
    store_release(send_dbrec, to_be32(next_producer_index));

    const std::uint32_t doorbell_index = to_be32(next_producer_index << 8);
    const std::uint32_t doorbell_qpn = to_be32(qpn << 8);
    const std::uint64_t doorbell =
        (static_cast<std::uint64_t>(doorbell_qpn) << 32) | doorbell_index;
    store_release(blueflame, doorbell);
}

}  // namespace

int main() {
    check_cuda(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");
    check_cuda(cudaSetDevice(kGpuIndex), "cudaSetDevice");

    const aboleth::probe::FabricAddress fabric =
        aboleth::probe::resolve_fabric_address(kGpuIndex);
    aboleth::probe::report_fabric_address(fabric);

    int device_count = 0;
    ibv_device** devices = ibv_get_device_list(&device_count);
    if (devices == nullptr) {
        fail("ibv_get_device_list");
    }
    ibv_device* selected =
        aboleth::probe::find_fabric_device(devices, device_count, fabric);

    ibv_context* context = ibv_open_device(selected);
    if (context == nullptr) {
        fail("ibv_open_device");
    }

    ibv_pd* pd = ibv_alloc_pd(context);
    if (pd == nullptr) {
        fail("ibv_alloc_pd");
    }
    ibv_cq* send_cq = ibv_create_cq(context, 16, nullptr, nullptr, 0);
    ibv_cq* peer_cq = ibv_create_cq(context, 16, nullptr, nullptr, 0);
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

    void* source = nullptr;
    void* destination = nullptr;
    check_cuda(cudaMalloc(&source, kTransferBytes), "cudaMalloc(source)");
    check_cuda(cudaMalloc(&destination, kTransferBytes), "cudaMalloc(destination)");

    std::vector<std::uint8_t> expected(kTransferBytes);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<std::uint8_t>((index * 17 + 3) & 0xff);
    }
    check_cuda(
        cudaMemcpy(source, expected.data(), expected.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy(source)");
    check_cuda(cudaMemset(destination, 0, kTransferBytes), "cudaMemset(destination)");

    ibv_mr* source_mr =
        ibv_reg_mr(pd, source, kTransferBytes, IBV_ACCESS_LOCAL_WRITE);
    ibv_mr* destination_mr = ibv_reg_mr(
        pd,
        destination,
        kTransferBytes,
        IBV_ACCESS_LOCAL_WRITE |
            IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ);
    if (source_mr == nullptr || destination_mr == nullptr) {
        fail("ibv_reg_mr(GPU memory)");
    }

    mlx5dv_qp direct_qp{};
    mlx5dv_obj direct_object{};
    direct_object.qp.in = sender;
    direct_object.qp.out = &direct_qp;
    if (mlx5dv_init_obj(&direct_object, MLX5DV_OBJ_QP) != 0) {
        fail("mlx5dv_init_obj(QP)");
    }
    if (direct_qp.sq.buf == nullptr || direct_qp.dbrec == nullptr || direct_qp.bf.reg == nullptr) {
        std::fprintf(stderr, "Direct-verbs QP did not expose SQ, DBREC, and BF mappings\n");
        return EXIT_FAILURE;
    }

    const std::size_t send_queue_bytes =
        static_cast<std::size_t>(direct_qp.sq.wqe_cnt) * direct_qp.sq.stride;
    std::printf(
        "QP %u: SQ=%p (%u WQEs, stride %u), DBREC=%p, BF=%p (%u bytes)\n",
        sender->qp_num,
        direct_qp.sq.buf,
        direct_qp.sq.wqe_cnt,
        direct_qp.sq.stride,
        static_cast<void*>(direct_qp.dbrec),
        direct_qp.bf.reg,
        direct_qp.bf.size);

    RegisteredMapping send_queue_mapping = map_host_pages(
        direct_qp.sq.buf,
        send_queue_bytes,
        cudaHostRegisterMapped);

    RegisteredMapping dbrec_mapping{};
    bool dbrec_has_separate_mapping = false;
    if (!mapping_contains(
            send_queue_mapping,
            direct_qp.dbrec,
            2 * sizeof(std::uint32_t))) {
        dbrec_mapping = map_host_pages(
            direct_qp.dbrec,
            2 * sizeof(std::uint32_t),
            cudaHostRegisterMapped);
        dbrec_has_separate_mapping = true;
    }

    RegisteredMapping blueflame_mapping = map_host_pages(
        direct_qp.bf.reg,
        sizeof(std::uint64_t),
        cudaHostRegisterMapped | cudaHostRegisterIoMemory);

    auto* device_send_queue = static_cast<std::uint8_t*>(
        mapped_pointer(send_queue_mapping, direct_qp.sq.buf));
    const RegisteredMapping& dbrec_owner =
        dbrec_has_separate_mapping ? dbrec_mapping : send_queue_mapping;
    auto* device_send_dbrec = static_cast<std::uint32_t*>(
        mapped_pointer(dbrec_owner, direct_qp.dbrec)) + MLX5_SND_DBR;
    auto* device_blueflame = static_cast<std::uint64_t*>(
        mapped_pointer(blueflame_mapping, direct_qp.bf.reg));

    std::printf(
        "Launching GPU WQE post: source=%p lkey=0x%x destination=%p rkey=0x%x\n",
        source,
        source_mr->lkey,
        destination,
        destination_mr->rkey);

    post_rdma_write<<<1, 1>>>(
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
        kTransferBytes);
    check_cuda(cudaGetLastError(), "post_rdma_write launch");
    check_cuda(cudaDeviceSynchronize(), "post_rdma_write execution");

    ibv_wc completion{};
    int polled = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        polled = ibv_poll_cq(send_cq, 1, &completion);
        if (polled != 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    bool success = false;
    if (polled < 0) {
        std::fprintf(stderr, "ibv_poll_cq failed\n");
    } else if (polled == 0) {
        std::fprintf(stderr, "Timed out waiting for GPU-posted WQE completion\n");
    } else if (completion.status != IBV_WC_SUCCESS) {
        std::fprintf(
            stderr,
            "WQE completed with status %s (%d), vendor error 0x%x\n",
            ibv_wc_status_str(completion.status),
            completion.status,
            completion.vendor_err);
    } else {
        std::vector<std::uint8_t> observed(kTransferBytes);
        check_cuda(
            cudaMemcpy(
                observed.data(),
                destination,
                observed.size(),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(destination)");
        success = observed == expected;
        if (!success) {
            std::fprintf(stderr, "CQ completed, but destination payload did not match\n");
        }
    }

    std::printf(
        "GPU-issued RDMA write: %s\n",
        success ? "PASS" : "FAIL");

    check_cuda(cudaHostUnregister(blueflame_mapping.host_base), "cudaHostUnregister(BF)");
    if (dbrec_has_separate_mapping) {
        check_cuda(cudaHostUnregister(dbrec_mapping.host_base), "cudaHostUnregister(DBREC)");
    }
    check_cuda(cudaHostUnregister(send_queue_mapping.host_base), "cudaHostUnregister(SQ)");

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
