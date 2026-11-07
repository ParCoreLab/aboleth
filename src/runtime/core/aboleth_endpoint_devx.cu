// The GPU-resident staged endpoint: DEVX queues over cudaMalloc memory.
//
// Staged submission's host-pinned layout makes every WQE store a posted PCIe
// write and every CQE poll a PCIe read. This module moves the send queue,
// both doorbell records, and the completion queue into GPU memory — the
// NVSHMEM IBGDA placement — by creating the CQ and QP through DEVX with
// dmabuf-registered umems (driver 595 has no nvidia_peermem; dmabuf is the
// only registration path). The GPU then builds WQEs and polls CQEs at HBM
// latency; the NIC fetches them with one peer read; and the only host
// involvement left is the agent forwarding the UAR doorbell store, exactly
// as before (the descriptor's blueflame word still points at the staged
// cell).
//
// The PRM command sequence (CREATE_CQ, CREATE_QP, RST2INIT/INIT2RTR/RTR2RTS)
// mirrors NVSHMEM ibgda.cpp and the mini-gda bring-up
// (Dissecting-IBGDA/mgda_host.cpp), field for field where this runtime's
// verbs path set the equivalent attribute. Everything here is
// probe-with-fallback: any refusal (missing DEVX exports, no dmabuf export,
// RoCE link layer, firmware rejection) reports a reason and the caller
// builds the classic host-pinned staged endpoint instead.

#include "aboleth_connection.h"
#include "aboleth_endpoint_rc.h"

#include "../aboleth_runtime_common.h"
#include "../aboleth_runtime_internal.h"
#include "aboleth_devx_prm.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include "../dl/shims.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

namespace dl = aboleth::dl;

namespace aboleth::runtime {

namespace {

constexpr std::uint32_t kWqebbBytes = 64;
constexpr std::uint32_t kCqeBytes = 64;
// Doorbell page layout: QP dbrec (2 x be32) at 0, CQ dbrec at 64.
constexpr std::uint64_t kCompletionDbrOffset = 64;

std::size_t page_size() {
    static const std::size_t size =
        static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    return size;
}

std::size_t round_up_page(std::size_t bytes) {
    const std::size_t page = page_size();
    return (bytes + page - 1) / page * page;
}

std::uint32_t ceil_pow2(std::uint32_t value) {
    std::uint32_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

std::uint32_t log2_exact(std::uint32_t value) {
    std::uint32_t log = 0;
    while ((1u << log) < value) {
        ++log;
    }
    return log;
}

// Page-aligned cudaMalloc: over-allocate one page and keep the raw base for
// cudaFree. The umem and the descriptor both use the aligned view.
bool gpu_alloc_aligned(
    std::size_t bytes,
    void** raw,
    std::uint64_t* aligned,
    std::string& reason) {
    const cudaError_t status = cudaMalloc(raw, bytes + page_size());
    if (status != cudaSuccess) {
        reason = std::string("cudaMalloc(queue): ") +
                 cudaGetErrorString(status);
        return false;
    }
    const std::uint64_t base = reinterpret_cast<std::uint64_t>(*raw);
    *aligned = (base + page_size() - 1) / page_size() * page_size();
    return true;
}

mlx5dv_devx_umem* register_gpu_umem(
    ibv_context* context,
    std::uint64_t address,
    std::size_t bytes,
    std::string& reason) {
    int fd = -1;
    const CUresult status =
        dl::cuda().cuMemGetHandleForAddressRange(
            &fd,
            static_cast<CUdeviceptr>(address),
            round_up_page(bytes),
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            0);
    if (status != CUDA_SUCCESS) {
        reason = "cuMemGetHandleForAddressRange(DMA_BUF) failed (CUresult " +
                 std::to_string(static_cast<int>(status)) + ")";
        return nullptr;
    }
    mlx5dv_devx_umem_in in{};
    in.addr = nullptr;  // offset within the dma-buf
    in.size = round_up_page(bytes);
    in.access = IBV_ACCESS_LOCAL_WRITE;
    in.pgsz_bitmap =
        ~static_cast<std::uint64_t>(page_size() - 1);
    in.comp_mask = MLX5DV_UMEM_MASK_DMABUF;
    in.dmabuf_fd = fd;
    mlx5dv_devx_umem* umem =
        dl::mlx5dv_devx().mlx5dv_devx_umem_reg_ex(context, &in);
    if (umem == nullptr) {
        reason = std::string("mlx5dv_devx_umem_reg_ex(dmabuf): ") +
                 std::strerror(errno);
    }
    close(fd);
    return umem;
}

std::string syndrome_of(const char* what, const void* out) {
    const auto* words = static_cast<const std::uint32_t*>(out);
    char buffer[128];
    std::snprintf(
        buffer, sizeof(buffer),
        "%s: status=0x%02x syndrome=0x%08x",
        what,
        be32toh(words[0]) >> 24,
        be32toh(words[1]));
    return buffer;
}

}  // namespace

DevxQueueState::~DevxQueueState() {
    const auto& devx = dl::mlx5dv_devx();
    if (queue_obj != nullptr) {
        devx.mlx5dv_devx_obj_destroy(queue_obj);
    }
    if (completion_obj != nullptr) {
        devx.mlx5dv_devx_obj_destroy(completion_obj);
    }
    if (send_umem != nullptr) {
        devx.mlx5dv_devx_umem_dereg(send_umem);
    }
    if (completion_umem != nullptr) {
        devx.mlx5dv_devx_umem_dereg(completion_umem);
    }
    if (dbr_umem != nullptr) {
        devx.mlx5dv_devx_umem_dereg(dbr_umem);
    }
    if (uar != nullptr) {
        devx.mlx5dv_devx_free_uar(uar);
    }
    if (send_raw != nullptr) {
        cudaFree(send_raw);
    }
    if (completion_raw != nullptr) {
        cudaFree(completion_raw);
    }
    if (dbr_raw != nullptr) {
        cudaFree(dbr_raw);
    }
}

bool devx_staged_queues_requested() {
    const char* configured =
        std::getenv("ABOLETH_RDMA_STAGED_QUEUES");
    const std::string queues =
        configured == nullptr ? "" : configured;
    if (queues.empty() || queues == "gpu") {
        return true;
    }
    if (queues == "host") {
        return false;
    }
    throw_error(
        "ABOLETH_RDMA_STAGED_QUEUES must be gpu or host");
}

bool create_devx_staged_queues(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    aboleth::abi::RdmaEndpointDescriptor& descriptor,
    OwnedEndpointContext& endpoint,
    const OwnedEndpointRequest& request,
    std::string& reason) {
    if (!dl::mlx5dv_devx_available()) {
        reason = "libmlx5 lacks the DEVX exports";
        return false;
    }
    if (runtime->port.link_layer != IBV_LINK_LAYER_INFINIBAND) {
        // The DEVX RTR mailbox here wires LID routing only; RoCE needs the
        // GRH address-path fields.
        reason = "GPU-resident queues are wired for InfiniBand LID routing";
        return false;
    }

    const std::uint32_t requested =
        request.send_wqe_requests != 0
            ? request.send_wqe_requests
            : kMaximumTransfers + 1;
    const std::uint32_t wqe_count = ceil_pow2(requested);
    if (wqe_count >= 65536) {
        throw_error(
            "staged submission caps qp_depth below 65536 entries");
    }
    const std::uint32_t cqe_count =
        ceil_pow2(std::max(kMaximumTransfers, requested));

    auto state = std::make_unique<DevxQueueState>();
    state->port_num = runtime->port.port_num;

    const std::size_t send_bytes =
        round_up_page(static_cast<std::size_t>(wqe_count) * kWqebbBytes);
    const std::size_t completion_bytes =
        round_up_page(static_cast<std::size_t>(cqe_count) * kCqeBytes);
    const std::size_t dbr_bytes = round_up_page(4096);
    if (!gpu_alloc_aligned(
            send_bytes, &state->send_raw, &state->send_buffer, reason) ||
        !gpu_alloc_aligned(
            completion_bytes, &state->completion_raw,
            &state->completion_buffer, reason) ||
        !gpu_alloc_aligned(
            dbr_bytes, &state->dbr_raw, &state->dbr_buffer, reason)) {
        return false;
    }
    cuda_check(
        cudaMemset(
            reinterpret_cast<void*>(state->send_buffer), 0, send_bytes),
        "cudaMemset(send queue)");
    // 0xff = CQE opcode INVALID with owner 1: never valid for the first
    // consumer sweep, exactly libmlx5's CQ buffer initialization.
    cuda_check(
        cudaMemset(
            reinterpret_cast<void*>(state->completion_buffer), 0xff,
            completion_bytes),
        "cudaMemset(completion queue)");
    cuda_check(
        cudaMemset(
            reinterpret_cast<void*>(state->dbr_buffer), 0, dbr_bytes),
        "cudaMemset(doorbell records)");

    state->send_umem = register_gpu_umem(
        runtime->verbs_context, state->send_buffer, send_bytes, reason);
    if (state->send_umem == nullptr) {
        return false;
    }
    state->completion_umem = register_gpu_umem(
        runtime->verbs_context, state->completion_buffer, completion_bytes,
        reason);
    if (state->completion_umem == nullptr) {
        return false;
    }
    state->dbr_umem = register_gpu_umem(
        runtime->verbs_context, state->dbr_buffer, dbr_bytes, reason);
    if (state->dbr_umem == nullptr) {
        return false;
    }

    // The host agent performs the doorbell store, so a non-cacheable UAR is
    // the right mapping (NVSHMEM's own preference for CPU-rung doorbells);
    // fall back to a BlueFlame slot when NC allocation is refused.
    // ABOLETH_DEVX_UAR=bf asks for a BlueFlame UAR instead: a dedicated
    // page per endpoint, where NC hands every endpoint of the context the
    // same register (rdma-core's shared NC UAR).
    const auto& devx = dl::mlx5dv_devx();
    const char* uar_kind = std::getenv("ABOLETH_DEVX_UAR");
    const bool want_bf =
        uar_kind != nullptr && std::strcmp(uar_kind, "bf") == 0;
    // A refused BF page (UAR budget) falls back to the shared NC register;
    // either kind refused falls back to the other.
    const auto allocate_uar = [&](std::uint32_t type) {
        return devx.mlx5dv_devx_alloc_uar(runtime->verbs_context, type);
    };
    bool uar_nc = !want_bf;
    state->uar = allocate_uar(
        want_bf ? MLX5DV_UAR_ALLOC_TYPE_BF : MLX5DV_UAR_ALLOC_TYPE_NC);
    if (state->uar == nullptr) {
        uar_nc = !uar_nc;
        state->uar = allocate_uar(
            uar_nc ? MLX5DV_UAR_ALLOC_TYPE_NC : MLX5DV_UAR_ALLOC_TYPE_BF);
    }
    if (state->uar != nullptr &&
        std::getenv("ABOLETH_DOORBELL_PLACEMENT_LOG") != nullptr) {
        std::fprintf(
            stderr,
            "[aboleth-doorbell] gpu %d devx uar %s page %u reg %p\n",
            gpu_index, uar_nc ? "nc" : "bf", state->uar->page_id,
            state->uar->reg_addr);
        std::fflush(stderr);
    }
    if (state->uar == nullptr) {
        reason = std::string("mlx5dv_devx_alloc_uar: ") +
                 std::strerror(errno);
        return false;
    }

    std::uint32_t eqn = 0;
    if (devx.mlx5dv_devx_query_eqn(
            runtime->verbs_context, 0, &eqn) != 0) {
        reason = std::string("mlx5dv_devx_query_eqn: ") +
                 std::strerror(errno);
        return false;
    }
    mlx5dv_pd pd_out{};
    mlx5dv_obj pd_obj{};
    pd_obj.pd.in = runtime->pd;
    pd_obj.pd.out = &pd_out;
    if (dl::mlx5dv().mlx5dv_init_obj(&pd_obj, MLX5DV_OBJ_PD) != 0) {
        reason = "mlx5dv_init_obj(PD) failed";
        return false;
    }

    {
        std::uint8_t in[DEVX_ST_SZ_BYTES(create_cq_in)] = {};
        std::uint8_t out[DEVX_ST_SZ_BYTES(create_cq_out)] = {};
        DEVX_SET(create_cq_in, in, opcode, MLX5_CMD_OP_CREATE_CQ);
        DEVX_SET(create_cq_in, in, cq_umem_id,
                 state->completion_umem->umem_id);
        DEVX_SET(create_cq_in, in, cq_umem_valid, 1);
        void* cqc = DEVX_ADDR_OF(create_cq_in, in, cq_context);
        DEVX_SET(cqc, cqc, log_cq_size, log2_exact(cqe_count));
        DEVX_SET(cqc, cqc, cqe_sz, 0);  // 64-byte CQEs
        DEVX_SET(cqc, cqc, uar_page, state->uar->page_id);
        DEVX_SET(cqc, cqc, c_eqn, eqn);
        // Polling-only CQ, never armed: overrun-ignore, NOT collapsed —
        // the completion path walks a normal owner-parity CQE ring.
        DEVX_SET(cqc, cqc, oi, 1);
        DEVX_SET(cqc, cqc, log_page_size, 0);
        DEVX_SET(cqc, cqc, dbr_umem_id, state->dbr_umem->umem_id);
        DEVX_SET(cqc, cqc, dbr_umem_valid, 1);
        DEVX_SET64(cqc, cqc, dbr_addr, kCompletionDbrOffset);
        state->completion_obj = devx.mlx5dv_devx_obj_create(
            runtime->verbs_context, in, sizeof(in), out, sizeof(out));
        if (state->completion_obj == nullptr) {
            reason = syndrome_of("CREATE_CQ", out);
            return false;
        }
        state->cqn = DEVX_GET(create_cq_out, out, cqn);
    }

    {
        std::uint8_t in[DEVX_ST_SZ_BYTES(create_qp_in)] = {};
        std::uint8_t out[DEVX_ST_SZ_BYTES(create_qp_out)] = {};
        DEVX_SET(create_qp_in, in, opcode, MLX5_CMD_OP_CREATE_QP);
        DEVX_SET(create_qp_in, in, wq_umem_id, state->send_umem->umem_id);
        DEVX_SET(create_qp_in, in, wq_umem_valid, 1);
        void* qpc = DEVX_ADDR_OF(create_qp_in, in, qpc);
        DEVX_SET(qpc, qpc, st, MLX5_QPC_ST_RC);
        DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
        DEVX_SET(qpc, qpc, pd, pd_out.pdn);
        DEVX_SET(qpc, qpc, uar_page, state->uar->page_id);
        DEVX_SET(qpc, qpc, cqn_snd, state->cqn);
        DEVX_SET(qpc, qpc, cqn_rcv, state->cqn);
        DEVX_SET(qpc, qpc, log_sq_size, log2_exact(wqe_count));
        // Send-only: a zero-size RQ keeps the SQ at wq-umem offset 0.
        DEVX_SET(qpc, qpc, rq_type, MLX5_QPC_RQ_TYPE_ZERO_SIZE_RQ);
        DEVX_SET(qpc, qpc, log_rq_size, 0);
        DEVX_SET(qpc, qpc, cs_req, 0);
        DEVX_SET(qpc, qpc, cs_res, 0);
        DEVX_SET(qpc, qpc, dbr_umem_id, state->dbr_umem->umem_id);
        DEVX_SET(qpc, qpc, dbr_umem_valid, 1);
        DEVX_SET64(qpc, qpc, dbr_addr, 0);
        DEVX_SET(qpc, qpc, log_page_size, 0);
        DEVX_SET(qpc, qpc, page_offset, 0);
        state->queue_obj = devx.mlx5dv_devx_obj_create(
            runtime->verbs_context, in, sizeof(in), out, sizeof(out));
        if (state->queue_obj == nullptr) {
            reason = syndrome_of("CREATE_QP", out);
            return false;
        }
        state->qpn = DEVX_GET(create_qp_out, out, qpn);
    }

    {
        // RST -> INIT, mirroring qp_to_init's access flags. pm_state,
        // counter_set_id and atomic_mode are not optional on current
        // firmware (CX-7 refuses their absence).
        std::uint8_t in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {};
        std::uint8_t out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {};
        DEVX_SET(rst2init_qp_in, in, opcode, MLX5_CMD_OP_RST2INIT_QP);
        DEVX_SET(rst2init_qp_in, in, qpn, state->qpn);
        void* qpc = DEVX_ADDR_OF(rst2init_qp_in, in, qpc);
        DEVX_SET(qpc, qpc, rwe, 1);
        DEVX_SET(qpc, qpc, rre, 1);
        DEVX_SET(qpc, qpc, rae, 1);
        DEVX_SET(qpc, qpc, atomic_mode,
                 ABOLETH_QPC_ATOMIC_MODE_UP_TO_64BIT);
        DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num,
                 state->port_num);
        DEVX_SET(qpc, qpc, primary_address_path.pkey_index, 0);
        DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
        DEVX_SET(qpc, qpc, counter_set_id, 0);
        if (devx.mlx5dv_devx_obj_modify(
                state->queue_obj, in, sizeof(in), out, sizeof(out)) != 0) {
            reason = syndrome_of("RST2INIT_QP", out);
            return false;
        }
    }

    // The staged doorbell cell and its agent, exactly as the host-queue
    // staged path wires them — the hook writes this endpoint's DEVX UAR.
    cuda_check(
        cudaHostAlloc(
            &endpoint.doorbell_cell,
            sizeof(std::uint64_t),
            cudaHostAllocPortable | cudaHostAllocMapped),
        "cudaHostAlloc(doorbell cell)");
    *endpoint.doorbell_cell = 0;
    void* cell_device = nullptr;
    cuda_check(
        cudaHostGetDevicePointer(
            &cell_device, endpoint.doorbell_cell, 0),
        "cudaHostGetDevicePointer(doorbell cell)");

    descriptor.send_queue = state->send_buffer;
    descriptor.send_dbrec =
        state->dbr_buffer + MLX5_SND_DBR * sizeof(std::uint32_t);
    descriptor.blueflame =
        reinterpret_cast<std::uint64_t>(cell_device);
    descriptor.completion_queue = state->completion_buffer;
    descriptor.completion_dbrec =
        state->dbr_buffer + kCompletionDbrOffset;
    descriptor.send_stride = kWqebbBytes;
    descriptor.send_wqe_count = wqe_count;
    descriptor.qpn = state->qpn;
    descriptor.completion_entry_size = kCqeBytes;
    descriptor.completion_entry_count = cqe_count;

    endpoint.staged_submit = true;
    endpoint.queue_pair_number = state->qpn;
    endpoint.send_stride = kWqebbBytes;
    endpoint.send_wqe_count = wqe_count;
    // No libmlx5 negotiation exists here: GPU-built WQEs inline up to the
    // direct emitter's WQE ceiling regardless, so the report clamps rather
    // than refuses.
    constexpr std::uint32_t kInlineWqeCeiling = 972;
    endpoint.max_inline_bytes = std::min(
        kInlineWqeCeiling,
        request.max_inline_bytes != 0
            ? request.max_inline_bytes
            : static_cast<std::uint32_t>(
                  configured_index(
                      "ABOLETH_RDMA_INLINE_BYTES", kRdmaInlineBytes)));
    endpoint.agent = acquire_doorbell_agent(
        runtime.get(), gpu_index, pin_numa_local);
    endpoint.agent_token = endpoint.agent->add(
        reinterpret_cast<const volatile std::uint32_t*>(
            endpoint.doorbell_cell),
        make_mlx5_uar_submit(
            reinterpret_cast<volatile std::uint64_t*>(
                state->uar->reg_addr),
            state->qpn));
    endpoint.devx = std::move(state);
    return true;
}

void devx_qp_to_rtr(
    DevxQueueState& state,
    ibv_context*,
    std::uint32_t remote_qpn,
    std::uint32_t remote_psn,
    std::uint16_t remote_lid,
    ibv_mtu path_mtu) {
    std::uint8_t in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {};
    std::uint8_t out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {};
    DEVX_SET(init2rtr_qp_in, in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
    DEVX_SET(init2rtr_qp_in, in, qpn, state.qpn);
    void* qpc = DEVX_ADDR_OF(init2rtr_qp_in, in, qpc);
    // The ibv_mtu enum values ARE the PRM mtu encoding (256=1 .. 4096=5).
    DEVX_SET(qpc, qpc, mtu, static_cast<std::uint32_t>(path_mtu));
    DEVX_SET(qpc, qpc, log_msg_max, 30);
    DEVX_SET(qpc, qpc, remote_qpn, remote_qpn);
    DEVX_SET(qpc, qpc, next_rcv_psn, remote_psn);
    // qp_to_rtr's min_rnr_timer=12 and max_dest_rd_atomic.
    DEVX_SET(qpc, qpc, min_rnr_nak, 12);
    DEVX_SET(qpc, qpc, log_rra_max, log_rd_atomic_depth());
    DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, state.port_num);
    DEVX_SET(qpc, qpc, primary_address_path.rlid, remote_lid);
    if (dl::mlx5dv_devx().mlx5dv_devx_obj_modify(
            state.queue_obj, in, sizeof(in), out, sizeof(out)) != 0) {
        throw_error(syndrome_of("INIT2RTR_QP", out));
    }
}

void devx_qp_to_rts(
    DevxQueueState& state,
    std::uint32_t psn) {
    std::uint8_t in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {};
    std::uint8_t out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {};
    DEVX_SET(rtr2rts_qp_in, in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
    DEVX_SET(rtr2rts_qp_in, in, qpn, state.qpn);
    void* qpc = DEVX_ADDR_OF(rtr2rts_qp_in, in, qpc);
    // qp_to_rts: timeout 14, retry 7, rnr_retry 7, and max_rd_atomic.
    DEVX_SET(qpc, qpc, retry_count, 7);
    DEVX_SET(qpc, qpc, rnr_retry, 7);
    DEVX_SET(qpc, qpc, next_send_psn, psn);
    DEVX_SET(qpc, qpc, log_sra_max, log_rd_atomic_depth());
    DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, 14);
    if (dl::mlx5dv_devx().mlx5dv_devx_obj_modify(
            state.queue_obj, in, sizeof(in), out, sizeof(out)) != 0) {
        throw_error(syndrome_of("RTR2RTS_QP", out));
    }
}

void dump_queue_pair_context(
    DevxQueueState* devx, ibv_qp* verbs_qp, int gpu_index) {
    static std::atomic<int> dumped{0};
    if (std::getenv("ABOLETH_QPC_DUMP") == nullptr || dumped++ >= 2) {
        return;
    }
    std::uint8_t in[DEVX_ST_SZ_BYTES(query_qp_in)] = {};
    std::uint8_t out[DEVX_ST_SZ_BYTES(query_qp_out)] = {};
    DEVX_SET(query_qp_in, in, opcode, MLX5_CMD_OP_QUERY_QP);
    const std::uint32_t qpn =
        devx != nullptr ? devx->qpn : verbs_qp->qp_num;
    DEVX_SET(query_qp_in, in, qpn, qpn);
    const auto& api = dl::mlx5dv_devx();
    const int status =
        devx != nullptr
            ? api.mlx5dv_devx_obj_query(
                  devx->queue_obj, in, sizeof(in), out, sizeof(out))
            : api.mlx5dv_devx_qp_query(
                  verbs_qp, in, sizeof(in), out, sizeof(out));
    if (status != 0) {
        std::fprintf(stderr, "[aboleth-qpc] gpu %d query failed: %s\n",
                     gpu_index, syndrome_of("QUERY_QP", out).c_str());
        return;
    }
    const auto* qpc = static_cast<const std::uint8_t*>(
        DEVX_ADDR_OF(query_qp_out, out, qpc));
    std::string hex;
    char word[12];
    for (std::size_t offset = 0; offset < DEVX_ST_SZ_BYTES(qpc);
         offset += 4) {
        std::uint32_t value = 0;
        std::memcpy(&value, qpc + offset, 4);
        std::snprintf(word, sizeof(word), "%08x ", be32toh(value));
        hex += word;
    }
    std::fprintf(stderr, "[aboleth-qpc] gpu %d %s qpn %u qpc %s\n",
                 gpu_index, devx != nullptr ? "devx" : "verbs", qpn,
                 hex.c_str());
    std::fflush(stderr);
}

}  // namespace aboleth::runtime
