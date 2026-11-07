#include "aboleth_device.cuh"
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

// The producer's GPU, and the one whose HCA carries both directions of the
// ping-pong. GPU 1's memory is registered on this device's protection domain.
constexpr int kProducerGpuIndex = 0;
constexpr std::uint32_t kDefaultRounds = 10'000;
constexpr std::uint32_t kDefaultTransfers = 32;
constexpr std::uint32_t kMaximumTransfers = 256;
constexpr std::uint32_t kBytesPerTransfer = 256;
constexpr std::uint32_t kBufferSlots = 2;
constexpr std::uint32_t kWaitTimeoutSeconds = 5;

enum class FailureStage : std::uint32_t {
    None = 0,
    WaitForReusableSlot = 1,
    SenderCompletion = 2,
    FinalAcknowledgment = 3,
    WaitForReady = 4,
    DataValidation = 5,
    AcknowledgmentCompletion = 6,
};

struct ProducerResult {
    FailureStage failure{FailureStage::None};
    std::uint32_t failed_round{};
    std::uint32_t rounds_sent{};
    aboleth::WaitStatus wait_status{
        aboleth::WaitStatus::Success};
    std::uint64_t total_cycles{};
};

struct ConsumerResult {
    FailureStage failure{FailureStage::None};
    std::uint32_t failed_round{};
    std::uint32_t failed_byte{};
    std::uint32_t rounds_consumed{};
    aboleth::WaitStatus wait_status{
        aboleth::WaitStatus::Success};
    std::uint64_t total_cycles{};
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
        std::fprintf(
            stderr,
            "%s: %s\n",
            what,
            cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

struct RegisteredRegion {
    void* host_base{};
    std::size_t bytes{};
    unsigned int flags{};
};

class PortableMappingRegistry {
public:
    void register_region(
        void* pointer,
        std::size_t bytes,
        unsigned int flags) {
        const auto page_size =
            static_cast<std::size_t>(
                sysconf(_SC_PAGESIZE));
        const auto start =
            reinterpret_cast<std::uintptr_t>(
                pointer);
        const auto base =
            start & ~(page_size - 1);
        const auto end =
            (start + bytes + page_size - 1) &
            ~(page_size - 1);
        const unsigned int portable_flags =
            flags | cudaHostRegisterPortable;

        for (const auto& region : regions_) {
            const auto region_base =
                reinterpret_cast<std::uintptr_t>(
                    region.host_base);
            const auto region_end =
                region_base + region.bytes;
            const bool overlaps =
                base < region_end &&
                end > region_base;
            if (!overlaps) {
                continue;
            }
            if (portable_flags == region.flags &&
                base >= region_base &&
                end <= region_end) {
                return;
            }
            fail_message(
                "Overlapping portable CUDA mappings are incompatible");
        }

        RegisteredRegion region{
            reinterpret_cast<void*>(base),
            end - base,
            portable_flags,
        };
        check_cuda(
            cudaHostRegister(
                region.host_base,
                region.bytes,
                region.flags),
            "cudaHostRegister(portable)");
        regions_.push_back(region);
    }

    void* device_pointer(void* pointer) const {
        const auto address =
            reinterpret_cast<std::uintptr_t>(
                pointer);
        for (const auto& region : regions_) {
            const auto base =
                reinterpret_cast<std::uintptr_t>(
                    region.host_base);
            if (address < base ||
                address >= base + region.bytes) {
                continue;
            }
            void* device_base = nullptr;
            check_cuda(
                cudaHostGetDevicePointer(
                    &device_base,
                    region.host_base,
                    0),
                "cudaHostGetDevicePointer");
            return reinterpret_cast<void*>(
                reinterpret_cast<std::uintptr_t>(
                    device_base) +
                address -
                base);
        }
        fail_message(
            "Requested pointer was not registered");
    }

    void unmap_all() {
        for (auto region = regions_.rbegin();
             region != regions_.rend();
             ++region) {
            check_cuda(
                cudaHostUnregister(
                    region->host_base),
                "cudaHostUnregister");
        }
        regions_.clear();
    }

private:
    std::vector<RegisteredRegion> regions_;
};

struct DirectEndpointObjects {
    mlx5dv_qp qp{};
    mlx5dv_cq cq{};
};

struct RdmaRuntime {
    ibv_device** devices{};
    ibv_context* context{};
    ibv_pd* pd{};
    ibv_cq* cq0{};
    ibv_cq* cq1{};
    ibv_qp* qp0{};
    ibv_qp* qp1{};
    ibv_mr* mr0{};
    ibv_mr* mr1{};
    DirectEndpointObjects direct0{};
    DirectEndpointObjects direct1{};
    PortableMappingRegistry mappings;
};

struct EndpointPair {
    aboleth::Endpoint gpu0_to_gpu1{};
    aboleth::Endpoint gpu1_to_gpu0{};
};

ibv_qp* create_rc_qp(
    ibv_pd* pd,
    ibv_cq* cq) {
    ibv_qp_init_attr attr{};
    attr.send_cq = cq;
    attr.recv_cq = cq;
    attr.cap.max_send_wr =
        kMaximumTransfers + 1;
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

void qp_to_rts(
    ibv_qp* qp,
    std::uint32_t psn) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = psn;
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

void query_direct_objects(
    ibv_qp* qp,
    ibv_cq* cq,
    DirectEndpointObjects& direct) {
    mlx5dv_obj object{};
    object.qp.in = qp;
    object.qp.out = &direct.qp;
    object.cq.in = cq;
    object.cq.out = &direct.cq;
    if (mlx5dv_init_obj(
            &object,
            MLX5DV_OBJ_QP |
                MLX5DV_OBJ_CQ) != 0) {
        fail("mlx5dv_init_obj(QP|CQ)");
    }
    if (direct.qp.sq.buf == nullptr ||
        direct.qp.dbrec == nullptr ||
        direct.qp.bf.reg == nullptr ||
        direct.cq.buf == nullptr ||
        direct.cq.dbrec == nullptr) {
        fail_message(
            "Direct-verbs resources are incomplete");
    }
}

void register_direct_objects(
    PortableMappingRegistry& mappings,
    DirectEndpointObjects& direct) {
    mappings.register_region(
        direct.qp.sq.buf,
        static_cast<std::size_t>(
            direct.qp.sq.wqe_cnt) *
            direct.qp.sq.stride,
        cudaHostRegisterMapped);
    mappings.register_region(
        direct.cq.buf,
        static_cast<std::size_t>(
            direct.cq.cqe_cnt) *
            direct.cq.cqe_size,
        cudaHostRegisterMapped);
    mappings.register_region(
        direct.qp.dbrec,
        2 * sizeof(std::uint32_t),
        cudaHostRegisterMapped);
    mappings.register_region(
        direct.cq.dbrec,
        2 * sizeof(std::uint32_t),
        cudaHostRegisterMapped);
    mappings.register_region(
        direct.qp.bf.reg,
        sizeof(std::uint64_t),
        cudaHostRegisterMapped |
            cudaHostRegisterIoMemory);
}

aboleth::Endpoint make_device_endpoint(
    PortableMappingRegistry& mappings,
    DirectEndpointObjects& direct,
    std::uint32_t qpn,
    std::uint32_t source_lkey,
    std::uint32_t destination_rkey,
    std::uint64_t timeout_cycles) {
    aboleth::Endpoint endpoint{};
    endpoint.backend = aboleth::Backend::Rdma;
    auto& rdma = endpoint.rdma;
    rdma.send_queue =
        static_cast<std::uint8_t*>(
            mappings.device_pointer(
                direct.qp.sq.buf));
    rdma.send_stride = direct.qp.sq.stride;
    rdma.send_wqe_count =
        direct.qp.sq.wqe_cnt;
    rdma.send_dbrec =
        static_cast<std::uint32_t*>(
            mappings.device_pointer(
                direct.qp.dbrec)) +
        MLX5_SND_DBR;
    rdma.blueflame =
        static_cast<std::uint64_t*>(
            mappings.device_pointer(
                direct.qp.bf.reg));
    rdma.qpn = qpn;
    rdma.source_lkey = source_lkey;
    rdma.destination_rkey =
        destination_rkey;
    rdma.completion_queue =
        static_cast<std::uint8_t*>(
            mappings.device_pointer(
                direct.cq.buf));
    rdma.completion_entry_size =
        direct.cq.cqe_size;
    rdma.completion_entry_count =
        direct.cq.cqe_cnt;
    rdma.completion_dbrec =
        static_cast<std::uint32_t*>(
            mappings.device_pointer(
                direct.cq.dbrec));
    rdma.completion_timeout_cycles =
        timeout_cycles;
    return endpoint;
}

EndpointPair create_rdma_endpoints(
    RdmaRuntime& runtime,
    void* allocation0,
    std::size_t allocation0_bytes,
    void* allocation1,
    std::size_t allocation1_bytes,
    std::uint64_t timeout_cycles) {
    const aboleth::probe::FabricAddress fabric =
        aboleth::probe::resolve_fabric_address(
            kProducerGpuIndex);
    aboleth::probe::report_fabric_address(fabric);

    int device_count = 0;
    runtime.devices =
        ibv_get_device_list(&device_count);
    if (runtime.devices == nullptr) {
        fail("ibv_get_device_list");
    }
    ibv_device* selected =
        aboleth::probe::find_fabric_device(
            runtime.devices,
            device_count,
            fabric);

    runtime.context =
        ibv_open_device(selected);
    if (runtime.context == nullptr) {
        fail("ibv_open_device");
    }

    runtime.pd =
        ibv_alloc_pd(runtime.context);
    if (runtime.pd == nullptr) {
        fail("ibv_alloc_pd");
    }
    runtime.cq0 =
        ibv_create_cq(
            runtime.context,
            256,
            nullptr,
            nullptr,
            0);
    runtime.cq1 =
        ibv_create_cq(
            runtime.context,
            256,
            nullptr,
            nullptr,
            0);
    if (runtime.cq0 == nullptr ||
        runtime.cq1 == nullptr) {
        fail("ibv_create_cq");
    }

    runtime.qp0 =
        create_rc_qp(runtime.pd, runtime.cq0);
    runtime.qp1 =
        create_rc_qp(runtime.pd, runtime.cq1);
    qp_to_init(runtime.qp0, fabric);
    qp_to_init(runtime.qp1, fabric);

    constexpr std::uint32_t psn0 = 0x123456;
    constexpr std::uint32_t psn1 = 0x654321;
    qp_to_rtr(
        runtime.qp0,
        runtime.qp1->qp_num,
        psn1,
        fabric);
    qp_to_rtr(
        runtime.qp1,
        runtime.qp0->qp_num,
        psn0,
        fabric);
    qp_to_rts(runtime.qp0, psn0);
    qp_to_rts(runtime.qp1, psn1);

    runtime.mr0 =
        ibv_reg_mr(
            runtime.pd,
            allocation0,
            allocation0_bytes,
            IBV_ACCESS_LOCAL_WRITE |
                IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ);
    runtime.mr1 =
        ibv_reg_mr(
            runtime.pd,
            allocation1,
            allocation1_bytes,
            IBV_ACCESS_LOCAL_WRITE |
                IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ);
    if (runtime.mr0 == nullptr ||
        runtime.mr1 == nullptr) {
        fail("ibv_reg_mr(GPU memory)");
    }

    query_direct_objects(
        runtime.qp0,
        runtime.cq0,
        runtime.direct0);
    query_direct_objects(
        runtime.qp1,
        runtime.cq1,
        runtime.direct1);

    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    register_direct_objects(
        runtime.mappings,
        runtime.direct0);
    register_direct_objects(
        runtime.mappings,
        runtime.direct1);

    EndpointPair endpoints{};
    endpoints.gpu0_to_gpu1 =
        make_device_endpoint(
            runtime.mappings,
            runtime.direct0,
            runtime.qp0->qp_num,
            runtime.mr0->lkey,
            runtime.mr1->rkey,
            timeout_cycles);
    check_cuda(
        cudaSetDevice(1),
        "cudaSetDevice(1)");
    endpoints.gpu1_to_gpu0 =
        make_device_endpoint(
            runtime.mappings,
            runtime.direct1,
            runtime.qp1->qp_num,
            runtime.mr1->lkey,
            runtime.mr0->rkey,
            timeout_cycles);
    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    return endpoints;
}

void destroy_rdma_runtime(
    RdmaRuntime& runtime) {
    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    runtime.mappings.unmap_all();
    if (runtime.mr1 != nullptr) {
        ibv_dereg_mr(runtime.mr1);
    }
    if (runtime.mr0 != nullptr) {
        ibv_dereg_mr(runtime.mr0);
    }
    if (runtime.qp1 != nullptr) {
        ibv_destroy_qp(runtime.qp1);
    }
    if (runtime.qp0 != nullptr) {
        ibv_destroy_qp(runtime.qp0);
    }
    if (runtime.cq1 != nullptr) {
        ibv_destroy_cq(runtime.cq1);
    }
    if (runtime.cq0 != nullptr) {
        ibv_destroy_cq(runtime.cq0);
    }
    if (runtime.pd != nullptr) {
        ibv_dealloc_pd(runtime.pd);
    }
    if (runtime.context != nullptr) {
        ibv_close_device(runtime.context);
    }
    if (runtime.devices != nullptr) {
        ibv_free_device_list(runtime.devices);
    }
}

__device__ __forceinline__ std::uint8_t pattern(
    std::uint32_t round,
    std::size_t byte_index) {
    return static_cast<std::uint8_t>(
        (byte_index * 17 +
         static_cast<std::size_t>(round) * 29 +
         3) &
        0xff);
}

__global__ void persistent_producer(
    aboleth::Endpoint forward,
    aboleth::LocalBuffer local,
    aboleth::RemoteBuffer remote,
    std::uint64_t* acknowledgment,
    std::uint32_t rounds,
    std::uint32_t transfers,
    std::uint32_t bytes_per_transfer,
    std::uint64_t timeout_cycles,
    ProducerResult* result) {
    if (blockIdx.x != 0 ||
        threadIdx.x != 0) {
        return;
    }

    aboleth::QueueState forward_state{};
    const std::size_t batch_bytes =
        static_cast<std::size_t>(transfers) *
        bytes_per_transfer;
    const std::uint64_t start = clock64();

    for (std::uint32_t round = 1;
         round <= rounds;
         ++round) {
        if (round > kBufferSlots) {
            const auto reusable =
                aboleth::wait_signal(
                    acknowledgment,
                    round - kBufferSlots,
                    timeout_cycles);
            if (reusable.status !=
                aboleth::WaitStatus::Success) {
                result->failure =
                    FailureStage::WaitForReusableSlot;
                result->failed_round = round;
                result->wait_status =
                    reusable.status;
                result->total_cycles =
                    clock64() - start;
                return;
            }
        }

        const std::uint32_t slot =
            (round - 1) % kBufferSlots;
        const std::size_t slot_offset =
            static_cast<std::size_t>(slot) *
            batch_bytes;
        for (std::size_t index = 0;
             index < batch_bytes;
             ++index) {
            const_cast<std::uint8_t*>(
                local.data)[slot_offset + index] =
                pattern(round, index);
        }

        aboleth::Transfer transfer{};
        for (std::uint32_t index = 0;
             index < transfers;
             ++index) {
            const std::size_t offset =
                slot_offset +
                static_cast<std::size_t>(index) *
                    bytes_per_transfer;
            transfer = aboleth::put(
                forward,
                forward_state,
                remote,
                offset,
                local,
                offset,
                bytes_per_transfer);
        }
        *const_cast<std::uint64_t*>(
            local.signal_value) = round;
        transfer = aboleth::signal(
            forward,
            forward_state,
            remote,
            local);
        const auto completion =
            aboleth::wait(
                forward,
                forward_state,
                transfer);
        if (completion.status !=
            aboleth::WaitStatus::Success) {
            result->failure =
                FailureStage::SenderCompletion;
            result->failed_round = round;
            result->wait_status =
                completion.status;
            result->total_cycles =
                clock64() - start;
            return;
        }
        result->rounds_sent = round;
    }

    const auto final_ack =
        aboleth::wait_signal(
            acknowledgment,
            rounds,
            timeout_cycles);
    if (final_ack.status !=
        aboleth::WaitStatus::Success) {
        result->failure =
            FailureStage::FinalAcknowledgment;
        result->failed_round = rounds;
        result->wait_status =
            final_ack.status;
    }
    result->total_cycles =
        clock64() - start;
}

__global__ void persistent_consumer(
    aboleth::Endpoint reverse,
    aboleth::LocalBuffer local,
    aboleth::RemoteBuffer remote,
    std::uint64_t* ready,
    std::uint32_t rounds,
    std::uint32_t transfers,
    std::uint32_t bytes_per_transfer,
    std::uint64_t timeout_cycles,
    ConsumerResult* result) {
    if (blockIdx.x != 0 ||
        threadIdx.x != 0) {
        return;
    }

    aboleth::QueueState reverse_state{};
    const std::size_t batch_bytes =
        static_cast<std::size_t>(transfers) *
        bytes_per_transfer;
    const std::uint64_t start = clock64();

    for (std::uint32_t round = 1;
         round <= rounds;
         ++round) {
        const auto ready_result =
            aboleth::wait_signal(
                ready,
                round,
                timeout_cycles);
        if (ready_result.status !=
            aboleth::WaitStatus::Success) {
            result->failure =
                FailureStage::WaitForReady;
            result->failed_round = round;
            result->wait_status =
                ready_result.status;
            result->total_cycles =
                clock64() - start;
            return;
        }

        const std::uint32_t slot =
            (round - 1) % kBufferSlots;
        const std::size_t slot_offset =
            static_cast<std::size_t>(slot) *
            batch_bytes;
        for (std::size_t index = 0;
             index < batch_bytes;
             ++index) {
            if (local.data[slot_offset + index] !=
                pattern(round, index)) {
                result->failure =
                    FailureStage::DataValidation;
                result->failed_round = round;
                result->failed_byte =
                    static_cast<std::uint32_t>(
                        index);
                result->total_cycles =
                    clock64() - start;
                return;
            }
        }

        *const_cast<std::uint64_t*>(
            local.signal_value) = round;
        const auto acknowledgment =
            aboleth::signal(
                reverse,
                reverse_state,
                remote,
                local);
        const auto completion =
            aboleth::wait(
                reverse,
                reverse_state,
                acknowledgment);
        if (completion.status !=
            aboleth::WaitStatus::Success) {
            result->failure =
                FailureStage::AcknowledgmentCompletion;
            result->failed_round = round;
            result->wait_status =
                completion.status;
            result->total_cycles =
                clock64() - start;
            return;
        }
        result->rounds_consumed = round;
    }
    result->total_cycles =
        clock64() - start;
}

const char* backend_name(
    aboleth::Backend backend) {
    return backend == aboleth::Backend::P2P
        ? "p2p"
        : "rdma";
}

const char* failure_name(
    FailureStage stage) {
    switch (stage) {
        case FailureStage::None:
            return "none";
        case FailureStage::WaitForReusableSlot:
            return "wait-for-reusable-slot";
        case FailureStage::SenderCompletion:
            return "sender-completion";
        case FailureStage::FinalAcknowledgment:
            return "final-acknowledgment";
        case FailureStage::WaitForReady:
            return "wait-for-ready";
        case FailureStage::DataValidation:
            return "data-validation";
        case FailureStage::AcknowledgmentCompletion:
            return "acknowledgment-completion";
    }
    return "unknown";
}

aboleth::Backend parse_backend(
    int argc,
    char** argv) {
    if (argc < 2) {
        fail_message(
            "Usage: aboleth_pingpong [p2p|rdma] [rounds] [transfers]");
    }
    const std::string value = argv[1];
    if (value == "p2p") {
        return aboleth::Backend::P2P;
    }
    if (value == "rdma") {
        return aboleth::Backend::Rdma;
    }
    fail_message(
        "Usage: aboleth_pingpong [p2p|rdma] [rounds] [transfers]");
}

std::uint32_t parse_positive(
    int argc,
    char** argv,
    int index,
    std::uint32_t default_value,
    std::uint32_t maximum,
    const char* name) {
    if (argc <= index) {
        return default_value;
    }
    char* end = nullptr;
    const unsigned long value =
        std::strtoul(argv[index], &end, 10);
    if (end == argv[index] ||
        *end != '\0' ||
        value == 0 ||
        value > maximum) {
        std::fprintf(
            stderr,
            "%s must be between 1 and %u\n",
            name,
            maximum);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<std::uint32_t>(value);
}

void enable_peer_access(
    int source,
    int destination) {
    check_cuda(
        cudaSetDevice(source),
        "cudaSetDevice(peer source)");
    int supported = 0;
    check_cuda(
        cudaDeviceCanAccessPeer(
            &supported,
            source,
            destination),
        "cudaDeviceCanAccessPeer");
    if (supported == 0) {
        fail_message(
            "Required CUDA peer access is unavailable");
    }
    const cudaError_t status =
        cudaDeviceEnablePeerAccess(
            destination,
            0);
    if (status ==
        cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    } else {
        check_cuda(
            status,
            "cudaDeviceEnablePeerAccess");
    }
}

void disable_peer_access(
    int source,
    int destination) {
    check_cuda(
        cudaSetDevice(source),
        "cudaSetDevice(peer source)");
    const cudaError_t status =
        cudaDeviceDisablePeerAccess(destination);
    if (status ==
        cudaErrorPeerAccessNotEnabled) {
        cudaGetLastError();
    } else {
        check_cuda(
            status,
            "cudaDeviceDisablePeerAccess");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const aboleth::Backend backend =
        parse_backend(argc, argv);
    const std::uint32_t rounds =
        parse_positive(
            argc,
            argv,
            2,
            kDefaultRounds,
            1'000'000,
            "rounds");
    const std::uint32_t transfers =
        parse_positive(
            argc,
            argv,
            3,
            kDefaultTransfers,
            kMaximumTransfers,
            "transfers");

    int device_count = 0;
    check_cuda(
        cudaGetDeviceCount(&device_count),
        "cudaGetDeviceCount");
    if (device_count < 2) {
        fail_message(
            "Ping-pong test requires two GPUs");
    }

    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    check_cuda(
        cudaSetDeviceFlags(cudaDeviceMapHost),
        "cudaSetDeviceFlags(0)");
    check_cuda(
        cudaSetDevice(1),
        "cudaSetDevice(1)");
    check_cuda(
        cudaSetDeviceFlags(cudaDeviceMapHost),
        "cudaSetDeviceFlags(1)");

    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(
            &properties,
            0),
        "cudaGetDeviceProperties");
#if CUDART_VERSION >= 13000
    int device_clock_khz = 0;
    check_cuda(
        cudaDeviceGetAttribute(
            &device_clock_khz,
            cudaDevAttrClockRate,
            0),
        "cudaDeviceGetAttribute(clockRate)");
#else
    const int device_clock_khz = properties.clockRate;
#endif
    const std::uint64_t timeout_cycles =
        static_cast<std::uint64_t>(
            device_clock_khz) *
        1000ULL *
        kWaitTimeoutSeconds;

    const std::size_t batch_bytes =
        static_cast<std::size_t>(transfers) *
        kBytesPerTransfer;
    const std::size_t data_bytes =
        kBufferSlots * batch_bytes;
    const std::size_t allocation_bytes =
        data_bytes +
        2 * sizeof(std::uint64_t);

    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    std::uint8_t* allocation0 = nullptr;
    check_cuda(
        cudaMalloc(
            &allocation0,
            allocation_bytes),
        "cudaMalloc(GPU 0 allocation)");
    check_cuda(
        cudaMemset(
            allocation0,
            0,
            allocation_bytes),
        "cudaMemset(GPU 0 allocation)");
    auto* data0 = allocation0;
    auto* acknowledgment0 =
        reinterpret_cast<std::uint64_t*>(
            allocation0 + data_bytes);
    auto* ready_value0 = acknowledgment0 + 1;

    check_cuda(
        cudaSetDevice(1),
        "cudaSetDevice(1)");
    std::uint8_t* allocation1 = nullptr;
    check_cuda(
        cudaMalloc(
            &allocation1,
            allocation_bytes),
        "cudaMalloc(GPU 1 allocation)");
    check_cuda(
        cudaMemset(
            allocation1,
            0,
            allocation_bytes),
        "cudaMemset(GPU 1 allocation)");
    auto* data1 = allocation1;
    auto* ready1 =
        reinterpret_cast<std::uint64_t*>(
            allocation1 + data_bytes);
    auto* acknowledgment_value1 = ready1 + 1;

    bool peer_enabled = false;
    if (backend == aboleth::Backend::P2P) {
        enable_peer_access(0, 1);
        enable_peer_access(1, 0);
        peer_enabled = true;
    }

    EndpointPair endpoints{};
    endpoints.gpu0_to_gpu1.backend = backend;
    endpoints.gpu1_to_gpu0.backend = backend;
    RdmaRuntime rdma_runtime{};
    if (backend == aboleth::Backend::Rdma) {
        endpoints = create_rdma_endpoints(
            rdma_runtime,
            allocation0,
            allocation_bytes,
            allocation1,
            allocation_bytes,
            timeout_cycles);
    }

    aboleth::LocalBuffer producer_local{
        data0,
        ready_value0,
    };
    aboleth::RemoteBuffer producer_remote{
        data1,
        ready1,
    };
    aboleth::LocalBuffer consumer_local{
        data1,
        acknowledgment_value1,
    };
    aboleth::RemoteBuffer consumer_remote{
        data0,
        acknowledgment0,
    };

    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    ProducerResult* producer_result_device =
        nullptr;
    cudaStream_t producer_stream{};
    check_cuda(
        cudaMalloc(
            &producer_result_device,
            sizeof(ProducerResult)),
        "cudaMalloc(producer result)");
    check_cuda(
        cudaMemset(
            producer_result_device,
            0,
            sizeof(ProducerResult)),
        "cudaMemset(producer result)");
    check_cuda(
        cudaStreamCreateWithFlags(
            &producer_stream,
            cudaStreamNonBlocking),
        "cudaStreamCreate(producer)");

    check_cuda(
        cudaSetDevice(1),
        "cudaSetDevice(1)");
    ConsumerResult* consumer_result_device =
        nullptr;
    cudaStream_t consumer_stream{};
    check_cuda(
        cudaMalloc(
            &consumer_result_device,
            sizeof(ConsumerResult)),
        "cudaMalloc(consumer result)");
    check_cuda(
        cudaMemset(
            consumer_result_device,
            0,
            sizeof(ConsumerResult)),
        "cudaMemset(consumer result)");
    check_cuda(
        cudaStreamCreateWithFlags(
            &consumer_stream,
            cudaStreamNonBlocking),
        "cudaStreamCreate(consumer)");

    persistent_consumer<<<1, 1, 0, consumer_stream>>>(
        endpoints.gpu1_to_gpu0,
        consumer_local,
        consumer_remote,
        ready1,
        rounds,
        transfers,
        kBytesPerTransfer,
        timeout_cycles,
        consumer_result_device);
    check_cuda(
        cudaGetLastError(),
        "persistent_consumer launch");

    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    persistent_producer<<<1, 1, 0, producer_stream>>>(
        endpoints.gpu0_to_gpu1,
        producer_local,
        producer_remote,
        acknowledgment0,
        rounds,
        transfers,
        kBytesPerTransfer,
        timeout_cycles,
        producer_result_device);
    check_cuda(
        cudaGetLastError(),
        "persistent_producer launch");

    check_cuda(
        cudaStreamSynchronize(
            producer_stream),
        "persistent_producer execution");
    ProducerResult producer_result{};
    check_cuda(
        cudaMemcpy(
            &producer_result,
            producer_result_device,
            sizeof(producer_result),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(producer result)");

    check_cuda(
        cudaSetDevice(1),
        "cudaSetDevice(1)");
    check_cuda(
        cudaStreamSynchronize(
            consumer_stream),
        "persistent_consumer execution");
    ConsumerResult consumer_result{};
    check_cuda(
        cudaMemcpy(
            &consumer_result,
            consumer_result_device,
            sizeof(consumer_result),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(consumer result)");

    std::uint64_t ready_value = 0;
    check_cuda(
        cudaMemcpy(
            &ready_value,
            ready1,
            sizeof(ready_value),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(ready signal)");
    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    std::uint64_t acknowledgment_value = 0;
    check_cuda(
        cudaMemcpy(
            &acknowledgment_value,
            acknowledgment0,
            sizeof(acknowledgment_value),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(acknowledgment)");

    const bool success =
        producer_result.failure ==
            FailureStage::None &&
        consumer_result.failure ==
            FailureStage::None &&
        producer_result.rounds_sent ==
            rounds &&
        consumer_result.rounds_consumed ==
            rounds &&
        ready_value == rounds &&
        acknowledgment_value == rounds;

    std::printf(
        "Backend: %s, rounds: %u, transfers/round: %u, bytes/transfer: %u\n",
        backend_name(backend),
        rounds,
        transfers,
        kBytesPerTransfer);
    std::printf(
        "Producer: failure=%s round=%u completed=%u cycles=%llu\n",
        failure_name(producer_result.failure),
        producer_result.failed_round,
        producer_result.rounds_sent,
        static_cast<unsigned long long>(
            producer_result.total_cycles));
    std::printf(
        "Consumer: failure=%s round=%u byte=%u completed=%u cycles=%llu\n",
        failure_name(consumer_result.failure),
        consumer_result.failed_round,
        consumer_result.failed_byte,
        consumer_result.rounds_consumed,
        static_cast<unsigned long long>(
            consumer_result.total_cycles));
    std::printf(
        "Signals: ready=%llu acknowledgment=%llu\n",
        static_cast<unsigned long long>(
            ready_value),
        static_cast<unsigned long long>(
            acknowledgment_value));
    std::printf(
        "Persistent Aboleth ping-pong: %s\n",
        success ? "PASS" : "FAIL");

    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    cudaFree(producer_result_device);
    cudaStreamDestroy(producer_stream);
    if (backend == aboleth::Backend::Rdma) {
        destroy_rdma_runtime(rdma_runtime);
    }
    if (peer_enabled) {
        disable_peer_access(0, 1);
        disable_peer_access(1, 0);
    }
    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    cudaFree(allocation0);

    check_cuda(
        cudaSetDevice(1),
        "cudaSetDevice(1)");
    cudaFree(consumer_result_device);
    cudaStreamDestroy(consumer_stream);
    cudaFree(allocation1);
    return success
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
