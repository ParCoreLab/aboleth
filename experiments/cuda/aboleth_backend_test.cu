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

// The producer's GPU, and the one whose HCA carries the transfer. The consumer
// runs on GPU 1; its memory is registered on this device's protection domain.
constexpr int kProducerGpuIndex = 0;
constexpr std::uint32_t kDefaultTransferCount = 32;
constexpr std::uint32_t kMaximumTransferCount = 256;
constexpr std::size_t kBytesPerTransfer = 256;
constexpr std::uint32_t kCompletionTimeoutSeconds = 5;
constexpr std::uint64_t kReadySignal =
    0x41424f4c45544801ULL;

struct ProducerResult {
    std::uint64_t put_cycles;
    std::uint64_t signal_cycles;
    std::uint64_t wait_cycles;
    std::uint64_t total_cycles;
    aboleth::WaitResult wait;
};

struct ConsumerResult {
    aboleth::SignalWaitResult signal_wait;
    std::uint32_t data_valid;
    std::uint32_t first_mismatch;
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

struct RegisteredMapping {
    void* host_base{};
    std::size_t bytes{};
    void* device_base{};
    unsigned int flags{};
};

class MappingRegistry {
public:
    void* map(
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

        for (const auto& mapping : mappings_) {
            const auto mapping_base =
                reinterpret_cast<std::uintptr_t>(
                    mapping.host_base);
            const auto mapping_end =
                mapping_base + mapping.bytes;
            const bool overlaps =
                base < mapping_end &&
                end > mapping_base;
            if (!overlaps) {
                continue;
            }
            if (flags == mapping.flags &&
                base >= mapping_base &&
                end <= mapping_end) {
                const auto offset =
                    start - mapping_base;
                return reinterpret_cast<void*>(
                    reinterpret_cast<std::uintptr_t>(
                        mapping.device_base) +
                    offset);
            }
            fail_message(
                "Overlapping CUDA host registrations are incompatible");
        }

        RegisteredMapping mapping{
            reinterpret_cast<void*>(base),
            end - base,
            nullptr,
            flags,
        };
        check_cuda(
            cudaHostRegister(
                mapping.host_base,
                mapping.bytes,
                flags),
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
            reinterpret_cast<std::uintptr_t>(
                mapping.device_base) +
            offset);
    }

    void unmap_all() {
        for (auto mapping = mappings_.rbegin();
             mapping != mappings_.rend();
             ++mapping) {
            check_cuda(
                cudaHostUnregister(
                    mapping->host_base),
                "cudaHostUnregister");
        }
        mappings_.clear();
    }

private:
    std::vector<RegisteredMapping> mappings_;
};

struct RdmaHostResources {
    ibv_device** devices{};
    ibv_context* context{};
    ibv_pd* pd{};
    ibv_cq* send_cq{};
    ibv_cq* peer_cq{};
    ibv_qp* sender{};
    ibv_qp* peer{};
    ibv_mr* source_mr{};
    ibv_mr* destination_mr{};
    MappingRegistry mappings;
};

ibv_qp* create_rc_qp(
    ibv_pd* pd,
    ibv_cq* cq) {
    ibv_qp_init_attr attr{};
    attr.send_cq = cq;
    attr.recv_cq = cq;
    attr.cap.max_send_wr =
        kMaximumTransferCount + 1;
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
    std::uint32_t local_psn) {
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

aboleth::Endpoint create_rdma_endpoint(
    RdmaHostResources& resources,
    void* source,
    void* destination,
    std::size_t bytes,
    std::uint64_t timeout_cycles) {
    const aboleth::probe::FabricAddress fabric =
        aboleth::probe::resolve_fabric_address(
            kProducerGpuIndex);
    aboleth::probe::report_fabric_address(fabric);

    int device_count = 0;
    resources.devices =
        ibv_get_device_list(&device_count);
    if (resources.devices == nullptr) {
        fail("ibv_get_device_list");
    }
    ibv_device* selected =
        aboleth::probe::find_fabric_device(
            resources.devices,
            device_count,
            fabric);

    resources.context =
        ibv_open_device(selected);
    if (resources.context == nullptr) {
        fail("ibv_open_device");
    }

    resources.pd =
        ibv_alloc_pd(resources.context);
    if (resources.pd == nullptr) {
        fail("ibv_alloc_pd");
    }
    resources.send_cq =
        ibv_create_cq(
            resources.context,
            kMaximumTransferCount,
            nullptr,
            nullptr,
            0);
    resources.peer_cq =
        ibv_create_cq(
            resources.context,
            1,
            nullptr,
            nullptr,
            0);
    if (resources.send_cq == nullptr ||
        resources.peer_cq == nullptr) {
        fail("ibv_create_cq");
    }

    resources.sender =
        create_rc_qp(
            resources.pd,
            resources.send_cq);
    resources.peer =
        create_rc_qp(
            resources.pd,
            resources.peer_cq);
    qp_to_init(resources.sender, fabric);
    qp_to_init(resources.peer, fabric);

    constexpr std::uint32_t sender_psn =
        0x123456;
    constexpr std::uint32_t peer_psn =
        0x654321;
    qp_to_rtr(
        resources.sender,
        resources.peer->qp_num,
        peer_psn,
        fabric);
    qp_to_rtr(
        resources.peer,
        resources.sender->qp_num,
        sender_psn,
        fabric);
    qp_to_rts(
        resources.sender,
        sender_psn);
    qp_to_rts(
        resources.peer,
        peer_psn);

    resources.source_mr =
        ibv_reg_mr(
            resources.pd,
            source,
            bytes,
            IBV_ACCESS_LOCAL_WRITE);
    resources.destination_mr =
        ibv_reg_mr(
            resources.pd,
            destination,
            bytes,
            IBV_ACCESS_LOCAL_WRITE |
                IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ);
    if (resources.source_mr == nullptr ||
        resources.destination_mr == nullptr) {
        fail("ibv_reg_mr(GPU memory)");
    }

    mlx5dv_qp direct_qp{};
    mlx5dv_cq direct_cq{};
    mlx5dv_obj direct_object{};
    direct_object.qp.in =
        resources.sender;
    direct_object.qp.out = &direct_qp;
    direct_object.cq.in =
        resources.send_cq;
    direct_object.cq.out = &direct_cq;
    if (mlx5dv_init_obj(
            &direct_object,
            MLX5DV_OBJ_QP |
                MLX5DV_OBJ_CQ) != 0) {
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

    const std::size_t send_queue_bytes =
        static_cast<std::size_t>(
            direct_qp.sq.wqe_cnt) *
        direct_qp.sq.stride;
    const std::size_t completion_queue_bytes =
        static_cast<std::size_t>(
            direct_cq.cqe_cnt) *
        direct_cq.cqe_size;

    aboleth::Endpoint endpoint{};
    endpoint.backend = aboleth::Backend::Rdma;
    auto& rdma = endpoint.rdma;
    rdma.send_queue =
        static_cast<std::uint8_t*>(
            resources.mappings.map(
                direct_qp.sq.buf,
                send_queue_bytes,
                cudaHostRegisterMapped));
    rdma.completion_queue =
        static_cast<std::uint8_t*>(
            resources.mappings.map(
                direct_cq.buf,
                completion_queue_bytes,
                cudaHostRegisterMapped));
    rdma.send_dbrec =
        static_cast<std::uint32_t*>(
            resources.mappings.map(
                direct_qp.dbrec,
                2 * sizeof(std::uint32_t),
                cudaHostRegisterMapped)) +
        MLX5_SND_DBR;
    rdma.completion_dbrec =
        static_cast<std::uint32_t*>(
            resources.mappings.map(
                direct_cq.dbrec,
                2 * sizeof(std::uint32_t),
                cudaHostRegisterMapped));
    rdma.blueflame =
        static_cast<std::uint64_t*>(
            resources.mappings.map(
                direct_qp.bf.reg,
                sizeof(std::uint64_t),
                cudaHostRegisterMapped |
                    cudaHostRegisterIoMemory));
    rdma.send_stride = direct_qp.sq.stride;
    rdma.send_wqe_count =
        direct_qp.sq.wqe_cnt;
    rdma.qpn = resources.sender->qp_num;
    rdma.source_lkey =
        resources.source_mr->lkey;
    rdma.destination_rkey =
        resources.destination_mr->rkey;
    rdma.completion_entry_size =
        direct_cq.cqe_size;
    rdma.completion_entry_count =
        direct_cq.cqe_cnt;
    rdma.completion_timeout_cycles =
        timeout_cycles;
    return endpoint;
}

void destroy_rdma_resources(
    RdmaHostResources& resources) {
    resources.mappings.unmap_all();
    if (resources.destination_mr != nullptr) {
        ibv_dereg_mr(
            resources.destination_mr);
    }
    if (resources.source_mr != nullptr) {
        ibv_dereg_mr(resources.source_mr);
    }
    if (resources.peer != nullptr) {
        ibv_destroy_qp(resources.peer);
    }
    if (resources.sender != nullptr) {
        ibv_destroy_qp(resources.sender);
    }
    if (resources.peer_cq != nullptr) {
        ibv_destroy_cq(resources.peer_cq);
    }
    if (resources.send_cq != nullptr) {
        ibv_destroy_cq(resources.send_cq);
    }
    if (resources.pd != nullptr) {
        ibv_dealloc_pd(resources.pd);
    }
    if (resources.context != nullptr) {
        ibv_close_device(resources.context);
    }
    if (resources.devices != nullptr) {
        ibv_free_device_list(
            resources.devices);
    }
}

__global__ void producer_kernel(
    aboleth::Endpoint endpoint,
    aboleth::LocalBuffer local,
    aboleth::RemoteBuffer remote,
    std::uint32_t transfer_count,
    std::uint32_t bytes_per_transfer,
    ProducerResult* result) {
    if (blockIdx.x != 0 ||
        threadIdx.x != 0) {
        return;
    }

    aboleth::QueueState state{};
    aboleth::Transfer transfer{};
    const std::uint64_t start = clock64();
    for (std::uint32_t index = 0;
         index < transfer_count;
         ++index) {
        transfer = aboleth::put(
            endpoint,
            state,
            remote,
            static_cast<std::size_t>(index) *
                bytes_per_transfer,
            local,
            static_cast<std::size_t>(index) *
                bytes_per_transfer,
            bytes_per_transfer);
    }
    const std::uint64_t put_end = clock64();
    transfer = aboleth::signal(
        endpoint,
        state,
        remote,
        local);
    const std::uint64_t signal_end = clock64();
    const aboleth::WaitResult wait_result =
        aboleth::wait(
            endpoint,
            state,
            transfer);
    const std::uint64_t wait_end = clock64();

    result->put_cycles = put_end - start;
    result->signal_cycles =
        signal_end - put_end;
    result->wait_cycles =
        wait_end - signal_end;
    result->total_cycles = wait_end - start;
    result->wait = wait_result;
}

__global__ void consumer_kernel(
    aboleth::RemoteBuffer remote,
    std::uint32_t transfer_count,
    std::uint32_t bytes_per_transfer,
    std::uint64_t expected_signal,
    std::uint64_t timeout_cycles,
    ConsumerResult* result) {
    if (blockIdx.x != 0 ||
        threadIdx.x != 0) {
        return;
    }

    result->signal_wait =
        aboleth::wait_signal(
            remote.signal,
            expected_signal,
            timeout_cycles);
    if (result->signal_wait.status !=
        aboleth::WaitStatus::Success) {
        result->data_valid = 0;
        result->first_mismatch =
            0xffffffffU;
        return;
    }

    const std::size_t total_bytes =
        static_cast<std::size_t>(
            transfer_count) *
        bytes_per_transfer;
    result->data_valid = 1;
    result->first_mismatch = 0xffffffffU;
    for (std::size_t index = 0;
         index < total_bytes;
         ++index) {
        const std::uint8_t expected =
            static_cast<std::uint8_t>(
                (index * 17 + 3) & 0xff);
        if (remote.data[index] != expected) {
            result->data_valid = 0;
            result->first_mismatch =
                static_cast<std::uint32_t>(
                    index);
            return;
        }
    }
}

const char* backend_name(
    aboleth::Backend backend) {
    return backend == aboleth::Backend::P2P
        ? "p2p"
        : "rdma";
}

const char* wait_status_name(
    aboleth::WaitStatus status) {
    switch (status) {
        case aboleth::WaitStatus::Success:
            return "success";
        case aboleth::WaitStatus::CompletionTimeout:
            return "completion-timeout";
        case aboleth::WaitStatus::CompletionError:
            return "completion-error";
        case aboleth::WaitStatus::UnexpectedWqeCounter:
            return "unexpected-wqe-counter";
        case aboleth::WaitStatus::InvalidTransfer:
            return "invalid-transfer";
    }
    return "unknown";
}

aboleth::Backend parse_backend(
    int argc,
    char** argv) {
    if (argc < 2) {
        fail_message(
            "Usage: aboleth_backend_test [p2p|rdma] [transfer-count]");
    }
    const std::string argument = argv[1];
    if (argument == "p2p") {
        return aboleth::Backend::P2P;
    }
    if (argument == "rdma") {
        return aboleth::Backend::Rdma;
    }
    fail_message(
        "Usage: aboleth_backend_test [p2p|rdma] [transfer-count]");
}

std::uint32_t parse_transfer_count(
    int argc,
    char** argv) {
    if (argc < 3) {
        return kDefaultTransferCount;
    }
    char* end = nullptr;
    const unsigned long parsed =
        std::strtoul(argv[2], &end, 10);
    if (end == argv[2] ||
        *end != '\0' ||
        parsed == 0 ||
        parsed > kMaximumTransferCount) {
        fail_message(
            "Transfer count must be between 1 and 256");
    }
    return static_cast<std::uint32_t>(
        parsed);
}

}  // namespace

int main(int argc, char** argv) {
    const aboleth::Backend backend =
        parse_backend(argc, argv);
    const std::uint32_t transfer_count =
        parse_transfer_count(argc, argv);
    const std::size_t total_bytes =
        static_cast<std::size_t>(
            transfer_count) *
        kBytesPerTransfer;
    const std::size_t allocation_bytes =
        total_bytes +
        sizeof(std::uint64_t);

    check_cuda(
        cudaSetDeviceFlags(cudaDeviceMapHost),
        "cudaSetDeviceFlags");
    check_cuda(cudaSetDevice(0), "cudaSetDevice(0)");

    int cuda_device_count = 0;
    check_cuda(
        cudaGetDeviceCount(&cuda_device_count),
        "cudaGetDeviceCount");
    if (cuda_device_count < 2) {
        fail_message(
            "Producer-consumer test requires two GPUs");
    }

    cudaDeviceProp device_properties{};
    check_cuda(
        cudaGetDeviceProperties(
            &device_properties,
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
    const int device_clock_khz = device_properties.clockRate;
#endif
    const std::uint64_t timeout_cycles =
        static_cast<std::uint64_t>(
            device_clock_khz) *
        1000ULL *
        kCompletionTimeoutSeconds;

    std::uint8_t* source_allocation = nullptr;
    std::uint8_t* destination_allocation =
        nullptr;
    check_cuda(
        cudaMalloc(
            &source_allocation,
            allocation_bytes),
        "cudaMalloc(source allocation)");
    auto* source = source_allocation;
    auto* source_signal =
        reinterpret_cast<std::uint64_t*>(
            source_allocation +
            total_bytes);

    bool peer_access_enabled = false;
    if (backend == aboleth::Backend::P2P) {
        int can_access_peer = 0;
        check_cuda(
            cudaDeviceCanAccessPeer(
                &can_access_peer,
                0,
                1),
            "cudaDeviceCanAccessPeer");
        if (can_access_peer == 0) {
            fail_message(
                "GPU 0 cannot access GPU 1");
        }
        const cudaError_t peer_status =
            cudaDeviceEnablePeerAccess(1, 0);
        if (peer_status ==
            cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
        } else {
            check_cuda(
                peer_status,
                "cudaDeviceEnablePeerAccess");
        }
        peer_access_enabled = true;
    }

    check_cuda(
        cudaSetDevice(1),
        "cudaSetDevice(1)");
    check_cuda(
        cudaMalloc(
            &destination_allocation,
            allocation_bytes),
        "cudaMalloc(destination allocation)");
    check_cuda(
        cudaMemset(
            destination_allocation,
            0,
            allocation_bytes),
        "cudaMemset(destination allocation)");
    auto* destination =
        destination_allocation;
    auto* destination_signal =
        reinterpret_cast<std::uint64_t*>(
            destination_allocation +
            total_bytes);
    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");

    std::vector<std::uint8_t> expected(
        total_bytes);
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
        cudaMemcpy(
            source_signal,
            &kReadySignal,
            sizeof(kReadySignal),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(source signal)");

    RdmaHostResources rdma_resources{};
    aboleth::Endpoint endpoint{};
    endpoint.backend = backend;
    if (backend == aboleth::Backend::Rdma) {
        endpoint = create_rdma_endpoint(
            rdma_resources,
            source_allocation,
            destination_allocation,
            allocation_bytes,
            timeout_cycles);
    }

    aboleth::LocalBuffer local{
        source,
        source_signal,
    };
    aboleth::RemoteBuffer remote{
        destination,
        destination_signal,
    };

    ProducerResult* device_producer_result =
        nullptr;
    cudaStream_t producer_stream{};
    check_cuda(
        cudaMalloc(
            &device_producer_result,
            sizeof(ProducerResult)),
        "cudaMalloc(producer result)");
    check_cuda(
        cudaMemset(
            device_producer_result,
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
    ConsumerResult* device_consumer_result =
        nullptr;
    cudaStream_t consumer_stream{};
    check_cuda(
        cudaMalloc(
            &device_consumer_result,
            sizeof(ConsumerResult)),
        "cudaMalloc(consumer result)");
    check_cuda(
        cudaMemset(
            device_consumer_result,
            0,
            sizeof(ConsumerResult)),
        "cudaMemset(consumer result)");
    check_cuda(
        cudaStreamCreateWithFlags(
            &consumer_stream,
            cudaStreamNonBlocking),
        "cudaStreamCreate(consumer)");

    consumer_kernel<<<1, 1, 0, consumer_stream>>>(
        remote,
        transfer_count,
        static_cast<std::uint32_t>(
            kBytesPerTransfer),
        kReadySignal,
        timeout_cycles,
        device_consumer_result);
    check_cuda(
        cudaGetLastError(),
        "consumer_kernel launch");

    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    producer_kernel<<<1, 1, 0, producer_stream>>>(
        endpoint,
        local,
        remote,
        transfer_count,
        static_cast<std::uint32_t>(
            kBytesPerTransfer),
        device_producer_result);
    check_cuda(
        cudaGetLastError(),
        "producer_kernel launch");
    check_cuda(
        cudaStreamSynchronize(
            producer_stream),
        "producer_kernel execution");

    ProducerResult producer_result{};
    check_cuda(
        cudaMemcpy(
            &producer_result,
            device_producer_result,
            sizeof(producer_result),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(producer result)");

    check_cuda(
        cudaSetDevice(1),
        "cudaSetDevice(1)");
    check_cuda(
        cudaStreamSynchronize(
            consumer_stream),
        "consumer_kernel execution");
    ConsumerResult consumer_result{};
    check_cuda(
        cudaMemcpy(
            &consumer_result,
            device_consumer_result,
            sizeof(consumer_result),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(consumer result)");

    std::vector<std::uint8_t> observed(
        total_bytes);
    check_cuda(
        cudaMemcpy(
            observed.data(),
            destination,
            observed.size(),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(destination)");
    const bool payload_matches =
        observed == expected;
    const bool consumer_valid =
        consumer_result.signal_wait.status ==
            aboleth::WaitStatus::Success &&
        consumer_result.data_valid != 0;

    check_cuda(
        cudaSetDevice(0),
        "cudaSetDevice(0)");
    const bool success =
        producer_result.wait.status ==
            aboleth::WaitStatus::Success &&
        consumer_valid &&
        payload_matches;
    std::printf(
        "Backend: %s, transfers: %u, bytes/transfer: %zu\n",
        backend_name(backend),
        transfer_count,
        kBytesPerTransfer);
    std::printf(
        "Producer wait: status=%s opcode=%u owner=%u wqe_counter=%u polls=%llu\n",
        wait_status_name(
            producer_result.wait.status),
        producer_result.wait.completion_opcode,
        producer_result.wait.completion_owner,
        producer_result.wait.completion_wqe_counter,
        static_cast<unsigned long long>(
            producer_result.wait.poll_iterations));
    std::printf(
        "Consumer signal: status=%s value=0x%llx polls=%llu data=%s mismatch=%u\n",
        wait_status_name(
            consumer_result.signal_wait.status),
        static_cast<unsigned long long>(
            consumer_result.signal_wait.observed_value),
        static_cast<unsigned long long>(
            consumer_result.signal_wait.poll_iterations),
        consumer_result.data_valid
            ? "VALID"
            : "INVALID",
        consumer_result.first_mismatch);
    std::printf(
        "Producer cycles: put=%llu signal=%llu wait=%llu total=%llu\n",
        static_cast<unsigned long long>(
            producer_result.put_cycles),
        static_cast<unsigned long long>(
            producer_result.signal_cycles),
        static_cast<unsigned long long>(
            producer_result.wait_cycles),
        static_cast<unsigned long long>(
            producer_result.total_cycles));
    std::printf(
        "Host payload check: %s\n",
        payload_matches ? "MATCH" : "MISMATCH");
    std::printf(
        "Aboleth producer-consumer: %s\n",
        success ? "PASS" : "FAIL");

    cudaFree(device_producer_result);
    cudaStreamDestroy(producer_stream);
    if (backend == aboleth::Backend::Rdma) {
        destroy_rdma_resources(
            rdma_resources);
    }
    if (peer_access_enabled) {
        const cudaError_t disable_status =
            cudaDeviceDisablePeerAccess(1);
        if (disable_status != cudaSuccess &&
            disable_status !=
                cudaErrorPeerAccessNotEnabled) {
            check_cuda(
                disable_status,
                "cudaDeviceDisablePeerAccess");
        } else if (
            disable_status ==
            cudaErrorPeerAccessNotEnabled) {
            cudaGetLastError();
        }
    }
    cudaFree(source_allocation);

    check_cuda(
        cudaSetDevice(1),
        "cudaSetDevice(1)");
    cudaFree(device_consumer_result);
    cudaStreamDestroy(consumer_stream);
    cudaFree(destination_allocation);
    return success
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
