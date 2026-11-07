// Device primitives for the EFA backend, compiled to LLVM bitcode and linked
// into generated Triton kernels.
//
// Every wrapper takes and returns flat 64- and 32-bit scalars because Triton
// extern calls carry scalars only, and each reads what it needs from the
// endpoint descriptor rather than taking a long argument list. The wrappers
// are inlined by the LLVM linker, so the generated kernel contains the
// efa-dp-direct instruction sequence directly with no call overhead.
//
// Queue-state mutation (batch start, flush, completion pop) must be
// serialized per queue; Aboleth's planned mode gives one producer CTA per
// endpoint, and the caller elects a single thread for those wrappers.

#include "aboleth/Target/EfaDpDirectDevice.cuh"
#include "aboleth/Target/EfaAbi.h"

namespace {

using namespace aboleth::abi;

struct Endpoint {
    efa_cuda_qp *queue_pair;
    efa_cuda_cq *send_cq;
    efa_cuda_cq *recv_cq;
    std::uint64_t *state;
    std::uint64_t timeout_cycles;
    std::uint16_t address_handle;
    std::uint32_t remote_qpn;
    std::uint32_t queue_key;
};

__device__ inline Endpoint readEndpoint(std::uint64_t address) {
    const std::uint64_t *words =
        reinterpret_cast<const std::uint64_t *>(address);
    Endpoint endpoint;
    endpoint.queue_pair =
        reinterpret_cast<efa_cuda_qp *>(words[kEfaWordQueuePair]);
    endpoint.send_cq =
        reinterpret_cast<efa_cuda_cq *>(words[kEfaWordSendCq]);
    endpoint.recv_cq =
        reinterpret_cast<efa_cuda_cq *>(words[kEfaWordRecvCq]);
    endpoint.state =
        reinterpret_cast<std::uint64_t *>(words[kEfaWordCompletionState]);
    endpoint.timeout_cycles = words[kEfaWordTimeoutCycles];
    const std::uint64_t handle = words[kEfaWordAddressHandle];
    endpoint.address_handle = static_cast<std::uint16_t>(handle);
    endpoint.remote_qpn = static_cast<std::uint32_t>(handle >> 32);
    endpoint.queue_key =
        static_cast<std::uint32_t>(words[kEfaWordQueueKey]);
    return endpoint;
}

__device__ inline std::uint64_t readClock() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value));
    return value;
}

// Consume completions until `target` of them have retired on this endpoint.
// SRD completions arrive out of order, so the retired head is a count rather
// than an index; each popped entry retires exactly one command.
__device__ inline std::uint64_t advanceCompletions(
    const Endpoint &endpoint,
    std::uint64_t target,
    efa_cuda_cq *cq) {
    std::uint64_t *retired = &endpoint.state[kEfaStateRetired];
    const std::uint64_t deadline =
        endpoint.timeout_cycles == 0 ? 0 : readClock() + endpoint.timeout_cycles;
    while (*retired < target) {
        void *completion = efa_cuda_cq_poll(cq, 0);
        if (completion == nullptr) {
            if (deadline != 0 && readClock() > deadline) {
                endpoint.state[kEfaStateError] = kEfaStatusTimeout;
                return kEfaStatusTimeout;
            }
            continue;
        }
        const std::uint32_t vendor_error =
            efa_cuda_wc_read_vendor_err(completion);
        efa_cuda_cq_pop(cq, 1);
        if (vendor_error != 0) {
            // The error word carries the adapter's completion status in its
            // upper bits so a failed run names the hardware reason (e.g.
            // 0x602 = BAD_LENGTH) instead of a bare remote-error code.
            endpoint.state[kEfaStateError] =
                kEfaStatusRemoteError |
                (static_cast<std::uint64_t>(vendor_error) << 8);
            return kEfaStatusRemoteError;
        }
        *retired += 1;
    }
    return kEfaStatusOk;
}

} // namespace

extern "C" {

// Open a batch of `count` commands. Serialized per queue by the caller.
__device__ std::uint64_t aboleth_efa_begin_batch(
    std::uint64_t endpoint_address,
    std::uint32_t count) {
    Endpoint endpoint = readEndpoint(endpoint_address);
    efa_cuda_start_sq_batch(endpoint.queue_pair, static_cast<int>(count));
    return kEfaStatusOk;
}

__device__ std::uint64_t aboleth_efa_post_write(
    std::uint64_t endpoint_address,
    std::uint32_t slot,
    std::uint64_t local_address,
    std::uint32_t local_key,
    std::uint64_t remote_address,
    std::uint32_t remote_key,
    std::uint32_t bytes,
    std::uint32_t request_id) {
    Endpoint endpoint = readEndpoint(endpoint_address);
    struct efa_io_tx_wqe request;
    if (efa_cuda_init_rdma_write_wr(
            &request, static_cast<std::uint16_t>(request_id), remote_key,
            remote_address) != 0)
        return kEfaStatusInvalid;
    efa_cuda_wr_set_remote(
        &request, endpoint.address_handle, endpoint.remote_qpn,
        endpoint.queue_key);
    // A disabled transfer still occupies its slot as a zero-length write, so a
    // batch always produces the number of completions the compiler planned.
    if (efa_cuda_wr_set_sge(
            &request, local_key, local_address, bytes) != 0)
        return kEfaStatusInvalid;
    // clang folds the union's scatter-gather stores into a zero memset when
    // optimizing the copy below (the vendor macros mix volatile field reads
    // with union writes); forcing the descriptor to escape to memory keeps
    // the placed bytes complete. Zero-cost compiler barrier.
    asm volatile("" ::"l"(&request) : "memory");
    efa_cuda_sq_batch_place_wr(
        endpoint.queue_pair, static_cast<int>(slot), &request);
    return kEfaStatusOk;
}

__device__ std::uint64_t aboleth_efa_post_write_imm(
    std::uint64_t endpoint_address,
    std::uint32_t slot,
    std::uint64_t local_address,
    std::uint32_t local_key,
    std::uint64_t remote_address,
    std::uint32_t remote_key,
    std::uint32_t bytes,
    std::uint32_t request_id,
    std::uint32_t immediate) {
    Endpoint endpoint = readEndpoint(endpoint_address);
    struct efa_io_tx_wqe request;
    if (efa_cuda_init_rdma_write_imm_wr(
            &request, static_cast<std::uint16_t>(request_id), remote_key,
            remote_address, immediate) != 0)
        return kEfaStatusInvalid;
    efa_cuda_wr_set_remote(
        &request, endpoint.address_handle, endpoint.remote_qpn,
        endpoint.queue_key);
    if (efa_cuda_wr_set_sge(
            &request, local_key, local_address, bytes) != 0)
        return kEfaStatusInvalid;
    // clang folds the union's scatter-gather stores into a zero memset when
    // optimizing the copy below (the vendor macros mix volatile field reads
    // with union writes); forcing the descriptor to escape to memory keeps
    // the placed bytes complete. Zero-cost compiler barrier.
    asm volatile("" ::"l"(&request) : "memory");
    efa_cuda_sq_batch_place_wr(
        endpoint.queue_pair, static_cast<int>(slot), &request);
    return kEfaStatusOk;
}

// Publish an 8-byte epoch from a registered source region. The adapter
// rejects inline data on RDMA writes (efa_cuda_wr_set_inline_data is
// send-only), so a signal is an ordinary write whose source is the epoch
// cell the compiler already keeps registered.
__device__ std::uint64_t aboleth_efa_post_inline(
    std::uint64_t endpoint_address,
    std::uint32_t slot,
    std::uint64_t source_address,
    std::uint32_t source_key,
    std::uint64_t remote_address,
    std::uint32_t remote_key,
    std::uint32_t request_id) {
    Endpoint endpoint = readEndpoint(endpoint_address);
    struct efa_io_tx_wqe request;
    if (efa_cuda_init_rdma_write_wr(
            &request, static_cast<std::uint16_t>(request_id), remote_key,
            remote_address) != 0)
        return kEfaStatusInvalid;
    efa_cuda_wr_set_remote(
        &request, endpoint.address_handle, endpoint.remote_qpn,
        endpoint.queue_key);
    if (efa_cuda_wr_set_sge(
            &request, source_key, source_address,
            sizeof(std::uint64_t)) != 0)
        return kEfaStatusInvalid;
    // clang folds the union's scatter-gather stores into a zero memset when
    // optimizing the copy below (the vendor macros mix volatile field reads
    // with union writes); forcing the descriptor to escape to memory keeps
    // the placed bytes complete. Zero-cost compiler barrier.
    asm volatile("" ::"l"(&request) : "memory");
    efa_cuda_sq_batch_place_wr(
        endpoint.queue_pair, static_cast<int>(slot), &request);
    return kEfaStatusOk;
}

// Ring the send doorbell for everything placed since the batch opened.
__device__ std::uint64_t aboleth_efa_flush(
    std::uint64_t endpoint_address) {
    Endpoint endpoint = readEndpoint(endpoint_address);
    efa_cuda_flush_sq_wrs(endpoint.queue_pair);
    return kEfaStatusOk;
}

// Retire submissions until `target` commands have completed on this endpoint.
__device__ std::uint64_t aboleth_efa_completion_advance(
    std::uint64_t endpoint_address,
    std::uint64_t target) {
    Endpoint endpoint = readEndpoint(endpoint_address);
    return advanceCompletions(endpoint, target, endpoint.send_cq);
}

// Receiver side of immediate-counting publication: consume arrival
// completions, replenish the receive queue, and advance the local epoch once
// the writes belonging to a publication have all landed.
__device__ std::uint64_t aboleth_efa_receive_advance(
    std::uint64_t endpoint_address,
    std::uint64_t expected,
    std::uint64_t signal_address,
    std::uint32_t writes_per_epoch) {
    Endpoint endpoint = readEndpoint(endpoint_address);
    if (endpoint.recv_cq == nullptr || writes_per_epoch == 0)
        return kEfaStatusInvalid;
    std::uint64_t *cell = reinterpret_cast<std::uint64_t *>(signal_address);
    std::uint64_t *arrivals = &endpoint.state[kEfaStatePublish];
    const std::uint64_t deadline =
        endpoint.timeout_cycles == 0 ? 0 : readClock() + endpoint.timeout_cycles;
    while (*cell < expected) {
        void *completion = efa_cuda_cq_poll(endpoint.recv_cq, 0);
        if (completion == nullptr) {
            if (deadline != 0 && readClock() > deadline) {
                endpoint.state[kEfaStateError] = kEfaStatusTimeout;
                return kEfaStatusTimeout;
            }
            continue;
        }
        const bool carries_epoch = efa_cuda_wc_has_imm(completion);
        const std::uint32_t immediate =
            carries_epoch ? efa_cuda_wc_read_imm_data(completion) : 0u;
        efa_cuda_cq_pop(endpoint.recv_cq, 1);
        if (!carries_epoch)
            continue;
        *arrivals += 1;
        if (*arrivals % writes_per_epoch == 0) {
            // Every write of this publication has landed, so the epoch it
            // carried is now safe to observe.
            __threadfence_system();
            *cell = immediate;
        }
    }
    return kEfaStatusOk;
}

} // extern "C"
