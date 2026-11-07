#include <cstdint>

#include "aboleth/Target/RdmaDevice.cuh"

using aboleth::abi::RdmaEndpointDescriptor;
namespace rdma = aboleth::cuda::rdma;

namespace {

__device__ inline const RdmaEndpointDescriptor& endpoint(
    std::uint64_t address) {
    return *reinterpret_cast<const RdmaEndpointDescriptor*>(
        address);
}

} // namespace

extern "C" {

__device__ std::uint64_t aboleth_rdma_reserve_batch(
    std::uint64_t endpoint_address,
    std::uint32_t count) {
    return rdma::reserve_count(
        endpoint(endpoint_address), count);
}

__device__ std::uint64_t aboleth_rdma_reserve(
    std::uint64_t endpoint_address,
    std::uint32_t planned,
    std::uint32_t cta_exclusive) {
    const auto& value = endpoint(endpoint_address);
    return planned
        ? rdma::reserve_planned(
              value, cta_exclusive != 0)
        : rdma::reserve(value);
}

__device__ std::uint64_t aboleth_rdma_reserve_planned(
    std::uint64_t endpoint_address,
    std::uint32_t count,
    std::uint32_t cta_exclusive) {
    return rdma::reserve_planned(
        endpoint(endpoint_address),
        cta_exclusive != 0,
        count);
}

__device__ std::uint64_t aboleth_rdma_post_nop(
    std::uint64_t endpoint_address,
    std::uint64_t index) {
    rdma::write_nop(
        endpoint(endpoint_address), index);
    return 0;
}

__device__ std::uint64_t aboleth_rdma_post_write(
    std::uint64_t endpoint_address,
    std::uint64_t index,
    std::uint64_t source,
    std::uint32_t source_key,
    std::uint64_t destination,
    std::uint32_t destination_key,
    std::uint32_t bytes,
    std::uint32_t predicate,
    std::uint32_t request_completion,
    std::uint32_t control_word) {
    if (predicate)
        rdma::write_rdma(
            endpoint(endpoint_address),
            index,
            reinterpret_cast<const void*>(source),
            source_key,
            destination,
            destination_key,
            bytes,
            request_completion != 0,
            control_word);
    else
        rdma::write_nop(
            endpoint(endpoint_address), index);
    return 0;
}

__device__ std::uint64_t aboleth_rdma_post_inline_u64(
    std::uint64_t endpoint_address,
    std::uint64_t index,
    std::uint64_t destination,
    std::uint32_t destination_key,
    std::uint64_t value,
    std::uint32_t request_completion,
    std::uint32_t control_word) {
    rdma::write_inline<std::uint64_t>(
        endpoint(endpoint_address),
        index,
        destination,
        destination_key,
        value,
        request_completion != 0,
        control_word);
    return 0;
}

__device__ std::uint64_t aboleth_rdma_post_atomic_signal(
    std::uint64_t endpoint_address,
    std::uint64_t index,
    std::uint64_t destination,
    std::uint32_t destination_key,
    std::uint64_t epoch,
    std::uint32_t request_completion) {
    rdma::write_atomic_signal(
        endpoint(endpoint_address),
        index,
        destination,
        destination_key,
        epoch,
        request_completion != 0);
    return 0;
}

__device__ std::uint64_t aboleth_rdma_publish(
    std::uint64_t endpoint_address,
    std::uint64_t lock_address,
    std::uint64_t base,
    std::uint64_t end,
    std::uint32_t ordered_head_only) {
    const auto& value = endpoint(endpoint_address);
    auto* lock = reinterpret_cast<std::uint32_t*>(
        lock_address);
    if (ordered_head_only)
        rdma::publish_head(value, lock, end);
    else
        rdma::publish(value, lock, base, end);
    return 0;
}

__device__ std::uint64_t aboleth_rdma_publish_head(
    std::uint64_t endpoint_address,
    std::uint64_t lock_address,
    std::uint64_t head) {
    rdma::publish_head(
        endpoint(endpoint_address),
        reinterpret_cast<std::uint32_t*>(lock_address),
        head);
    return 0;
}

__device__ std::uint64_t aboleth_rdma_poll(
    std::uint64_t endpoint_address,
    std::uint64_t base,
    std::uint64_t end) {
    rdma::poll(endpoint(endpoint_address), base, end);
    return 0;
}

__device__ std::uint64_t aboleth_rdma_drain(
    std::uint64_t endpoint_address,
    std::uint64_t lock_address) {
    rdma::drain(
        endpoint(endpoint_address),
        reinterpret_cast<std::uint32_t*>(lock_address));
    return 0;
}

} // extern "C"
