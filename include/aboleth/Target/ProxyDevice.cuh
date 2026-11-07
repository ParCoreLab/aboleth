#pragma once

// Device-side host-proxy submission: a kernel posts commands to a proxy
// endpoint's queue (ProxyAbi.h) and a host thread executes them on the
// NIC. Used by the Tier-1 proxy benchmark arm (experiments/bench).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "aboleth/Target/ProxyAbi.h"

namespace aboleth::cuda {
namespace proxy_detail {

__device__ __forceinline__ std::uint64_t load_acquire(
    const std::uint64_t* pointer) {
    std::uint64_t value;
    asm volatile(
        "ld.acquire.sys.global.L1::no_allocate.u64 %0, [%1];"
        : "=l"(value)
        : "l"(pointer)
        : "memory");
    return value;
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

__device__ __forceinline__ void fail(
    unsigned int code) {
    printf(
        "Aboleth proxy failure code=%u block=%u thread=%u\n",
        code,
        blockIdx.x,
        threadIdx.x);
    asm volatile("trap;");
}

__device__ __forceinline__ std::uint64_t submit(
    const abi::ProxyEndpointDescriptor& descriptor,
    const abi::ProxyCommand& command) {
    if (descriptor.abi_version !=
            abi::kProxyEndpointVersion ||
        descriptor.slot_count !=
            abi::kProxySlotCount ||
        descriptor.maximum_commands <
            1) {
        fail(1);
    }
    auto* producer =
        reinterpret_cast<unsigned long long*>(
            descriptor.producer_index_address);
    auto* queue =
        reinterpret_cast<abi::ProxyQueueState*>(
            descriptor.queue_state);
    const auto base =
        atomicAdd(producer, 1ULL);
    const auto target = base + 1;
    const auto start = clock64();
    while (target >
           load_acquire(
               &queue->completed_sequence) +
               descriptor.slot_count) {
        if (load_acquire(
                &queue->status) != 0) {
            fail(2);
        }
        if (clock64() - start >
            descriptor.completion_timeout_cycles) {
            fail(3);
        }
        __nanosleep(64);
    }
    auto& slot =
        queue->slots[
            base % descriptor.slot_count];
    slot.command_count = 1;
    slot.reserved = 0;
    slot.commands[0] = command;
    __threadfence_system();
    store_release(
        &slot.ready_sequence,
        target);
    return target;
}

__device__ __forceinline__ void wait_completed(
    const abi::ProxyEndpointDescriptor& descriptor,
    std::uint64_t target) {
    const auto* queue =
        reinterpret_cast<
            const abi::ProxyQueueState*>(
            descriptor.queue_state);
    const auto start = clock64();
    while (load_acquire(
               &queue->completed_sequence) <
           target) {
        if (load_acquire(
                &queue->status) != 0) {
            fail(4);
        }
        if (clock64() - start >
            descriptor.completion_timeout_cycles) {
            fail(5);
        }
        __nanosleep(64);
    }
    if (load_acquire(&queue->status) != 0) {
        fail(6);
    }
}

__device__ __forceinline__ std::uint64_t
snapshot_producer(
    const abi::ProxyEndpointDescriptor& descriptor) {
    return load_acquire(
        reinterpret_cast<
            const std::uint64_t*>(
            descriptor.producer_index_address));
}

__device__ __forceinline__ void drain(
    const abi::ProxyEndpointDescriptor& descriptor) {
    wait_completed(
        descriptor,
        snapshot_producer(descriptor));
}

__device__ __forceinline__ void write(
    const abi::ProxyEndpointDescriptor& descriptor,
    const void* source,
    std::uint32_t source_key,
    std::uint64_t destination,
    std::uint32_t destination_key,
    std::uint32_t bytes) {
    abi::ProxyCommand command{};
    command.source_address =
        reinterpret_cast<std::uint64_t>(source);
    command.destination_address = destination;
    command.bytes = bytes;
    command.source_key = source_key;
    command.destination_key = destination_key;
    command.opcode =
        abi::kProxyCommandWrite;
    submit(descriptor, command);
}

__device__ __forceinline__ void inline_write(
    const abi::ProxyEndpointDescriptor& descriptor,
    std::uint64_t destination,
    std::uint32_t destination_key,
    std::uint64_t value,
    std::uint32_t bytes,
    std::uint32_t opcode) {
    abi::ProxyCommand command{};
    command.source_address = value;
    command.destination_address = destination;
    command.bytes = bytes;
    command.destination_key = destination_key;
    command.opcode = opcode;
    submit(descriptor, command);
}

__device__ __forceinline__ void signal(
    const abi::ProxyEndpointDescriptor& descriptor,
    std::uint64_t destination,
    std::uint32_t destination_key,
    std::uint64_t value) {
    abi::ProxyCommand command{};
    command.source_address = value;
    command.destination_address = destination;
    command.bytes = sizeof(value);
    command.destination_key = destination_key;
    command.opcode =
        abi::kProxyCommandStrongSignal;
    const auto target =
        submit(descriptor, command);
    wait_completed(descriptor, target);
}

}  // namespace proxy_detail
}  // namespace aboleth::cuda
