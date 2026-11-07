#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "aboleth/Target/RdmaAbi.h"

namespace aboleth::cuda::rdma {


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

struct alignas(16) WqeAtomic {
    std::uint64_t swap_add;
    std::uint64_t compare;
};

static_assert(sizeof(WqeCtrl) == 16);
static_assert(sizeof(WqeRemoteAddress) == 16);
static_assert(sizeof(WqeData) == 16);
static_assert(sizeof(WqeAtomic) == 16);

__device__ __forceinline__ std::uint32_t to_be32(std::uint32_t value) {
    std::uint32_t result;
    asm(
        "{ .reg .b32 ignored; prmt.b32 %0, %1, ignored, 0x0123; }"
        : "=r"(result)
        : "r"(value));
    return result;
}

__device__ __forceinline__ std::uint16_t from_be16(std::uint16_t value) {
    return static_cast<std::uint16_t>((value >> 8) | (value << 8));
}

__device__ __forceinline__ std::uint64_t to_be64(std::uint64_t value) {
    std::uint64_t result;
    asm(
        "{ .reg .b32 low, high, new_low, new_high, ignored; "
        "mov.b64 {low,high}, %1; "
        "prmt.b32 new_high, low, ignored, 0x0123; "
        "prmt.b32 new_low, high, ignored, 0x0123; "
        "mov.b64 %0, {new_low,new_high}; }"
        : "=l"(result)
        : "l"(value));
    return result;
}

__device__ __forceinline__ void store_no_allocate(
    int4* pointer, const int4& value) {
    asm volatile(
        "st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(pointer), "r"(value.x), "r"(value.y),
          "r"(value.z), "r"(value.w)
        : "memory");
}

__device__ __forceinline__ void store_release(
    std::uint32_t* pointer, std::uint32_t value) {
    asm volatile(
        "st.release.sys.global.L1::no_allocate.b32 [%0], %1;"
        :
        : "l"(pointer), "r"(value)
        : "memory");
}

__device__ __forceinline__ void store_relaxed(
    std::uint32_t* pointer, std::uint32_t value) {
    asm volatile(
        "st.relaxed.sys.global.L1::no_allocate.b32 [%0], %1;"
        :
        : "l"(pointer), "r"(value)
        : "memory");
}

__device__ __forceinline__ void store_release(
    std::uint64_t* pointer, std::uint64_t value) {
    asm volatile(
        "st.release.sys.global.L1::no_allocate.b64 [%0], %1;"
        :
        : "l"(pointer), "l"(value)
        : "memory");
}

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

__device__ __forceinline__ std::uint8_t load_volatile(
    const std::uint8_t* pointer) {
    std::uint16_t value;
    asm volatile(
        "ld.volatile.global.b8 %0, [%1];"
        : "=h"(value)
        : "l"(pointer)
        : "memory");
    return static_cast<std::uint8_t>(value);
}

__device__ __forceinline__ std::uint16_t load_volatile(
    const std::uint16_t* pointer) {
    std::uint16_t value;
    asm volatile(
        "ld.volatile.global.b16 %0, [%1];"
        : "=h"(value)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ void fail() {
    asm volatile("trap;");
}

__device__ __forceinline__ std::uint64_t* completion_state(
    const abi::RdmaEndpointDescriptor& endpoint, std::uint32_t word) {
    return reinterpret_cast<std::uint64_t*>(
        endpoint.completion_consumer_address) + word;
}

__device__ __forceinline__ std::uint64_t reserve_count(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint32_t count) {
    auto* producer = reinterpret_cast<unsigned long long*>(
        endpoint.producer_index_address);
    const auto base = atomicAdd(
        producer,
        static_cast<unsigned long long>(count));
    const auto end = base + count;
    while (end > load_acquire(completion_state(endpoint, 1)) +
                     endpoint.send_wqe_count)
        __nanosleep(64);
    return base;
}

__device__ __forceinline__ std::uint64_t reserve(
    const abi::RdmaEndpointDescriptor& endpoint) {
    return reserve_count(endpoint, 1);
}

__device__ __forceinline__ std::uint64_t reserve_planned(
    const abi::RdmaEndpointDescriptor& endpoint,
    bool cta_exclusive = false,
    std::uint32_t count = 1) {
    // preflight() proves capacity for the complete phase.
    const auto amount = static_cast<unsigned long long>(count);
    if (cta_exclusive)
        return atomicAdd_block(
            reinterpret_cast<unsigned long long*>(
                endpoint.producer_index_address),
            amount);
    return atomicAdd(
        reinterpret_cast<unsigned long long*>(
            endpoint.producer_index_address),
        amount);
}

__device__ __forceinline__ std::uint8_t* wqe_address(
    const abi::RdmaEndpointDescriptor& endpoint, std::uint64_t index) {
    const auto slot =
        index & (endpoint.send_wqe_count - 1);
    return reinterpret_cast<std::uint8_t*>(endpoint.send_queue) +
           slot * endpoint.send_stride;
}

__device__ __forceinline__ void ring_doorbell(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint32_t* lock, std::uint64_t head) {
    if (lock != nullptr) {
        while (atomicCAS(lock, 0u, 1u) != 0u)
            __nanosleep(64);
    }
    const auto head16 = static_cast<std::uint32_t>(head) & 0xffff;
    // The later BlueFlame release orders this DB record and the SQ stores.
    store_relaxed(
        reinterpret_cast<std::uint32_t*>(endpoint.send_dbrec),
        to_be32(head16));
    const auto doorbell_index = to_be32(head16 << 8);
    const auto doorbell_qpn = to_be32(endpoint.qpn << 8);
    const auto doorbell =
        (static_cast<std::uint64_t>(doorbell_qpn) << 32) |
        doorbell_index;
    store_release(
        reinterpret_cast<std::uint64_t*>(endpoint.blueflame),
        doorbell);
    if (lock != nullptr) {
        __threadfence_system();
        atomicExch(lock, 0u);
    }
}

__device__ __forceinline__ void publish(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint32_t* doorbell_lock,
    std::uint64_t base, std::uint64_t end) {
    auto* ready = completion_state(endpoint, 2);
    while (load_acquire(ready) != base)
        __nanosleep(64);
    __threadfence_system();
    ring_doorbell(
        endpoint, doorbell_lock, end);
    store_release(ready, end);
}

__device__ __forceinline__ void publish_head(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint32_t* doorbell_lock,
    std::uint64_t head) {
    ring_doorbell(
        endpoint, doorbell_lock, head);
    store_release(
        completion_state(endpoint, 2),
        head);
}

__device__ __forceinline__ void write_rdma(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint64_t index, const void* source, std::uint32_t source_lkey,
    std::uint64_t destination, std::uint32_t destination_rkey,
    std::uint32_t bytes, bool request_completion,
    std::uint32_t control_word = 0) {
    auto* wqe = wqe_address(endpoint, index);
    WqeCtrl ctrl{};
    ctrl.opmod_idx_opcode =
        to_be32(
            ((static_cast<std::uint32_t>(index) & 0xffff) << 8) |
            0x08);
    ctrl.qpn_ds = to_be32((endpoint.qpn << 8) | 3);
    ctrl.signature_dci_fm =
        control_word |
        (request_completion ? 0x08000000 : 0);
    WqeRemoteAddress remote{};
    remote.address = to_be64(destination);
    remote.rkey = to_be32(destination_rkey);
    WqeData data{};
    data.byte_count = to_be32(bytes);
    data.lkey = to_be32(source_lkey);
    data.address = to_be64(reinterpret_cast<std::uint64_t>(source));
    store_no_allocate(
        reinterpret_cast<int4*>(wqe),
        *reinterpret_cast<const int4*>(&ctrl));
    store_no_allocate(
        reinterpret_cast<int4*>(wqe + 16),
        *reinterpret_cast<const int4*>(&remote));
    store_no_allocate(
        reinterpret_cast<int4*>(wqe + 32),
        *reinterpret_cast<const int4*>(&data));
}

template <typename dtype_t>
__device__ __forceinline__ void write_inline(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint64_t index, std::uint64_t destination,
    std::uint32_t destination_rkey, dtype_t value,
    bool request_completion,
    std::uint32_t control_word = 0) {
    static_assert(sizeof(dtype_t) == 4 || sizeof(dtype_t) == 8);
    auto* wqe = wqe_address(endpoint, index);
    WqeCtrl ctrl{};
    ctrl.opmod_idx_opcode =
        to_be32(
            ((static_cast<std::uint32_t>(index) & 0xffff) << 8) |
            0x08);
    ctrl.qpn_ds = to_be32((endpoint.qpn << 8) | 3);
    ctrl.signature_dci_fm =
        control_word |
        (request_completion ? 0x08000000 : 0);
    WqeRemoteAddress remote{};
    remote.address = to_be64(destination);
    remote.rkey = to_be32(destination_rkey);
    store_no_allocate(
        reinterpret_cast<int4*>(wqe),
        *reinterpret_cast<const int4*>(&ctrl));
    store_no_allocate(
        reinterpret_cast<int4*>(wqe + 16),
        *reinterpret_cast<const int4*>(&remote));
    int4 inline_data{};
    inline_data.x = static_cast<int>(
        to_be32(0x80000000u | sizeof(dtype_t)));
    if constexpr (sizeof(dtype_t) == 8) {
        const auto bits =
            static_cast<std::uint64_t>(value);
        inline_data.y = static_cast<int>(
            static_cast<std::uint32_t>(bits));
        inline_data.z = static_cast<int>(
            static_cast<std::uint32_t>(
                bits >> 32));
    } else {
        inline_data.y = static_cast<int>(
            static_cast<std::uint32_t>(value));
    }
    store_no_allocate(
        reinterpret_cast<int4*>(wqe + 32),
        inline_data);
}

__device__ __forceinline__ void write_nop(
    const abi::RdmaEndpointDescriptor& endpoint, std::uint64_t index) {
    auto* wqe = wqe_address(endpoint, index);
    WqeCtrl ctrl{};
    ctrl.opmod_idx_opcode =
        to_be32(
            (static_cast<std::uint32_t>(index) & 0xffff) << 8);
    ctrl.qpn_ds = to_be32((endpoint.qpn << 8) | 1);
    ctrl.signature_dci_fm = 0x08000000;
    store_no_allocate(
        reinterpret_cast<int4*>(wqe),
        *reinterpret_cast<const int4*>(&ctrl));
}

__device__ __forceinline__ void write_atomic_signal(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint64_t index, std::uint64_t destination,
    std::uint32_t destination_rkey, std::uint64_t epoch,
    bool request_completion) {
    auto* wqe = wqe_address(endpoint, index);
    WqeCtrl ctrl{};
    ctrl.opmod_idx_opcode =
        to_be32(
            ((static_cast<std::uint32_t>(index) & 0xffff) << 8) |
            0x11);
    ctrl.qpn_ds = to_be32((endpoint.qpn << 8) | 4);
    ctrl.signature_dci_fm =
        request_completion ? 0x08000000 : 0;
    WqeRemoteAddress remote{};
    remote.address = to_be64(destination);
    remote.rkey = to_be32(destination_rkey);
    WqeAtomic atomic{};
    atomic.swap_add = to_be64(epoch);
    atomic.compare = to_be64(epoch - 1);
    WqeData result{};
    result.byte_count = to_be32(sizeof(std::uint64_t));
    result.lkey = to_be32(endpoint.atomic_result_lkey);
    result.address = to_be64(
        endpoint.completion_consumer_address +
        abi::kRdmaAtomicResultOffset);
    store_no_allocate(
        reinterpret_cast<int4*>(wqe),
        *reinterpret_cast<const int4*>(&ctrl));
    store_no_allocate(
        reinterpret_cast<int4*>(wqe + 16),
        *reinterpret_cast<const int4*>(&remote));
    store_no_allocate(
        reinterpret_cast<int4*>(wqe + 32),
        *reinterpret_cast<const int4*>(&atomic));
    store_no_allocate(
        reinterpret_cast<int4*>(wqe + 48),
        *reinterpret_cast<const int4*>(&result));
}

__device__ __forceinline__ void poll(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint64_t base, std::uint64_t end) {
    auto* completed = completion_state(endpoint, 1);
    while (load_acquire(completed) != base)
        __nanosleep(64);

    auto* consumer_state = completion_state(endpoint, 0);
    const auto consumer = load_acquire(consumer_state);
    const auto cqe_index =
        consumer % endpoint.completion_entry_count;
    const auto cqe_offset =
        cqe_index * endpoint.completion_entry_size +
        (endpoint.completion_entry_size == 128 ? 64 : 0);
    const auto* cqe =
        reinterpret_cast<const std::uint8_t*>(endpoint.completion_queue) +
        cqe_offset;
    const auto* op_own = cqe + 63;
    const auto expected_owner =
        static_cast<std::uint8_t>(
            (consumer / endpoint.completion_entry_count) & 1);
    const auto start = clock64();
    std::uint8_t observed = 0xff;
    while (clock64() - start < endpoint.completion_timeout_cycles) {
        observed = load_volatile(op_own);
        if ((observed >> 4) != 15 &&
            (observed & 1) == expected_owner)
            break;
        __nanosleep(64);
    }
    const auto opcode = observed >> 4;
    const auto owner = observed & 1;
    const auto counter = from_be16(load_volatile(
        reinterpret_cast<const std::uint16_t*>(cqe + 60)));
    if (opcode != 0 || owner != expected_owner ||
        counter != (static_cast<std::uint16_t>(end - 1) & 0xffff)) {
        printf(
            "Aboleth CQ failure rank-qpn=%u opcode=%u owner=%u/%u counter=%u/%u base=%llu end=%llu\n",
            endpoint.qpn, opcode, owner, expected_owner,
            counter,
            static_cast<unsigned>(
                static_cast<std::uint16_t>(end - 1)),
            static_cast<unsigned long long>(base),
            static_cast<unsigned long long>(end));
        *completion_state(endpoint, 3) =
            opcode == 15 ? 1 : (opcode != 0 ? 2 : 3);
        fail();
    }

    const auto next_consumer = consumer + 1;
    store_release(consumer_state, next_consumer);
    store_release(
        reinterpret_cast<std::uint32_t*>(endpoint.completion_dbrec),
        to_be32(static_cast<std::uint32_t>(next_consumer) & 0x00ffffff));
    store_release(completed, end);
}

__device__ __forceinline__ void drain(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint32_t* doorbell_lock) {
    const auto completion_base =
        load_acquire(completion_state(endpoint, 1));
    const auto base = reserve(endpoint);
    const auto end = base + 1;
    write_nop(endpoint, base);
    publish_head(
        endpoint, doorbell_lock, end);
    poll(endpoint, completion_base, end);
}

__device__ __forceinline__ void signal(
    const abi::RdmaEndpointDescriptor& endpoint,
    std::uint32_t* doorbell_lock,
    std::uint64_t destination, std::uint32_t rkey,
    std::uint64_t epoch, bool wait_completion) {
    const auto base = wait_completion
        ? reserve(endpoint)
        : reserve_planned(endpoint);
    const auto end = base + 1;
    if (wait_completion)
        write_atomic_signal(
            endpoint, base, destination, rkey, epoch,
            /*request_completion=*/true);
    else
        // A fenced write on relaxed-ordering registrations. Publication
        // planning selects the compare-and-swap above instead; only
        // hand-written callers of this primitive reach this branch.
        write_inline<std::uint64_t>(
            endpoint, base, destination, rkey,
            epoch,
            /*request_completion=*/false,
            /*strong ordering=*/0x60000000);
    if (wait_completion)
        publish(
            endpoint, doorbell_lock, base, end);
    else
        publish_head(
            endpoint, doorbell_lock, end);
    if (wait_completion)
        poll(endpoint, base, end);
}


} // namespace aboleth::cuda::rdma
