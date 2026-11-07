#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace aboleth {

enum class Backend : std::uint32_t {
    P2P = 0,
    Rdma = 1,
};

enum class WaitStatus : std::uint32_t {
    Success = 0,
    CompletionTimeout = 1,
    CompletionError = 2,
    UnexpectedWqeCounter = 3,
    InvalidTransfer = 4,
};

struct RdmaEndpoint {
    std::uint8_t* send_queue{};
    std::uint32_t send_stride{};
    std::uint32_t send_wqe_count{};
    std::uint32_t* send_dbrec{};
    std::uint64_t* blueflame{};
    std::uint32_t qpn{};
    std::uint32_t source_lkey{};
    std::uint32_t destination_rkey{};
    std::uint8_t* completion_queue{};
    std::uint32_t completion_entry_size{};
    std::uint32_t completion_entry_count{};
    std::uint32_t* completion_dbrec{};
    std::uint64_t completion_timeout_cycles{};
};

struct Endpoint {
    Backend backend{Backend::P2P};
    RdmaEndpoint rdma{};
};

struct LocalBuffer {
    const std::uint8_t* data{};
    const std::uint64_t* signal_value{};
};

struct RemoteBuffer {
    std::uint8_t* data{};
    std::uint64_t* signal{};
};

struct QueueState {
    std::uint32_t next_sequence{};
    std::uint32_t completion_consumer{};
};

struct Transfer {
    std::uint32_t sequence{};
};

struct WaitResult {
    WaitStatus status{WaitStatus::Success};
    std::uint32_t completion_opcode{};
    std::uint32_t completion_owner{};
    std::uint32_t completion_wqe_counter{};
    std::uint32_t completion_syndrome{};
    std::uint32_t completion_vendor_syndrome{};
    std::uint64_t poll_iterations{};
};

struct SignalWaitResult {
    WaitStatus status{WaitStatus::Success};
    std::uint64_t observed_value{};
    std::uint64_t poll_iterations{};
};

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

__device__ __forceinline__ std::uint32_t to_be32(
    std::uint32_t value) {
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

__device__ __forceinline__ std::uint16_t from_be16(
    std::uint16_t value) {
    return static_cast<std::uint16_t>(
        (value >> 8) |
        (value << 8));
}

__device__ __forceinline__ std::uint64_t to_be64(
    std::uint64_t value) {
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

__device__ __forceinline__ void store_no_allocate(
    std::uint32_t* pointer,
    std::uint32_t value) {
    asm volatile(
        "st.global.L1::no_allocate.b32 [%0], %1;"
        :
        : "l"(pointer), "r"(value)
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

__device__ __forceinline__ std::uint64_t load_acquire(
    const std::uint64_t* pointer) {
    std::uint64_t value;
    asm volatile(
        "ld.acquire.sys.global.L1::no_allocate.b64 %0, [%1];"
        : "=l"(value)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ void ring_doorbell(
    const RdmaEndpoint& endpoint,
    std::uint32_t producer_index) {
    store_release(
        endpoint.send_dbrec,
        to_be32(producer_index & 0xffff));
    const std::uint32_t doorbell_index =
        to_be32(
            (producer_index & 0xffff) << 8);
    const std::uint32_t doorbell_qpn =
        to_be32(endpoint.qpn << 8);
    const std::uint64_t doorbell =
        (static_cast<std::uint64_t>(
             doorbell_qpn) << 32) |
        doorbell_index;
    store_release(endpoint.blueflame, doorbell);
}

__device__ __forceinline__ Transfer put(
    const Endpoint& endpoint,
    QueueState& state,
    void* destination,
    const void* source,
    std::uint32_t bytes) {
    const std::uint32_t sequence =
        state.next_sequence++;

    if (endpoint.backend == Backend::P2P) {
        auto* destination_bytes =
            static_cast<std::uint8_t*>(destination);
        const auto* source_bytes =
            static_cast<const std::uint8_t*>(source);
        std::uint32_t offset = 0;
        for (; offset + sizeof(int4) <= bytes;
             offset += sizeof(int4)) {
            const int4 value =
                *reinterpret_cast<const int4*>(
                    source_bytes + offset);
            *reinterpret_cast<int4*>(
                destination_bytes + offset) = value;
        }
        for (; offset < bytes; ++offset) {
            destination_bytes[offset] =
                source_bytes[offset];
        }
        return Transfer{sequence + 1};
    }

    const auto& rdma = endpoint.rdma;
    auto* wqe =
        rdma.send_queue +
        (sequence % rdma.send_wqe_count) *
            rdma.send_stride;
    auto* ctrl_pointer =
        reinterpret_cast<WqeCtrl*>(wqe);
    auto* remote_pointer =
        reinterpret_cast<WqeRemoteAddress*>(
            wqe + sizeof(WqeCtrl));
    auto* data_pointer =
        reinterpret_cast<WqeData*>(
            wqe +
            sizeof(WqeCtrl) +
            sizeof(WqeRemoteAddress));

    constexpr std::uint32_t opcode_rdma_write = 0x08;
    constexpr std::uint32_t data_segments = 3;

    WqeCtrl ctrl{};
    ctrl.opmod_idx_opcode =
        to_be32(
            ((sequence & 0xffff) << 8) |
            opcode_rdma_write);
    ctrl.qpn_ds =
        to_be32(
            (rdma.qpn << 8) |
            data_segments);

    WqeRemoteAddress remote{};
    remote.address =
        to_be64(
            reinterpret_cast<std::uint64_t>(
                destination));
    remote.rkey =
        to_be32(rdma.destination_rkey);

    WqeData data{};
    data.byte_count = to_be32(bytes);
    data.lkey = to_be32(rdma.source_lkey);
    data.address =
        to_be64(
            reinterpret_cast<std::uint64_t>(
                source));

    store_no_allocate(
        reinterpret_cast<int4*>(ctrl_pointer),
        *reinterpret_cast<const int4*>(&ctrl));
    store_no_allocate(
        reinterpret_cast<int4*>(remote_pointer),
        *reinterpret_cast<const int4*>(&remote));
    store_no_allocate(
        reinterpret_cast<int4*>(data_pointer),
        *reinterpret_cast<const int4*>(&data));
    return Transfer{sequence + 1};
}

__device__ __forceinline__ Transfer put(
    const Endpoint& endpoint,
    QueueState& state,
    const RemoteBuffer& remote,
    std::size_t remote_offset,
    const LocalBuffer& local,
    std::size_t local_offset,
    std::uint32_t bytes) {
    return put(
        endpoint,
        state,
        remote.data + remote_offset,
        local.data + local_offset,
        bytes);
}

__device__ __forceinline__ Transfer signal(
    const Endpoint& endpoint,
    QueueState& state,
    const RemoteBuffer& remote,
    const LocalBuffer& local) {
    if (endpoint.backend == Backend::P2P) {
        const std::uint32_t sequence =
            state.next_sequence++;
        __threadfence_system();
        store_release(
            remote.signal,
            *local.signal_value);
        return Transfer{sequence + 1};
    }
    return put(
        endpoint,
        state,
        remote.signal,
        local.signal_value,
        sizeof(std::uint64_t));
}

__device__ __forceinline__ SignalWaitResult wait_signal(
    const std::uint64_t* signal,
    std::uint64_t expected_value,
    std::uint64_t timeout_cycles) {
    SignalWaitResult result{};
    const std::uint64_t start = clock64();
    while (clock64() - start < timeout_cycles) {
        const std::uint64_t observed =
            load_acquire(signal);
        ++result.poll_iterations;
        if (observed >= expected_value) {
            result.status = WaitStatus::Success;
            result.observed_value = observed;
            return result;
        }
        __nanosleep(64);
    }
    result.status = WaitStatus::CompletionTimeout;
    result.observed_value =
        load_acquire(signal);
    return result;
}

__device__ __forceinline__ WaitResult wait(
    const Endpoint& endpoint,
    QueueState& state,
    Transfer transfer) {
    WaitResult result{};
    if (transfer.sequence == 0 ||
        transfer.sequence > state.next_sequence) {
        result.status = WaitStatus::InvalidTransfer;
        return result;
    }

    if (endpoint.backend == Backend::P2P) {
        __threadfence_system();
        result.status = WaitStatus::Success;
        return result;
    }

    constexpr std::uint8_t cqe_request = 0;
    constexpr std::uint8_t cqe_request_error = 13;
    constexpr std::uint8_t cqe_invalid = 15;
    constexpr std::uint8_t wqe_ctrl_cq_update = 2 << 2;

    const auto& rdma = endpoint.rdma;
    const std::uint32_t last_wqe_index =
        transfer.sequence - 1;
    auto* last_wqe =
        rdma.send_queue +
        (last_wqe_index % rdma.send_wqe_count) *
            rdma.send_stride;
    auto* signal_word =
        reinterpret_cast<std::uint32_t*>(
            last_wqe + 8);
    store_no_allocate(
        signal_word,
        static_cast<std::uint32_t>(
            wqe_ctrl_cq_update) << 24);

    __threadfence_system();
    ring_doorbell(rdma, transfer.sequence);
    const std::uint64_t poll_start = clock64();

    const std::uint32_t consumer =
        state.completion_consumer;
    const std::uint32_t entry_index =
        consumer %
        rdma.completion_entry_count;
    const std::size_t cqe64_offset =
        rdma.completion_entry_size == 128
            ? 64
            : 0;
    const auto* cqe =
        rdma.completion_queue +
        static_cast<std::size_t>(entry_index) *
            rdma.completion_entry_size +
        cqe64_offset;
    const auto* wqe_counter_pointer =
        reinterpret_cast<const std::uint16_t*>(
            cqe + 60);
    const auto* op_own_pointer =
        reinterpret_cast<const std::uint8_t*>(
            cqe + 63);
    const std::uint8_t expected_owner =
        static_cast<std::uint8_t>(
            (consumer /
             rdma.completion_entry_count) &
            1);

    std::uint8_t op_own = 0xff;
    std::uint8_t opcode = cqe_invalid;
    bool completion_ready = false;
    while (clock64() - poll_start <
           rdma.completion_timeout_cycles) {
        op_own = load_no_allocate(op_own_pointer);
        opcode = op_own >> 4;
        const std::uint8_t owner =
            op_own & 1;
        ++result.poll_iterations;
        if (opcode != cqe_invalid &&
            owner == expected_owner) {
            completion_ready = true;
            break;
        }
        __nanosleep(64);
    }

    result.completion_opcode = opcode;
    result.completion_owner = op_own & 1;
    if (!completion_ready) {
        result.status =
            WaitStatus::CompletionTimeout;
        return result;
    }

    __threadfence_system();
    result.completion_wqe_counter =
        from_be16(
            load_no_allocate(
                wqe_counter_pointer));
    if (opcode == cqe_request_error ||
        opcode != cqe_request) {
        result.completion_vendor_syndrome =
            load_no_allocate(cqe + 54);
        result.completion_syndrome =
            load_no_allocate(cqe + 55);
        result.status =
            WaitStatus::CompletionError;
        return result;
    }
    if (result.completion_wqe_counter !=
        static_cast<std::uint16_t>(
            transfer.sequence - 1)) {
        result.status =
            WaitStatus::UnexpectedWqeCounter;
        return result;
    }

    ++state.completion_consumer;
    store_release(
        rdma.completion_dbrec,
        to_be32(
            state.completion_consumer &
            0xffffff));
    result.status = WaitStatus::Success;
    return result;
}

}  // namespace aboleth
