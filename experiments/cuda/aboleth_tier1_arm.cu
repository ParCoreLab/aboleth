// One arm of the Tier-1 wire differential.
//
// This file is compiled twice by `aboleth_tier1_differential.py`, against two
// different `aboleth/device.cuh`:
//
//   facade    — `include/` (the hand-written primitives; header-only)
//   generated — the generated library's include tree, device-linked against
//               libaboleth_tier1_<arch>.a (the compiler-emitted ops)
//
// The launchers are identical either way, which is the point: the calling
// code cannot tell the implementations apart, and the send-queue snapshot
// decides whether the wire can.
//
// Compiled at test time by the differential; not part of the CMake build.

#include <cuda_runtime.h>

#include <cstdint>

#include <aboleth/device.cuh>

#ifndef ABOLETH_TIER1_GENERATED
// The facade has no put_many; its arm plays one put per descriptor, which is
// exactly the per-call-doorbell behaviour the composite op exists to replace.
namespace aboleth {
struct PutDesc {
    std::uint64_t destination;
    std::uint64_t source;
    std::uint32_t bytes;
    std::uint32_t destination_key;
    std::uint32_t source_key;
    std::uint32_t _pad;
};
} // namespace aboleth
#endif

namespace {

__device__ __forceinline__ aboleth::DeviceContext context_of(
    const std::uint64_t* table) {
    return aboleth::DeviceContext{table, nullptr};
}

} // namespace

// Every kernel takes the one-peer endpoint table by pointer and issues from a
// single thread, matching the facade's per-thread contract.

__global__ void tier1_arm_put_kernel(
    const std::uint64_t* endpoints,
    std::uint64_t destination, std::uint64_t source, std::uint32_t bytes,
    std::uint32_t source_key, std::uint32_t destination_key,
    int quiet_after) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    const auto ctx = context_of(endpoints);
    aboleth::put(
        ctx, 0, {destination, destination_key}, {source, source_key}, bytes);
    if (quiet_after)
        aboleth::quiet(ctx, 0);
}

__global__ void tier1_arm_put_inline_kernel(
    const std::uint64_t* endpoints,
    std::uint64_t destination, std::uint32_t destination_key,
    std::uint64_t value, int quiet_after) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    const auto ctx = context_of(endpoints);
    aboleth::put_inline(ctx, 0, {destination, destination_key}, value);
    if (quiet_after)
        aboleth::quiet(ctx, 0);
}

__global__ void tier1_arm_signal_kernel(
    const std::uint64_t* endpoints,
    std::uint64_t flag, std::uint32_t flag_key,
    std::uint64_t epoch, int quiet_after) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    const auto ctx = context_of(endpoints);
    aboleth::signal(ctx, 0, {flag, flag_key}, epoch);
    if (quiet_after)
        aboleth::quiet(ctx, 0);
}

__global__ void tier1_arm_put_signal_kernel(
    const std::uint64_t* endpoints,
    std::uint64_t destination, std::uint64_t source, std::uint32_t bytes,
    std::uint32_t source_key, std::uint32_t destination_key,
    std::uint64_t flag, std::uint32_t flag_key, std::uint64_t epoch,
    int quiet_after) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    const auto ctx = context_of(endpoints);
#ifdef ABOLETH_TIER1_PUT_SIGNAL
    aboleth::put_signal(
        ctx, 0, {destination, destination_key}, {source, source_key}, bytes,
        {flag, flag_key}, epoch);
#else
    // The facade has no fused op; put-then-signal is the sequence whose wire
    // bytes the fused batch must reproduce.
    aboleth::put(
        ctx, 0, {destination, destination_key}, {source, source_key}, bytes);
    aboleth::signal(ctx, 0, {flag, flag_key}, epoch);
#endif
    if (quiet_after)
        aboleth::quiet(ctx, 0);
}

__global__ void tier1_arm_put_many_kernel(
    const std::uint64_t* endpoints,
    const aboleth::PutDesc* descriptors, std::uint32_t count,
    int quiet_after) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    const auto ctx = context_of(endpoints);
#ifdef ABOLETH_TIER1_GENERATED
    aboleth::put_many(ctx, 0, descriptors, count);
#else
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& desc = descriptors[index];
        aboleth::put(
            ctx, 0,
            {desc.destination, desc.destination_key},
            {desc.source, desc.source_key},
            desc.bytes);
    }
#endif
    if (quiet_after)
        aboleth::quiet(ctx, 0);
}

__global__ void tier1_arm_wait_signal_kernel(
    const std::uint64_t* flag, std::uint64_t epoch) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    aboleth::DeviceContext ctx{nullptr, nullptr};
    aboleth::wait_signal(ctx, flag, epoch);
}

__global__ void tier1_arm_quiet_kernel(const std::uint64_t* endpoints) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    aboleth::quiet(context_of(endpoints), 0);
}

namespace {

int launch_status() {
    const auto launched = cudaGetLastError();
    if (launched != cudaSuccess)
        return static_cast<int>(launched);
    return static_cast<int>(cudaDeviceSynchronize());
}

} // namespace

extern "C" int tier1_arm_put(
    std::uint64_t endpoints, std::uint64_t destination, std::uint64_t source,
    std::uint32_t bytes, std::uint32_t source_key,
    std::uint32_t destination_key, int quiet_after) {
    tier1_arm_put_kernel<<<1, 32>>>(
        reinterpret_cast<const std::uint64_t*>(endpoints),
        destination, source, bytes, source_key, destination_key,
        quiet_after);
    return launch_status();
}

extern "C" int tier1_arm_put_inline(
    std::uint64_t endpoints, std::uint64_t destination,
    std::uint32_t destination_key, std::uint64_t value, int quiet_after) {
    tier1_arm_put_inline_kernel<<<1, 32>>>(
        reinterpret_cast<const std::uint64_t*>(endpoints),
        destination, destination_key, value, quiet_after);
    return launch_status();
}

extern "C" int tier1_arm_signal(
    std::uint64_t endpoints, std::uint64_t flag, std::uint32_t flag_key,
    std::uint64_t epoch, int quiet_after) {
    tier1_arm_signal_kernel<<<1, 32>>>(
        reinterpret_cast<const std::uint64_t*>(endpoints),
        flag, flag_key, epoch, quiet_after);
    return launch_status();
}

extern "C" int tier1_arm_put_signal(
    std::uint64_t endpoints, std::uint64_t destination, std::uint64_t source,
    std::uint32_t bytes, std::uint32_t source_key,
    std::uint32_t destination_key, std::uint64_t flag,
    std::uint32_t flag_key, std::uint64_t epoch, int quiet_after) {
    tier1_arm_put_signal_kernel<<<1, 32>>>(
        reinterpret_cast<const std::uint64_t*>(endpoints),
        destination, source, bytes, source_key, destination_key,
        flag, flag_key, epoch, quiet_after);
    return launch_status();
}

extern "C" int tier1_arm_put_many(
    std::uint64_t endpoints, std::uint64_t descriptors, std::uint32_t count,
    int quiet_after) {
    tier1_arm_put_many_kernel<<<1, 32>>>(
        reinterpret_cast<const std::uint64_t*>(endpoints),
        reinterpret_cast<const aboleth::PutDesc*>(descriptors), count,
        quiet_after);
    return launch_status();
}

extern "C" int tier1_arm_wait_signal(
    std::uint64_t flag, std::uint64_t epoch) {
    tier1_arm_wait_signal_kernel<<<1, 32>>>(
        reinterpret_cast<const std::uint64_t*>(flag), epoch);
    return launch_status();
}

extern "C" int tier1_arm_quiet(std::uint64_t endpoints) {
    tier1_arm_quiet_kernel<<<1, 32>>>(
        reinterpret_cast<const std::uint64_t*>(endpoints));
    return launch_status();
}
