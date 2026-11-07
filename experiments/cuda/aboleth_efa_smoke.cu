// End-to-end check of the GPU-initiated EFA path with no compiler involved.
//
// Two GPUs, one SRD endpoint each, a kernel on the first that posts an RDMA
// write and an epoch signal through the same device primitives the compiler
// emits calls to, and a byte comparison on the second. If this passes, the
// host control plane and the device datapath agree.

#include "runtime/efa/aboleth_efa_runtime.h"
#include "runtime/efa/aboleth_efa_device.cu"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::uint32_t kPayloadBytes = 4096;
// Buffers are allocated larger than the base transfer so the host-posted
// matrix can exercise payloads up to 1 MiB from the same registrations.
constexpr std::uint32_t kPayloadCapacity = 1u << 20;
constexpr std::uint64_t kEpoch = 7;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr,
            "%s: %s\n",
            operation,
            cudaGetErrorString(status));
        std::exit(1);
    }
}

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "aboleth_efa_smoke: %s\n", message);
    std::fprintf(
        stderr, "  last error: %s\n", aboleth::efa::last_error());
    std::exit(1);
}

// One CTA, one thread: the batch protocol requires queue-state changes to be
// serialized per queue, and the compiler elects a single thread the same way.
__global__ void transfer_kernel(
    std::uint64_t endpoint,
    std::uint64_t source_address,
    std::uint32_t source_key,
    std::uint64_t destination_address,
    std::uint32_t destination_key,
    std::uint32_t bytes,
    std::uint64_t epoch_source_address,
    std::uint32_t epoch_source_key,
    std::uint64_t signal_address,
    std::uint32_t signal_key,
    std::uint64_t* status_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    std::uint64_t status = 0;
    status |= aboleth_efa_begin_batch(endpoint, 1);
    status |= aboleth_efa_post_write(
        endpoint,
        /*slot=*/0,
        source_address,
        source_key,
        destination_address,
        destination_key,
        bytes,
        /*request_id=*/0);
    status |= aboleth_efa_flush(endpoint);
    // Sender-fenced publication: retire the payload before the epoch travels.
    status |= aboleth_efa_completion_advance(endpoint, 1);
    status |= aboleth_efa_begin_batch(endpoint, 1);
    status |= aboleth_efa_post_inline(
        endpoint,
        /*slot=*/0,
        epoch_source_address,
        epoch_source_key,
        signal_address,
        signal_key,
        /*request_id=*/1);
    status |= aboleth_efa_flush(endpoint);
    status |= aboleth_efa_completion_advance(endpoint, 2);
    *status_out = status;
}

// Host-posted mode: the host places the descriptors and rings the doorbell
// through its own mapping of the same send queue, and the kernel only
// consumes completions from the queue in GPU memory. This is the diagnostic
// split for platforms (EFA gen2 / p5) whose fabric drops GPU stores into the
// adapter BAR: everything except those two stores still runs on the GPU.
__global__ void completion_kernel(
    std::uint64_t endpoint,
    std::uint64_t target,
    std::uint64_t* status_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    *status_out |= aboleth_efa_completion_advance(endpoint, target);
}

// Staged-submission matrix: the kernel posts a whole batch through the same
// primitives the compiler emits and retires its completions, while the
// runtime's doorbell agent moves the descriptors to the adapter.
__global__ void batch_kernel(
    std::uint64_t endpoint,
    std::uint32_t count,
    std::uint64_t source_address,
    std::uint32_t source_key,
    std::uint64_t destination_address,
    std::uint32_t destination_key,
    std::uint32_t bytes,
    std::uint32_t request_base,
    std::uint64_t target,
    std::uint64_t* status_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    std::uint64_t status = 0;
    status |= aboleth_efa_begin_batch(endpoint, count);
    for (std::uint32_t index = 0; index < count; ++index) {
        status |= aboleth_efa_post_write(
            endpoint,
            /*slot=*/index,
            source_address,
            source_key,
            destination_address,
            destination_key,
            bytes,
            /*request_id=*/(request_base + index) & 63);
    }
    status |= aboleth_efa_flush(endpoint);
    status |= aboleth_efa_completion_advance(endpoint, target);
    *status_out |= status;
}

struct Peer {
    int device;
    EfaRuntimeHandle* runtime;
    EfaEndpointHandle* endpoint;
    void* descriptor;
    void* payload;
    EfaBufferHandle* payload_registration;
    std::uint64_t* signal;
    EfaBufferHandle* signal_registration;
    std::uint64_t* epoch_source;
    EfaBufferHandle* epoch_source_registration;
};

Peer create_peer(int device) {
    Peer peer{};
    peer.device = device;
    check(cudaSetDevice(device), "cudaSetDevice");
    peer.runtime = aboleth::efa::runtime_create(device, nullptr);
    if (peer.runtime == nullptr) {
        fail("could not open an EFA device");
    }
    check(
        cudaMalloc(&peer.descriptor, 128),
        "cudaMalloc(descriptor)");
    peer.endpoint = aboleth::efa::endpoint_create(
        peer.runtime,
        /*send_depth=*/64,
        /*recv_depth=*/64,
        peer.descriptor,
        128);
    if (peer.endpoint == nullptr) {
        fail("could not create an EFA endpoint");
    }
    check(
        cudaMalloc(&peer.payload, kPayloadCapacity),
        "cudaMalloc(payload)");
    check(
        cudaMemset(peer.payload, 0, kPayloadCapacity),
        "cudaMemset(payload)");
    peer.payload_registration = aboleth::efa::register_buffer(
        peer.runtime, peer.payload, kPayloadCapacity);
    if (peer.payload_registration == nullptr) {
        fail("could not register the payload buffer");
    }
    check(
        cudaMalloc(&peer.signal, sizeof(std::uint64_t)),
        "cudaMalloc(signal)");
    check(
        cudaMemset(peer.signal, 0, sizeof(std::uint64_t)),
        "cudaMemset(signal)");
    peer.signal_registration = aboleth::efa::register_buffer(
        peer.runtime, peer.signal, sizeof(std::uint64_t));
    if (peer.signal_registration == nullptr) {
        fail("could not register the signal cell");
    }
    // Signals travel as ordinary writes from a registered epoch cell; EFA has
    // no inline data on RDMA writes.
    check(
        cudaMalloc(&peer.epoch_source, sizeof(std::uint64_t)),
        "cudaMalloc(epoch source)");
    check(
        cudaMemset(peer.epoch_source, 0, sizeof(std::uint64_t)),
        "cudaMemset(epoch source)");
    peer.epoch_source_registration = aboleth::efa::register_buffer(
        peer.runtime, peer.epoch_source, sizeof(std::uint64_t));
    if (peer.epoch_source_registration == nullptr) {
        fail("could not register the epoch source cell");
    }
    return peer;
}

// Post `posts` host-posted writes of `bytes` each from the sender payload to
// the receiver payload, retiring completions on the GPU in chunks so the
// send queue never overfills. Returns false on a rejected post or a nonzero
// kernel status.
bool host_post_payload(
    Peer& sender,
    Peer& receiver,
    std::uint64_t endpoint_words,
    std::uint64_t* status_device,
    std::uint64_t& retired_target,
    std::uint32_t bytes,
    std::uint32_t posts,
    std::uint32_t chunk) {
    std::uint32_t issued = 0;
    while (issued < posts) {
        const std::uint32_t batch =
            chunk < posts - issued ? chunk : posts - issued;
        check(cudaSetDevice(sender.device), "cudaSetDevice(matrix)");
        completion_kernel<<<1, 32>>>(
            endpoint_words, retired_target + batch, status_device);
        check(cudaGetLastError(), "completion_kernel launch(matrix)");
        for (std::uint32_t index = 0; index < batch; ++index) {
            if (aboleth::efa::endpoint_host_post_write(
                    sender.endpoint,
                    reinterpret_cast<std::uint64_t>(sender.payload),
                    aboleth::efa::buffer_local_key(
                        sender.payload_registration),
                    reinterpret_cast<std::uint64_t>(receiver.payload),
                    aboleth::efa::buffer_remote_key(
                        receiver.payload_registration),
                    bytes,
                    /*request_id=*/
                    static_cast<std::uint32_t>(
                        (retired_target + index) & 63)) != 0) {
                return false;
            }
        }
        check(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize(matrix)");
        retired_target += batch;
        issued += batch;
    }
    std::uint64_t status = 0;
    check(
        cudaMemcpy(
            &status,
            status_device,
            sizeof(status),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(matrix status)");
    return status == 0;
}

// Staged-mode poster: the kernel builds and posts the batch itself; the
// doorbell agent submits it. Same signature as host_post_payload so the
// matrix below runs against either.
bool staged_post_payload(
    Peer& sender,
    Peer& receiver,
    std::uint64_t endpoint_words,
    std::uint64_t* status_device,
    std::uint64_t& retired_target,
    std::uint32_t bytes,
    std::uint32_t posts,
    std::uint32_t chunk) {
    std::uint32_t issued = 0;
    while (issued < posts) {
        const std::uint32_t batch =
            chunk < posts - issued ? chunk : posts - issued;
        check(cudaSetDevice(sender.device), "cudaSetDevice(matrix)");
        batch_kernel<<<1, 32>>>(
            endpoint_words,
            batch,
            reinterpret_cast<std::uint64_t>(sender.payload),
            aboleth::efa::buffer_local_key(
                sender.payload_registration),
            reinterpret_cast<std::uint64_t>(receiver.payload),
            aboleth::efa::buffer_remote_key(
                receiver.payload_registration),
            bytes,
            static_cast<std::uint32_t>(retired_target & 63),
            retired_target + batch,
            status_device);
        check(cudaGetLastError(), "batch_kernel launch");
        check(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize(matrix)");
        retired_target += batch;
        issued += batch;
    }
    std::uint64_t status = 0;
    check(
        cudaMemcpy(
            &status,
            status_device,
            sizeof(status),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(matrix status)");
    return status == 0;
}

using MatrixPoster = bool (*)(
    Peer&,
    Peer&,
    std::uint64_t,
    std::uint64_t*,
    std::uint64_t&,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t);

// The matrix the handoff asks for, in whichever mode moves bytes on this
// platform: payload sizes, zero-length transfers, send-queue wraparound
// (the producer crosses the 64-slot queue, flipping the descriptor phase),
// completion-queue wraparound (total completions exceed the queue), and the
// completion-timeout path. Timing is coarse — each op includes a kernel
// launch — but bounds the submission path.
bool run_matrix(
    Peer& sender,
    Peer& receiver,
    std::uint64_t endpoint_words,
    std::uint64_t* status_device,
    std::uint64_t& retired_target,
    const std::vector<std::uint8_t>& expected,
    MatrixPoster poster) {
    bool ok = true;

    const std::uint32_t sizes[] = {
        8, 256, 4096, 65536, kPayloadCapacity};
    for (const std::uint32_t bytes : sizes) {
        check(cudaSetDevice(receiver.device), "cudaSetDevice(zero)");
        check(
            cudaMemset(receiver.payload, 0, kPayloadCapacity),
            "cudaMemset(matrix)");
        bool case_ok = poster(
            sender, receiver, endpoint_words, status_device,
            retired_target, bytes, /*posts=*/1, /*chunk=*/1);
        std::vector<std::uint8_t> observed(bytes);
        check(cudaSetDevice(receiver.device), "cudaSetDevice(verify)");
        check(
            cudaMemcpy(
                observed.data(),
                receiver.payload,
                bytes,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(matrix verify)");
        case_ok = case_ok &&
            std::memcmp(
                observed.data(), expected.data(), bytes) == 0;
        if (bytes < kPayloadCapacity) {
            std::uint8_t tail = 0xff;
            check(
                cudaMemcpy(
                    &tail,
                    static_cast<std::uint8_t*>(receiver.payload) +
                        bytes,
                    1,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(matrix tail)");
            case_ok = case_ok && tail == 0;
        }
        std::printf(
            "matrix bytes=%u %s\n",
            bytes,
            case_ok ? "ok" : "FAIL");
        ok = ok && case_ok;
    }

    // A disabled transfer still occupies its slot as a zero-length write.
    check(cudaSetDevice(receiver.device), "cudaSetDevice(zero)");
    check(
        cudaMemset(receiver.payload, 0, kPayloadCapacity),
        "cudaMemset(zero-length)");
    bool case_ok = poster(
        sender, receiver, endpoint_words, status_device,
        retired_target, /*bytes=*/0, /*posts=*/1, /*chunk=*/1);
    std::uint64_t head = 0xff;
    check(cudaSetDevice(receiver.device), "cudaSetDevice(verify)");
    check(
        cudaMemcpy(
            &head,
            receiver.payload,
            sizeof(head),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(zero-length verify)");
    case_ok = case_ok && head == 0;
    std::printf(
        "matrix zero-length %s\n", case_ok ? "ok" : "FAIL");
    ok = ok && case_ok;

    // 96 more posts push the producer past the 64-slot send queue, so the
    // descriptor phase bit flips mid-run; the later timing posts push total
    // completions past the completion queue's capacity as well.
    case_ok = poster(
        sender, receiver, endpoint_words, status_device,
        retired_target, /*bytes=*/8, /*posts=*/96, /*chunk=*/16);
    std::printf(
        "matrix wraparound producer=%llu %s\n",
        static_cast<unsigned long long>(retired_target),
        case_ok ? "ok" : "FAIL");
    ok = ok && case_ok;

    const auto stamp = [] {
        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        return static_cast<double>(now.tv_sec) * 1e6 +
            static_cast<double>(now.tv_nsec) * 1e-3;
    };
    double begin = stamp();
    case_ok = poster(
        sender, receiver, endpoint_words, status_device,
        retired_target, /*bytes=*/8, /*posts=*/64, /*chunk=*/16);
    double elapsed = stamp() - begin;
    std::printf(
        "matrix timing 8B x64: %.1f us/op %s\n",
        elapsed / 64.0,
        case_ok ? "ok" : "FAIL");
    ok = ok && case_ok;
    begin = stamp();
    case_ok = poster(
        sender, receiver, endpoint_words, status_device,
        retired_target, kPayloadCapacity, /*posts=*/8, /*chunk=*/4);
    elapsed = stamp() - begin;
    std::printf(
        "matrix timing 1MiB x8: %.1f us/op (%.2f GB/s) %s\n",
        elapsed / 8.0,
        8.0 * kPayloadCapacity / (elapsed * 1e3),
        case_ok ? "ok" : "FAIL");
    ok = ok && case_ok;

    // The timeout path runs on the receiver's endpoint, whose send queue is
    // idle, so the sender's latched error word stays clean for the final
    // verdict.
    check(cudaSetDevice(receiver.device), "cudaSetDevice(timeout)");
    std::uint64_t* receiver_status = nullptr;
    check(
        cudaMalloc(&receiver_status, sizeof(std::uint64_t)),
        "cudaMalloc(timeout status)");
    check(
        cudaMemset(receiver_status, 0, sizeof(std::uint64_t)),
        "cudaMemset(timeout status)");
    completion_kernel<<<1, 32>>>(
        aboleth::efa::endpoint_descriptor_address(receiver.endpoint),
        /*target=*/1,
        receiver_status);
    check(cudaGetLastError(), "completion_kernel launch(timeout)");
    check(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize(timeout)");
    std::uint64_t timeout_status = 0;
    check(
        cudaMemcpy(
            &timeout_status,
            receiver_status,
            sizeof(timeout_status),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(timeout status)");
    check(cudaFree(receiver_status), "cudaFree(timeout status)");
    std::uint64_t receiver_error = 0;
    aboleth::efa::endpoint_state(
        receiver.endpoint, nullptr, nullptr, &receiver_error);
    case_ok = timeout_status == 1 && receiver_error == 1;
    std::printf(
        "matrix completion-timeout %s\n", case_ok ? "ok" : "FAIL");
    ok = ok && case_ok;

    return ok;
}

} // namespace

int main() {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::fprintf(
            stderr,
            "aboleth_efa_smoke: needs two GPUs, found %d\n",
            device_count);
        return 1;
    }

    Peer sender = create_peer(0);
    Peer receiver = create_peer(1);
    std::printf(
        "sender=gpu0/%s receiver=gpu1/%s submit=%s\n",
        aboleth::efa::runtime_device_name(sender.runtime),
        aboleth::efa::runtime_device_name(receiver.runtime),
        aboleth::efa::runtime_staged_submit(sender.runtime)
            ? "staged"
            : "gpu");

    // Exchange addresses the way a bootstrap would, then connect both ways.
    const auto sender_address =
        aboleth::efa::endpoint_address(sender.endpoint);
    const auto receiver_address =
        aboleth::efa::endpoint_address(receiver.endpoint);
    std::uint8_t packed[aboleth::efa::kAddressBytes]{};
    aboleth::efa::pack_address(
        receiver_address, packed, sizeof(packed));
    aboleth::efa::endpoint_connect(
        sender.endpoint,
        aboleth::efa::unpack_address(packed, sizeof(packed)));
    aboleth::efa::endpoint_connect(
        receiver.endpoint, sender_address);

    // A pattern the receiver cannot already hold.
    std::vector<std::uint8_t> expected(kPayloadCapacity);
    for (std::uint32_t index = 0; index < kPayloadCapacity; ++index) {
        expected[index] =
            static_cast<std::uint8_t>((index * 31 + 7) & 0xff);
    }
    check(cudaSetDevice(sender.device), "cudaSetDevice(sender)");
    check(
        cudaMemcpy(
            sender.payload,
            expected.data(),
            kPayloadCapacity,
            cudaMemcpyHostToDevice),
        "cudaMemcpy(payload)");
    check(
        cudaMemcpy(
            sender.epoch_source,
            &kEpoch,
            sizeof(kEpoch),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(epoch source)");

    std::uint64_t* status_device = nullptr;
    check(
        cudaMalloc(&status_device, sizeof(std::uint64_t)),
        "cudaMalloc(status)");
    check(
        cudaMemset(status_device, 0, sizeof(std::uint64_t)),
        "cudaMemset(status)");

    const std::uint64_t endpoint_words =
        aboleth::efa::endpoint_descriptor_address(sender.endpoint);
    bool matrix_ok = true;
    if (std::getenv("ABOLETH_EFA_HOST_POST") != nullptr) {
        // Diagnostic split: descriptors and doorbells from the host through
        // the same queues, completions consumed by the GPU. The kernel is
        // launched first and spins on the completion queue while the host
        // posts, mirroring how a producer CTA overlaps the adapter.
        std::printf("mode=host-post\n");
        completion_kernel<<<1, 32>>>(
            endpoint_words, /*target=*/1, status_device);
        check(cudaGetLastError(), "completion_kernel launch");
        if (aboleth::efa::endpoint_host_post_write(
                sender.endpoint,
                reinterpret_cast<std::uint64_t>(sender.payload),
                aboleth::efa::buffer_local_key(
                    sender.payload_registration),
                reinterpret_cast<std::uint64_t>(receiver.payload),
                aboleth::efa::buffer_remote_key(
                    receiver.payload_registration),
                kPayloadBytes,
                /*request_id=*/0) != 0) {
            fail("host-posted payload write was rejected");
        }
        check(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize(payload)");
        // The payload has retired on the sender, so the epoch may travel:
        // the same sender-fenced ordering the GPU path enforces.
        completion_kernel<<<1, 32>>>(
            endpoint_words, /*target=*/2, status_device);
        check(cudaGetLastError(), "completion_kernel launch");
        if (aboleth::efa::endpoint_host_post_write(
                sender.endpoint,
                reinterpret_cast<std::uint64_t>(sender.epoch_source),
                aboleth::efa::buffer_local_key(
                    sender.epoch_source_registration),
                reinterpret_cast<std::uint64_t>(receiver.signal),
                aboleth::efa::buffer_remote_key(
                    receiver.signal_registration),
                sizeof(std::uint64_t),
                /*request_id=*/1) != 0) {
            fail("host-posted signal write was rejected");
        }
        check(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize(signal)");
        std::uint64_t retired_target = 2;
        matrix_ok = run_matrix(
            sender, receiver, endpoint_words, status_device,
            retired_target, expected, host_post_payload);
    } else {
        transfer_kernel<<<1, 32>>>(
            endpoint_words,
            reinterpret_cast<std::uint64_t>(sender.payload),
            aboleth::efa::buffer_local_key(sender.payload_registration),
            reinterpret_cast<std::uint64_t>(receiver.payload),
            aboleth::efa::buffer_remote_key(
                receiver.payload_registration),
            kPayloadBytes,
            reinterpret_cast<std::uint64_t>(sender.epoch_source),
            aboleth::efa::buffer_local_key(
                sender.epoch_source_registration),
            reinterpret_cast<std::uint64_t>(receiver.signal),
            aboleth::efa::buffer_remote_key(
                receiver.signal_registration),
            status_device);
        check(cudaGetLastError(), "transfer_kernel launch");
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (aboleth::efa::runtime_staged_submit(sender.runtime)) {
            // The kernel posted everything itself; run the matrix the same
            // way, with the doorbell agent as the only host participant.
            std::uint64_t retired_target = 2;
            matrix_ok = run_matrix(
                sender, receiver, endpoint_words, status_device,
                retired_target, expected, staged_post_payload);
        }
    }

    std::uint64_t status = 0;
    check(
        cudaMemcpy(
            &status,
            status_device,
            sizeof(status),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(status)");

    std::uint64_t producer = 0;
    std::uint64_t retired = 0;
    std::uint64_t error = 0;
    aboleth::efa::endpoint_state(
        sender.endpoint, &producer, &retired, &error);
    std::printf(
        "status=%llu producer=%llu retired=%llu error=%llu\n",
        static_cast<unsigned long long>(status),
        static_cast<unsigned long long>(producer),
        static_cast<unsigned long long>(retired),
        static_cast<unsigned long long>(error));

    check(cudaSetDevice(receiver.device), "cudaSetDevice(receiver)");
    std::vector<std::uint8_t> observed(kPayloadBytes);
    check(
        cudaMemcpy(
            observed.data(),
            receiver.payload,
            kPayloadBytes,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(observed)");
    std::uint64_t epoch = 0;
    check(
        cudaMemcpy(
            &epoch,
            receiver.signal,
            sizeof(epoch),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(epoch)");

    const bool payload_matches =
        std::memcmp(
            observed.data(), expected.data(), kPayloadBytes) == 0;
    std::printf(
        "payload=%s epoch=%llu\n",
        payload_matches ? "MATCH" : "MISMATCH",
        static_cast<unsigned long long>(epoch));

    const bool passed =
        payload_matches && epoch == kEpoch && status == 0 &&
        error == 0 && matrix_ok;
    std::printf(
        "aboleth_efa_smoke=%s\n", passed ? "PASS" : "FAIL");

    aboleth::efa::unregister_buffer(sender.payload_registration);
    aboleth::efa::unregister_buffer(sender.signal_registration);
    aboleth::efa::unregister_buffer(sender.epoch_source_registration);
    aboleth::efa::unregister_buffer(receiver.payload_registration);
    aboleth::efa::unregister_buffer(receiver.signal_registration);
    aboleth::efa::unregister_buffer(receiver.epoch_source_registration);
    aboleth::efa::endpoint_destroy(sender.endpoint);
    aboleth::efa::endpoint_destroy(receiver.endpoint);
    aboleth::efa::runtime_destroy(sender.runtime);
    aboleth::efa::runtime_destroy(receiver.runtime);
    return passed ? 0 : 1;
}
