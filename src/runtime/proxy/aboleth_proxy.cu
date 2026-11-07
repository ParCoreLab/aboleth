// The CPU-proxied datapath and the host thread that drives it.
//
// A proxy endpoint is a queue a kernel writes commands into; this file is
// everything on the other side of that queue. The datapath translates a batch
// slot into provider work — RC work requests on mlx5, RMA writes and a
// transmit counter on EFA-OFI — and publishes the completion the kernel spins
// on. Above it, one ProxyWorker per (runtime, GPU, shard) owns the progress
// thread and the shared completion queue every endpoint on that shard uses, so
// a process with many endpoints still has one poller per shard.
//
// The endpoint type itself, and the two entry points the rest of the runtime
// calls to build and cross-connect endpoints, are declared in aboleth_proxy.h.
// The libfabric objects behind an EFA proxy endpoint are built by
// efa/aboleth_efa_endpoints.cu; only the datapath that drives them is here.

#include "aboleth_proxy.h"

#include "../aboleth_runtime_common.h"
#include "../aboleth_runtime_internal.h"
#include "../core/aboleth_connection.h"
#include "../core/aboleth_endpoint_rc.h"
#include "../efa/aboleth_efa_internal.h"

#include "aboleth/Target/ProxyAbi.h"
#include "aboleth/runtime_c.h"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#if ABOLETH_HAS_EFA
// The EFA arm of the datapath is the only libfabric consumer here.
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#endif

#include "../dl/shims.h"

namespace dl = aboleth::dl;

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

// aboleth::runtime is where the shared internals live (see
// aboleth_runtime_internal.h). The anonymous-namespace halves of the datapath
// below reach them unqualified, as the runtime translation unit this code came
// from does.
using namespace aboleth::runtime;

namespace aboleth::runtime {

ProxyProviderKind configured_proxy_provider() {
    const char* value =
        std::getenv("ABOLETH_PROXY_PROVIDER");
    if (value == nullptr || value[0] == '\0' ||
        std::strcmp(value, "verbs") == 0 ||
        std::strcmp(value, "mlx5") == 0) {
        return ProxyProviderKind::VerbsRc;
    }
    if (std::strcmp(value, "efa") == 0 ||
        std::strcmp(value, "efa-ofi") == 0) {
#if ABOLETH_HAS_EFA
        return ProxyProviderKind::EfaOfi;
#else
        throw_error(kEfaUnavailable);
#endif
    }
    throw_error(
        "ABOLETH_PROXY_PROVIDER must be verbs or efa");
}

}  // namespace aboleth::runtime

namespace {

#if ABOLETH_HAS_EFA

bool poll_efa_proxy_completion(
    ProxyEndpointContext* endpoint,
    void* expected_context) {
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(
            kTimeoutSeconds);
    while (std::chrono::steady_clock::now() <
           deadline) {
        fi_cq_entry completion{};
        const ssize_t count =
            fi_cq_read(
                endpoint->efa_tx_cq,
                &completion,
                1);
        if (count == 1) {
            if (completion.op_context ==
                expected_context) {
                return true;
            }
            if (endpoint->debug) {
                std::fprintf(
                    stderr,
                    "aboleth EFA proxy unexpected completion context\n");
            }
            return false;
        }
        if (count == -FI_EAVAIL) {
            fi_cq_err_entry error{};
            fi_cq_readerr(
                endpoint->efa_tx_cq,
                &error,
                0);
            if (endpoint->debug) {
                char provider_error[256]{};
                const char* detail =
                    fi_cq_strerror(
                        endpoint->efa_tx_cq,
                        error.prov_errno,
                        error.err_data,
                        provider_error,
                        sizeof(provider_error));
                std::fprintf(
                    stderr,
                    "aboleth EFA proxy CQ error: %s (%d), provider %s (%d)\n",
                    dl::fabric().fi_strerror(error.err),
                    error.err,
                    detail == nullptr
                        ? "unknown"
                        : detail,
                    error.prov_errno);
                std::fflush(stderr);
            }
            return false;
        }
        if (count != -FI_EAGAIN) {
            if (endpoint->debug) {
                std::fprintf(
                    stderr,
                    "aboleth EFA proxy CQ read failed: %s\n",
                    dl::fabric().fi_strerror(
                        static_cast<int>(-count)));
                std::fflush(stderr);
            }
            return false;
        }
        fi_cq_data_entry ignored{};
        const ssize_t receive_count =
            fi_cq_read(
                endpoint->efa_rx_cq,
                &ignored,
                1);
        if (receive_count == -FI_EAVAIL) {
            fi_cq_err_entry error{};
            fi_cq_readerr(
                endpoint->efa_rx_cq,
                &error,
                0);
            if (endpoint->debug) {
                std::fprintf(
                    stderr,
                    "aboleth EFA proxy RX CQ error: %s\n",
                    dl::fabric().fi_strerror(error.err));
                std::fflush(stderr);
            }
            return false;
        }
        const cudaError_t flush_status =
            cudaDeviceFlushGPUDirectRDMAWrites(
                cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                cudaFlushGPUDirectRDMAWritesToOwner);
        if (flush_status != cudaSuccess &&
            flush_status !=
                cudaErrorNotSupported) {
            return false;
        }
        std::this_thread::yield();
    }
    if (endpoint->debug) {
        std::fprintf(
            stderr,
            "aboleth EFA proxy completion timed out\n");
        std::fflush(stderr);
    }
    return false;
}

std::uint64_t process_efa_proxy_batch(
    ProxyEndpointContext* endpoint,
    std::uint64_t target,
    const aboleth::abi::ProxyBatchSlot& slot) {
    if (endpoint->efa_endpoint == nullptr ||
        endpoint->efa_peer ==
            FI_ADDR_NOTAVAIL ||
        slot.command_count == 0 ||
        slot.command_count >
            aboleth::abi::kProxyMaximumCommands) {
        return 3;
    }

    for (std::uint32_t index = 0;
         index < slot.command_count;
         ++index) {
        const auto& command =
            slot.commands[index];
        if (command.opcode ==
            aboleth::abi::kProxyCommandNop) {
            continue;
        }
        const bool is_write =
            command.opcode ==
            aboleth::abi::kProxyCommandWrite;
        const bool is_signal =
            command.opcode ==
                aboleth::abi::
                    kProxyCommandWeakSignal ||
            command.opcode ==
                aboleth::abi::
                    kProxyCommandStrongSignal;
        const bool valid_signal_bytes =
            command.bytes == 1 ||
            command.bytes == 2 ||
            command.bytes == 4 ||
            command.bytes == 8;
        if ((!is_write && !is_signal) ||
            command.bytes == 0 ||
            (is_signal &&
             !valid_signal_bytes)) {
            return 3;
        }
        if (endpoint->debug) {
            std::fprintf(
                stderr,
                "aboleth EFA command batch=%llu index=%u opcode=%u bytes=%u src=%#llx lkey=%u dst=%#llx rkey=%u\n",
                static_cast<unsigned long long>(
                    target),
                index,
                command.opcode,
                command.bytes,
                static_cast<unsigned long long>(
                    command.source_address),
                command.source_key,
                static_cast<unsigned long long>(
                    command.destination_address),
                command.destination_key);
            std::fflush(stderr);
        }

        std::uint64_t inline_value =
            command.source_address;
        iovec local_iov{};
        void* local_descriptor = nullptr;
        if (is_signal) {
            local_iov.iov_base =
                &inline_value;
            local_iov.iov_len =
                command.bytes;
        } else {
            EfaMemoryRecord memory{};
            {
                std::lock_guard<std::mutex> lock(
                    endpoint->runtime->efa->
                        memory_mutex);
                const auto found =
                    endpoint->runtime->efa->
                        memory.find(
                            command.source_key);
                if (found ==
                    endpoint->runtime->efa->
                        memory.end()) {
                    return 3;
                }
                memory = found->second;
            }
            const auto begin =
                reinterpret_cast<
                    std::uintptr_t>(
                    memory.address);
            const auto source =
                static_cast<std::uintptr_t>(
                    command.source_address);
            if (source < begin ||
                source - begin >
                    memory.bytes ||
                command.bytes >
                    memory.bytes -
                        (source - begin)) {
                return 3;
            }
            local_iov.iov_base =
                reinterpret_cast<void*>(
                    command.source_address);
            local_iov.iov_len =
                command.bytes;
            local_descriptor =
                memory.descriptor;
        }

        fi_rma_iov remote_iov{};
        remote_iov.addr =
            command.destination_address;
        remote_iov.len = command.bytes;
        remote_iov.key =
            command.destination_key;
        struct {
            std::uint64_t batch;
            std::uint32_t command;
        } operation_context{target, index};
        fi_msg_rma message{};
        message.msg_iov = &local_iov;
        message.desc =
            is_signal
            ? nullptr
            : &local_descriptor;
        message.iov_count = 1;
        message.addr = endpoint->efa_peer;
        message.rma_iov = &remote_iov;
        message.rma_iov_count = 1;
        message.context = &operation_context;
        const std::uint64_t flags =
            FI_COMPLETION |
            FI_DELIVERY_COMPLETE |
            (is_signal ? FI_INJECT : 0);

        const auto post_deadline =
            std::chrono::steady_clock::now() +
            std::chrono::seconds(
                kTimeoutSeconds);
        for (;;) {
            const ssize_t status =
                fi_writemsg(
                    endpoint->efa_endpoint,
                    &message,
                    flags);
            if (status == 0) {
                break;
            }
            if (status != -FI_EAGAIN) {
                return 3;
            }
            if (std::chrono::steady_clock::now() >=
                post_deadline) {
                return 1;
            }
            fi_cq_entry ignored{};
            const ssize_t progress =
                fi_cq_read(
                    endpoint->efa_tx_cq,
                    &ignored,
                    1);
            if (progress == -FI_EAVAIL) {
                fi_cq_err_entry error{};
                fi_cq_readerr(
                    endpoint->efa_tx_cq,
                    &error,
                    0);
                return 2;
            }
            if (progress < 0 &&
                progress != -FI_EAGAIN) {
                return 2;
            }
            std::this_thread::yield();
        }
        if (!poll_efa_proxy_completion(
                endpoint,
                &operation_context)) {
            return 2;
        }
    }
    return 0;
}

#endif  // ABOLETH_HAS_EFA

std::uint64_t process_proxy_batch(
    ProxyEndpointContext* endpoint,
    std::uint64_t target,
    const aboleth::abi::ProxyBatchSlot& slot) {
#if ABOLETH_HAS_EFA
    if (endpoint->provider ==
        ProxyProviderKind::EfaOfi) {
        return process_efa_proxy_batch(
            endpoint,
            target,
            slot);
    }
#endif
    const std::uint32_t command_count =
        slot.command_count;
    if (command_count == 0 ||
        command_count >
            aboleth::abi::kProxyMaximumCommands) {
        return 3;
    }

    std::size_t request_count = 0;
    for (std::uint32_t index = 0;
         index < command_count;
         ++index) {
        const auto& command =
            slot.commands[index];
        if (command.opcode ==
            aboleth::abi::kProxyCommandNop) {
            continue;
        }
        const bool is_write =
            command.opcode ==
            aboleth::abi::kProxyCommandWrite;
        const bool is_weak_signal =
            command.opcode ==
            aboleth::abi::kProxyCommandWeakSignal;
        const bool is_strong_signal =
            command.opcode ==
            aboleth::abi::kProxyCommandStrongSignal;
        const bool is_signal =
            is_weak_signal || is_strong_signal;
        if ((!is_write && !is_signal) ||
            command.bytes == 0) {
            return 3;
        }

        auto& scatter =
            endpoint->scatter_entries[
                request_count];
        if (is_strong_signal) {
            if (command.bytes !=
                    sizeof(std::uint64_t) ||
                endpoint->atomic_result_mr == nullptr) {
                return 3;
            }
            scatter.addr =
                reinterpret_cast<std::uint64_t>(
                    &endpoint->atomic_result);
            scatter.length =
                sizeof(std::uint64_t);
            scatter.lkey =
                endpoint->atomic_result_mr->lkey;
        } else if (is_weak_signal) {
            if (command.bytes !=
                sizeof(std::uint64_t)) {
                return 3;
            }
            endpoint->inline_values[
                request_count] =
                command.source_address;
            scatter.addr =
                reinterpret_cast<std::uint64_t>(
                    &endpoint->inline_values[
                        request_count]);
            scatter.length =
                sizeof(std::uint64_t);
            scatter.lkey = 0;
        } else {
            scatter.addr =
                command.source_address;
            scatter.length = command.bytes;
            scatter.lkey = command.source_key;
        }

        auto& request =
            endpoint->work_requests[
                request_count];
        request.wr_id = target;
        request.send_flags = 0;
        if (is_strong_signal) {
            request.opcode =
                IBV_WR_ATOMIC_CMP_AND_SWP;
            request.wr.atomic.remote_addr =
                command.destination_address;
            request.wr.atomic.compare_add =
                command.source_address - 1;
            request.wr.atomic.swap =
                command.source_address;
            request.wr.atomic.rkey =
                command.destination_key;
        } else {
            request.opcode = IBV_WR_RDMA_WRITE;
            request.wr.rdma.remote_addr =
                command.destination_address;
            request.wr.rdma.rkey =
                command.destination_key;
        }
        if (is_weak_signal) {
            request.send_flags =
                IBV_SEND_INLINE;
        }
        ++request_count;
    }
    if (request_count == 0) {
        return 3;
    }

    auto& final_request =
        endpoint->work_requests[
            request_count - 1];
    final_request.send_flags |=
        IBV_SEND_SIGNALED;
    ibv_send_wr* saved_next =
        final_request.next;
    final_request.next = nullptr;
    ibv_send_wr* bad_request = nullptr;
    const int post_status = ibv_post_send(
            endpoint->sender,
            &endpoint->work_requests[0],
            &bad_request);
    final_request.next = saved_next;
    if (post_status != 0) {
        return 3;
    }
    return 0;
}

void publish_proxy_completion(
    ProxyEndpointContext* endpoint,
    std::uint64_t target,
    std::uint64_t status) {
    if (status != 0) {
        endpoint->failure_status = status;
        atomic_store_release(
            &endpoint->queue_host->status,
            status);
    }
    atomic_store_release(
        &endpoint->queue_host->
            completed_sequence,
        target);
    endpoint->next_sequence = target;
    endpoint->outstanding = false;
    if (endpoint->debug) {
        std::fprintf(
            stderr,
            "aboleth proxy GPU %d completed batch %llu status %llu\n",
            endpoint->gpu_index,
            static_cast<unsigned long long>(
                target),
            static_cast<unsigned long long>(
                status));
        std::fflush(stderr);
    }
}

#if ABOLETH_HAS_EFA

bool valid_efa_proxy_command(
    const aboleth::abi::ProxyCommand& command) {
    const bool write =
        command.opcode ==
        aboleth::abi::kProxyCommandWrite;
    const bool signal =
        command.opcode ==
            aboleth::abi::
                kProxyCommandWeakSignal ||
        command.opcode ==
            aboleth::abi::
                kProxyCommandStrongSignal;
    const bool valid_signal_bytes =
        command.bytes == 1 ||
        command.bytes == 2 ||
        command.bytes == 4 ||
        command.bytes == 8;
    return command.opcode ==
               aboleth::abi::kProxyCommandNop ||
        ((write || signal) &&
         command.bytes > 0 &&
         (!signal || valid_signal_bytes));
}

int post_efa_proxy_command(
    ProxyEndpointContext* endpoint,
    EfaProxyBatchState& batch,
    const aboleth::abi::ProxyCommand& command,
    std::uint32_t command_index,
    bool more) {
    std::uint64_t inline_value =
        command.source_address;

    const bool signal =
        command.opcode ==
            aboleth::abi::
                kProxyCommandWeakSignal ||
        command.opcode ==
            aboleth::abi::
                kProxyCommandStrongSignal;
    iovec local_iov{};
    void* local_descriptor = nullptr;
    if (signal) {
        if (endpoint->runtime->efa->
                fabric_name ==
            "efa-direct") {
            const auto slot_index =
                (batch.target - 1) %
                aboleth::abi::
                    kProxySlotCount;
            auto& stored_value =
                endpoint->
                    efa_signal_values[
                        slot_index *
                            aboleth::abi::
                                kProxyMaximumCommands +
                        command_index];
            stored_value = inline_value;
            local_iov.iov_base =
                &stored_value;
            local_descriptor =
                fi_mr_desc(
                    endpoint->
                        efa_signal_mr);
        } else {
            local_iov.iov_base =
                &inline_value;
        }
        local_iov.iov_len =
            command.bytes;
    } else {
        EfaMemoryRecord memory{};
        {
            std::lock_guard<std::mutex> lock(
                endpoint->runtime->efa->
                    memory_mutex);
            const auto found =
                endpoint->runtime->efa->
                    memory.find(
                        command.source_key);
            if (found ==
                endpoint->runtime->efa->
                    memory.end()) {
                return -1;
            }
            memory = found->second;
        }
        const auto begin =
            reinterpret_cast<std::uintptr_t>(
                memory.address);
        const auto source =
            static_cast<std::uintptr_t>(
                command.source_address);
        if (source < begin ||
            source - begin >
                memory.bytes ||
            command.bytes >
                memory.bytes -
                    (source - begin)) {
            return -1;
        }
        local_iov.iov_base =
            reinterpret_cast<void*>(
                command.source_address);
        local_iov.iov_len =
            command.bytes;
        local_descriptor =
            memory.descriptor;
    }

    fi_rma_iov remote_iov{};
    remote_iov.addr =
        command.destination_address;
    remote_iov.len = command.bytes;
    remote_iov.key =
        command.destination_key;
    fi_msg_rma message{};
    message.msg_iov = &local_iov;
    message.desc =
        local_descriptor == nullptr
            ? nullptr
            : &local_descriptor;
    message.iov_count = 1;
    message.addr = endpoint->efa_peer;
    message.rma_iov = &remote_iov;
    message.rma_iov_count = 1;
    message.context = nullptr;
    std::uint64_t flags =
        FI_DELIVERY_COMPLETE |
        (signal &&
                 endpoint->runtime->efa->
                         fabric_name !=
                     "efa-direct"
             ? FI_INJECT
             : 0);
    if (more) {
        flags |= FI_MORE;
    }
    const ssize_t status =
        fi_writemsg(
            endpoint->efa_endpoint,
            &message,
            flags);
    if (status == 0) {
        ++batch.pending_completions;
        ++batch.counter_target;
        if (endpoint->debug) {
            std::fprintf(
                stderr,
                "aboleth EFA post gpu=%d fabric=%s target=%llu counter=%llu opcode=%u bytes=%u more=%d\n",
                endpoint->gpu_index,
                endpoint->runtime->efa->
                    fabric_name.c_str(),
                static_cast<unsigned long long>(
                    batch.target),
                static_cast<unsigned long long>(
                    batch.counter_target),
                command.opcode,
                command.bytes,
                more ? 1 : 0);
            std::fflush(stderr);
        }
        return 0;
    }
    if (status == -FI_EAGAIN) {
        return 1;
    }
    if (endpoint->debug) {
        std::fprintf(
            stderr,
            "aboleth EFA post failed gpu=%d fabric=%s status=%s (%zd)\n",
            endpoint->gpu_index,
            endpoint->runtime->efa->
                fabric_name.c_str(),
            dl::fabric().fi_strerror(
                static_cast<int>(-status)),
            status);
        std::fflush(stderr);
    }
    return -1;
}

bool progress_efa_proxy_completions(
    ProxyEndpointContext* endpoint) {
    std::array<fi_cq_entry, 64>
        completions{};
    const ssize_t count =
        fi_cq_read(
            endpoint->efa_tx_cq,
            completions.data(),
            completions.size());
    if (count == -FI_EAGAIN) {
        return false;
    }
    if (count == -FI_EAVAIL) {
        fi_cq_err_entry error{};
        fi_cq_readerr(
            endpoint->efa_tx_cq,
            &error,
            0);
        endpoint->efa_cq_shard->
            failure_status.store(
                2,
                std::memory_order_release);
        return true;
    }
    if (count < 0) {
        endpoint->efa_cq_shard->
            failure_status.store(
                2,
                std::memory_order_release);
        return true;
    }
    if (count > 0) {
        endpoint->efa_cq_shard->
            failure_status.store(
                2,
                std::memory_order_release);
    }
    return count > 0;
}

bool progress_efa_proxy_endpoint(
    ProxyEndpointContext* endpoint) {
    bool progressed =
        progress_efa_proxy_completions(
            endpoint);
    const auto shard_failure =
        endpoint->efa_cq_shard->
            failure_status.load(
                std::memory_order_acquire);
    if (shard_failure != 0) {
        endpoint->failure_status =
            shard_failure;
    }
    if (fi_cntr_readerr(
            endpoint->efa_tx_counter) != 0) {
        endpoint->failure_status = 2;
    }
    auto& current_batch =
        endpoint->efa_batches[
            endpoint->next_sequence %
            aboleth::abi::
                kProxySlotCount];
    if (current_batch.active &&
        current_batch.target ==
            endpoint->next_sequence + 1 &&
        current_batch.counter_target >
            current_batch.counter_base &&
        fi_cntr_read(
            endpoint->efa_tx_counter) >=
            current_batch.counter_target) {
        __atomic_store_n(
            &current_batch.completed_commands,
            current_batch.pending_completions,
            __ATOMIC_RELEASE);
        progressed = true;
    }

    fi_cq_data_entry ignored{};
    const ssize_t receive_count =
        fi_cq_read(
            endpoint->efa_rx_cq,
            &ignored,
            1);
    if (receive_count == -FI_EAVAIL) {
        fi_cq_err_entry error{};
        fi_cq_readerr(
            endpoint->efa_rx_cq,
            &error,
            0);
        endpoint->efa_cq_shard->
            failure_status.store(
                2,
                std::memory_order_release);
    } else if (
        receive_count < 0 &&
        receive_count != -FI_EAGAIN) {
        endpoint->efa_cq_shard->
            failure_status.store(
                2,
                std::memory_order_release);
    } else if (receive_count > 0) {
        endpoint->efa_cq_shard->
            failure_status.store(
                2,
                std::memory_order_release);
    }
    const auto receive_failure =
        endpoint->efa_cq_shard->
            failure_status.load(
                std::memory_order_acquire);
    if (receive_failure != 0) {
        endpoint->failure_status =
            receive_failure;
    }

    if (endpoint->failure_status != 0) {
        atomic_store_release(
            &endpoint->queue_host->status,
            endpoint->failure_status);
        return true;
    }

    std::uint32_t posted = 0;
    while (posted <
           endpoint->efa_post_budget) {
        const auto target =
            endpoint->
                efa_accepted_sequence + 1;
        if (target !=
            endpoint->next_sequence + 1) {
            break;
        }
        auto& batch =
            endpoint->efa_batches[
                (target - 1) %
                aboleth::abi::
                    kProxySlotCount];
        const auto& slot =
            endpoint->queue_host->slots[
                (target - 1) %
                aboleth::abi::
                    kProxySlotCount];
        if (!batch.active) {
            if (atomic_load_acquire(
                    &slot.ready_sequence) !=
                target) {
                break;
            }
            if (slot.command_count == 0 ||
                slot.command_count >
                    aboleth::abi::
                        kProxyMaximumCommands) {
                endpoint->failure_status = 3;
                atomic_store_release(
                    &endpoint->queue_host->status,
                    endpoint->failure_status);
                return true;
            }
            batch = {};
            batch.target = target;
            batch.command_count =
                slot.command_count;
            batch.active = true;
            batch.counter_base =
                fi_cntr_read(
                    endpoint->
                        efa_tx_counter);
            batch.counter_target =
                batch.counter_base;
            ++endpoint->efa_batch_count;
            endpoint->efa_command_count +=
                slot.command_count;
            for (std::uint32_t command_index = 0;
                 command_index <
                     slot.command_count;
                 ++command_index) {
                const auto& command =
                    slot.commands[command_index];
                endpoint->efa_payload_bytes +=
                    command.bytes;
                if (command_index == 0) {
                    continue;
                }
                const auto& previous =
                    slot.commands[
                        command_index - 1];
                if (previous.destination_address +
                        previous.bytes ==
                    command.destination_address) {
                    ++endpoint->
                        efa_adjacent_destinations;
                    if (previous.source_address +
                            previous.bytes ==
                        command.source_address &&
                        previous.opcode ==
                            aboleth::abi::
                                kProxyCommandWrite &&
                        command.opcode ==
                            aboleth::abi::
                                kProxyCommandWrite) {
                        ++endpoint->
                            efa_adjacent_source_and_destination;
                    }
                }
            }
        }

        bool blocked = false;
        while (batch.next_command <
                   batch.command_count &&
               posted <
                   endpoint->
                       efa_post_budget) {
            const auto command_index =
                batch.next_command;
            const auto& command =
                slot.commands[command_index];
            if (!valid_efa_proxy_command(
                    command)) {
                endpoint->failure_status = 3;
                atomic_store_release(
                    &endpoint->queue_host->status,
                    endpoint->failure_status);
                return true;
            }
            if (command.opcode ==
                aboleth::abi::
                    kProxyCommandNop) {
                ++batch.next_command;
                continue;
            }
            const bool strong =
                command.opcode ==
                aboleth::abi::
                    kProxyCommandStrongSignal;
            if (strong &&
                (endpoint->next_sequence + 1 !=
                     target ||
                 __atomic_load_n(
                     &batch.completed_commands,
                     __ATOMIC_ACQUIRE) !=
                     batch.pending_completions)) {
                blocked = true;
                break;
            }
            auto next_post =
                command_index + 1;
            while (next_post <
                       batch.command_count &&
                   slot.commands[next_post].
                           opcode ==
                       aboleth::abi::
                           kProxyCommandNop) {
                ++next_post;
            }
            const bool has_later_post =
                next_post <
                batch.command_count;
            const bool more =
                !strong &&
                !batch.force_flush &&
                has_later_post &&
                batch.posts_since_flush + 1 <
                    endpoint->efa_post_batch &&
                slot.commands[next_post].
                        opcode !=
                    aboleth::abi::
                        kProxyCommandStrongSignal;
            const int post_status =
                post_efa_proxy_command(
                    endpoint,
                    batch,
                    command,
                    command_index,
                    more);
            if (post_status > 0) {
                batch.force_flush = true;
                blocked = true;
                break;
            }
            if (post_status < 0) {
                endpoint->failure_status = 3;
                atomic_store_release(
                    &endpoint->queue_host->status,
                    endpoint->failure_status);
                return true;
            }
            ++batch.next_command;
            batch.force_flush = false;
            if (more) {
                ++batch.posts_since_flush;
            } else {
                batch.posts_since_flush = 0;
            }
            ++posted;
            progressed = true;
            if (strong) {
                blocked = true;
                break;
            }
        }
        if (batch.next_command ==
            batch.command_count) {
            endpoint->
                efa_accepted_sequence =
                target;
        }
        if (blocked ||
            batch.next_command <
                batch.command_count) {
            break;
        }
    }

    for (;;) {
        const auto target =
            endpoint->next_sequence + 1;
        auto& batch =
            endpoint->efa_batches[
                (target - 1) %
                aboleth::abi::
                    kProxySlotCount];
        if (!batch.active ||
            batch.target != target ||
            batch.next_command !=
                batch.command_count ||
            __atomic_load_n(
                &batch.completed_commands,
                __ATOMIC_ACQUIRE) !=
                batch.pending_completions) {
            break;
        }
        publish_proxy_completion(
            endpoint,
            target,
            0);
        batch = {};
        progressed = true;
    }
    return progressed;
}

#endif  // ABOLETH_HAS_EFA

bool progress_proxy_endpoint(
    ProxyEndpointContext* endpoint) {
#if ABOLETH_HAS_EFA
    if (endpoint->provider ==
        ProxyProviderKind::EfaOfi) {
        return progress_efa_proxy_endpoint(
            endpoint);
    }
#endif
    if (endpoint->outstanding) {
        return false;
    }

    const std::uint64_t target =
        endpoint->next_sequence + 1;
    auto& slot =
        endpoint->queue_host->slots[
            endpoint->next_sequence %
            aboleth::abi::kProxySlotCount];
    if (atomic_load_acquire(
            &slot.ready_sequence) !=
        target) {
        return false;
    }
    if (endpoint->failure_status != 0) {
        publish_proxy_completion(
            endpoint,
            target,
            endpoint->failure_status);
        return true;
    }
    if (endpoint->debug) {
        std::fprintf(
            stderr,
            "aboleth proxy GPU %d posting batch %llu (%u commands)\n",
            endpoint->gpu_index,
            static_cast<unsigned long long>(
                target),
            slot.command_count);
        std::fflush(stderr);
    }
    const std::uint64_t status =
        process_proxy_batch(
            endpoint,
            target,
            slot);
    if (status != 0) {
        publish_proxy_completion(
            endpoint,
            target,
            status);
        return true;
    }
#if ABOLETH_HAS_EFA
    if (endpoint->provider ==
        ProxyProviderKind::EfaOfi) {
        publish_proxy_completion(
            endpoint,
            target,
            0);
        return true;
    }
#endif
    endpoint->outstanding = true;
    endpoint->outstanding_target = target;
    endpoint->completion_deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(
            kTimeoutSeconds);
    return true;
}

}  // namespace

namespace aboleth::runtime {

struct ProxyWorker {
    ProxyWorker(
        SharedRuntimeContext* runtime,
        int gpu,
        int shard)
        : gpu_index(gpu),
          worker_shard(shard) {
        if (runtime->proxy_provider ==
            ProxyProviderKind::VerbsRc) {
            send_cq =
                dl::verbs().ibv_create_cq(
                    runtime->verbs_context,
                    kMaximumTransfers,
                    nullptr,
                    nullptr,
                    0);
            if (send_cq == nullptr &&
                worker_shard == 0) {
                throw_error(
                    "ibv_create_cq(shared proxy) failed");
            }
        }
        worker =
            std::thread(
                [this] { run(); });
    }

    ~ProxyWorker() {
        stop.store(
            true,
            std::memory_order_relaxed);
        if (worker.joinable()) {
            worker.join();
        }
        if (send_cq != nullptr) {
            dl::verbs().ibv_destroy_cq(send_cq);
        }
    }

    void add(ProxyEndpointContext* endpoint) {
        std::uint64_t target = 0;
        {
            std::lock_guard<std::mutex> lock(
                mutex);
            endpoints.push_back(endpoint);
            target =
                generation.fetch_add(
                    1,
                    std::memory_order_release) +
                1;
        }
        while (observed_generation.load(
                   std::memory_order_acquire) <
               target) {
        }
    }

    void remove(ProxyEndpointContext* endpoint) {
        std::uint64_t target = 0;
        {
            std::lock_guard<std::mutex> lock(
                mutex);
            endpoints.erase(
                std::remove(
                    endpoints.begin(),
                    endpoints.end(),
                    endpoint),
                endpoints.end());
            target =
                generation.fetch_add(
                    1,
                    std::memory_order_release) +
                1;
        }
        while (observed_generation.load(
                   std::memory_order_acquire) <
               target) {
        }
    }

    void run() {
        if (send_cq == nullptr) {
            cudaSetDevice(gpu_index);
            pin_efa_proxy_worker(
                gpu_index,
                worker_shard);
            int ordering = 0;
            if (cudaDeviceGetAttribute(
                    &ordering,
                    cudaDevAttrGPUDirectRDMAWritesOrdering,
                    gpu_index) ==
                    cudaSuccess &&
                ordering <
                    cudaGPUDirectRDMAWritesOrderingOwner) {
                efa_needs_flush = true;
            }
        }
        if (std::getenv(
                "ABOLETH_PROXY_DEBUG") !=
            nullptr) {
            std::fprintf(
                stderr,
                "aboleth shared proxy worker started on GPU %d\n",
                gpu_index);
            std::fflush(stderr);
        }
        std::vector<ProxyEndpointContext*>
            active_endpoints;
        std::uint64_t local_generation =
            std::numeric_limits<
                std::uint64_t>::max();
        while (!stop.load(
            std::memory_order_relaxed)) {
            const std::uint64_t current_generation =
                generation.load(
                    std::memory_order_acquire);
            if (current_generation !=
                local_generation) {
                {
                    std::lock_guard<std::mutex> lock(
                        mutex);
                    active_endpoints = endpoints;
                    local_generation =
                        current_generation;
                }
                observed_generation.store(
                    local_generation,
                    std::memory_order_release);
            }
            for (auto* endpoint : active_endpoints) {
                progress_proxy_endpoint(
                    endpoint);
            }
            if (send_cq == nullptr &&
                worker_shard == 0 &&
                efa_needs_flush) {
                const cudaError_t flush_status =
                    cudaDeviceFlushGPUDirectRDMAWrites(
                        cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                        cudaFlushGPUDirectRDMAWritesToOwner);
                if (flush_status != cudaSuccess &&
                    flush_status !=
                        cudaErrorNotSupported) {
                    for (auto* endpoint :
                         active_endpoints) {
                        endpoint->failure_status = 2;
                        atomic_store_release(
                            &endpoint->
                                 queue_host->status,
                            2);
                    }
                }
            }
            if (send_cq != nullptr) {
                std::array<ibv_wc, 16>
                    completions{};
                const int count =
                    ibv_poll_cq(
                        send_cq,
                        completions.size(),
                        completions.data());
                if (count < 0) {
                    for (auto* endpoint :
                         active_endpoints) {
                        if (endpoint->outstanding) {
                            publish_proxy_completion(
                                endpoint,
                                endpoint->
                                    outstanding_target,
                                2);
                        }
                    }
                } else {
                    for (int index = 0;
                         index < count;
                         ++index) {
                        const auto& completion =
                            completions[index];
                        auto match =
                            std::find_if(
                                active_endpoints.begin(),
                                active_endpoints.end(),
                                [&](auto* endpoint) {
                                    return endpoint->
                                               outstanding &&
                                        endpoint->sender->
                                                qp_num ==
                                            completion.qp_num;
                                });
                        if (match ==
                            active_endpoints.end()) {
                            continue;
                        }
                        auto* endpoint = *match;
                        const std::uint64_t status =
                            completion.status ==
                                    IBV_WC_SUCCESS &&
                                completion.wr_id ==
                                    endpoint->
                                        outstanding_target
                            ? 0
                            : 2;
                        publish_proxy_completion(
                            endpoint,
                            endpoint->
                                outstanding_target,
                            status);
                    }
                }
            }
            ++loop_iterations;
            if ((loop_iterations & 1023U) == 0) {
                const auto now =
                    std::chrono::steady_clock::now();
                for (auto* endpoint :
                     active_endpoints) {
                    if (endpoint->outstanding &&
                        now >= endpoint->
                            completion_deadline) {
                        publish_proxy_completion(
                            endpoint,
                            endpoint->
                                outstanding_target,
                            1);
                    }
                }
            }
        }
    }

    int gpu_index{};
    int worker_shard{};
    bool efa_needs_flush{};
    ibv_cq* send_cq{};
    std::mutex mutex;
    std::vector<ProxyEndpointContext*>
        endpoints;
    std::atomic<std::uint64_t> generation{0};
    std::atomic<std::uint64_t>
        observed_generation{0};
    std::atomic<bool> stop{false};
    std::uint32_t loop_iterations{};
    std::thread worker;
};

}  // namespace aboleth::runtime

namespace {

std::mutex proxy_worker_registry_mutex;
std::map<
    std::tuple<
        SharedRuntimeContext*,
        int,
        int>,
    std::weak_ptr<ProxyWorker>>
    proxy_worker_registry;

std::shared_ptr<ProxyWorker>
get_proxy_worker(
    SharedRuntimeContext* runtime,
    int gpu_index,
    int worker_shard = 0) {
    std::lock_guard<std::mutex> lock(
        proxy_worker_registry_mutex);
    const auto key =
        std::make_tuple(
            runtime,
            gpu_index,
            worker_shard);
    auto& weak = proxy_worker_registry[key];
    auto worker = weak.lock();
    if (!worker) {
        worker =
            std::make_shared<ProxyWorker>(
                runtime,
                gpu_index,
                worker_shard);
        weak = worker;
    }
    return worker;
}

}  // namespace

namespace aboleth::runtime {

ProxyEndpointContext::~ProxyEndpointContext() {
    if (provider ==
            ProxyProviderKind::EfaOfi &&
        producer_index != nullptr &&
        queue_host != nullptr) {
        cudaSetDevice(gpu_index);
        std::uint64_t target = 0;
        if (cudaMemcpy(
                &target,
                producer_index,
                sizeof(target),
                cudaMemcpyDeviceToHost) ==
            cudaSuccess) {
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(
                    kTimeoutSeconds);
            while (atomic_load_acquire(
                       &queue_host->
                           completed_sequence) <
                       target &&
                   atomic_load_acquire(
                       &queue_host->status) == 0 &&
                   std::chrono::steady_clock::now() <
                       deadline) {
                std::this_thread::yield();
            }
        }
    }
    if (worker) {
        worker->remove(this);
    }
    if (provider ==
            ProxyProviderKind::EfaOfi &&
        std::getenv(
            "ABOLETH_PROXY_STATS") !=
            nullptr &&
        efa_command_count > 0) {
        std::fprintf(
            stderr,
            "aboleth EFA stats gpu=%d batches=%llu commands=%llu bytes=%llu adjacent_dst=%llu adjacent_both=%llu\n",
            gpu_index,
            static_cast<unsigned long long>(
                efa_batch_count),
            static_cast<unsigned long long>(
                efa_command_count),
            static_cast<unsigned long long>(
                efa_payload_bytes),
            static_cast<unsigned long long>(
                efa_adjacent_destinations),
            static_cast<unsigned long long>(
                efa_adjacent_source_and_destination));
        std::fflush(stderr);
    }
    cudaSetDevice(gpu_index);
    if (producer_index != nullptr) {
        cudaFree(producer_index);
    }
    if (queue_host != nullptr) {
        cudaFreeHost(queue_host);
    }
    if (atomic_result_mr != nullptr) {
        dl::verbs().ibv_dereg_mr(atomic_result_mr);
    }
    if (sender != nullptr) {
        dl::verbs().ibv_destroy_qp(sender);
    }
#if ABOLETH_HAS_EFA
    if (efa_signal_mr != nullptr) {
        fi_close(&efa_signal_mr->fid);
    }
    if (efa_endpoint != nullptr) {
        fi_close(&efa_endpoint->fid);
    }
    if (efa_tx_counter != nullptr) {
        fi_close(&efa_tx_counter->fid);
    }
#endif
}

std::unique_ptr<ProxyEndpointContext>
create_proxy_endpoint(
    const std::shared_ptr<SharedRuntimeContext>& runtime,
    int gpu_index,
    aboleth::abi::ProxyEndpointDescriptor&
        descriptor,
    int worker_shard) {
    auto endpoint =
        std::make_unique<
            ProxyEndpointContext>();
    endpoint->runtime = runtime;
    endpoint->gpu_index = gpu_index;
    endpoint->worker_shard =
        worker_shard;
    endpoint->provider =
        runtime->proxy_provider;
    endpoint->debug =
        std::getenv("ABOLETH_PROXY_DEBUG") !=
        nullptr;
    endpoint->efa_post_batch =
        static_cast<std::uint32_t>(
            configured_index(
                "ABOLETH_EFA_POST_BATCH",
                8));
    if (endpoint->efa_post_batch == 0 ||
        endpoint->efa_post_batch > 64) {
        throw_error(
            "ABOLETH_EFA_POST_BATCH must be in [1, 64]");
    }
    endpoint->efa_post_budget =
        static_cast<std::uint32_t>(
            configured_index(
                "ABOLETH_EFA_POST_BUDGET",
                16));
    if (endpoint->efa_post_budget == 0 ||
        endpoint->efa_post_budget > 64) {
        throw_error(
            "ABOLETH_EFA_POST_BUDGET must be in [1, 64]");
    }
    cuda_check(
        cudaSetDevice(gpu_index),
        "cudaSetDevice(proxy endpoint)");

    endpoint->worker =
        get_proxy_worker(
            runtime.get(),
            gpu_index,
            worker_shard);
    if (endpoint->provider ==
        ProxyProviderKind::EfaOfi) {
#if ABOLETH_HAS_EFA
        create_efa_proxy_endpoint_resources(
            endpoint.get());
#endif
    } else {
        endpoint->send_cq =
            endpoint->worker->send_cq;
        endpoint->sender =
            create_rc_qp(
                runtime->pd,
                endpoint->send_cq,
                kProxyInlineBytes);
        qp_to_init(
            endpoint->sender,
            runtime->port);
        endpoint->psn = allocate_psn();
        endpoint->atomic_result_mr =
            dl::verbs().reg_mr(
                runtime->pd,
                &endpoint->atomic_result,
                sizeof(endpoint->atomic_result),
                IBV_ACCESS_LOCAL_WRITE);
        if (endpoint->atomic_result_mr == nullptr) {
            throw_error(
                "ibv_reg_mr(proxy atomic result) failed");
        }
    }

    cuda_check(
        cudaHostAlloc(
            reinterpret_cast<void**>(
                &endpoint->queue_host),
            sizeof(
                aboleth::abi::ProxyQueueState),
            cudaHostAllocMapped |
                cudaHostAllocPortable),
        "cudaHostAlloc(proxy queue)");
    std::memset(
        endpoint->queue_host,
        0,
        sizeof(
            aboleth::abi::ProxyQueueState));
    cuda_check(
        cudaHostGetDevicePointer(
            reinterpret_cast<void**>(
                &endpoint->queue_device),
            endpoint->queue_host,
            0),
        "cudaHostGetDevicePointer(proxy queue)");
    cuda_check(
        cudaMalloc(
            &endpoint->producer_index,
            sizeof(std::uint64_t)),
        "cudaMalloc(proxy producer)");
    cuda_check(
        cudaMemset(
            endpoint->producer_index,
            0,
            sizeof(std::uint64_t)),
        "cudaMemset(proxy producer)");

    descriptor.queue_state =
        reinterpret_cast<std::uint64_t>(
            endpoint->queue_device);
    descriptor.producer_index_address =
        reinterpret_cast<std::uint64_t>(
            endpoint->producer_index);
    descriptor.completion_timeout_cycles =
        cuda_clock_rate_khz(gpu_index) *
        1000ULL *
        kTimeoutSeconds;
    descriptor.slot_count =
        aboleth::abi::kProxySlotCount;
    descriptor.slot_stride =
        sizeof(
            aboleth::abi::ProxyBatchSlot);
    descriptor.maximum_commands =
        aboleth::abi::kProxyMaximumCommands;
    descriptor.abi_version =
        aboleth::abi::kProxyEndpointVersion;

    if (endpoint->provider ==
        ProxyProviderKind::VerbsRc) {
        for (std::size_t index = 0;
             index <
                 aboleth::abi::kProxyMaximumCommands;
             ++index) {
            auto& request =
                endpoint->work_requests[index];
            request = {};
            request.sg_list =
                &endpoint->scatter_entries[index];
            request.num_sge = 1;
            request.opcode = IBV_WR_RDMA_WRITE;
            if (index + 1 <
                aboleth::abi::kProxyMaximumCommands) {
                request.next =
                    &endpoint->work_requests[
                        index + 1];
            }
        }
    }

    endpoint->worker->add(
        endpoint.get());
    return endpoint;
}

void connect_proxy_endpoints(
    ProxyEndpointContext* first,
    ProxyEndpointContext* second) {
    if (first->provider != second->provider) {
        throw_error(
            "proxy endpoint providers do not match");
    }
#if ABOLETH_HAS_EFA
    if (first->provider ==
        ProxyProviderKind::EfaOfi) {
        connect_efa_proxy_endpoint(
            first,
            second->efa_address.data(),
            second->efa_address_length);
        connect_efa_proxy_endpoint(
            second,
            first->efa_address.data(),
            first->efa_address_length);
        return;
    }
#endif
    qp_to_rtr(
        first->sender,
        second->sender->qp_num,
        second->psn,
        first->runtime->port,
        second->runtime->port,
        std::min(
            first->runtime->path_mtu,
            second->runtime->path_mtu));
    qp_to_rtr(
        second->sender,
        first->sender->qp_num,
        first->psn,
        second->runtime->port,
        first->runtime->port,
        std::min(
            first->runtime->path_mtu,
            second->runtime->path_mtu));
    qp_to_rts(
        first->sender,
        first->psn);
    qp_to_rts(
        second->sender,
        second->psn);
}

}  // namespace aboleth::runtime
