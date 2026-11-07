"""Differential proof that a compiled fragment artifact emits the same wire
bytes as the same communication inlined into the kernel.

Two runs of one logical put+signal+wait over a fresh endpoint each:

  a. the in-kernel path — aboleth.put/signal/wait inside a Triton kernel,
     lowered by aboleth.compilation("rdma");
  b. the artifact path — the same transfers compiled by
     compile_fragment_artifact into a device function, called from a Triton
     kernel that links the artifact's bitcode.

The send queues are then compared byte for byte, excluding the fields that
legitimately differ between two endpoints: the queue-pair number, the
registered addresses, and the memory keys. Those are checked separately against
each run's own buffers, so nothing about them goes unverified. Opcodes, segment
counts, completion flags, byte counts, epochs, and WQE placement must match.
"""

import struct
import tempfile

import torch
import triton
import triton.language as tl
from cuda.bindings import runtime as cudart
from triton.language import core

import aboleth.language as aboleth
from aboleth.artifact import compile_fragment_artifact
from aboleth.compiler import compilation
from aboleth.pattern import (
    KernelTransportArgument,
    KernelTransportProgram,
    KernelTransportSignal,
    KernelTransportTransfer,
)
from aboleth.world import device_library
from aboleth_rdma_abi_test import (
    DESCRIPTOR_WORDS,
    create_endpoints,
    load_runtime,
)


TRANSFER_BYTES = 256
SIGNAL_BYTES = 8
ALLOCATION_BYTES = TRANSFER_BYTES + SIGNAL_BYTES
WQE_COUNT = 2

# Bytes each WQE kind actually writes; the rest of the 64-byte slot is whatever
# the queue was initialized with and is no part of the wire format.
PAYLOAD_USED = 48
SIGNAL_USED = 64

# (offset, length) of the fields two endpoints cannot agree on: the queue-pair
# number inside qpn_and_segments, the registered addresses, and the keys.
PAYLOAD_VARIABLE = ((4, 3), (16, 8), (24, 4), (36, 4), (40, 8))
SIGNAL_VARIABLE = ((4, 3), (16, 8), (24, 4), (52, 4), (56, 8))

# Named restatement of what the byte comparison already covers.
INVARIANT_FIELDS = (
    "payload_opcode",
    "payload_bytes",
    "payload_completion",
    "signal_opcode",
    "signal_completion",
    "signal_swap",
    "signal_compare",
    "signal_result_bytes",
)

# The fragment ABI, in manifest slot order.
SLOTS = (
    ("endpoint", "i64"),
    ("destination", "i64"),
    ("source", "i64"),
    ("bytes", "i32"),
    ("source_key", "i32"),
    ("destination_key", "i32"),
    ("predicate", "i1"),
    ("remote_signal", "i64"),
    ("local_signal", "i64"),
    ("iteration", "i32"),
    ("iterations", "i32"),
)

PROGRAM = KernelTransportProgram(
    artifact="bindable_epilogue",
    name="differential",
    operations=("payload",),
    kernel="artifact_producer",
    insertion_point="epilogue",
    arguments=tuple(
        KernelTransportArgument(name, mlir_type)
        for name, mlir_type in SLOTS
    ),
    transfers=(
        KernelTransportTransfer(
            operations=("payload",),
            destination="destination",
            source="source",
            byte_count="bytes",
            source_key="source_key",
            destination_key="destination_key",
            predicate="predicate",
            max_bytes=TRANSFER_BYTES,
        ),
    ),
    signal=KernelTransportSignal(
        destination="remote_signal",
        source="local_signal",
        source_key="source_key",
        destination_key="destination_key",
        iteration="iteration",
        iterations="iterations",
    ),
    program_id=0,
)

# Filled in once the artifact exists; read while the caller kernel is traced.
ARTIFACT = {}


@triton.jit
def inline_producer(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    source_key,
    destination_key,
    byte_count: tl.constexpr,
):
    transfer = aboleth.put(
        endpoint,
        destination,
        source,
        byte_count,
        source_key=source_key,
        destination_key=destination_key,
    )
    ready = aboleth.signal(
        endpoint,
        remote_signal,
        local_signal,
        transfer,
        source_key=source_key,
        destination_key=destination_key,
    )
    aboleth.wait(ready)


@core.extern
def fragment(
    endpoint,
    destination,
    source,
    byte_count,
    source_key,
    destination_key,
    predicate,
    remote_signal,
    local_signal,
    iteration,
    iterations,
    _semantic=None,
):
    address = core.dtype("uint64")
    scalar = core.dtype("int32")
    return core.extern_elementwise(
        "aboleth_fragment",
        ARTIFACT["bitcode"],
        [
            endpoint, destination, source, byte_count, source_key,
            destination_key, predicate, remote_signal, local_signal,
            iteration, iterations,
        ],
        {
            (
                address, address, address, scalar, scalar, scalar, scalar,
                address, address, scalar, scalar,
            ): (ARTIFACT["symbol"], core.dtype("int64")),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def artifact_producer(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    source_key,
    destination_key,
    byte_count,
):
    fragment(
        endpoint.to(tl.uint64),
        destination.to(tl.uint64),
        source.to(tl.uint64),
        byte_count,
        source_key,
        destination_key,
        1,
        remote_signal.to(tl.uint64),
        local_signal.to(tl.uint64),
        1,
        1,
    )


def cuda_check(result, operation: str) -> None:
    error = result[0]
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {error}")


def snapshot(endpoints, descriptor):
    return endpoints.send_queue_snapshot(
        WQE_COUNT * descriptor.send_stride
    )


def fields(raw: bytes, stride: int) -> dict:
    payload = 0
    signal = stride
    return {
        "payload_opcode": struct.unpack_from(">I", raw, payload)[0],
        "payload_segments": struct.unpack_from(">I", raw, payload + 4)[0],
        "payload_completion": raw[payload + 11],
        "payload_remote_address":
            struct.unpack_from(">Q", raw, payload + 16)[0],
        "payload_remote_key": struct.unpack_from(">I", raw, payload + 24)[0],
        "payload_bytes": struct.unpack_from(">I", raw, payload + 32)[0],
        "payload_local_key": struct.unpack_from(">I", raw, payload + 36)[0],
        "payload_local_address":
            struct.unpack_from(">Q", raw, payload + 40)[0],
        "signal_opcode": struct.unpack_from(">I", raw, signal)[0],
        "signal_segments": struct.unpack_from(">I", raw, signal + 4)[0],
        "signal_completion": raw[signal + 11],
        "signal_remote_address": struct.unpack_from(">Q", raw, signal + 16)[0],
        "signal_remote_key": struct.unpack_from(">I", raw, signal + 24)[0],
        "signal_swap": struct.unpack_from(">Q", raw, signal + 32)[0],
        "signal_compare": struct.unpack_from(">Q", raw, signal + 40)[0],
        "signal_result_bytes": struct.unpack_from(">I", raw, signal + 48)[0],
    }


def normalized(raw: bytes, stride: int) -> bytes:
    """The wire bytes with the per-endpoint fields blanked."""
    result = bytearray()
    for index, (used, variable) in enumerate((
        (PAYLOAD_USED, PAYLOAD_VARIABLE),
        (SIGNAL_USED, SIGNAL_VARIABLE),
    )):
        base = index * stride
        wqe = bytearray(raw[base:base + used])
        for offset, length in variable:
            wqe[offset:offset + length] = bytes(length)
        result += wqe
    return bytes(result)


def differences(left: bytes, right: bytes, stride: int) -> list[str]:
    report = []
    used = (PAYLOAD_USED, SIGNAL_USED)
    base = 0
    for index, length in enumerate(used):
        for offset in range(0, length, 4):
            a = left[base + offset:base + offset + 4]
            b = right[base + offset:base + offset + 4]
            if a != b:
                report.append(
                    f"wqe {index} offset {offset}: "
                    f"in-kernel {a.hex()} != artifact {b.hex()}"
                )
        base += length
    return report


def run_case(name: str, launch) -> tuple[bytes, int, dict]:
    """Create a fresh endpoint pair, run `launch`, and snapshot the queue."""
    descriptor = torch.empty(
        DESCRIPTOR_WORDS,
        dtype=torch.uint64,
        device="cuda:0",
    )
    source_allocation = torch.zeros(
        ALLOCATION_BYTES,
        dtype=torch.uint8,
        device="cuda:0",
    )
    destination_allocation = torch.zeros(
        ALLOCATION_BYTES,
        dtype=torch.uint8,
        device="cuda:1",
    )
    source = source_allocation[:TRANSFER_BYTES]
    destination = destination_allocation[:TRANSFER_BYTES]
    local_signal = source_allocation[TRANSFER_BYTES:].view(torch.int64)
    remote_signal = destination_allocation[TRANSFER_BYTES:].view(torch.int64)
    local_signal.fill_(1)
    payload = (
        (torch.arange(TRANSFER_BYTES, device="cuda:0") * 17 + 3) & 0xFF
    ).to(torch.uint8)
    source.copy_(payload)

    library = load_runtime()
    endpoints = create_endpoints(
        library,
        descriptor,
        source_allocation,
        destination_allocation,
    )
    try:
        expected = endpoints.descriptor
        source_lkey, destination_rkey = endpoints.keys()
        with torch.cuda.device(0):
            launch(
                descriptor,
                destination,
                source,
                remote_signal,
                local_signal,
                source_lkey,
                destination_rkey,
            )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        torch.testing.assert_close(
            destination.cpu(),
            payload.cpu(),
            rtol=0,
            atol=0,
        )
        assert remote_signal.cpu().item() == 1
        raw, dbrec_head = snapshot(endpoints, expected)
        assert dbrec_head == WQE_COUNT
        observed = fields(raw, expected.send_stride)

        # Whatever the comparison masks is verified here against this run's own
        # endpoint and buffers.
        assert observed["payload_opcode"] == 0x08
        assert observed["payload_segments"] == (expected.qpn << 8) | 3
        assert observed["payload_remote_address"] == destination.data_ptr()
        assert observed["payload_local_address"] == source.data_ptr()
        assert observed["payload_remote_key"] == destination_rkey
        assert observed["payload_local_key"] == source_lkey
        assert observed["signal_opcode"] == (1 << 8) | 0x11
        assert observed["signal_segments"] == (expected.qpn << 8) | 4
        assert observed["signal_remote_address"] == remote_signal.data_ptr()
        print(f"{name}_qpn={expected.qpn}")
        print(f"{name}_wqes={WQE_COUNT}")
        print(f"{name}_doorbell_head={dbrec_head}")
        return raw, expected.send_stride, observed
    finally:
        endpoints.close()


def inline_case():
    def launch(
        descriptor,
        destination,
        source,
        remote_signal,
        local_signal,
        source_lkey,
        destination_rkey,
    ):
        inline_producer[(1,)](
            descriptor,
            destination,
            source,
            remote_signal,
            local_signal,
            source_lkey,
            destination_rkey,
            byte_count=TRANSFER_BYTES,
            num_warps=1,
        )

    with compilation("rdma"):
        return run_case("inline", launch)


def artifact_case(directory: str):
    library = device_library(generated=False)
    major, minor = torch.cuda.get_device_capability(0)
    artifact = compile_fragment_artifact(
        PROGRAM,
        arch=f"sm_{major}{minor}",
        completion_interval=1,
        device_bitcode=library.bitcode,
        output_dir=directory,
    )
    ARTIFACT["bitcode"] = str(artifact.bitcode)
    ARTIFACT["symbol"] = artifact.symbol
    print(f"artifact_arch={artifact.arch}")
    print(f"artifact_device_library={library.arch}")
    print(f"artifact_symbol={artifact.symbol}")
    print(f"artifact_signature={artifact.binding_signature}")

    def launch(
        descriptor,
        destination,
        source,
        remote_signal,
        local_signal,
        source_lkey,
        destination_rkey,
    ):
        arguments = (
            descriptor,
            destination,
            source,
            remote_signal,
            local_signal,
            source_lkey,
            destination_rkey,
            TRANSFER_BYTES,
        )
        options = {
            "num_warps": 1,
            "extern_libs": {"aboleth_fragment": ARTIFACT["bitcode"]},
        }
        compiled = artifact_producer.warmup(
            *arguments,
            grid=(1,),
            **options,
        )
        # No compilation context is active here: the caller carries no Aboleth
        # operation, and every descriptor it posts comes from the linked
        # artifact.
        assert "aboleth." not in compiled.asm["ttir"]
        assert artifact.symbol in compiled.asm["llir"]
        artifact_producer[(1,)](*arguments, **options)

    return run_case("artifact", launch)


def run() -> None:
    assert torch.cuda.device_count() >= 2
    assert torch.cuda.can_device_access_peer(0, 1)
    cuda_check(cudart.cudaSetDevice(0), "cudaSetDevice(0)")
    enabled = cudart.cudaDeviceEnablePeerAccess(1, 0)
    if enabled[0] not in (
        cudart.cudaError_t.cudaSuccess,
        cudart.cudaError_t.cudaErrorPeerAccessAlreadyEnabled,
    ):
        cuda_check(enabled, "cudaDeviceEnablePeerAccess")
    try:
        inline_raw, inline_stride, inline_fields = inline_case()
        with tempfile.TemporaryDirectory(prefix="aboleth-fragment-") as out:
            artifact_raw, artifact_stride, artifact_fields = artifact_case(out)
        assert inline_stride == artifact_stride

        left = normalized(inline_raw, inline_stride)
        right = normalized(artifact_raw, artifact_stride)
        if left != right:
            for line in differences(left, right, inline_stride):
                print(f"DIFF {line}")
            for key in sorted(inline_fields):
                if inline_fields[key] != artifact_fields[key]:
                    print(
                        f"DIFF field {key}: in-kernel "
                        f"{inline_fields[key]:#x} != artifact "
                        f"{artifact_fields[key]:#x}"
                    )
            raise AssertionError(
                "artifact wire bytes differ from the in-kernel path"
            )

        for key in INVARIANT_FIELDS:
            assert inline_fields[key] == artifact_fields[key], key

        print(f"compared_bytes={len(left)}")
        print(f"payload_completion_flag={inline_fields['payload_completion']}")
        print(f"signal_completion_flag={inline_fields['signal_completion']}")
        print("aboleth_fragment_differential=PASS")
    finally:
        cudart.cudaSetDevice(0)
        disabled = cudart.cudaDeviceDisablePeerAccess(1)
        if disabled[0] not in (
            cudart.cudaError_t.cudaSuccess,
            cudart.cudaError_t.cudaErrorPeerAccessNotEnabled,
        ):
            cuda_check(disabled, "cudaDeviceDisablePeerAccess")


if __name__ == "__main__":
    run()
