"""Differential proof that the generated Tier-1 ops write the facade's bytes.

Each case runs one Tier-1 op sequence over a fresh endpoint pair per arm:

  a. generated — `experiments/cuda/aboleth_tier1_arm.cu` compiled against the
     GENERATED `aboleth/device.cuh` and device-linked with
     libaboleth_tier1_<arch>.a (every op a one-op transport program emitted
     by the real RDMA lowering in detached mode);
  b. facade — the same arm source compiled against `include/` (the
     hand-written primitives).

The send queues are compared byte for byte, excluding only the fields two
endpoints cannot agree on (queue-pair number, registered addresses, memory
keys), each of which is verified separately against its own run's buffers.
Every case ends with `quiet`, so the trailing drain NOP and the CQ
consumption are part of every comparison and the payload's arrival is
guaranteed before it is checked.

A Triton arm additionally calls the generated `put`/`quiet` through
`extern_elementwise` on the per-op bitcode — the same wire bytes must come
out of the bitcode consumer as out of the archive consumer.

This is the retirement gate of the hand-written Tier-1 device code: the
hand-written device code may be cut only while this differential is green.
"""

import ctypes
import os
import pathlib
import shutil
import struct
import subprocess
import tempfile

import torch
import triton
import triton.language as tl
from cuda.bindings import runtime as cudart
from triton.language import core

from aboleth.device_library import generate_device_library
from aboleth.tier1 import PUT_DESC_BYTES
from aboleth_rdma_abi_test import (
    DESCRIPTOR_WORDS,
    create_endpoints,
    load_runtime,
)

SLOT_BYTES = 128
PUT_MANY_COUNT = 8
SIGNAL_BYTES = 8

# Bytes each WQE kind actually writes; the rest of the 64-byte slot is
# whatever the queue was initialized with and is no part of the wire format.
# (The facade's drain NOP writes only its ctrl segment; the compiler's writes
# a full placeholder WQE — the 16 used bytes are the wire contract.)
WRITE_USED = 48
INLINE_USED = 48
ATOMIC_USED = 64
NOP_USED = 16

# (offset, length) of the fields two endpoints cannot agree on.
WRITE_VARIABLE = ((4, 3), (16, 8), (24, 4), (36, 4), (40, 8))
INLINE_VARIABLE = ((4, 3), (16, 8), (24, 4))
# The atomic's result lands in the endpoint's own completion state.
ATOMIC_VARIABLE = ((4, 3), (16, 8), (24, 4), (52, 4), (56, 8))
NOP_VARIABLE = ((4, 3),)

CQ_UPDATE = 0x08000000
RDMA_WRITE_OPCODE = 0x08
COMPARE_SWAP_OPCODE = 0x11
NOP_OPCODE = 0x00
INLINE_HEADER = 0x80000000 | SIGNAL_BYTES

CASES = ("put", "put_inline", "signal", "put_signal", "put_many")

# Each slot: (kind, used bytes, endpoint-variable fields). WRITE and INLINE
# use the same byte count, so the kind tag is what tells a gather WQE from an
# inline one in a mixed batch (put_signal).
_WRITE = ("write", WRITE_USED, WRITE_VARIABLE)
_INLINE = ("inline", INLINE_USED, INLINE_VARIABLE)
_ATOMIC = ("atomic", ATOMIC_USED, ATOMIC_VARIABLE)
_NOP = ("nop", NOP_USED, NOP_VARIABLE)

# A signal is strong publication: with relaxed-ordering registrations (the
# default library) it is a responder compare-and-swap of the epoch.
LAYOUTS = {
    "put": (_WRITE, _NOP),
    "put_inline": (_INLINE, _NOP),
    "signal": (_ATOMIC, _NOP),
    # The fused batch: payload put + strong epoch, then the drain NOP.
    "put_signal": (_WRITE, _ATOMIC, _NOP),
    "put_many": (*(_WRITE,) * PUT_MANY_COUNT, _NOP),
}

INLINE_VALUE = 0x1122334455667788

# Filled once the generated library exists; read while Triton traces.
TRITON_LIBS = {}


def arch() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}"


def nvcc() -> str:
    configured = os.environ.get("ABOLETH_NVCC")
    if configured and pathlib.Path(configured).is_file():
        return configured
    for home in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ):
        if not home:
            continue
        candidate = pathlib.Path(home) / "bin" / "nvcc"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("nvcc")
    if found is None:
        raise FileNotFoundError(
            "nvcc not found; set ABOLETH_NVCC or CUDA_HOME"
        )
    return found


def bind(shared: pathlib.Path):
    library = ctypes.CDLL(str(shared))
    u64, u32, i32 = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int
    library.tier1_arm_put.argtypes = [u64, u64, u64, u32, u32, u32, i32]
    library.tier1_arm_put_inline.argtypes = [u64, u64, u32, u64, i32]
    library.tier1_arm_signal.argtypes = [u64, u64, u32, u64, i32]
    library.tier1_arm_put_signal.argtypes = [
        u64, u64, u64, u32, u32, u32, u64, u32, u64, i32]
    library.tier1_arm_put_many.argtypes = [u64, u64, u32, i32]
    library.tier1_arm_wait_signal.argtypes = [u64, u64]
    library.tier1_arm_quiet.argtypes = [u64]
    for name in (
        "tier1_arm_put", "tier1_arm_put_inline", "tier1_arm_signal",
        "tier1_arm_put_signal",
        "tier1_arm_put_many", "tier1_arm_wait_signal", "tier1_arm_quiet",
    ):
        getattr(library, name).restype = i32
    return library


def compile_arm(directory: pathlib.Path, generated=None):
    """Compile the arm source against one of the two headers."""
    root = pathlib.Path(__file__).resolve().parents[2]
    source = root / "experiments" / "cuda" / "aboleth_tier1_arm.cu"
    name = "generated" if generated is not None else "facade"
    shared = directory / f"libaboleth_tier1_arm_{name}.so"
    command = [
        nvcc(), "-O3", "-std=c++17", f"-arch={arch()}",
        "-Xcompiler", "-fPIC", "-shared",
    ]
    if generated is not None:
        command += [
            "-rdc=true",
            "-I", str(generated.include_dir),
            str(source),
            str(generated.archive),
        ]
    else:
        command += ["-I", str(root / "include"), str(source)]
    command += ["-o", str(shared)]
    result = subprocess.run(
        [str(argument) for argument in command],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc failed on the {name} arm: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    print(f"{name}_arm={shared.name}")
    return bind(shared)


def slot_name(case: str, index: int) -> str:
    layout = LAYOUTS[case]
    if index == len(layout) - 1:
        return "drain"
    return f"slot{index}"


def fields(case: str, raw: bytes, stride: int) -> dict:
    result = {}
    for index, (kind, used, _) in enumerate(LAYOUTS[case]):
        base = index * stride
        name = slot_name(case, index)
        result[f"{name}_opcode"] = struct.unpack_from(">I", raw, base)[0]
        result[f"{name}_segments"] = struct.unpack_from(
            ">I", raw, base + 4)[0]
        result[f"{name}_control"] = struct.unpack_from(
            "<I", raw, base + 8)[0]
        if kind == "nop":
            continue
        result[f"{name}_remote_address"] = struct.unpack_from(
            ">Q", raw, base + 16)[0]
        result[f"{name}_remote_key"] = struct.unpack_from(
            ">I", raw, base + 24)[0]
        if kind == "inline":
            result[f"{name}_inline_header"] = struct.unpack_from(
                ">I", raw, base + 32)[0]
            result[f"{name}_payload"] = struct.unpack_from(
                "<Q", raw, base + 36)[0]
        elif kind == "atomic":
            result[f"{name}_swap"] = struct.unpack_from(
                ">Q", raw, base + 32)[0]
            result[f"{name}_compare"] = struct.unpack_from(
                ">Q", raw, base + 40)[0]
            result[f"{name}_result_bytes"] = struct.unpack_from(
                ">I", raw, base + 48)[0]
        else:
            result[f"{name}_bytes"] = struct.unpack_from(
                ">I", raw, base + 32)[0]
            result[f"{name}_local_key"] = struct.unpack_from(
                ">I", raw, base + 36)[0]
            result[f"{name}_local_address"] = struct.unpack_from(
                ">Q", raw, base + 40)[0]
    return result


def normalized(case: str, raw: bytes, stride: int) -> bytes:
    result = bytearray()
    for index, (_, used, variable) in enumerate(LAYOUTS[case]):
        base = index * stride
        wqe = bytearray(raw[base:base + used])
        for offset, length in variable:
            wqe[offset:offset + length] = bytes(length)
        result += wqe
    return bytes(result)


def differences(case: str, left: bytes, right: bytes) -> list[str]:
    report = []
    base = 0
    for index, (_, used, _variable) in enumerate(LAYOUTS[case]):
        for offset in range(0, used, 4):
            a = left[base + offset:base + offset + 4]
            b = right[base + offset:base + offset + 4]
            if a != b:
                report.append(
                    f"{slot_name(case, index)} offset {offset}: "
                    f"generated {a.hex()} != facade {b.hex()}"
                )
        base += used
    return report


class Run:
    """A fresh endpoint pair plus the buffers one case writes."""

    def __init__(self):
        self.descriptor = torch.empty(
            DESCRIPTOR_WORDS, dtype=torch.uint64, device="cuda:0"
        )
        payload_bytes = PUT_MANY_COUNT * SLOT_BYTES
        self.source_allocation = torch.zeros(
            payload_bytes + SIGNAL_BYTES, dtype=torch.uint8, device="cuda:0"
        )
        self.destination_allocation = torch.zeros(
            payload_bytes + SIGNAL_BYTES, dtype=torch.uint8, device="cuda:1"
        )
        self.source = self.source_allocation[:payload_bytes]
        self.destination = self.destination_allocation[:payload_bytes]
        self.remote_signal = (
            self.destination_allocation[payload_bytes:].view(torch.int64)
        )
        payload = (
            (torch.arange(payload_bytes, device="cuda:0") * 31 + 7) & 0xFF
        ).to(torch.uint8)
        self.source.copy_(payload)
        self.payload = payload
        self.library = load_runtime()
        self.endpoints = create_endpoints(
            self.library,
            self.descriptor,
            self.source_allocation,
            self.destination_allocation,
        )
        self.expected = self.endpoints.descriptor
        self.source_lkey, self.destination_rkey = self.endpoints.keys()
        self.table = torch.tensor(
            [self.descriptor.data_ptr()],
            dtype=torch.uint64,
            device="cuda:0",
        )

    def snapshot(self, case: str):
        return self.endpoints.send_queue_snapshot(
            len(LAYOUTS[case]) * self.expected.send_stride
        )

    def close(self):
        self.endpoints.close()


def check(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with CUDA status {status}")


def failure_word(run: Run) -> int:
    """Completion-state word 3: the drain's recorded failure code."""
    words = (ctypes.c_uint64 * 4)()
    run.library.aboleth_runtime_endpoint_pair_completion_snapshot(
        run.endpoints._pair, 0, words, 4
    )
    return words[3]


def launch_case(case: str, arm, run: Run) -> None:
    # The runtime's pair bring-up may leave another device current; every
    # launcher must run where its table and descriptor live.
    table = run.table.data_ptr()
    if case == "put":
        with torch.cuda.device(0):
            check(arm.tier1_arm_put(
                table, run.destination.data_ptr(), run.source.data_ptr(),
                SLOT_BYTES, run.source_lkey, run.destination_rkey, 1,
            ), "tier1_arm_put")
    elif case == "put_inline":
        with torch.cuda.device(0):
            check(arm.tier1_arm_put_inline(
                table, run.remote_signal.data_ptr(), run.destination_rkey,
                INLINE_VALUE, 1,
            ), "tier1_arm_put_inline")
    elif case == "signal":
        with torch.cuda.device(0):
            check(arm.tier1_arm_signal(
                table, run.remote_signal.data_ptr(), run.destination_rkey,
                1, 1,
            ), "tier1_arm_signal")
        # The published epoch satisfies a waiter on the destination device —
        # the wait_signal op, exercised where it is actually used.
        with torch.cuda.device(1):
            check(arm.tier1_arm_wait_signal(
                run.remote_signal.data_ptr(), 1,
            ), "tier1_arm_wait_signal")
    elif case == "put_signal":
        with torch.cuda.device(0):
            check(arm.tier1_arm_put_signal(
                table, run.destination.data_ptr(), run.source.data_ptr(),
                SLOT_BYTES, run.source_lkey, run.destination_rkey,
                run.remote_signal.data_ptr(), run.destination_rkey, 1, 1,
            ), "tier1_arm_put_signal")
        with torch.cuda.device(1):
            check(arm.tier1_arm_wait_signal(
                run.remote_signal.data_ptr(), 1,
            ), "tier1_arm_wait_signal")
    elif case == "put_many":
        descriptors = torch.zeros(
            PUT_MANY_COUNT * PUT_DESC_BYTES,
            dtype=torch.uint8,
            device="cuda:0",
        )
        packed = bytearray()
        for index in range(PUT_MANY_COUNT):
            packed += struct.pack(
                "<QQIIII",
                run.destination.data_ptr() + index * SLOT_BYTES,
                run.source.data_ptr() + index * SLOT_BYTES,
                SLOT_BYTES,
                run.destination_rkey,
                run.source_lkey,
                0,
            )
        descriptors.copy_(torch.frombuffer(
            bytes(packed), dtype=torch.uint8
        ).to("cuda:0"))
        with torch.cuda.device(0):
            check(arm.tier1_arm_put_many(
                table, descriptors.data_ptr(), PUT_MANY_COUNT, 1,
            ), "tier1_arm_put_many")
    else:
        raise ValueError(case)


def verify_case(case: str, name: str, run: Run) -> tuple[bytes, dict]:
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    layout = LAYOUTS[case]
    raw, dbrec_head = run.snapshot(case)
    assert dbrec_head == len(layout), (case, dbrec_head)
    observed = fields(case, raw, run.expected.send_stride)
    qpn = run.expected.qpn

    if case in ("put", "put_signal"):
        torch.testing.assert_close(
            run.destination[:SLOT_BYTES].cpu(),
            run.payload[:SLOT_BYTES].cpu(),
            rtol=0, atol=0,
        )
        if case == "put_signal":
            assert run.remote_signal.cpu().item() == 1
    elif case in ("put_inline", "signal"):
        expected_cell = INLINE_VALUE if case == "put_inline" else 1
        assert run.remote_signal.cpu().item() == expected_cell
    elif case == "put_many":
        torch.testing.assert_close(
            run.destination.cpu(), run.payload.cpu(), rtol=0, atol=0,
        )

    for index, (kind, used, _) in enumerate(layout):
        slot = slot_name(case, index)
        if kind == "nop":
            assert observed[f"{slot}_opcode"] == (index << 8) | NOP_OPCODE
            assert observed[f"{slot}_segments"] == (qpn << 8) | 1
            assert observed[f"{slot}_control"] == CQ_UPDATE
            continue
        if kind == "atomic":
            assert observed[f"{slot}_opcode"] == (
                (index << 8) | COMPARE_SWAP_OPCODE
            )
            assert observed[f"{slot}_segments"] == (qpn << 8) | 4
            assert observed[f"{slot}_control"] == 0
            assert observed[f"{slot}_remote_key"] == run.destination_rkey
            assert observed[f"{slot}_remote_address"] == (
                run.remote_signal.data_ptr()
            )
            assert observed[f"{slot}_swap"] == 1
            assert observed[f"{slot}_compare"] == 0
            assert observed[f"{slot}_result_bytes"] == 8
            continue
        assert observed[f"{slot}_opcode"] == (
            (index << 8) | RDMA_WRITE_OPCODE
        )
        assert observed[f"{slot}_segments"] == (qpn << 8) | 3
        assert observed[f"{slot}_remote_key"] == run.destination_rkey
        if kind == "inline":
            assert observed[f"{slot}_inline_header"] == INLINE_HEADER
            assert observed[f"{slot}_payload"] == INLINE_VALUE
            assert observed[f"{slot}_remote_address"] == (
                run.remote_signal.data_ptr()
            )
            assert observed[f"{slot}_control"] == 0
        else:
            assert observed[f"{slot}_control"] == 0
            assert observed[f"{slot}_bytes"] == SLOT_BYTES
            assert observed[f"{slot}_local_key"] == run.source_lkey
            assert observed[f"{slot}_remote_address"] == (
                run.destination.data_ptr() + index * SLOT_BYTES
            )
            assert observed[f"{slot}_local_address"] == (
                run.source.data_ptr() + index * SLOT_BYTES
            )

    # Exactly one WQE requests a CQE: the drain's carrier.
    signalled = [
        slot_name(case, index)
        for index in range(len(layout))
        if observed[f"{slot_name(case, index)}_control"] & CQ_UPDATE
    ]
    assert signalled == ["drain"], (case, signalled)

    producer, consumer, _, _ = run.endpoints.state()
    assert producer == len(layout), (case, producer)
    assert consumer == 1, (case, consumer)

    print(f"{case}_{name}_wqes={len(layout)}")
    print(f"{case}_{name}_doorbell_head={dbrec_head}")
    return raw, observed


def run_arm(case: str, name: str, arm) -> tuple[bytes, int, dict]:
    run = Run()
    try:
        launch_case(case, arm, run)
        raw, observed = verify_case(case, name, run)
        return raw, run.expected.send_stride, observed
    finally:
        run.close()


# --- The Triton consumer: the same generated ops through the bitcode. ------


@core.extern
def triton_put(
    endpoint, destination, source, bytes_, source_key, destination_key,
    predicate, _semantic=None,
):
    address = core.dtype("uint64")
    scalar = core.dtype("int32")
    return core.extern_elementwise(
        "aboleth_tier1_put",
        TRITON_LIBS["put"],
        [
            endpoint, destination, source, bytes_,
            source_key, destination_key, predicate,
        ],
        {
            (
                address, address, address, scalar, scalar, scalar, scalar,
            ): ("aboleth_tier1_put_rdma", core.dtype("int64")),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def triton_producer(
    endpoint, destination, source, source_key, destination_key,
    slot_bytes: tl.constexpr,
):
    triton_put(
        endpoint.to(tl.uint64),
        destination.to(tl.uint64),
        source.to(tl.uint64),
        slot_bytes,
        source_key,
        destination_key,
        1,
    )


WARP = 32


def run_triton(generated_arm) -> None:
    """The generated ops through Triton `extern_libs`, per-thread.

    A Triton extern call executes on every active thread — there is no way to
    spell a lane guard around it — so with num_warps=1 the producer issues 32
    independent puts. That is the ops' per-thread contract working as
    documented: every reservation gets its own slot and the queue stays
    coherent. Slot order is a race between lanes, so this arm is functional
    rather than byte-gated (the byte gate is the CUDA arms above), and the
    drain runs once from a single-thread launcher — quiet is single-consumer
    per endpoint, from Triton exactly as from CUDA.
    """
    run = Run()
    try:
        options = {
            "num_warps": 1,
            "extern_libs": {
                "aboleth_tier1_put": TRITON_LIBS["put"],
            },
        }
        arguments = (
            run.descriptor,
            run.destination,
            run.source,
            run.source_lkey,
            run.destination_rkey,
        )
        with torch.cuda.device(0):
            compiled = triton_producer.warmup(
                *arguments, slot_bytes=SLOT_BYTES, grid=(1,), **options,
            )
            assert "aboleth." not in compiled.asm["ttir"]
            assert "aboleth_tier1_put_rdma" in compiled.asm["llir"]
            triton_producer[(1,)](
                *arguments, slot_bytes=SLOT_BYTES, **options,
            )
            check(
                generated_arm.tier1_arm_quiet(run.table.data_ptr()),
                "tier1_arm_quiet",
            )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        torch.testing.assert_close(
            run.destination[:SLOT_BYTES].cpu(),
            run.payload[:SLOT_BYTES].cpu(),
            rtol=0, atol=0,
        )
        # 32 puts plus the drain's signalled NOP, which consumed the one CQE.
        producer, consumer, _, _ = run.endpoints.state()
        raw, dbrec_head = run.endpoints.send_queue_snapshot(
            (WARP + 1) * run.expected.send_stride
        )
        assert producer == WARP + 1, producer
        assert consumer == 1, consumer
        assert dbrec_head == WARP + 1, dbrec_head
        assert failure_word(run) == 0
        opcodes = [
            struct.unpack_from(">I", raw, index * run.expected.send_stride)[0]
            & 0xFF
            for index in range(WARP + 1)
        ]
        assert opcodes.count(RDMA_WRITE_OPCODE) == WARP
        assert opcodes.count(NOP_OPCODE) == 1
        print(f"put_triton_per_thread_wqes={WARP}")
        print(f"put_triton_doorbell_head={dbrec_head}")
    finally:
        run.close()


def compare(case: str, name: str, generated, facade) -> None:
    generated_raw, generated_stride, generated_fields = generated
    facade_raw, facade_stride, facade_fields = facade
    assert generated_stride == facade_stride
    left = normalized(case, generated_raw, generated_stride)
    right = normalized(case, facade_raw, facade_stride)
    if left != right:
        for line in differences(case, left, right):
            print(f"DIFF {case} {line}")
        for key in sorted(generated_fields):
            if generated_fields[key] != facade_fields[key]:
                print(
                    f"DIFF {case} field {key}: generated "
                    f"{generated_fields[key]:#x} != facade "
                    f"{facade_fields[key]:#x}"
                )
        raise AssertionError(
            f"the generated {name} arm's wire bytes differ from the "
            f"hand-written facade for {case}"
        )
    print(f"{case}_{name}_compared_bytes={len(left)}")


def cuda_check(result, operation: str) -> None:
    error = result[0]
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {error}")


def run() -> None:
    assert torch.cuda.device_count() >= 2
    assert torch.cuda.can_device_access_peer(0, 1)
    # The Triton arm passes device-1 pointers to a device-0 launch.
    cuda_check(cudart.cudaSetDevice(0), "cudaSetDevice(0)")
    enabled = cudart.cudaDeviceEnablePeerAccess(1, 0)
    if enabled[0] not in (
        cudart.cudaError_t.cudaSuccess,
        cudart.cudaError_t.cudaErrorPeerAccessAlreadyEnabled,
    ):
        cuda_check(enabled, "cudaDeviceEnablePeerAccess")
    with tempfile.TemporaryDirectory(prefix="aboleth-tier1-diff-") as out:
        directory = pathlib.Path(out)
        library = generate_device_library(
            arch(), output_dir=directory / "tier1", require_archive=True,
        )
        print(f"generated_library_arch={library.arch}")
        print(f"generated_library_digest={library.digest[:12]}")
        rdma_bitcodes = library.op_bitcodes["rdma"]
        TRITON_LIBS["put"] = str(rdma_bitcodes["put"])
        TRITON_LIBS["quiet"] = str(rdma_bitcodes["quiet"])
        generated_arm = compile_arm(directory, generated=library)
        facade_arm = compile_arm(directory)

        for case in CASES:
            facade = run_arm(case, "facade", facade_arm)
            generated = run_arm(case, "generated", generated_arm)
            compare(case, "generated", generated, facade)
        # Deferred completion across a kernel boundary: a put that quiets in
        # a LATER launch still reclaims — the contract callers actually rely
        # on when they batch quiets.
        control = Run()
        try:
            with torch.cuda.device(0):
                check(generated_arm.tier1_arm_put(
                    control.table.data_ptr(),
                    control.destination.data_ptr(),
                    control.source.data_ptr(),
                    SLOT_BYTES, control.source_lkey,
                    control.destination_rkey, 0,
                ), "control put")
                check(generated_arm.tier1_arm_quiet(
                    control.table.data_ptr(),
                ), "control quiet")
            torch.cuda.synchronize(0)
            producer, consumer, _, _ = control.endpoints.state()
            assert (producer, consumer) == (2, 1), (producer, consumer)
            assert failure_word(control) == 0
            print("cross_kernel_quiet=PASS")
        finally:
            control.close()
        run_triton(generated_arm)

    print("aboleth_tier1_differential=PASS")


if __name__ == "__main__":
    run()
