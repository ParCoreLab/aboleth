"""Hardware proof that a register-direct inline payload is the same transfer as a
staged one, and that its queue accounting is in WQEBBs.

For each payload size, three runs over a fresh endpoint pair each:

  a. the inline path — a Triton kernel computes the payload in registers and
     calls `aboleth.put_inline`, so the bytes travel registers → WQE inline
     segment. No staging store, no lkey, and no adapter read of the payload;
  b. the staged path — the same values written into a registered source buffer
     and sent by `aboleth.put`, the form this milestone replaces;
  c. the forwarded path — the staged source is written by scalar stores in the
     kernel and `aboleth-forward-inline-payloads` turns the put into an inline
     transfer, which is checked in the scheduled IR as well as on the wire.

An inactive predicate is checked on both sides of the one-WQEBB boundary too: it
becomes a NOP that keeps the write's segment count, and the batch completing is
the adapter accepting a command that occupies several WQEBBs without
transferring.

The receiver's bytes must be identical for all three: that is the semantic
equivalence claim. The sender's send queue is then read back and checked field by
field, which is where the accounting claim lives — the inline-data header, the ds
segment count, the WQE index the following command lands on, and the doorbell
head all have to reflect the WQEBBs the payload consumed rather than the number
of commands posted.

`aboleth_fragment_differential.py` is the same style of proof for the artifact
path.
"""

import contextlib
import io
import os
import struct

import torch
import triton
import triton.language as tl
from cuda.bindings import runtime as cudart

import aboleth.language as aboleth
from aboleth.compiler import DEFAULT_INLINE_BYTES, compilation
from aboleth_rdma_abi_test import (
    DESCRIPTOR_WORDS,
    create_endpoints,
    load_runtime,
)


# Every size the granted inline capacity has to cover. 8 is one 64-bit word, 28
# is the largest payload that still fits one WQEBB, and the rest cross that
# boundary: 64 needs two WQEBBs, 128 three, 256 five.
SIZES = (8, 16, 28, 64, 128, 256)
MAX_BYTES = max(SIZES)

# Predication is checked on both sides of the one-WQEBB boundary, because that is
# where the placeholder NOP has to keep the write's segment count.
PREDICATED_SIZES = (28, 256)
SIGNAL_BYTES = 8
ALLOCATION_BYTES = MAX_BYTES + SIGNAL_BYTES

# The forwarded case stages 3 x i64 through a scratch buffer.
FORWARDED_WORDS = 3
FORWARDED_BYTES = FORWARDED_WORDS * 8

# mlx5 wire constants.
RDMA_WRITE_OPCODE = 0x08
ATOMIC_OPCODE = 0x11
CQ_UPDATE = 0x08000000
INLINE_SEGMENT = 0x80000000


def lowered_header(size: int) -> int:
    """The inline segment header as the lowered TTIR spells it: canonicalized
    after lowering, the byte swap folds into a big-endian i32 constant."""
    header = (INLINE_SEGMENT | size).to_bytes(4, "little")
    return int.from_bytes(header, "big", signed=True)
WQE_BLOCK_BYTES = 64
WQE_SEGMENT_BYTES = 16
INLINE_HEADER_OFFSET = 32

# The payload each value carries, mirrored by the kernels below so the receiver
# can be checked against a value the host computed independently.
MULTIPLIER = 6364136223846793005
INCREMENT = 2654435761
MASK64 = (1 << 64) - 1


def inline_segment_bytes(payload_bytes: int) -> int:
    used = 4 + payload_bytes
    return -(-used // WQE_SEGMENT_BYTES) * WQE_SEGMENT_BYTES


def inline_wqe_segments(payload_bytes: int) -> int:
    """The ds field: control, remote address, and the inline-data segment."""
    return 2 + inline_segment_bytes(payload_bytes) // WQE_SEGMENT_BYTES


def inline_wqe_blocks(payload_bytes: int) -> int:
    """WQEBBs the command occupies, which is what it costs the producer index."""
    used = INLINE_HEADER_OFFSET + inline_segment_bytes(payload_bytes)
    return -(-used // WQE_BLOCK_BYTES)


def expected_payload(seed: int, words: int, tail: bool) -> bytes:
    """The bytes the kernels below compute, derived here independently."""
    payload = bytearray()
    for index in range(words):
        value = (seed * MULTIPLIER + index * INCREMENT) & MASK64
        payload += struct.pack("<Q", value)
    if tail:
        value = (seed * 2654435761 + 12345) & 0xFFFFFFFF
        payload += struct.pack("<I", value)
    return bytes(payload)


def payload_shape(size: int) -> tuple[int, bool]:
    """How many 64-bit values and whether a 32-bit tail completes `size`."""
    words, remainder = divmod(size, 8)
    if remainder == 0:
        return words, False
    assert remainder == 4, size
    return words, True


@triton.jit
def inline_producer(
    endpoint,
    destination,
    remote_signal,
    local_signal,
    source_key,
    destination_key,
    seed,
    active,
    words: tl.constexpr,
    tail: tl.constexpr,
):
    """Compute the payload in registers and send it without staging it."""
    base = seed.to(tl.uint64)
    values = ()
    for index in tl.static_range(0, words):
        values = values + (base * 6364136223846793005 +
                           index * 2654435761, )
    if tail:
        values = values + ((base * 2654435761 + 12345).to(tl.uint32), )
    transfer = aboleth.put_inline(
        endpoint,
        destination,
        *values,
        predicate=active,
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


@triton.jit
def staged_producer(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    source_key,
    destination_key,
    byte_count: tl.constexpr,
):
    """The form this milestone replaces: the payload is already in a buffer."""
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


@triton.jit
def forwarded_producer(
    endpoint,
    destination,
    scratch,
    remote_signal,
    local_signal,
    source_key,
    destination_key,
    seed,
    words: tl.constexpr,
):
    """Stage the payload with scalar stores, then send the buffer.

    The forwarding pass is what turns this into an inline transfer; the source is
    written and sent exactly as an unoptimized kernel would do it.
    """
    base = seed.to(tl.uint64)
    for index in tl.static_range(0, words):
        tl.store(
            scratch + index,
            base * 6364136223846793005 + index * 2654435761,
        )
    transfer = aboleth.put(
        endpoint,
        destination,
        scratch,
        words * 8,
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


def cuda_check(result, operation: str) -> None:
    error = result[0]
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {error}")


def snapshot(endpoints, descriptor, blocks):
    return endpoints.send_queue_snapshot(
        blocks * descriptor.send_stride
    )


def wqe_fields(raw: bytes, stride: int, block: int) -> dict:
    """The control fields of the WQE that begins at `block`."""
    base = block * stride
    return {
        "opcode": struct.unpack_from(">I", raw, base)[0],
        "segments": struct.unpack_from(">I", raw, base + 4)[0],
        "control": struct.unpack_from("<I", raw, base + 8)[0],
        "remote_address": struct.unpack_from(">Q", raw, base + 16)[0],
        "remote_key": struct.unpack_from(">I", raw, base + 24)[0],
    }


def inline_payload(raw: bytes, stride: int, payload_bytes: int) -> bytes:
    """The payload bytes as they sit in the queue, following its WQEBB wrap.

    The command starts at block zero of the snapshot, so the block index inside
    the WQE is also the snapshot's block index.
    """
    payload = bytearray()
    for offset in range(36, 36 + payload_bytes):
        block, inside = divmod(offset, WQE_BLOCK_BYTES)
        payload.append(raw[block * stride + inside])
    return bytes(payload)


def run_case(name: str, launch, blocks: int, payload: bytes):
    """Create a fresh endpoint pair, run `launch`, and read back both sides."""
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
    source = source_allocation[:MAX_BYTES]
    destination = destination_allocation[:MAX_BYTES]
    local_signal = source_allocation[MAX_BYTES:].view(torch.int64)
    remote_signal = destination_allocation[MAX_BYTES:].view(torch.int64)
    local_signal.fill_(1)

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
        granted = endpoints.max_inline()
        if granted < len(payload):
            raise RuntimeError(
                f"the adapter granted {granted} inline bytes, which cannot "
                f"carry this {len(payload)}-byte payload"
            )
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

        received = bytes(destination[:len(payload)].cpu().numpy().tobytes())
        assert remote_signal.cpu().item() == 1
        raw, dbrec_head = snapshot(endpoints, expected, blocks)
        print(f"{name}_qpn={expected.qpn}")
        print(f"{name}_granted_inline_bytes={granted}")
        print(f"{name}_doorbell_head={dbrec_head}")
        return {
            "received": received,
            "raw": raw,
            "stride": expected.send_stride,
            "dbrec_head": dbrec_head,
            "qpn": expected.qpn,
            "granted": granted,
            "remote_key": destination_rkey,
            "local_key": source_lkey,
            "destination": destination.data_ptr(),
            "source": source.data_ptr(),
            "remote_signal": remote_signal.data_ptr(),
        }
    finally:
        endpoints.close()


def run_inline(name: str, size: int, seed: int, payload: bytes, active: int):
    words, tail = payload_shape(size)
    blocks = inline_wqe_blocks(size) + 1

    def launch(descriptor, destination, source, remote_signal, local_signal,
               source_lkey, destination_rkey):
        arguments = (
            descriptor,
            destination,
            remote_signal,
            local_signal,
            source_lkey,
            destination_rkey,
            seed,
            active,
        )
        options = {"words": words, "tail": tail, "num_warps": 1}
        compiled = inline_producer.warmup(*arguments, grid=(1,), **options)
        lowered = compiled.asm["ttir"]
        assert "aboleth." not in lowered
        assert f"arith.constant {lowered_header(size)} : i32" in lowered, size
        inline_producer[(1,)](*arguments, **options)

    with compilation("rdma"):
        return run_case(name, launch, blocks, payload)


def predicated_case(size: int, seed: int, payload: bytes):
    """An inactive predicate is a NOP that keeps the write's segment count.

    That is what lets the command occupy the same WQEBBs whether or not it
    transfers, and the batch completing at all is the adapter accepting it: the
    publication that follows only runs if the NOP advanced the queue by the
    reservation's own count.
    """
    observed = run_inline(f"predicated{size}", size, seed, payload, active=0)
    stride = observed["stride"]
    write = wqe_fields(observed["raw"], stride, 0)
    signal = wqe_fields(observed["raw"], stride, inline_wqe_blocks(size))

    assert write["opcode"] == 0, hex(write["opcode"])
    assert write["segments"] == (observed["qpn"] << 8) | inline_wqe_segments(
        size), (write["segments"], inline_wqe_segments(size))
    assert write["control"] == 0, hex(write["control"])
    assert signal["opcode"] == (
        (inline_wqe_blocks(size) << 8) | ATOMIC_OPCODE
    ), hex(signal["opcode"])
    assert observed["dbrec_head"] == inline_wqe_blocks(size) + 1, (
        observed["dbrec_head"]
    )
    if observed["received"] != bytes(len(payload)):
        raise AssertionError(
            f"an inactive {size}-byte inline put still wrote "
            f"{observed['received'].hex()}"
        )
    print(f"predicated{size}_opcode=nop")
    print(f"predicated{size}_segments={inline_wqe_segments(size)}")
    print(f"predicated{size}_wqebbs={inline_wqe_blocks(size)}")
    return observed


def inline_case(size: int, seed: int, payload: bytes):
    observed = run_inline(f"inline{size}", size, seed, payload, active=1)

    # The wire form: one inline write, then the publication two commands' worth
    # of WQEBBs later only if the payload consumed that many.
    stride = observed["stride"]
    write = wqe_fields(observed["raw"], stride, 0)
    signal = wqe_fields(observed["raw"], stride, inline_wqe_blocks(size))
    header = struct.unpack_from(
        ">I", observed["raw"], INLINE_HEADER_OFFSET)[0]

    assert write["opcode"] == RDMA_WRITE_OPCODE, hex(write["opcode"])
    assert write["segments"] == (observed["qpn"] << 8) | inline_wqe_segments(
        size), (write["segments"], inline_wqe_segments(size))
    assert write["control"] == 0, hex(write["control"])
    assert write["remote_address"] == observed["destination"]
    assert write["remote_key"] == observed["remote_key"]
    assert header == INLINE_SEGMENT | size, hex(header)
    assert inline_payload(observed["raw"], stride, size) == payload

    # The following command's own WQE index is what proves the reservation
    # advanced in WQEBBs: a command-counting reservation would put it at one.
    assert signal["opcode"] == (
        (inline_wqe_blocks(size) << 8) | ATOMIC_OPCODE
    ), hex(signal["opcode"])
    assert signal["remote_address"] == observed["remote_signal"]
    assert signal["control"] & CQ_UPDATE
    assert observed["dbrec_head"] == inline_wqe_blocks(size) + 1, (
        observed["dbrec_head"]
    )

    print(f"inline{size}_wqebbs={inline_wqe_blocks(size)}")
    print(f"inline{size}_segments={inline_wqe_segments(size)}")
    print(f"inline{size}_inline_header={header:#x}")
    print(f"inline{size}_signal_wqe_index={signal['opcode'] >> 8}")
    return observed


def staged_case(size: int, payload: bytes):
    def launch(descriptor, destination, source, remote_signal, local_signal,
               source_lkey, destination_rkey):
        source[:size].copy_(
            torch.frombuffer(bytearray(payload), dtype=torch.uint8).cuda(0)
        )
        torch.cuda.synchronize(0)
        staged_producer[(1,)](
            descriptor,
            destination,
            source,
            remote_signal,
            local_signal,
            source_lkey,
            destination_rkey,
            byte_count=size,
            num_warps=1,
        )

    with compilation("rdma"):
        observed = run_case(f"staged{size}", launch, 2, payload)

    write = wqe_fields(observed["raw"], observed["stride"], 0)
    assert write["opcode"] == RDMA_WRITE_OPCODE
    assert write["segments"] == (observed["qpn"] << 8) | 3
    assert observed["dbrec_head"] == 2, observed["dbrec_head"]
    print(f"staged{size}_wqebbs=1")
    return observed


def forwarded_case(seed: int, payload: bytes):
    """The forwarding pass, proven in the scheduled IR and on the wire."""
    blocks = inline_wqe_blocks(FORWARDED_BYTES) + 1
    scheduled = io.StringIO()

    def launch(descriptor, destination, source, remote_signal, local_signal,
               source_lkey, destination_rkey):
        scratch = source.view(torch.int64)
        arguments = (
            descriptor,
            destination,
            scratch,
            remote_signal,
            local_signal,
            source_lkey,
            destination_rkey,
            seed,
        )
        options = {"words": FORWARDED_WORDS, "num_warps": 1}
        previous = os.environ.get("ABOLETH_DUMP_SCHEDULED_TTIR")
        os.environ["ABOLETH_DUMP_SCHEDULED_TTIR"] = "1"
        try:
            with contextlib.redirect_stdout(scheduled):
                compiled = forwarded_producer.warmup(
                    *arguments, grid=(1,), **options)
        finally:
            if previous is None:
                os.environ.pop("ABOLETH_DUMP_SCHEDULED_TTIR", None)
            else:
                os.environ["ABOLETH_DUMP_SCHEDULED_TTIR"] = previous
        lowered = compiled.asm["ttir"]
        assert "aboleth." not in lowered
        assert (
            f"arith.constant {lowered_header(FORWARDED_BYTES)} : i32" in lowered
        )
        forwarded_producer[(1,)](*arguments, **options)

    with compilation("rdma", inline_bytes=DEFAULT_INLINE_BYTES):
        observed = run_case("forwarded", launch, blocks, payload)

    # The scheduled module is the put after forwarding and batch formation: an
    # inline command, and no store left staging the payload.
    text = scheduled.getvalue()
    assert "aboleth.post_inline" in text, text[:2000]
    assert "aboleth.post_put" not in text, text[:2000]
    assert "tt.store" not in text, text[:2000]

    stride = observed["stride"]
    header = struct.unpack_from(">I", observed["raw"], INLINE_HEADER_OFFSET)[0]
    assert header == INLINE_SEGMENT | FORWARDED_BYTES, hex(header)
    assert inline_payload(observed["raw"], stride, FORWARDED_BYTES) == payload
    print(f"forwarded_bytes={FORWARDED_BYTES}")
    print(f"forwarded_inline_header={header:#x}")
    print("forwarded_staging_stores=0")
    return observed


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
        for index, size in enumerate(SIZES):
            seed = 0x9E3779B1 + index
            words, tail = payload_shape(size)
            payload = expected_payload(seed, words, tail)
            assert len(payload) == size

            inline = inline_case(size, seed, payload)
            staged = staged_case(size, payload)
            if inline["received"] != payload:
                raise AssertionError(
                    f"the inline {size}-byte payload did not arrive: "
                    f"{inline['received'].hex()} != {payload.hex()}"
                )
            if inline["received"] != staged["received"]:
                raise AssertionError(
                    f"the inline and staged {size}-byte forms delivered "
                    f"different bytes: {inline['received'].hex()} != "
                    f"{staged['received'].hex()}"
                )
            print(f"equivalent{size}_bytes={size}")

        # The predicated form on both sides of the one-WQEBB boundary: a NOP that
        # keeps the write's segment count has to be a command the adapter
        # accepts, or a batch containing one would never complete.
        for size in PREDICATED_SIZES:
            seed = 0x2545F491 + size
            words, tail = payload_shape(size)
            predicated_case(size, seed, expected_payload(seed, words, tail))

        seed = 0x51ED2701
        payload = expected_payload(seed, FORWARDED_WORDS, False)
        forwarded = forwarded_case(seed, payload)
        if forwarded["received"] != payload:
            raise AssertionError(
                "the forwarded payload did not arrive: "
                f"{forwarded['received'].hex()} != {payload.hex()}"
            )

        print(f"sizes={','.join(str(size) for size in SIZES)}")
        print("aboleth_inline_differential=PASS")
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
