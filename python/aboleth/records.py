"""Semantic record transport shared by compiler-planned Tier-2 libraries.

The artifact supplies transport partitions, lane count and slice coefficients.
The caller supplies addresses and executes the artifact's launches. This module
has no lane selection, Queue IR, explicit fences or runtime schedule knobs.
"""

import triton
import triton.language as tl
import aboleth.language as ops


@triton.jit
def send_record_slice(
    endpoints,
    payload,
    payload_key,
    metadata,
    metadata_key,
    destination_data,
    destination_meta,
    destination_counts,
    destination_flags,
    count,
    epoch,
    peer,
    lane,
    RANK: tl.constexpr,
    PEER_EXTENT: tl.constexpr,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    META_WIDTH: tl.constexpr,
    LANES: tl.constexpr,
    ROW_QUOTIENT: tl.constexpr,
    ROW_REMAINDER: tl.constexpr,
    REUSE: tl.constexpr,
    SEND_COUNT: tl.constexpr,
    READY_STRIDE: tl.constexpr,
):
    endpoint = ops.endpoint(endpoints, lane * PEER_EXTENT + tl.program_id(0))
    # All offsets stay i64. The two coefficients are emitted by C++, and tile
    # the indivisible record domain exactly even for non-power-of-two bounds.
    begin = ROW_QUOTIENT * lane.to(tl.int64) + ROW_REMAINDER * lane // LANES
    end = (
        ROW_QUOTIENT * (lane.to(tl.int64) + 1)
        + ROW_REMAINDER * (lane + 1) // LANES
    )
    ELEMENT_BYTES: tl.constexpr = (
        payload.dtype.element_ty.primitive_bitwidth // 8
    )
    source = ops.region(
        payload + peer.to(tl.int64) * CAPACITY * WIDTH, payload_key
    ).valid(count, CAPACITY, WIDTH * ELEMENT_BYTES)
    remote = ops.remote(destination_data, peer, payload.dtype.element_ty)
    remote_flag = ops.remote(destination_flags, peer, tl.int64)
    header = ops.put(
        endpoint,
        remote.at((RANK * CAPACITY + begin) * WIDTH),
        source.at(begin * WIDTH),
        ((end - begin) * WIDTH * ELEMENT_BYTES).to(tl.int32),
    )
    if META_WIDTH > 0:
        meta = ops.region(
            metadata + peer.to(tl.int64) * CAPACITY * META_WIDTH, metadata_key
        ).valid(count, CAPACITY, META_WIDTH * 4)
        remote_meta = ops.remote(destination_meta, peer, tl.int32)
        header = ops.put(
            endpoint,
            remote_meta.at((RANK * CAPACITY + begin) * META_WIDTH),
            meta.at(begin * META_WIDTH),
            ((end - begin) * META_WIDTH * 4).to(tl.int32),
            after=header,
        )
    if SEND_COUNT:
        remote_count = ops.remote(destination_counts, peer, tl.int64)
        header = ops.put_inline(
            endpoint, remote_count.at(RANK), source.count, after=header
        )
    published = ops.signal(
        endpoint,
        remote_flag.at(RANK * READY_STRIDE + lane),
        epoch,
        after=header,
    )
    ops.wait(published, iteration=epoch, reuse=REUSE)


@triton.jit(do_not_specialize=["active"])
def send_records(
    endpoints,
    payload,
    payload_key,
    metadata,
    metadata_key,
    destination_data,
    destination_meta,
    destination_counts,
    destination_flags,
    counts,
    epoch_pointer,
    active,
    RANK: tl.constexpr,
    PEERS: tl.constexpr,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    META_WIDTH: tl.constexpr,
    LANES: tl.constexpr,
    ROW_QUOTIENT: tl.constexpr,
    ROW_REMAINDER: tl.constexpr,
    REUSE: tl.constexpr,
    READY_STRIDE: tl.constexpr = 0,
    COUNT_PRESENT: tl.constexpr = True,
):
    if active != 0:
        STRIDE: tl.constexpr = LANES if READY_STRIDE == 0 else READY_STRIDE
        peer_index, lane = tl.program_id(0), tl.program_id(1)
        peer = tl.full((), PEERS[0], tl.int64)
        for index in tl.static_range(1, len(PEERS)):
            peer = tl.where(peer_index == index, PEERS[index], peer)
        count, epoch = tl.load(counts + peer), tl.load(epoch_pointer)
        # Keep each complete token chain in one lexical block. The only
        # specialization is the semantic fact that lane zero owns the count.
        if not COUNT_PRESENT:
            send_record_slice(
                endpoints,
                payload,
                payload_key,
                metadata,
                metadata_key,
                destination_data,
                destination_meta,
                destination_counts,
                destination_flags,
                count,
                epoch,
                peer,
                lane,
                RANK,
                len(PEERS),
                CAPACITY,
                WIDTH,
                META_WIDTH,
                LANES,
                ROW_QUOTIENT,
                ROW_REMAINDER,
                REUSE,
                False,
                STRIDE,
            )
        elif LANES == 1 or lane == 0:
            send_record_slice(
                endpoints,
                payload,
                payload_key,
                metadata,
                metadata_key,
                destination_data,
                destination_meta,
                destination_counts,
                destination_flags,
                count,
                epoch,
                peer,
                lane,
                RANK,
                len(PEERS),
                CAPACITY,
                WIDTH,
                META_WIDTH,
                LANES,
                ROW_QUOTIENT,
                ROW_REMAINDER,
                REUSE,
                True,
                STRIDE,
            )
        else:
            send_record_slice(
                endpoints,
                payload,
                payload_key,
                metadata,
                metadata_key,
                destination_data,
                destination_meta,
                destination_counts,
                destination_flags,
                count,
                epoch,
                peer,
                lane,
                RANK,
                len(PEERS),
                CAPACITY,
                WIDTH,
                META_WIDTH,
                LANES,
                ROW_QUOTIENT,
                ROW_REMAINDER,
                REUSE,
                False,
                STRIDE,
            )


@triton.jit(do_not_specialize=["active"])
def await_records(
    flags,
    epoch_pointer,
    active,
    PEERS: tl.constexpr,
    LANES: tl.constexpr,
    READY_STRIDE: tl.constexpr = 0,
    LANES_PER_PROGRAM: tl.constexpr = 1,
):
    if active != 0:
        STRIDE: tl.constexpr = LANES if READY_STRIDE == 0 else READY_STRIDE
        program = tl.program_id(0)
        peer = tl.full((), PEERS[0], tl.int64)
        for index in tl.static_range(1, len(PEERS)):
            peer = tl.where(program == index, PEERS[index], peer)
        epoch = tl.load(epoch_pointer)
        # This kernel only joins publications; it publishes no dependency of
        # another lane. Grouping preserves the same acquire on every signal.
        for offset in tl.static_range(LANES_PER_PROGRAM):
            lane = tl.program_id(1) * LANES_PER_PROGRAM + offset
            if lane < LANES:
                ops.wait_signal(flags + peer * STRIDE + lane, epoch)
