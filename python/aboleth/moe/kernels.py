"""BF16 MoE compute around compiler-planned semantic record transport.

Rank deduplication, route metadata and weighted rank gather are the algorithm.
Transport, capacities and storage bindings come from the compiled artifact.
Only the three used compute kernels are retained from the measured native
implementation; no protocol/launch/environment choices live here.
"""

import triton
import triton.language as tl
from ..records import send_records, await_records, send_record_slice


@triton.jit
def pack_records(
    tokens,
    expert_ids,
    expert_weights,
    primary_ranks,
    record_tokens,
    record_metadata,
    record_counts,
    source_rank_slots,
    overflow,
    num_tokens,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    METADATA_WORDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    QUANT_GROUP: tl.constexpr = 0,
    ROUND_SCALE: tl.constexpr = False,
    AMAX_FLOOR: tl.constexpr = 1e-4,
    DESTINATION_BASE: tl.constexpr = 0,
):
    # A grid narrower than WORLD_SIZE packs destinations from
    # DESTINATION_BASE on (the fused dispatch packs only its own rank's).
    token = tl.program_id(0)
    destination = tl.program_id(1) + DESTINATION_BASE
    token_live = token < num_tokens
    selected = tl.zeros((), tl.int32)
    for choice in tl.static_range(TOP_K):
        expert = tl.load(
            expert_ids + token * TOP_K + choice,
            mask=token_live,
            other=0,
        ).to(tl.int32)
        expert_valid = (expert >= 0) & (expert < NUM_EXPERTS)
        primary = tl.load(
            primary_ranks + expert,
            mask=expert_valid,
            other=-1,
        )
        actual_destination = primary
        belongs_destination = actual_destination == destination
        selected += expert_valid & belongs_destination
    active = token_live & (selected > 0)
    slot = tl.atomic_add(
        record_counts + destination,
        active.to(tl.int32),
    )
    valid = active & (slot < MAX_TOKENS)
    tl.atomic_max(overflow, (active & ~valid).to(tl.int32))
    tl.store(
        source_rank_slots + token * WORLD_SIZE + destination,
        tl.where(valid, slot, -1),
        mask=token_live,
    )
    record = destination * MAX_TOKENS + slot
    hidden = tl.arange(0, BLOCK_H)
    values = tl.load(
        tokens + tl.cast(token, tl.int64) * HIDDEN + hidden,
        mask=token_live & (hidden < HIDDEN),
        other=0.0,
    )
    if QUANT_GROUP > 0:
        SCALE_COLS: tl.constexpr = HIDDEN // QUANT_GROUP
        BLOCK_GROUPS: tl.constexpr = BLOCK_H // QUANT_GROUP
        grouped = tl.reshape(
            values.to(tl.float32), (BLOCK_GROUPS, QUANT_GROUP)
        )
        amax = tl.max(tl.maximum(tl.abs(grouped), AMAX_FLOOR), axis=1)
        scale = tl.div_rn(amax, 448.0)
        if ROUND_SCALE:
            scale = tl.maximum(scale, 1e-10)
            bits = scale.to(tl.int32, bitcast=True)
            exponent = (bits >> 23) & 0xFF
            mantissa = bits & 0x7FFFFF
            exponent = tl.where(mantissa != 0, exponent + 1, exponent)
            scale = (exponent << 23).to(tl.float32, bitcast=True)
        quantized = tl.clamp(
            tl.div_rn(grouped, scale[:, None]), -448.0, 448.0
        ).to(record_tokens.dtype.element_ty)
        values = tl.reshape(quantized, (BLOCK_H,))
        group = tl.arange(0, BLOCK_GROUPS)
        tl.store(
            record_metadata + record * METADATA_WORDS + 2 + 2 * TOP_K + group,
            scale.to(tl.int32, bitcast=True),
            mask=valid & (group < SCALE_COLS),
        )
    tl.store(
        record_tokens + tl.cast(record, tl.int64) * HIDDEN + hidden,
        values,
        mask=valid & (hidden < HIDDEN),
    )
    metadata = record * METADATA_WORDS
    tl.store(record_metadata + metadata, token, mask=valid)
    tl.store(record_metadata + metadata + 1, selected, mask=valid)
    packed = tl.zeros((), tl.int32)
    for choice in tl.static_range(TOP_K):
        expert = tl.load(
            expert_ids + token * TOP_K + choice,
            mask=token_live,
            other=0,
        ).to(tl.int32)
        expert_valid = (expert >= 0) & (expert < NUM_EXPERTS)
        primary = tl.load(
            primary_ranks + expert,
            mask=expert_valid,
            other=-1,
        )
        actual_destination = primary
        belongs_destination = actual_destination == destination
        weight = tl.load(
            expert_weights + token * TOP_K + choice,
            mask=token_live,
            other=0.0,
        )
        belongs = valid & expert_valid & belongs_destination
        tl.store(
            record_metadata + metadata + 2 + packed,
            expert,
            mask=belongs,
        )
        weight_bits = weight.to(tl.int32, bitcast=True)
        tl.store(
            record_metadata + metadata + 2 + TOP_K + packed,
            weight_bits,
            mask=belongs,
        )
        packed += belongs.to(tl.int32)


@triton.jit
def unpack_records(
    received_metadata,
    received_counts,
    expert_counts,
    row_experts,
    row_weights,
    overflow,
    TOP_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    METADATA_WORDS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    PLACEHOLDER: tl.constexpr,
):
    row = tl.program_id(0)
    source_rank = row // MAX_TOKENS
    record_slot = row % MAX_TOKENS
    source_row = row
    output_row = row
    count = tl.load(received_counts + source_rank)
    live = (
        (source_rank < WORLD_SIZE)
        & (record_slot < count)
        & (tl.load(overflow) == 0)
    )
    metadata = source_row * METADATA_WORDS
    selection_count = tl.load(
        received_metadata + metadata + 1,
        mask=live,
        other=0,
    )
    selection_valid = (selection_count >= 0) & (selection_count <= TOP_K)
    live &= selection_valid
    for choice in tl.static_range(TOP_K):
        choice_live = live & (choice < selection_count)
        expert = tl.load(
            received_metadata + metadata + 2 + choice,
            mask=choice_live,
            other=-1,
        ).to(tl.int32)
        weight_bits = tl.load(
            received_metadata + metadata + 2 + TOP_K + choice,
            mask=choice_live,
            other=0,
        )
        weight = weight_bits.to(tl.float32, bitcast=True)
        expert_valid = choice_live & (expert >= 0) & (expert < NUM_EXPERTS)
        tl.atomic_max(
            overflow,
            (choice_live & ~expert_valid).to(tl.int32),
        )
        route = output_row * TOP_K + choice
        tl.store(
            row_experts + route,
            tl.where(expert_valid, expert, PLACEHOLDER),
        )
        tl.store(
            row_weights + route,
            tl.where(expert_valid, weight, 0.0),
        )
        tl.atomic_add(
            expert_counts + expert,
            expert_valid.to(tl.int32),
            mask=expert_valid,
        )


@triton.jit
def unpack_records_fp8(
    received_metadata,
    received_counts,
    expert_counts,
    row_experts,
    row_weights,
    overflow,
    row_scales,
    TOP_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    METADATA_WORDS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    PLACEHOLDER: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    BLOCK_SCALES: tl.constexpr,
):
    unpack_records(
        received_metadata,
        received_counts,
        expert_counts,
        row_experts,
        row_weights,
        overflow,
        TOP_K,
        WORLD_SIZE,
        MAX_TOKENS,
        METADATA_WORDS,
        NUM_EXPERTS,
        PLACEHOLDER,
    )
    row = tl.program_id(0)
    source = row // MAX_TOKENS
    live = (row % MAX_TOKENS < tl.load(received_counts + source)) & (
        tl.load(overflow) == 0
    )
    group = tl.arange(0, BLOCK_SCALES)
    bits = tl.load(
        received_metadata + row * METADATA_WORDS + 2 + 2 * TOP_K + group,
        mask=live & (group < SCALE_COLS),
        other=0,
    )
    scale = tl.where(live, bits.to(tl.float32, bitcast=True), 1.0)
    tl.store(
        row_scales + row * SCALE_COLS + group, scale, mask=group < SCALE_COLS
    )


@triton.jit
def gather_partials(
    returned_partials,
    local_rank_partials,
    source_rank_slots,
    output,
    overflow,
    num_tokens,
    HIDDEN: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
):
    token = tl.program_id(0)
    hidden = tl.arange(0, BLOCK_H)
    token_live = (tl.load(overflow) == 0) & (token < num_tokens)
    accumulator = tl.zeros([BLOCK_H], tl.float32)
    for destination in tl.static_range(WORLD_SIZE):
        slot = tl.load(
            source_rank_slots + token * WORLD_SIZE + destination,
            mask=token_live,
            other=-1,
        )
        contribution_live = token_live & (slot >= 0)
        if destination == LOCAL_RANK:
            partials = local_rank_partials
        else:
            partials = returned_partials
        value = tl.load(
            partials
            + tl.cast(destination * MAX_TOKENS + slot, tl.int64) * HIDDEN
            + hidden,
            mask=contribution_live & (hidden < HIDDEN),
            other=0.0,
        ).to(tl.float32)
        accumulator += value
    tl.store(
        output + tl.cast(token, tl.int64) * HIDDEN + hidden,
        accumulator,
        mask=token_live & (hidden < HIDDEN),
    )


@triton.jit
def _route_block(
    expert_ids,
    primary_ranks,
    token,
    token_live,
    destination,
    TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    """Per token of a block: how many of its choices `destination` owns."""
    selected = tl.zeros(token.shape, tl.int32)
    for choice in tl.static_range(TOP_K):
        expert = tl.load(
            expert_ids + token * TOP_K + choice, mask=token_live, other=-1
        ).to(tl.int32)
        expert_valid = (expert >= 0) & (expert < NUM_EXPERTS)
        owner = tl.load(primary_ranks + expert, mask=expert_valid, other=-1)
        selected += (expert_valid & (owner == destination)).to(tl.int32)
    return selected


@triton.jit
def _pack_destination(
    tokens,
    expert_ids,
    expert_weights,
    primary_ranks,
    record_tokens,
    record_metadata,
    source_rank_slots,
    overflow,
    num_tokens,
    destination,
    begin,
    end,
    WRITE_SLOTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    METADATA_WORDS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    QUANT_GROUP: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
):
    """Pack one destination's records in token order; returns their count.

    The same records as `pack_records`, with slots from a running prefix sum
    instead of an atomic: every program that packs for a destination derives
    the same slots, so the lanes of one peer each write their own row slice
    [begin, end) and only one of them (WRITE_SLOTS) writes the slot map.
    Slots follow token order, so a slice is one token range: a wide scan of
    the routes counts the records and finds that range, and only the range is
    packed, in [BLOCK_T, BLOCK_C] tiles.
    """
    SCAN: tl.constexpr = 128
    count = tl.zeros((), tl.int32)
    first = num_tokens  # first token of the scan block holding slot `begin`
    first_count = tl.zeros((), tl.int32)  # records before that block
    last = num_tokens  # end of the scan block holding slot `end - 1`
    for block in range(0, num_tokens, SCAN):
        token = block + tl.arange(0, SCAN)
        token_live = token < num_tokens
        selected = _route_block(
            expert_ids, primary_ranks, token, token_live, destination,
            TOP_K, NUM_EXPERTS,
        )
        active = token_live & (selected > 0)
        if WRITE_SLOTS:
            slot = count + tl.cumsum(active.to(tl.int32), 0) - 1
            valid = active & (slot < MAX_TOKENS)
            tl.atomic_max(overflow, tl.max((active & ~valid).to(tl.int32), 0))
            tl.store(
                source_rank_slots + token * WORLD_SIZE + destination,
                tl.where(valid, slot, -1),
                mask=token_live,
            )
        after = count + tl.sum(active.to(tl.int32), 0)
        starts = (first == num_tokens) & (after > begin)
        first_count = tl.where(starts, count, first_count)
        first = tl.where(starts, block, first)
        last = tl.where(
            (last == num_tokens) & (after >= end),
            tl.minimum(block + SCAN, num_tokens),
            last,
        )
        count = after
    rows = tl.arange(0, BLOCK_T)
    columns = tl.arange(0, BLOCK_C)
    # Quantization groups per hidden chunk (unused without quantization).
    GROUPS: tl.constexpr = BLOCK_C // (QUANT_GROUP + (QUANT_GROUP == 0))
    running = first_count
    for block in range(first, last, BLOCK_T):
        token = block + rows
        token_live = token < last
        selected = _route_block(
            expert_ids, primary_ranks, token, token_live, destination,
            TOP_K, NUM_EXPERTS,
        )
        active = token_live & (selected > 0)
        slot = running + tl.cumsum(active.to(tl.int32), 0) - 1
        running += tl.sum(active.to(tl.int32), 0)
        mine = active & (slot < MAX_TOKENS) & (slot >= begin) & (slot < end)
        if tl.max(mine.to(tl.int32), 0) > 0:
            record = destination * MAX_TOKENS + slot
            metadata = record * METADATA_WORDS
            tl.store(record_metadata + metadata, token, mask=mine)
            tl.store(record_metadata + metadata + 1, selected, mask=mine)
            packed = tl.zeros((BLOCK_T,), tl.int32)
            for choice in tl.static_range(TOP_K):
                expert = tl.load(
                    expert_ids + token * TOP_K + choice, mask=token_live, other=-1
                ).to(tl.int32)
                expert_valid = (expert >= 0) & (expert < NUM_EXPERTS)
                owner = tl.load(
                    primary_ranks + expert, mask=expert_valid, other=-1
                )
                belongs = mine & expert_valid & (owner == destination)
                weight = tl.load(
                    expert_weights + token * TOP_K + choice,
                    mask=token_live,
                    other=0.0,
                )
                tl.store(
                    record_metadata + metadata + 2 + packed, expert, mask=belongs
                )
                tl.store(
                    record_metadata + metadata + 2 + TOP_K + packed,
                    weight.to(tl.int32, bitcast=True),
                    mask=belongs,
                )
                packed += belongs.to(tl.int32)
            # Rows in hidden chunks: a [BLOCK_T, BLOCK_C] tile per step.
            source = tokens + tl.cast(token, tl.int64)[:, None] * HIDDEN
            target = record_tokens + tl.cast(record, tl.int64)[:, None] * HIDDEN
            for chunk in tl.static_range(0, HIDDEN, BLOCK_C):
                column = chunk + columns
                values = tl.load(
                    source + column[None, :],
                    mask=mine[:, None] & (column[None, :] < HIDDEN),
                    other=0.0,
                )
                if QUANT_GROUP > 0:
                    grouped = tl.reshape(
                        values.to(tl.float32), (BLOCK_T, GROUPS, QUANT_GROUP)
                    )
                    amax = tl.max(tl.maximum(tl.abs(grouped), AMAX_FLOOR), axis=2)
                    scale = tl.div_rn(amax, 448.0)
                    if ROUND_SCALE:
                        scale = tl.maximum(scale, 1e-10)
                        bits = scale.to(tl.int32, bitcast=True)
                        exponent = (bits >> 23) & 0xFF
                        mantissa = bits & 0x7FFFFF
                        exponent = tl.where(mantissa != 0, exponent + 1, exponent)
                        scale = (exponent << 23).to(tl.float32, bitcast=True)
                    quantized = tl.clamp(
                        tl.div_rn(grouped, scale[:, :, None]), -448.0, 448.0
                    )
                    values = tl.reshape(quantized, (BLOCK_T, BLOCK_C))
                    group = chunk // QUANT_GROUP + tl.arange(0, GROUPS)
                    tl.store(
                        record_metadata
                        + metadata[:, None]
                        + 2
                        + 2 * TOP_K
                        + group[None, :],
                        scale.to(tl.int32, bitcast=True),
                        mask=mine[:, None]
                        & (group[None, :] < HIDDEN // QUANT_GROUP),
                    )
                tl.store(
                    target + column[None, :],
                    values.to(record_tokens.dtype.element_ty),
                    mask=mine[:, None] & (column[None, :] < HIDDEN),
                )
    return count


@triton.jit(do_not_specialize=["num_tokens", "active"])
def pack_send_records(
    endpoints,
    tokens,
    expert_ids,
    expert_weights,
    primary_ranks,
    payload,
    payload_key,
    metadata,
    metadata_key,
    source_rank_slots,
    overflow,
    destination_data,
    destination_meta,
    destination_counts,
    destination_flags,
    epoch_pointer,
    num_tokens,
    active,
    RANK: tl.constexpr,
    PEERS: tl.constexpr,
    CAPACITY: tl.constexpr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    METADATA_WORDS: tl.constexpr,
    LANES: tl.constexpr,
    ROW_QUOTIENT: tl.constexpr,
    ROW_REMAINDER: tl.constexpr,
    REUSE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    READY_STRIDE: tl.constexpr = 0,
    QUANT_GROUP: tl.constexpr = 0,
    ROUND_SCALE: tl.constexpr = False,
    AMAX_FLOOR: tl.constexpr = 1e-4,
):
    """Dispatch with packing fused before the puts (paper Listing 2's shape).

    One program per (peer, lane) of a transport partition: it packs (and
    quantizes) the peer's records of its row slice from the route, then
    sends them with the chain of `records.send_records`.
    """
    if active != 0:
        STRIDE: tl.constexpr = LANES if READY_STRIDE == 0 else READY_STRIDE
        peer_index, lane = tl.program_id(0), tl.program_id(1)
        peer = tl.full((), PEERS[0], tl.int64)
        for index in tl.static_range(1, len(PEERS)):
            peer = tl.where(peer_index == index, PEERS[index], peer)
        begin = ROW_QUOTIENT * lane + ROW_REMAINDER * lane // LANES
        end = ROW_QUOTIENT * (lane + 1) + ROW_REMAINDER * (lane + 1) // LANES
        count = _pack_destination(
            tokens,
            expert_ids,
            expert_weights,
            primary_ranks,
            payload,
            metadata,
            source_rank_slots,
            overflow,
            num_tokens,
            peer,
            begin,
            end,
            LANES == 1 or lane == 0,
            HIDDEN,
            TOP_K,
            NUM_EXPERTS,
            WORLD_SIZE,
            CAPACITY,
            METADATA_WORDS,
            BLOCK_T,
            BLOCK_C,
            QUANT_GROUP,
            ROUND_SCALE,
            AMAX_FLOOR,
        )
        epoch = tl.load(epoch_pointer)
        # The pack's records are in this program's own payload rows, so the
        # chain reads storage only this program wrote.
        if LANES == 1 or lane == 0:
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
                HIDDEN,
                METADATA_WORDS,
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
                HIDDEN,
                METADATA_WORDS,
                LANES,
                ROW_QUOTIENT,
                ROW_REMAINDER,
                REUSE,
                False,
                STRIDE,
            )
