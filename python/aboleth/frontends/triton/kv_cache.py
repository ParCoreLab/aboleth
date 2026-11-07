from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import triton
import triton.language as tl

from ... import language as aboleth
from ...patterns.kv_cache import (
    KVCacheMigrationSpec,
)
from ...patterns.kv_cache_transport import (
    CompiledKVCacheMigrationPlan,
    KVCacheLayoutContract,
    compile_kv_cache_migration,
)
from ...runtime import Runtime


@triton.jit
def migrate_pages_kernel(
    endpoint_handles,
    peer_indices,
    remote_page_ptrs,
    remote_ready_ptrs,
    source_pages,
    batch_source_rows,
    batch_destination_peers,
    batch_destination_rows,
    batch_consumers,
    batch_lengths,
    batch_count,
    ready,
    signal_values,
    rank,
    epoch,
    slot,
    world_size: tl.constexpr,
    slot_count: tl.constexpr,
    max_consumers: tl.constexpr,
    max_batches: tl.constexpr,
    max_pages: tl.constexpr,
    page_bytes: tl.constexpr,
    pages_per_transfer: tl.constexpr,
    transport_manifest: tl.constexpr,
):
    peer_slot = tl.program_id(0)
    consumer = tl.program_id(1)
    peer = tl.load(peer_indices + peer_slot)
    endpoint = tl.load(
        endpoint_handles + peer
    ).to(tl.pointer_type(tl.int64))
    remote_pages = tl.load(
        remote_page_ptrs + peer
    ).to(tl.pointer_type(tl.uint8))
    remote_ready = tl.load(
        remote_ready_ptrs + peer
    ).to(tl.pointer_type(tl.int64))
    item_count = tl.load(batch_count)
    transfers = ()
    for item in tl.static_range(0, max_batches):
        active = item < item_count
        source_row = tl.load(
            batch_source_rows + item,
            mask=active,
            other=0,
        )
        destination_peer = tl.load(
            batch_destination_peers + item,
            mask=active,
            other=0,
        )
        destination_row = tl.load(
            batch_destination_rows + item,
            mask=active,
            other=0,
        )
        destination_consumer = tl.load(
            batch_consumers + item,
            mask=active,
            other=0,
        )
        page_count = tl.load(
            batch_lengths + item,
            mask=active,
            other=0,
        )
        destination = (
            remote_pages
            + (
                slot * max_pages
                + destination_row
            )
            * page_bytes
        )
        source = source_pages + source_row * page_bytes
        predicate = (
            active
            & (destination_peer == peer)
            & (destination_consumer == consumer)
            & (page_count > 0)
        )
        transfers += ((
            destination,
            source,
            page_count * page_bytes,
            0,
            0,
            predicate,
        ),)
    ready_base = (
        slot * max_consumers * world_size
        + consumer * world_size
    )
    tl.store(signal_values + ready_base + peer, epoch)
    aboleth.invoke_transport_program(
        transport_manifest,
        endpoint,
        transfers,
        (
            remote_ready + ready_base + rank,
            signal_values + ready_base + peer,
            0,
            0,
            1,
            1,
        ),
    )


@triton.jit
def kv_cache_page_batch_copy_kernel(
    endpoint_handles,
    remote_page_ptrs,
    remote_ready_ptrs,
    source_pages,
    batch_source_rows,
    batch_destination_peers,
    batch_destination_rows,
    batch_consumers,
    batch_lengths,
    batch_count,
    batch_worker_done,
    consumer_batch_done,
    consumer_batch_counts,
    signal_values,
    rank,
    epoch,
    slot,
    world_size: tl.constexpr,
    max_consumers: tl.constexpr,
    max_pages: tl.constexpr,
    page_bytes: tl.constexpr,
    pages_per_transfer: tl.constexpr,
    max_batches: tl.constexpr,
    copy_workers: tl.constexpr,
    block_elements: tl.constexpr,
    transport_manifest: tl.constexpr,
):
    batch = tl.program_id(0)
    worker = tl.program_id(1)
    active = batch < tl.load(batch_count)
    peer = tl.load(
        batch_destination_peers + batch,
        mask=active,
        other=0,
    )
    source_row = tl.load(
        batch_source_rows + batch,
        mask=active,
        other=0,
    )
    destination_row = tl.load(
        batch_destination_rows + batch,
        mask=active,
        other=0,
    )
    consumer = tl.load(
        batch_consumers + batch,
        mask=active,
        other=0,
    )
    page_count = tl.load(
        batch_lengths + batch,
        mask=active,
        other=0,
    )
    remote_pages = tl.load(
        remote_page_ptrs + peer,
        mask=active,
        other=0,
    ).to(tl.pointer_type(tl.uint64))
    source_words = source_pages.to(
        tl.pointer_type(tl.uint64)
    )
    page_words = page_bytes // 8
    word_count = page_count * page_words
    offsets = tl.arange(0, block_elements)
    for block_start in tl.range(
        worker * block_elements,
        pages_per_transfer * page_words,
        copy_workers * block_elements,
    ):
        indices = block_start + offsets
        mask = active & (indices < word_count)
        source = (
            source_words
            + source_row * page_words
            + indices
        )
        destination = (
            remote_pages
            + (
                slot * max_pages
                + destination_row
            )
            * page_words
            + indices
        )
        values = tl.load(source, mask=mask, other=0)
        tl.store(destination, values, mask=mask)
    completed = tl.atomic_add(
        batch_worker_done
        + slot * max_batches
        + batch,
        active.to(tl.int32),
        sem="acq_rel",
        scope="gpu",
    )
    if active & (completed + 1 == copy_workers):
        aboleth.system_fence()
        tl.atomic_xchg(
            batch_worker_done
            + slot * max_batches
            + batch,
            0,
            sem="relaxed",
            scope="gpu",
        )
        consumer_index = (
            consumer * world_size + peer
        )
        batches_complete = tl.atomic_add(
            consumer_batch_done
            + slot * max_consumers * world_size
            + consumer_index,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        expected_batches = tl.load(
            consumer_batch_counts + consumer_index
        )
        if batches_complete + 1 == expected_batches:
            endpoint = tl.load(
                endpoint_handles + peer
            ).to(tl.pointer_type(tl.int64))
            remote_ready = tl.load(
                remote_ready_ptrs + peer
            ).to(tl.pointer_type(tl.int64))
            ready_base = (
                slot * max_consumers * world_size
                + consumer * world_size
            )
            tl.store(
                signal_values + ready_base + peer,
                epoch,
            )
            aboleth.invoke_transport_program(
                transport_manifest,
                endpoint,
                (),
                (
                    remote_ready + ready_base + rank,
                    signal_values + ready_base + peer,
                    0,
                    0,
                    1,
                    1,
                ),
            )
            tl.atomic_xchg(
                consumer_batch_done
                + slot * max_consumers * world_size
                + consumer_index,
                0,
                sem="relaxed",
                scope="gpu",
            )


@triton.jit
def kv_cache_consumer_publish_kernel(
    endpoint_handles,
    peer_indices,
    remote_ready_ptrs,
    ready,
    signal_values,
    rank,
    epoch,
    slot,
    world_size: tl.constexpr,
    max_consumers: tl.constexpr,
    transport_manifest: tl.constexpr,
):
    peer_slot = tl.program_id(0)
    consumer = tl.program_id(1)
    peer = tl.load(peer_indices + peer_slot)
    endpoint = tl.load(
        endpoint_handles + peer
    ).to(tl.pointer_type(tl.int64))
    remote_ready = tl.load(
        remote_ready_ptrs + peer
    ).to(tl.pointer_type(tl.int64))
    ready_base = (
        slot * max_consumers * world_size
        + consumer * world_size
    )
    tl.store(signal_values + ready_base + peer, epoch)
    aboleth.invoke_transport_program(
        transport_manifest,
        endpoint,
        (),
        (
            remote_ready + ready_base + rank,
            signal_values + ready_base + peer,
            0,
            0,
            1,
            1,
        ),
    )


@dataclass(frozen=True)
class KVCacheMigrationResult:
    pages: tuple[torch.Tensor, ...]
    storage: tuple[torch.Tensor, ...]
    ready: tuple[torch.Tensor, ...]
    compiled_plan: CompiledKVCacheMigrationPlan
    active_slot: int
    slot_history: tuple[int, ...]
    pages_per_transfer: int
    samples_us: tuple[float, ...]
    wall_samples_us: tuple[float, ...]


def _page_runs(
    entries: list[tuple[int, int, int, int]],
    pages_per_transfer: int,
) -> list[tuple[int, int, int, int, int]]:
    ordered = sorted(
        entries,
        key=lambda item: (
            item[3],
            item[1],
            item[0],
            item[2],
        ),
    )
    runs: list[list[int]] = []
    for source, peer, destination, consumer in ordered:
        if (
            runs
            and runs[-1][1] == peer
            and runs[-1][3] == consumer
            and runs[-1][0] + runs[-1][4] == source
            and runs[-1][2] + runs[-1][4] == destination
            and runs[-1][4] < pages_per_transfer
        ):
            runs[-1][4] += 1
        else:
            runs.append([
                source,
                peer,
                destination,
                consumer,
                1,
            ])
    return [
        (source, peer, destination, consumer, length)
        for source, peer, destination, consumer, length in runs
    ]


def _packed_tensor(
    values: list[int],
    *,
    size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    result = torch.zeros(
        size,
        dtype=dtype,
        device=device,
    )
    if values:
        result[: len(values)] = torch.tensor(
            values,
            dtype=dtype,
            device=device,
        )
    return result


def migrate_kv_cache_pages(
    source_pages: tuple[torch.Tensor, ...],
    source_rows: tuple[torch.Tensor, ...],
    destination_peers: tuple[torch.Tensor, ...],
    destination_rows: tuple[torch.Tensor, ...],
    counts: tuple[torch.Tensor, ...],
    *,
    consumer_ids: tuple[torch.Tensor, ...] | None = None,
    max_consumers: int = 1,
    slots: int | None = None,
    consumer_max_inflight: int = 1,
    replays: int = 1,
    transport: str = "p2p",
) -> KVCacheMigrationResult:
    source_pages = tuple(source_pages)
    source_rows = tuple(source_rows)
    destination_peers = tuple(destination_peers)
    destination_rows = tuple(destination_rows)
    counts = tuple(counts)
    consumer_ids = (
        tuple(consumer_ids)
        if consumer_ids is not None
        else None
    )
    world_size = len(source_pages)
    if world_size < 2:
        raise ValueError("KV migration requires at least two ranks")
    if max_consumers <= 0:
        raise ValueError("max_consumers must be positive")
    if consumer_max_inflight <= 0:
        raise ValueError(
            "consumer_max_inflight must be positive"
        )
    if replays <= 0:
        raise ValueError("replays must be positive")
    if transport != "p2p":
        raise NotImplementedError(
            "native KV migration currently supports P2P"
        )
    if not all(
        len(values) == world_size
        for values in (
            source_rows,
            destination_peers,
            destination_rows,
            counts,
        )
    ):
        raise ValueError(
            "KV migration inputs require one tensor per rank"
        )
    if (
        consumer_ids is not None
        and len(consumer_ids) != world_size
    ):
        raise ValueError(
            "consumer_ids require one tensor per rank"
        )

    max_pages, page_bytes = source_pages[0].shape
    if consumer_ids is None:
        consumer_ids = tuple(
            torch.zeros(
                max_pages,
                dtype=torch.int32,
                device=pages.device,
            )
            for pages in source_pages
        )
    for rank in range(world_size):
        pages = source_pages[rank]
        if (
            pages.shape != (max_pages, page_bytes) or
            pages.dtype != torch.uint8 or
            pages.device != torch.device(f"cuda:{rank}") or
            not pages.is_contiguous()
        ):
            raise ValueError(
                f"source_pages[{rank}] has an invalid contract"
            )
        for name, values, dtype in (
            ("source_rows", source_rows[rank], torch.int64),
            (
                "destination_peers",
                destination_peers[rank],
                torch.int32,
            ),
            (
                "destination_rows",
                destination_rows[rank],
                torch.int64,
            ),
            (
                "consumer_ids",
                consumer_ids[rank],
                torch.int32,
            ),
        ):
            if (
                values.shape != (max_pages,) or
                values.dtype != dtype or
                values.device != pages.device or
                not values.is_contiguous()
            ):
                raise ValueError(
                    f"{name}[{rank}] has an invalid contract"
                )
        if (
            counts[rank].shape != (1,) or
            counts[rank].dtype != torch.int32 or
            counts[rank].device != pages.device or
            not counts[rank].is_contiguous()
        ):
            raise ValueError(
                f"counts[{rank}] has an invalid contract"
            )

    destinations_seen = set()
    entries_by_rank = []
    for rank in range(world_size):
        item_count = int(counts[rank].item())
        if item_count < 0 or item_count > max_pages:
            raise ValueError(
                f"counts[{rank}] exceeds max_pages"
            )
        active_sources = source_rows[rank][
            :item_count
        ].cpu().tolist()
        active_peers = destination_peers[rank][
            :item_count
        ].cpu().tolist()
        active_rows = destination_rows[rank][
            :item_count
        ].cpu().tolist()
        active_consumers = consumer_ids[rank][
            :item_count
        ].cpu().tolist()
        entries = []
        for source, peer, destination, consumer in zip(
            active_sources,
            active_peers,
            active_rows,
            active_consumers,
        ):
            if source < 0 or source >= max_pages:
                raise ValueError(
                    f"source_rows[{rank}] contains an invalid page"
                )
            if peer < 0 or peer >= world_size:
                raise ValueError(
                    f"destination_peers[{rank}] contains an invalid rank"
                )
            if destination < 0 or destination >= max_pages:
                raise ValueError(
                    f"destination_rows[{rank}] contains an invalid page"
                )
            if consumer < 0 or consumer >= max_consumers:
                raise ValueError(
                    f"consumer_ids[{rank}] contains an invalid consumer"
                )
            key = (peer, destination)
            if key in destinations_seen:
                raise ValueError(
                    "KV migration destinations must be unique"
                )
            destinations_seen.add(key)
            entries.append((
                source,
                peer,
                destination,
                consumer,
            ))
        entries_by_rank.append(entries)

    spec = KVCacheMigrationSpec(
        max_pages=max_pages,
        page_bytes=page_bytes,
        num_ranks=world_size,
        slots=slots,
        max_consumers=max_consumers,
        unique_destinations=True,
        consumer_max_inflight=consumer_max_inflight,
    )
    compiled_plan = compile_kv_cache_migration(
        spec,
        transports=(transport,) * world_size,
        layout_contract=KVCacheLayoutContract(
            contiguous_page_runs=True,
            consumer_partitioned=True,
            graph_stable_addresses=True,
        ),
    )
    pages_per_transfer = (
        compiled_plan.physicalization.pages_per_transfer
    )
    max_batches = triton.cdiv(
        max_pages,
        pages_per_transfer,
    )
    runs_by_rank = [
        _page_runs(entries, pages_per_transfer)
        for entries in entries_by_rank
    ]
    contiguous_layout = all(
        len(runs) <= max_batches
        for runs in runs_by_rank
    )
    phase, selected_pages = compiled_plan.select_phase(
        contiguous_page_runs=contiguous_layout,
        consumer_partitioned=True,
    )
    if selected_pages != pages_per_transfer:
        pages_per_transfer = selected_pages
        max_batches = max_pages
        runs_by_rank = [
            _page_runs(entries, 1)
            for entries in entries_by_rank
        ]
    packed = []
    for rank, runs in enumerate(runs_by_rank):
        device = source_pages[rank].device
        consumer_batch_counts = [
            0
        ] * (max_consumers * world_size)
        for _source, peer, _destination, consumer, _length in runs:
            consumer_batch_counts[
                consumer * world_size + peer
            ] += 1
        packed.append({
            "source_rows": _packed_tensor(
                [run[0] for run in runs],
                size=max_batches,
                dtype=torch.int64,
                device=device,
            ),
            "destination_peers": _packed_tensor(
                [run[1] for run in runs],
                size=max_batches,
                dtype=torch.int32,
                device=device,
            ),
            "destination_rows": _packed_tensor(
                [run[2] for run in runs],
                size=max_batches,
                dtype=torch.int64,
                device=device,
            ),
            "consumers": _packed_tensor(
                [run[3] for run in runs],
                size=max_batches,
                dtype=torch.int32,
                device=device,
            ),
            "lengths": _packed_tensor(
                [run[4] for run in runs],
                size=max_batches,
                dtype=torch.int32,
                device=device,
            ),
            "count": torch.tensor(
                [len(runs)],
                dtype=torch.int32,
                device=device,
            ),
            "consumer_batch_counts": _packed_tensor(
                consumer_batch_counts,
                size=max_consumers * world_size,
                dtype=torch.int32,
                device=device,
            ),
        })
    destinations = tuple(
        torch.zeros(
            spec.slot_count,
            max_pages,
            page_bytes,
            dtype=pages.dtype,
            device=pages.device,
        )
        for pages in source_pages
    )
    ready = tuple(
        torch.zeros(
            spec.slot_count * max_consumers * world_size,
            dtype=torch.int64,
            device=f"cuda:{rank}",
        )
        for rank in range(world_size)
    )
    signal_values = tuple(
        torch.zeros_like(values)
        for values in ready
    )
    batch_worker_done = tuple(
        torch.zeros(
            spec.slot_count * max_batches,
            dtype=torch.int32,
            device=f"cuda:{rank}",
        )
        for rank in range(world_size)
    )
    consumer_batch_done = tuple(
        torch.zeros(
            spec.slot_count * max_consumers * world_size,
            dtype=torch.int32,
            device=f"cuda:{rank}",
        )
        for rank in range(world_size)
    )

    with Runtime() as runtime:
        with runtime.create_endpoint_mesh(
            tuple(range(world_size)),
            transport=transport,
        ) as mesh:
            peer_indices = tuple(
                torch.tensor(
                    [
                        peer
                        for peer in range(world_size)
                        if peer != rank
                    ],
                    dtype=torch.int32,
                    device=f"cuda:{rank}",
                )
                for rank in range(world_size)
            )
            remote_page_ptrs = tuple(
                torch.tensor(
                    [
                        destinations[peer].data_ptr()
                        for peer in range(world_size)
                    ],
                    dtype=torch.int64,
                    device=f"cuda:{rank}",
                )
                for rank in range(world_size)
            )
            remote_ready_ptrs = tuple(
                torch.tensor(
                    [
                        ready[peer].data_ptr()
                        for peer in range(world_size)
                    ],
                    dtype=torch.int64,
                    device=f"cuda:{rank}",
                )
                for rank in range(world_size)
            )
            parallel_copy = (
                phase.plan.phase_schedule(
                    phase.name
                ).strategy
                == "parallel_copy"
            )
            with mesh.endpoint(0, 1).compilation(
                pattern_plan=phase.plan,
                kernel=phase.binding.kernel,
            ):
                slot_history = []
                samples_us = []
                wall_samples_us = []
                for replay in range(replays):
                    wall_start = time.perf_counter()
                    epoch = replay + 1
                    slot = replay % spec.slot_count
                    slot_history.append(slot)
                    starts = [
                        torch.cuda.Event(enable_timing=True)
                        for _ in range(world_size)
                    ]
                    stops = [
                        torch.cuda.Event(enable_timing=True)
                        for _ in range(world_size)
                    ]
                    for rank in range(world_size):
                        with torch.cuda.device(rank):
                            starts[rank].record()
                    if parallel_copy:
                        copy_workers = 16
                        block_elements = 256
                        for rank in range(world_size):
                            with torch.cuda.device(rank):
                                if runs_by_rank[rank]:
                                    kv_cache_page_batch_copy_kernel[
                                        (
                                            max_batches,
                                            copy_workers,
                                        )
                                    ](
                                        mesh.tables[
                                            rank
                                        ].device_handles,
                                        remote_page_ptrs[rank],
                                        remote_ready_ptrs[rank],
                                        source_pages[rank],
                                        packed[rank][
                                            "source_rows"
                                        ],
                                        packed[rank][
                                            "destination_peers"
                                        ],
                                        packed[rank][
                                            "destination_rows"
                                        ],
                                        packed[rank]["consumers"],
                                        packed[rank]["lengths"],
                                        packed[rank]["count"],
                                        batch_worker_done[rank],
                                        consumer_batch_done[rank],
                                        packed[rank][
                                            "consumer_batch_counts"
                                        ],
                                        signal_values[rank],
                                        rank,
                                        epoch,
                                        slot,
                                        world_size=world_size,
                                        max_consumers=max_consumers,
                                        max_pages=max_pages,
                                        page_bytes=page_bytes,
                                        pages_per_transfer=(
                                            pages_per_transfer
                                        ),
                                        max_batches=max_batches,
                                        copy_workers=copy_workers,
                                        block_elements=block_elements,
                                        transport_manifest=(
                                            phase.manifest
                                        ),
                                        num_warps=8,
                                    )
                    else:
                        for rank in range(world_size):
                            with torch.cuda.device(rank):
                                migrate_pages_kernel[
                                    (
                                        world_size - 1,
                                        max_consumers,
                                    )
                                ](
                                    mesh.tables[
                                        rank
                                    ].device_handles,
                                    peer_indices[rank],
                                    remote_page_ptrs[rank],
                                    remote_ready_ptrs[rank],
                                    source_pages[rank],
                                    packed[rank]["source_rows"],
                                    packed[rank][
                                        "destination_peers"
                                    ],
                                    packed[rank][
                                        "destination_rows"
                                    ],
                                    packed[rank]["consumers"],
                                    packed[rank]["lengths"],
                                    packed[rank]["count"],
                                    ready[rank],
                                    signal_values[rank],
                                    rank,
                                    epoch,
                                    slot,
                                    world_size=world_size,
                                    slot_count=spec.slot_count,
                                    max_consumers=max_consumers,
                                    max_batches=max_batches,
                                    max_pages=max_pages,
                                    page_bytes=page_bytes,
                                    pages_per_transfer=(
                                        pages_per_transfer
                                    ),
                                    transport_manifest=(
                                        phase.manifest
                                    ),
                                    num_warps=1,
                                )
                    for rank in range(world_size):
                        with torch.cuda.device(rank):
                            stops[rank].record()
                    for rank in range(world_size):
                        torch.cuda.synchronize(rank)
                    wall_samples_us.append(
                        (time.perf_counter() - wall_start)
                        * 1e6
                    )
                    samples_us.append(max(
                        start.elapsed_time(stop) * 1000.0
                        for start, stop in zip(
                            starts,
                            stops,
                            strict=True,
                        )
                    ))

    active_slot = slot_history[-1]
    return KVCacheMigrationResult(
        pages=tuple(
            destination[active_slot]
            for destination in destinations
        ),
        storage=destinations,
        ready=ready,
        compiled_plan=compiled_plan,
        active_slot=active_slot,
        slot_history=tuple(slot_history),
        pages_per_transfer=pages_per_transfer,
        samples_us=tuple(samples_us),
        wall_samples_us=tuple(wall_samples_us),
    )
