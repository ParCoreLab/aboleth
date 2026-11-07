"""Persistent same-node KV-cache migration runtime for vLLM."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from ..bootstrap import TcpStoreBootstrap
from ..kv.copy import (
    canonicalize_block_runs,
    cuda_memcpy_async,
    pull_contiguous_kv_kernel,
    pull_kv_runs_kernel,
)
from ..kv.profile import (
    KVRuntimeOverrides,
    KVTargetProfile,
    select_kv_target_profile,
)
from ..patterns.kv_cache import KVCacheMigrationSpec
from ..patterns.kv_cache_transport import (
    CompiledKVCacheMigrationPlan,
    KVCacheLayoutContract,
    compile_kv_cache_migration,
)
from ..world import World


@dataclass(frozen=True)
class KVCachePullTicket:
    completion: torch.cuda.Event
    layer_events: dict[str, torch.cuda.Event]


class VLLMKVPeerRuntime:
    """Own one persistent P2P world and registered KV storage."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: int,
        host: str,
        port: int,
        timeout: float = 300.0,
        target_profile: KVTargetProfile | None = None,
        overrides: KVRuntimeOverrides | None = None,
    ):
        if world_size != 2:
            raise NotImplementedError(
                "the first vLLM KV gate supports 1P1D"
            )
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.target_profile = (
            target_profile
            or select_kv_target_profile(
                torch.cuda.get_device_name(device)
            )
        )
        self.overrides = overrides or KVRuntimeOverrides()
        self._copy_streams = ()
        self._completion_stream = None
        self.bootstrap = TcpStoreBootstrap(
            rank,
            world_size,
            host=host,
            port=port,
            timeout=timeout,
        )
        self.world = World(
            rank,
            world_size,
            device=device,
            bootstrap=self.bootstrap,
        )
        self.world.connect_all("p2p")
        self._registrations = []
        self._region_table = None
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._plan: CompiledKVCacheMigrationPlan | None = None
        self._local_layer_ptrs = None
        self._remote_layer_ptrs = None
        self._remote_layer_addresses = None
        self._page_bytes = 0
        self._max_blocks = 0
        self._last_transfer_us = 0.0
        self._last_physicalization = "none"
        self._last_transfer_bytes = 0
        self._last_run_count = 0

    @property
    def plan(self) -> CompiledKVCacheMigrationPlan:
        if self._plan is None:
            raise RuntimeError("KV caches are not registered")
        return self._plan

    @property
    def last_transfer_us(self) -> float:
        return self._last_transfer_us

    @property
    def last_physicalization(self) -> str:
        return self._last_physicalization

    @property
    def last_transfer_bytes(self) -> int:
        return self._last_transfer_bytes

    @property
    def last_run_count(self) -> int:
        return self._last_run_count

    @property
    def profile_transfers(self) -> bool:
        return self.overrides.profile_transfers

    @property
    def asynchronous_completion(self) -> bool:
        return self.plan.physicalization.asynchronous_completion

    @property
    def consumer_layer_pipeline(self) -> bool:
        return self.plan.physicalization.consumer_layer_pipeline

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        *,
        consumer_order: tuple[str, ...] | None = None,
    ) -> None:
        if self._region_table is not None:
            raise RuntimeError("KV caches were already registered")
        if not kv_caches:
            raise ValueError("KV cache registration cannot be empty")
        if consumer_order is None:
            ordered = dict(kv_caches)
            consumer_ordered = False
        else:
            if (
                len(set(consumer_order)) != len(consumer_order)
                or set(consumer_order) != set(kv_caches)
            ):
                raise ValueError(
                    "consumer order must name every KV layer exactly once"
                )
            ordered = {
                layer_name: kv_caches[layer_name]
                for layer_name in consumer_order
            }
            consumer_ordered = True
        tensors = tuple(ordered.values())
        if any(
            tensor.device != torch.device(
                f"cuda:{self.device}"
            )
            or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError(
                "KV cache tensors must be contiguous on the worker GPU"
            )
        max_blocks = tensors[0].shape[0]
        page_bytes = (
            tensors[0].stride(0)
            * tensors[0].element_size()
        )
        if any(
            tensor.shape[0] != max_blocks
            or tensor.stride(0) * tensor.element_size()
            != page_bytes
            for tensor in tensors
        ):
            raise NotImplementedError(
                "the first KV gate requires uniform layer pages"
            )
        registrations = {}
        region_names = tuple(
            f"kv_layer/{layer_name}"
            for layer_name in ordered
        )
        for region_name, tensor in zip(
            region_names,
            tensors,
            strict=True,
        ):
            registration = self.world.runtime.register(
                tensor,
                network=False,
            )
            self._registrations.append(registration)
            registrations[region_name] = self.world.region(
                region_name,
                registration,
                tensor,
            )
        self._region_table = self.world.exchange_regions(
            registrations,
            tag="vllm-kv-cache",
        )
        num_layers = len(tensors)
        self._remote_layer_addresses = tuple(
            tuple(
                self._region_table.local(
                    region_name
                ).address
                if peer == self.rank
                else self._region_table.remote(
                    peer,
                    region_name,
                ).address
                for region_name in region_names
            )
            for peer in range(self.world_size)
        )
        remote_by_layer = torch.stack([
            self._region_table.addresses(
                region_name
            )
            for region_name in region_names
        ])
        self._remote_layer_ptrs = (
            remote_by_layer.transpose(0, 1)
            .contiguous()
            .reshape(-1)
        )
        self._local_layer_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in tensors],
            dtype=torch.int64,
            device=f"cuda:{self.device}",
        )
        self._kv_caches = ordered
        self._page_bytes = page_bytes
        self._max_blocks = max_blocks
        self._plan = compile_kv_cache_migration(
            KVCacheMigrationSpec(
                max_pages=max_blocks,
                page_bytes=page_bytes,
                num_ranks=self.world_size,
                max_consumers=num_layers,
                consumer_asynchronous=True,
                consumer_max_inflight=1,
            ),
            transports=("p2p",) * self.world_size,
            layout_contract=KVCacheLayoutContract(
                consumer_ordered=consumer_ordered,
            ),
            target_profile=self.target_profile,
            generated_copy_workers=(
                self.overrides.generated_copy_workers
            ),
            generated_block_elements=(
                self.overrides.generated_block_elements
            ),
            copy_engine_threshold_bytes=(
                self.overrides.copy_engine_threshold_bytes
            ),
            copy_engine_streams=(
                self.overrides.copy_engine_streams
            ),
            asynchronous_completion=(
                self.overrides.asynchronous_completion
                if self.overrides.asynchronous_completion
                is not None
                else True
            ),
            consumer_layer_pipeline=(
                self.overrides.consumer_layer_pipeline
            ),
            consumers_per_readiness=(
                self.overrides.consumers_per_readiness
            ),
        )
        physicalization = self._plan.physicalization
        with torch.cuda.device(self.device):
            self._copy_streams = tuple(
                torch.cuda.Stream()
                for _ in range(
                    physicalization.copy_engine_streams
                )
            )
            self._completion_stream = torch.cuda.Stream()

    def pull_blocks(
        self,
        *,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_rank: int,
    ) -> float:
        started = time.perf_counter()
        completion = self.start_pull_blocks(
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
            remote_rank=remote_rank,
        )
        completion.synchronize()
        self._last_transfer_us = (
            time.perf_counter() - started
        ) * 1e6
        if self.profile_transfers:
            print(
                "aboleth_kv_transfer="
                f"physicalization={self.last_physicalization} "
                f"pages={len(local_block_ids)} "
                f"bytes={len(local_block_ids) * self._page_bytes * len(self._kv_caches)} "
                f"us={self._last_transfer_us:.1f}",
                flush=True,
            )
        return self._last_transfer_us

    def start_pull_blocks(
        self,
        *,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_rank: int,
    ) -> torch.cuda.Event:
        return self._start_pull_blocks(
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
            remote_rank=remote_rank,
            capture_layer_events=False,
        ).completion

    def start_layered_pull_blocks(
        self,
        *,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_rank: int,
    ) -> KVCachePullTicket:
        return self._start_pull_blocks(
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
            remote_rank=remote_rank,
            capture_layer_events=True,
        )

    def _start_pull_blocks(
        self,
        *,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_rank: int,
        capture_layer_events: bool,
    ) -> KVCachePullTicket:
        if self._plan is None:
            raise RuntimeError("KV caches are not registered")
        physicalization = self._plan.physicalization
        if len(local_block_ids) != len(remote_block_ids):
            raise ValueError(
                "local and remote KV block counts must match"
            )
        if not local_block_ids:
            self._last_transfer_us = 0.0
            completion = torch.cuda.Event()
            completion.record()
            return KVCachePullTicket(completion, {})
        if (
            min(local_block_ids) < 0
            or max(local_block_ids) >= self._max_blocks
            or min(remote_block_ids) < 0
            or max(remote_block_ids) >= self._max_blocks
        ):
            raise ValueError("KV block ID exceeds registered storage")
        if len(set(local_block_ids)) != len(local_block_ids):
            raise ValueError(
                "compiled KV pulls require unique destination blocks"
            )
        device = f"cuda:{self.device}"
        runs = canonicalize_block_runs(
            local_block_ids,
            remote_block_ids,
        )
        transfer_bytes = (
            runs.page_count
            * self._page_bytes
            * len(self._kv_caches)
        )
        self._last_transfer_bytes = transfer_bytes
        self._last_run_count = runs.count
        if self._plan.select_transfer_engine(
            transfer_bytes=transfer_bytes,
            run_count=runs.count,
        ) == "copy_engine":
            assert self._remote_layer_addresses is not None
            current_stream = torch.cuda.current_stream(
                self.device
            )
            remote_addresses = (
                self._remote_layer_addresses[remote_rank]
            )
            for copy_stream in self._copy_streams:
                copy_stream.wait_stream(current_stream)
            layer_events = {}
            layer_names = tuple(self._kv_caches)
            for layer, (
                layer_name,
                local_tensor,
                remote_address,
            ) in enumerate(zip(
                self._kv_caches,
                self._kv_caches.values(),
                remote_addresses,
                strict=True,
            )):
                copy_stream = self._copy_streams[
                    layer % len(self._copy_streams)
                ]
                for (
                    local_start,
                    remote_start,
                    run_length,
                ) in zip(
                    runs.local_starts,
                    runs.remote_starts,
                    runs.lengths,
                    strict=True,
                ):
                    run_bytes = (
                        run_length * self._page_bytes
                    )
                    local_offset = (
                        local_start * self._page_bytes
                    )
                    remote_offset = (
                        remote_start * self._page_bytes
                    )
                    cuda_memcpy_async(
                        local_tensor.data_ptr() + local_offset,
                        remote_address + remote_offset,
                        run_bytes,
                        copy_stream.cuda_stream,
                    )
                if (
                    capture_layer_events
                    and (
                        (layer + 1)
                        % physicalization.consumers_per_readiness
                        == 0
                        or layer + 1 == len(layer_names)
                    )
                ):
                    layer_event = torch.cuda.Event()
                    layer_event.record(copy_stream)
                    readiness_layer = layer_names[
                        (
                            layer
                            // physicalization.consumers_per_readiness
                        )
                        * physicalization.consumers_per_readiness
                    ]
                    layer_events[
                        readiness_layer
                    ] = layer_event
            self._last_physicalization = "copy_engine"
            completion = torch.cuda.Event()
            if len(self._copy_streams) == 1:
                completion.record(self._copy_streams[0])
            else:
                assert self._completion_stream is not None
                for copy_stream in self._copy_streams:
                    self._completion_stream.wait_stream(
                        copy_stream
                    )
                completion.record(self._completion_stream)
            return KVCachePullTicket(
                completion,
                layer_events,
            )
        if runs.count == 1:
            pull_contiguous_kv_kernel[
                (
                    len(self._kv_caches),
                    physicalization.generated_copy_workers,
                )
            ](
                self._remote_layer_ptrs,
                self._local_layer_ptrs,
                runs.local_starts[0],
                runs.remote_starts[0],
                runs.lengths[0],
                remote_rank,
                num_layers=len(self._kv_caches),
                page_words=self._page_bytes // 8,
                copy_workers=(
                    physicalization.generated_copy_workers
                ),
                block_elements=(
                    physicalization.generated_block_elements
                ),
                num_warps=8,
            )
            self._last_physicalization = "generated_contiguous"
        else:
            local_ids = torch.tensor(
                runs.local_starts,
                dtype=torch.int64,
                device=device,
            )
            remote_ids = torch.tensor(
                runs.remote_starts,
                dtype=torch.int64,
                device=device,
            )
            lengths = torch.tensor(
                runs.lengths,
                dtype=torch.int32,
                device=device,
            )
            pull_kv_runs_kernel[
                (
                    len(self._kv_caches),
                    runs.count,
                    physicalization.generated_copy_workers,
                )
            ](
                self._remote_layer_ptrs,
                self._local_layer_ptrs,
                local_ids,
                remote_ids,
                lengths,
                runs.count,
                remote_rank,
                num_layers=len(self._kv_caches),
                page_words=self._page_bytes // 8,
                copy_workers=(
                    physicalization.generated_copy_workers
                ),
                block_elements=(
                    physicalization.generated_block_elements
                ),
                num_warps=8,
            )
            self._last_physicalization = "generated_runs"
        completion = torch.cuda.Event()
        completion.record()
        return KVCachePullTicket(
            completion,
            (
                {
                    next(iter(self._kv_caches)): completion
                }
                if capture_layer_events
                else {}
            ),
        )

    def close(self) -> None:
        if self._region_table is not None:
            self._region_table.close()
            self._region_table = None
        for registration in reversed(self._registrations):
            registration.close()
        self._registrations.clear()
        self.world.close()
