from __future__ import annotations

from typing import Callable, Protocol

from .lifecycle import (
    CompletionEvent,
    KVCompletionNotification,
    KVLifecyclePoll,
    KVRequestLifecycle,
)


class LayeredPullTicket(Protocol):
    completion: CompletionEvent
    layer_events: dict[str, CompletionEvent]


class KVTransferRuntime(Protocol):
    asynchronous_completion: bool
    consumer_layer_pipeline: bool

    def pull_blocks(
        self,
        *,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_rank: int,
    ) -> float: ...

    def start_pull_blocks(
        self,
        *,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_rank: int,
    ) -> CompletionEvent: ...

    def start_layered_pull_blocks(
        self,
        *,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_rank: int,
    ) -> LayeredPullTicket: ...


class CompiledKVTransferManager:
    def __init__(
        self,
        runtime: KVTransferRuntime,
    ) -> None:
        self.runtime = runtime
        self.lifecycle = KVRequestLifecycle()

    @property
    def has_active_layers(self) -> bool:
        return self.lifecycle.has_active_layers

    @property
    def active_layer_count(self) -> int:
        return self.lifecycle.active_layer_count

    def begin_step(
        self,
        cancelled_request_ids: set[str],
    ) -> int:
        self.lifecycle.cancel(cancelled_request_ids)
        return self.lifecycle.activate_admitted()

    def start_transfer(
        self,
        *,
        request_id: str,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_rank: int,
        notification: KVCompletionNotification,
    ) -> bool:
        if not local_block_ids:
            return False
        layer_events = {}
        pipelined = False
        if (
            self.runtime.asynchronous_completion
            and self.runtime.consumer_layer_pipeline
        ):
            ticket = self.runtime.start_layered_pull_blocks(
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
                remote_rank=remote_rank,
            )
            completion = ticket.completion
            layer_events = ticket.layer_events
            pipelined = True
        elif self.runtime.asynchronous_completion:
            completion = self.runtime.start_pull_blocks(
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
                remote_rank=remote_rank,
            )
        else:
            self.runtime.pull_blocks(
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
                remote_rank=remote_rank,
            )
            return False
        self.lifecycle.add(
            request_id=request_id,
            completion=completion,
            notification=notification,
            layer_events=layer_events,
            pipelined=pipelined,
        )
        return True

    def wait_for_layer(
        self,
        layer_name: str,
        wait_event: Callable[[CompletionEvent], None],
    ) -> None:
        self.lifecycle.wait_for_layer(
            layer_name,
            wait_event,
        )

    def poll(self) -> KVLifecyclePoll:
        return self.lifecycle.poll()
