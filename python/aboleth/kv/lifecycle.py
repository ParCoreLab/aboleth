from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Protocol


class CompletionEvent(Protocol):
    def query(self) -> bool: ...


@dataclass(frozen=True)
class KVCompletionNotification:
    target: str
    payload: bytes


class KVRequestState(Enum):
    TRANSFERRING = auto()
    READY_TO_ADMIT = auto()
    ADMITTED = auto()
    ACTIVE = auto()
    RETIRED = auto()


@dataclass
class PendingKVRequest:
    request_id: str
    completion: CompletionEvent
    notification: KVCompletionNotification
    layer_events: dict[str, CompletionEvent] = field(
        default_factory=dict
    )
    state: KVRequestState = KVRequestState.TRANSFERRING
    notification_emitted: bool = False


@dataclass(frozen=True)
class KVLifecyclePoll:
    ready_request_ids: frozenset[str]
    notifications: tuple[KVCompletionNotification, ...]


class KVRequestLifecycle:
    def __init__(self) -> None:
        self._requests: dict[str, PendingKVRequest] = {}
        self._active_layers: dict[
            str,
            list[tuple[str, CompletionEvent]],
        ] = {}
        self._active_completions: dict[
            str,
            CompletionEvent,
        ] = {}

    @property
    def has_active_layers(self) -> bool:
        return bool(self._active_layers)

    @property
    def active_layer_count(self) -> int:
        return len(self._active_layers)

    @property
    def pending_count(self) -> int:
        return len(self._requests)

    def add(
        self,
        *,
        request_id: str,
        completion: CompletionEvent,
        notification: KVCompletionNotification,
        layer_events: dict[str, CompletionEvent] | None = None,
        pipelined: bool,
    ) -> None:
        if request_id in self._requests:
            raise ValueError(
                f"duplicate KV request lifecycle: {request_id}"
            )
        self._requests[request_id] = PendingKVRequest(
            request_id=request_id,
            completion=completion,
            notification=notification,
            layer_events=dict(layer_events or {}),
            state=(
                KVRequestState.READY_TO_ADMIT
                if pipelined
                else KVRequestState.TRANSFERRING
            ),
        )

    def cancel(self, request_ids: set[str]) -> None:
        if not request_ids:
            return
        for request_id in request_ids:
            request = self._requests.get(request_id)
            if request is None:
                continue
            request.state = KVRequestState.RETIRED
            request.layer_events.clear()
            self._active_completions.pop(
                request_id,
                None,
            )
        for layer_name, entries in tuple(
            self._active_layers.items()
        ):
            remaining = [
                entry
                for entry in entries
                if entry[0] not in request_ids
            ]
            if remaining:
                self._active_layers[layer_name] = remaining
            else:
                del self._active_layers[layer_name]
        self._discard_retired()

    def poll(self) -> KVLifecyclePoll:
        ready = set()
        notifications = []
        for request in tuple(self._requests.values()):
            completed = request.completion.query()
            if request.state is KVRequestState.READY_TO_ADMIT:
                ready.add(request.request_id)
                request.state = KVRequestState.ADMITTED
            elif (
                request.state is KVRequestState.TRANSFERRING
                and completed
            ):
                ready.add(request.request_id)
                request.state = KVRequestState.RETIRED
            if completed and not request.notification_emitted:
                notifications.append(request.notification)
                request.notification_emitted = True
        self._discard_retired()
        return KVLifecyclePoll(
            ready_request_ids=frozenset(ready),
            notifications=tuple(notifications),
        )

    def activate_admitted(self) -> int:
        activated = 0
        for request in tuple(self._requests.values()):
            if request.state is not KVRequestState.ADMITTED:
                continue
            if request.completion.query():
                request.state = KVRequestState.RETIRED
            else:
                for layer_name, event in (
                    request.layer_events.items()
                ):
                    self._active_layers.setdefault(
                        layer_name,
                        [],
                    ).append((request.request_id, event))
                self._active_completions[
                    request.request_id
                ] = request.completion
                request.state = KVRequestState.ACTIVE
            activated += 1
        self._discard_retired()
        return activated

    def wait_for_layer(
        self,
        layer_name: str,
        wait_event: Callable[[CompletionEvent], None],
    ) -> None:
        entries = self._active_layers.pop(
            layer_name,
            (),
        )
        if not entries:
            return
        if (
            self._active_completions
            and all(
                event.query()
                for event in self._active_completions.values()
            )
        ):
            self._active_layers.clear()
            self._retire_active()
            return
        for _, event in entries:
            if not event.query():
                wait_event(event)
        if not self._active_layers:
            self._retire_active()

    def _retire_active(self) -> None:
        active_ids = tuple(self._active_completions)
        self._active_completions.clear()
        for request_id in active_ids:
            request = self._requests.get(request_id)
            if (
                request is not None
                and request.state is KVRequestState.ACTIVE
            ):
                request.state = KVRequestState.RETIRED
        self._discard_retired()

    def _discard_retired(self) -> None:
        for request_id, request in tuple(
            self._requests.items()
        ):
            if (
                request.state is KVRequestState.RETIRED
                and request.notification_emitted
            ):
                del self._requests[request_id]
