from __future__ import annotations

import os
import struct
import time
from collections.abc import Iterable, Mapping
from datetime import timedelta
from multiprocessing.connection import wait as wait_connections
from typing import Protocol

import torch.distributed as dist
from torch.distributed import TCPStore

# One framing for every payload an exchange hands to `all_gather`: a version
# tag, the sending rank, and a run of length-prefixed records. What a record
# holds is the exchange's business — this settles only where one ends, so the
# bundles share their truncation checks and their version-skew message rather
# than each reinventing them.
_BUNDLE_HEADER = struct.Struct("<III")
_BUNDLE_RECORD = struct.Struct("<I")


def encode_bundle(
    version: int,
    rank: int,
    records: Iterable[bytes],
) -> bytes:
    """Frame one rank's records for a bootstrap exchange."""
    payload = bytearray()
    count = 0
    for record in records:
        payload.extend(_BUNDLE_RECORD.pack(len(record)))
        payload.extend(record)
        count += 1
    return bytes(_BUNDLE_HEADER.pack(version, rank, count) + payload)


def decode_bundle(
    payload: bytes,
    *,
    version: int,
    kind: str,
) -> tuple[int, tuple[bytes, ...]]:
    """The rank and records a peer framed, or why they cannot be read."""
    view = memoryview(payload)
    if len(view) < _BUNDLE_HEADER.size:
        raise ValueError(f"truncated Aboleth {kind} bundle")
    encoded_version, rank, count = _BUNDLE_HEADER.unpack_from(view)
    if encoded_version != version:
        raise ValueError(
            f"unsupported Aboleth {kind}-bundle version "
            f"{encoded_version}: this rank speaks version "
            f"{version}, so every rank must run the same "
            "Aboleth build"
        )
    offset = _BUNDLE_HEADER.size
    records = []
    for _ in range(count):
        if offset + _BUNDLE_RECORD.size > len(view):
            raise ValueError(f"truncated Aboleth {kind} record")
        length = _BUNDLE_RECORD.unpack_from(view, offset)[0]
        offset += _BUNDLE_RECORD.size
        end = offset + length
        if end > len(view):
            raise ValueError(f"truncated Aboleth {kind} record")
        records.append(bytes(view[offset:end]))
        offset = end
    if offset != len(view):
        raise ValueError(f"trailing bytes in Aboleth {kind} bundle")
    return int(rank), tuple(records)


class Bootstrap(Protocol):
    rank: int
    world_size: int

    def all_gather(
        self,
        tag: str,
        payload: bytes,
    ) -> dict[int, bytes]: ...


class TorchDistributedBootstrap:
    """Adapt an existing torch.distributed process group for Aboleth setup.

    This is an initialization-only collective. Runtime communication remains
    on Aboleth's P2P or network endpoints; the client-owned process group is
    used only to exchange topology, endpoint, and registered-region metadata.
    """

    def __init__(self, group):
        self._group = group
        self.rank = int(group.rank())
        self.world_size = int(group.size())
        self._sequence = 0
        if self.world_size < 1:
            raise ValueError("torch.distributed bootstrap needs at least one rank")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("torch.distributed bootstrap rank is outside the group")

    def all_gather(
        self,
        tag: str,
        payload: bytes,
    ) -> dict[int, bytes]:
        sequence = self._sequence
        self._sequence += 1
        local = (
            sequence,
            str(tag),
            self.rank,
            bytes(payload),
        )
        gathered = [None] * self.world_size
        dist.all_gather_object(
            gathered,
            local,
            group=self._group,
        )
        result = {}
        for expected_rank, record in enumerate(gathered):
            if not isinstance(record, tuple) or len(record) != 4:
                raise RuntimeError(
                    f"rank {expected_rank} returned an invalid "
                    f"Aboleth bootstrap record for {tag}"
                )
            peer_sequence, peer_tag, peer_rank, peer_payload = record
            if (
                int(peer_sequence) != sequence
                or peer_tag != tag
                or int(peer_rank) != expected_rank
            ):
                raise RuntimeError(
                    f"rank {expected_rank} entered a different "
                    f"Aboleth bootstrap collective than {tag}"
                )
            result[expected_rank] = bytes(peer_payload)
        return result


class PipeBootstrap:
    def __init__(
        self,
        rank: int,
        world_size: int,
        connection,
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._connection = connection

    def all_gather(
        self,
        tag: str,
        payload: bytes,
    ) -> dict[int, bytes]:
        self._connection.send(
            {
                "kind": "aboleth_all_gather",
                "tag": tag,
                "rank": self.rank,
                "payload": bytes(payload),
            }
        )
        response = self._connection.recv()
        if response.get("kind") == "aboleth_bootstrap_error":
            raise RuntimeError(response["error"])
        if (
            response.get("kind") != "aboleth_all_gather_result"
            or response.get("tag") != tag
        ):
            raise RuntimeError(f"invalid Aboleth bootstrap response for {tag}")
        gathered = {
            int(rank): bytes(value) for rank, value in response["payloads"].items()
        }
        if set(gathered) != set(range(self.world_size)):
            raise RuntimeError(f"incomplete Aboleth bootstrap result for {tag}")
        return gathered


class PipeBootstrapCoordinator:
    def __init__(
        self,
        connections: Mapping[int, object],
        world_size: int,
    ):
        self._connections = dict(connections)
        self.world_size = int(world_size)
        if set(self._connections) != set(range(self.world_size)):
            raise ValueError("bootstrap connections must cover every rank")

    def all_gather(
        self,
        tag: str,
        *,
        timeout: float = 300,
    ) -> dict[int, bytes]:
        deadline = time.monotonic() + timeout
        payloads = {}
        pending = {connection: rank for rank, connection in self._connections.items()}
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = sorted(pending.values())
                    raise TimeoutError(f"ranks {missing} did not reach bootstrap {tag}")
                ready = wait_connections(
                    tuple(pending),
                    timeout=remaining,
                )
                if not ready:
                    continue
                for connection in ready:
                    rank = pending.pop(connection)
                    message = connection.recv()
                    if message.get("ok") is False:
                        raise RuntimeError(message["error"])
                    if (
                        message.get("kind") != "aboleth_all_gather"
                        or message.get("tag") != tag
                        or int(message.get("rank", -1)) != rank
                    ):
                        raise RuntimeError(
                            f"rank {rank} sent an invalid bootstrap message for {tag}"
                        )
                    payloads[rank] = bytes(message["payload"])
        except Exception as error:
            response = {
                "kind": "aboleth_bootstrap_error",
                "tag": tag,
                "error": str(error),
            }
            for connection in self._connections.values():
                try:
                    connection.send(response)
                except (EOFError, OSError):
                    pass
            raise
        response = {
            "kind": "aboleth_all_gather_result",
            "tag": tag,
            "payloads": payloads,
        }
        for connection in self._connections.values():
            connection.send(response)
        return payloads


class TcpStoreBootstrap:
    """Multi-node bootstrap over a `torch.distributed.TCPStore`.

    Rank 0 hosts the store and the others attach to it; no process group is
    initialized, so this leaves the global torch world to its owner. Store
    keys are numbered by a per-instance counter, so every rank must issue
    the same bootstrap collectives (`all_gather` and `barrier` alike) in
    the same order, and rank 0 must outlive the others' final collective.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        host: str,
        port: int,
        timeout: float = 300.0,
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 1:
            raise ValueError("Aboleth bootstrap needs at least one rank")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Aboleth bootstrap rank is outside the world")
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self._sequence = 0
        self._store = TCPStore(
            self.host,
            self.port,
            self.world_size,
            self.rank == 0,
            timeout=timedelta(seconds=self.timeout),
        )

    @classmethod
    def from_env(
        cls,
        *,
        timeout: float = 300.0,
    ) -> TcpStoreBootstrap:
        settings = {}
        for name in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "WORLD_SIZE",
        ):
            value = os.environ.get(name)
            if not value:
                raise RuntimeError(
                    f"{name} is unset: the Aboleth TCP bootstrap "
                    "reads MASTER_ADDR, MASTER_PORT, RANK, and "
                    "WORLD_SIZE"
                )
            settings[name] = value
        return cls(
            int(settings["RANK"]),
            int(settings["WORLD_SIZE"]),
            host=settings["MASTER_ADDR"],
            port=int(settings["MASTER_PORT"]),
            timeout=timeout,
        )

    def all_gather(
        self,
        tag: str,
        payload: bytes,
    ) -> dict[int, bytes]:
        store = self._active_store()
        sequence = self._sequence
        self._sequence += 1
        store.set(
            self._key(sequence, tag, self.rank),
            bytes(payload),
        )
        gathered = {}
        for peer in range(self.world_size):
            try:
                gathered[peer] = bytes(store.get(self._key(sequence, tag, peer)))
            except Exception as error:
                raise TimeoutError(
                    f"rank {self.rank} did not receive rank {peer} at bootstrap {tag}"
                ) from error
        if set(gathered) != set(range(self.world_size)):
            raise RuntimeError(f"incomplete Aboleth bootstrap result for {tag}")
        return gathered

    def barrier(self, tag: str = "barrier") -> None:
        self.all_gather(tag, b"")

    def close(self) -> None:
        # Dropping rank 0's store stops the server the peers are attached to,
        # so the ranks should agree they are done before closing.
        self._store = None

    def _key(
        self,
        sequence: int,
        tag: str,
        rank: int,
    ) -> str:
        return f"aboleth/{sequence}/{tag}/{rank}"

    def _active_store(self) -> TCPStore:
        if self._store is None:
            raise RuntimeError("Aboleth TCP bootstrap is closed")
        return self._store
