from __future__ import annotations

import json
import os
import pathlib
import socket
import struct
from uuid import UUID
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

import torch

# The bootstraps live in their own module, which needs no runtime or CUDA;
# they stay importable from here, where they used to be defined.
from .bootstrap import (
    Bootstrap,
    decode_bundle,
    encode_bundle,
)
from .runtime import (
    AllocationHandle,
    AllocationProvider,
    LocalEndpoint,
    RegisteredBuffer,
    RemoteBuffer,
    Runtime,
    Transport,
    connection_info_from_bytes,
    connection_transport,
)

# Every exchange below is a bundle of length-prefixed records (see
# `aboleth.bootstrap.encode_bundle`); these say what one record holds, which
# is a fixed head followed by the one variable-length field it introduces.
_TOPOLOGY_BUNDLE_VERSION = 2
_CONNECTION_BUNDLE_VERSION = 3
_REGION_BUNDLE_VERSION = 3
_TOPOLOGY_RECORD = struct.Struct("<I")  # device, then the hostname
_CONNECTION_RECORD = struct.Struct("<II")  # peer, transport
_REGION_RECORD = struct.Struct("<IH")  # target peer, name length
_P2P_CONNECTION = struct.Struct("<i")  # a P2P peer's CUDA ordinal
_TRANSPORT_TO_WIRE = {
    Transport.P2P: 1,
    Transport.RDMA: 2,
    Transport.PROXY: 3,
    Transport.EFA: 4,
    Transport.IONIC: 5,
}
_TRANSPORT_FROM_WIRE = {value: key for key, value in _TRANSPORT_TO_WIRE.items()}


def _local_node_identity() -> str:
    """Distinguish machines whose image supplied the same hostname.

    Linux's boot ID is shared by local ranks and differs across running
    kernels. Keep the hostname for diagnostics and the existing portable
    fallback where a boot ID is unavailable.
    """
    hostname = socket.gethostname()
    try:
        boot_id = UUID(pathlib.Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text().strip())
    except (OSError, ValueError):
        return hostname
    return f"{hostname}:{boot_id}"


@dataclass(frozen=True)
class DeviceLibrary:
    """The prebuilt compatibility device library for one architecture.

    `bitcode` links into a client's own module (Triton `extern_libs`, clang
    `-mlink-builtin-bitcode`); `archive` device-links from nvcc
    (`nvcc -rdc=true client.cu <archive>`); `include_dir` carries
    aboleth/device.cuh, which is header-only and needs neither.
    """

    arch: str
    bitcode: pathlib.Path
    archive: pathlib.Path
    include_dir: pathlib.Path


def normalize_arch(
    arch: str | None = None,
    device: int | None = None,
) -> str:
    """The build's name for a capability.

    `sm_90` and `sm_90a` are one capability to the compiler frontend, but the
    build ships suffixed artifacts (ABOLETH_RDMA_BITCODE_ARCHS defaults to
    sm_90a) while `torch.cuda.get_device_capability` spells the bare form —
    so a capability-derived default is mapped to the suffixed variant the
    build actually emitted. Generated Tier-1 manifests are authoritative;
    the retiring RDMA bitcode names remain a compatibility fallback. An
    explicit `arch` that exists is kept verbatim.
    """
    if arch is None:
        if torch.version.hip:
            arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        else:
            major, minor = torch.cuda.get_device_capability(device)
            arch = f"sm_{major}{minor}"
    root = pathlib.Path(__file__).resolve().parents[2]
    build = pathlib.Path(os.environ.get("ABOLETH_BUILD_DIR", root / "build"))
    built_arches = {
        path.stem.removeprefix("aboleth_rdma_")
        for path in (build / "lib").glob("aboleth_rdma_*.bc")
    }
    for manifest in (build / "lib" / "tier1").glob("*/manifest.json"):
        try:
            recorded_arch = json.loads(manifest.read_text())["arch"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if isinstance(recorded_arch, str):
            built_arches.add(recorded_arch)
    if arch in built_arches:
        return arch
    variants = sorted(
        candidate
        for candidate in built_arches
        if candidate.startswith(arch) and candidate[len(arch):] in ("a", "f")
    )
    if variants:
        return variants[0]
    return arch


def _legacy_rdma_device_library(arch: str) -> DeviceLibrary:
    """Locate the handwritten RDMA compatibility library."""
    root = pathlib.Path(__file__).resolve().parents[2]
    build = pathlib.Path(os.environ.get("ABOLETH_BUILD_DIR", root / "build"))
    bitcode = build / "lib" / f"aboleth_rdma_{arch}.bc"
    if not bitcode.is_file():
        built = sorted(
            path.stem.removeprefix("aboleth_rdma_")
            for path in (build / "lib").glob("aboleth_rdma_*.bc")
        )
        raise FileNotFoundError(
            f"no device library for {arch}: {bitcode} is not built "
            f"(built: {', '.join(built) or 'none'}); add it with "
            "-DABOLETH_RDMA_BITCODE_ARCHS or select one with ABOLETH_RDMA_ARCH"
        )
    return DeviceLibrary(
        arch=arch,
        bitcode=bitcode,
        archive=build / "lib" / f"libaboleth_device_{arch}.a",
        include_dir=root / "include",
    )


def device_library(
    arch: str | None = None,
    *,
    generated: bool = True,
):
    """Return the generic Tier-1 library for one architecture.

    The default is the compiler-generated mixed RDMA+P2P artifact. The
    handwritten RDMA library remains reachable only as `generated=False`
    while differential tests use it as the retirement oracle.
    """
    arch = normalize_arch(arch)
    if generated:
        from .device_library import cached_device_library

        return cached_device_library(arch)
    return _legacy_rdma_device_library(arch)


@dataclass(frozen=True)
class TopologyInfo:
    rank: int
    device: int
    hostname: str

    def to_bytes(self) -> bytes:
        return encode_bundle(
            _TOPOLOGY_BUNDLE_VERSION,
            self.rank,
            [_TOPOLOGY_RECORD.pack(self.device) + self.hostname.encode("utf-8")],
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> TopologyInfo:
        rank, records = decode_bundle(
            data,
            version=_TOPOLOGY_BUNDLE_VERSION,
            kind="topology",
        )
        if len(records) != 1:
            raise ValueError("an Aboleth topology bundle carries one record")
        record = records[0]
        if len(record) < _TOPOLOGY_RECORD.size:
            raise ValueError("truncated Aboleth topology record")
        device = _TOPOLOGY_RECORD.unpack_from(record)[0]
        hostname = record[_TOPOLOGY_RECORD.size :].decode("utf-8")
        if not hostname:
            raise ValueError("topology hostname cannot be empty")
        return cls(
            rank=rank,
            device=int(device),
            hostname=hostname,
        )


@dataclass(frozen=True)
class ConnectionRecord:
    transport: Transport
    device: int
    # The peer's native connection info; P2P peers carry a device instead.
    info: object | None = None


@dataclass(frozen=True)
class AutoTransportPolicy:
    p2p_max_payload_bytes: int = 16 * 1024

    def __post_init__(self):
        if self.p2p_max_payload_bytes < 0:
            raise ValueError("P2P payload threshold cannot be negative")

    def select(
        self,
        *,
        can_p2p: bool,
        payload_bytes: int | None,
        remote_transport: Transport,
    ) -> Transport:
        if remote_transport not in (
            Transport.RDMA,
            Transport.PROXY,
            Transport.IONIC,
        ):
            raise ValueError("remote transport must be rdma or proxy")
        if payload_bytes is not None and payload_bytes < 0:
            raise ValueError("payload bytes cannot be negative")
        if can_p2p and (
            payload_bytes is None or payload_bytes <= self.p2p_max_payload_bytes
        ):
            return Transport.P2P
        return remote_transport


@dataclass(frozen=True)
class LocalRegion:
    name: str
    address: int
    length: int
    lkey: int
    device: int
    generation: int
    _owner: RegisteredBuffer

    def export(
        self,
        rank: int,
        provider: AllocationProvider,
    ) -> AllocationHandle:
        return self._owner.export_allocation(
            rank,
            provider=provider,
            offset=self.address - self._owner.address,
            length=self.length,
        )


def _encode_connections(
    rank: int,
    endpoints: Mapping[int, LocalEndpoint],
) -> bytes:
    records = []
    for peer in sorted(endpoints):
        endpoint = endpoints[peer]
        records.append(
            _CONNECTION_RECORD.pack(
                peer,
                _TRANSPORT_TO_WIRE[endpoint.transport],
            )
            + (
                _P2P_CONNECTION.pack(endpoint.device)
                if endpoint.transport == Transport.P2P
                else endpoint.connection_info.to_bytes()
            )
        )
    return encode_bundle(
        _CONNECTION_BUNDLE_VERSION,
        rank,
        records,
    )


def _decode_connections(
    data: bytes,
) -> tuple[int, dict[int, ConnectionRecord]]:
    rank, records = decode_bundle(
        data,
        version=_CONNECTION_BUNDLE_VERSION,
        kind="connection",
    )
    connections = {}
    for record in records:
        if len(record) < _CONNECTION_RECORD.size:
            raise ValueError("truncated Aboleth connection record")
        peer, transport_code = _CONNECTION_RECORD.unpack_from(record)
        encoded = record[_CONNECTION_RECORD.size :]
        if peer in connections:
            raise ValueError("duplicate peer in Aboleth connection bundle")
        try:
            transport = _TRANSPORT_FROM_WIRE[transport_code]
        except KeyError as error:
            raise ValueError(
                f"unknown connection transport {transport_code}"
            ) from error
        if transport == Transport.P2P:
            if len(encoded) != _P2P_CONNECTION.size:
                raise ValueError("invalid P2P connection-info size")
            connections[peer] = ConnectionRecord(
                transport=transport,
                device=_P2P_CONNECTION.unpack(encoded)[0],
            )
        else:
            info = connection_info_from_bytes(encoded)
            if connection_transport(info) != transport:
                raise ValueError("connection transport payload mismatch")
            connections[peer] = ConnectionRecord(
                transport=transport,
                device=-1,
                info=info,
            )
    return rank, connections


def _encode_regions(
    rank: int,
    regions: Mapping[str, LocalRegion],
    endpoints: Mapping[int, LocalEndpoint],
) -> bytes:
    records = []
    for target_peer in sorted(endpoints):
        provider = (
            AllocationProvider.CUDA_IPC
            if endpoints[target_peer].transport == Transport.P2P
            else AllocationProvider.IB_VERBS
        )
        for name in sorted(regions):
            encoded_name = name.encode("utf-8")
            if len(encoded_name) > 0xFFFF:
                raise ValueError("Aboleth region name is too large")
            records.append(
                _REGION_RECORD.pack(
                    target_peer,
                    len(encoded_name),
                )
                + encoded_name
                + regions[name]
                .export(
                    rank,
                    provider,
                )
                .to_bytes()
            )
    return encode_bundle(
        _REGION_BUNDLE_VERSION,
        rank,
        records,
    )


def _decode_regions(
    data: bytes,
    *,
    target_rank: int,
) -> tuple[int, dict[str, AllocationHandle]]:
    rank, records = decode_bundle(
        data,
        version=_REGION_BUNDLE_VERSION,
        kind="region",
    )
    regions = {}
    for record in records:
        if len(record) < _REGION_RECORD.size:
            raise ValueError("truncated Aboleth region record")
        target_peer, name_length = _REGION_RECORD.unpack_from(record)
        name_end = _REGION_RECORD.size + name_length
        if name_end > len(record):
            raise ValueError("truncated Aboleth region metadata")
        if target_peer != target_rank:
            continue
        name = record[_REGION_RECORD.size : name_end].decode("utf-8")
        if not name:
            raise ValueError("invalid region name in Aboleth bundle")
        if name in regions:
            raise ValueError("duplicate targeted region in Aboleth bundle")
        regions[name] = AllocationHandle.from_bytes(record[name_end:])
    return rank, regions


class PeerRegionTable:
    def __init__(
        self,
        world: World,
        local_regions: Mapping[str, LocalRegion],
        remote_regions: Mapping[
            int,
            Mapping[str, RemoteBuffer],
        ],
        strictly_ordered: bool = False,
    ):
        # Every verbs registration behind the remote regions carries no
        # relaxed ordering, so publication into them may use a fenced write.
        self.strictly_ordered = strictly_ordered
        self._world = world
        self._local_regions = dict(local_regions)
        self._remote_regions = {
            int(peer): dict(regions) for peer, regions in remote_regions.items()
        }
        self._tables: dict[
            tuple[str, str],
            torch.Tensor,
        ] = {}
        self._closed = False
        # The Tier-1 device library dispatches per peer on the low bits of
        # the handle (aboleth/device.cuh kKind*): descriptors are >=8-byte
        # aligned, so the transport kind rides the bits a pointer never
        # uses. Only Tier-1 clients consume this table.
        kind_bits = {
            Transport.RDMA: 0,
            Transport.P2P: 1,
            Transport.PROXY: 2,
            Transport.EFA: 3,
            Transport.IONIC: 4,
        }
        transports = world.peer_transports
        # Peer-major [world_size x endpoints_per_peer]: fan-out slots for one
        # peer are consecutive, matching the generated header's
        # `peer * kEndpointsPerPeer + blockIdx.x % kEndpointsPerPeer`.
        fan_out = world.endpoints_per_peer
        handles = []
        for peer in range(world.world_size):
            if peer == world.rank:
                handles.extend([0] * fan_out)
                continue
            kind = kind_bits[transports.get(peer, world.transport)]
            for slot in range(fan_out):
                handle = world.endpoint(
                    peer, slot=slot
                ).device_handle.data_ptr()
                handles.append(handle | kind)
        self._endpoint_handles = torch.tensor(
            handles,
            dtype=torch.uint64,
            device=f"cuda:{world.device}",
        )

    @property
    def endpoint_handles(self) -> torch.Tensor:
        self._ensure_open()
        return self._endpoint_handles

    @property
    def rank(self) -> int:
        return self._world.rank

    @property
    def world_size(self) -> int:
        return self._world.world_size

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def names(self) -> tuple[str, ...]:
        self._ensure_open()
        return tuple(sorted(self._local_regions))

    def local(self, name: str) -> LocalRegion:
        self._ensure_open()
        try:
            return self._local_regions[name]
        except KeyError as error:
            raise KeyError(f"unknown Aboleth region {name!r}") from error

    def remote(
        self,
        peer: int,
        name: str,
    ) -> RemoteBuffer:
        self._ensure_open()
        if peer == self._world.rank:
            raise ValueError("the local rank does not have a remote region")
        try:
            remote = self._remote_regions[peer][name]
            remote.ensure_open()
            return remote
        except KeyError as error:
            raise KeyError(f"rank {peer} has no Aboleth region {name!r}") from error

    def addresses(self, name: str) -> torch.Tensor:
        self._ensure_open()
        key = ("addresses", name)
        if key not in self._tables:
            values = [
                self.local(name).address
                if peer == self._world.rank
                else self.remote(peer, name).address
                for peer in range(self._world.world_size)
            ]
            self._tables[key] = torch.tensor(
                values,
                dtype=torch.int64,
                device=f"cuda:{self._world.device}",
            )
        return self._tables[key]

    def rkeys(self, name: str) -> torch.Tensor:
        self._ensure_open()
        key = ("rkeys", name)
        if key not in self._tables:
            values = [
                0 if peer == self._world.rank else self.remote(peer, name).rkey
                for peer in range(self._world.world_size)
            ]
            self._tables[key] = torch.tensor(
                values,
                dtype=torch.int64,
                device=f"cuda:{self._world.device}",
            )
        return self._tables[key]

    def regions(self, name: str) -> torch.Tensor:
        """Every rank's `name` region as (address, key) pairs, rank-major.

        A kernel reads a peer's entry with `aboleth.language.remote`. The local
        rank's entry is its own buffer and registration.
        """
        self._ensure_open()
        key = ("regions", name)
        if key not in self._tables:
            local = self.local(name)
            values = []
            for peer in range(self._world.world_size):
                if peer == self._world.rank:
                    values += [local.address, local.lkey]
                else:
                    remote = self.remote(peer, name)
                    values += [remote.address, remote.rkey]
            self._tables[key] = torch.tensor(
                values,
                dtype=torch.int64,
                device=f"cuda:{self._world.device}",
            )
        return self._tables[key]

    def generations(self, name: str) -> torch.Tensor:
        self._ensure_open()
        key = ("generations", name)
        if key not in self._tables:
            values = [
                self.local(name).generation
                if peer == self._world.rank
                else self.remote(peer, name).generation
                for peer in range(self._world.world_size)
            ]
            self._tables[key] = torch.tensor(
                values,
                dtype=torch.uint64,
                device=f"cuda:{self._world.device}",
            )
        return self._tables[key]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Aboleth peer-region table is closed")

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self._world.device)
        owners = {}
        for regions in self._remote_regions.values():
            for remote in regions.values():
                owner = remote._owner
                if owner is not None:
                    owners[id(owner)] = owner
        first_error = None
        for owner in owners.values():
            close = getattr(owner, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as error:  # noqa: BLE001
                    if first_error is None:
                        first_error = error
        self._remote_regions.clear()
        self._tables.clear()
        self._closed = True
        if first_error is not None:
            raise first_error


class World:
    # Each tag's suffix moves with its bundle version, so builds that frame
    # an exchange differently never meet over one bootstrap key.
    TOPOLOGY_TAG = "aboleth.world.topology.v2"
    CONNECTION_TAG = "aboleth.world.connections.v3"
    # ((pin, calibrate), ResolvedProfiles) after endpoint_profiles().
    _profiles = None

    @staticmethod
    def region_tag(tag: str) -> str:
        if not tag:
            raise ValueError("Aboleth region-exchange tag cannot be empty")
        return f"aboleth.world.regions.{tag}.v3"

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        device: int,
        bootstrap: Bootstrap,
        hca: str | None = None,
        runtime: Runtime | None = None,
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = int(device)
        if self.world_size < 2:
            raise ValueError("Aboleth World requires at least two ranks")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Aboleth rank is outside the world")
        if bootstrap.rank != self.rank or bootstrap.world_size != self.world_size:
            raise ValueError("bootstrap rank/world does not match World")
        self.bootstrap = bootstrap
        if runtime is not None and (
            runtime.device != self.device
            or hca is not None
        ):
            raise ValueError(
                "a shared World runtime must match the device "
                "and owns its HCA selection"
            )
        self.runtime = (
            Runtime(device=self.device, hca=hca)
            if runtime is None
            else runtime
        )
        self._owns_runtime = runtime is None
        self._transport: Transport | None = None
        self._topology: dict[int, TopologyInfo] = {}
        # peer -> per-slot endpoints; slot 0 is the primary every
        # single-endpoint consumer sees, extra slots exist only under
        # connect_all(endpoints_per_peer=N) fan-out.
        self._endpoints: dict[int, tuple[LocalEndpoint, ...]] = {}
        self._endpoints_per_peer = 1
        self._region_tables: dict[str, PeerRegionTable] = {}
        self._closed = False

    @property
    def hca(self) -> str:
        self._ensure_open()
        return self.runtime.hca

    @property
    def lid(self) -> int:
        self._ensure_open()
        return self.runtime.lid

    @property
    def transport(self) -> Transport | None:
        return self._transport

    @property
    def peer_transports(self) -> Mapping[int, Transport]:
        """The transport actually selected for each peer.

        Under `Transport.AUTO` this differs per peer: same-node peers resolve
        to P2P while remote peers resolve to the requested remote transport.
        """
        self._ensure_open()
        return MappingProxyType(
            {peer: slots[0].transport for peer, slots in self._endpoints.items()}
        )

    @property
    def transport_geography(self) -> tuple[tuple[int, str], ...]:
        """Canonical compile-time peer-rank -> transport facts."""
        self._ensure_open()
        return tuple(
            sorted(
                (peer, slots[0].transport.value)
                for peer, slots in self._endpoints.items()
            )
        )

    @property
    def gpu_resident_queues(self) -> bool:
        """Every connected rdma endpoint keeps its queues in GPU memory, so a
        GPU-scope release publishes its doorbell (False with none connected).
        """
        self._ensure_open()
        rdma = [
            endpoint
            for slots in getattr(self, "_endpoints", {}).values()
            for endpoint in slots
            if endpoint.transport == Transport.RDMA
        ]
        return bool(rdma) and all(
            endpoint.gpu_resident_queues for endpoint in rdma)

    @property
    def endpoints_per_peer(self) -> int:
        """The connect-time endpoint fan-out (1 unless requested)."""
        return self._endpoints_per_peer

    @property
    def topology(self) -> Mapping[int, TopologyInfo]:
        self._ensure_open()
        return MappingProxyType(self._topology)

    @property
    def endpoints(self) -> Mapping[int, LocalEndpoint]:
        """Each peer's primary (slot-0) endpoint."""
        self._ensure_open()
        return MappingProxyType(
            {peer: slots[0] for peer, slots in self._endpoints.items()}
        )

    def endpoint(self, peer: int, *, slot: int = 0) -> LocalEndpoint:
        self._ensure_open()
        try:
            slots = self._endpoints[int(peer)]
        except KeyError as error:
            raise KeyError(
                f"rank {self.rank} has no endpoint for peer {peer}"
            ) from error
        if not 0 <= slot < len(slots):
            raise IndexError(
                f"peer {peer} has {len(slots)} endpoint slot(s), "
                f"not slot {slot}"
            )
        return slots[slot]

    def connect_all(
        self,
        transport: str | Transport,
        *,
        remote_transport: str | Transport = Transport.RDMA,
        payload_bytes: int | None = None,
        auto_policy: AutoTransportPolicy | None = None,
        endpoints_per_peer: int = 1,
    ) -> None:
        self._ensure_open()
        if self._endpoints:
            raise RuntimeError("Aboleth World is already connected")
        endpoints_per_peer = int(endpoints_per_peer)
        if endpoints_per_peer < 1:
            raise ValueError("endpoints_per_peer must be >= 1")
        requested = Transport(transport)
        resolved_remote = Transport(remote_transport)
        selected_policy = AutoTransportPolicy() if auto_policy is None else auto_policy
        if not isinstance(selected_policy, AutoTransportPolicy):
            raise TypeError("auto_policy must be an AutoTransportPolicy")
        if resolved_remote not in (
            Transport.RDMA,
            Transport.PROXY,
            Transport.IONIC,
        ):
            raise ValueError("remote transport must be rdma or proxy")
        if requested not in (
            Transport.AUTO,
            Transport.P2P,
            Transport.RDMA,
            Transport.PROXY,
            Transport.IONIC,
        ):
            raise ValueError("unsupported World transport")

        local_topology = TopologyInfo(
            rank=self.rank,
            device=self.device,
            hostname=_local_node_identity(),
        )
        gathered_topology = self.bootstrap.all_gather(
            self.TOPOLOGY_TAG,
            local_topology.to_bytes(),
        )
        topology = {
            peer: TopologyInfo.from_bytes(payload)
            for peer, payload in gathered_topology.items()
        }
        for peer, info in topology.items():
            if info.rank != peer:
                raise RuntimeError(f"invalid topology bundle from rank {peer}")

        peer_transports = {}
        for peer, info in topology.items():
            if peer == self.rank:
                continue
            same_node = info.hostname == local_topology.hostname
            can_p2p = (
                same_node
                and torch.cuda.can_device_access_peer(
                    self.device,
                    info.device,
                )
                and torch.cuda.can_device_access_peer(
                    info.device,
                    self.device,
                )
            )
            if requested == Transport.AUTO:
                selected = selected_policy.select(
                    can_p2p=can_p2p,
                    payload_bytes=payload_bytes,
                    remote_transport=resolved_remote,
                )
            elif requested == Transport.P2P:
                if not same_node:
                    raise ValueError(f"rank {peer} is not on the local node")
                if not can_p2p:
                    raise ValueError(f"rank {peer} is not P2P accessible")
                selected = Transport.P2P
            else:
                selected = requested
            peer_transports[peer] = selected

        # Fan-out repeats the whole create/exchange/connect sequence per
        # slot: slot 0 keeps the historic bootstrap tag (byte-identical for
        # endpoints_per_peer=1), later slots ride slot-suffixed tags and
        # pair with the same slot on every peer.
        per_slot: list[dict[int, LocalEndpoint]] = []
        # Send-queue depth request for the RDMA endpoints (0: the adapter's
        # default), the knob a tuning sweep turns.
        qp_depth = int(os.environ.get("ABOLETH_QP_DEPTH", "0") or 0)
        try:
            for slot in range(endpoints_per_peer):
                endpoints: dict[int, LocalEndpoint] = {}
                per_slot.append(endpoints)
                for peer, selected in peer_transports.items():
                    endpoints[peer] = self.runtime.create_local_endpoint(
                        transport=selected,
                        qp_depth=(qp_depth if Transport(selected)
                                  in (Transport.RDMA, Transport.IONIC) else 0),
                    )
                tag = (
                    self.CONNECTION_TAG
                    if slot == 0
                    else f"{self.CONNECTION_TAG}:{slot}"
                )
                encoded = _encode_connections(
                    self.rank,
                    endpoints,
                )
                gathered = self.bootstrap.all_gather(
                    tag,
                    encoded,
                )
                for peer, endpoint in endpoints.items():
                    remote_rank, connections = _decode_connections(
                        gathered[peer]
                    )
                    if remote_rank != peer or set(connections) != set(
                        range(self.world_size)
                    ) - {peer}:
                        raise RuntimeError(
                            f"invalid endpoint bundle from rank {peer}"
                        )
                    connection = connections[self.rank]
                    if connection.transport != endpoint.transport:
                        raise RuntimeError(
                            "peer transport selection is asymmetric"
                        )
                    if endpoint.transport == Transport.P2P:
                        if connection.device != topology[peer].device:
                            raise RuntimeError(
                                f"P2P device mismatch for rank {peer}"
                            )
                        endpoint.connect_p2p(
                            peer=peer,
                            peer_device=connection.device,
                        )
                    else:
                        if connection.info is None:
                            raise RuntimeError(
                                "verbs connection omitted endpoint info"
                            )
                        endpoint.connect(
                            connection.info,
                            peer=peer,
                        )
        except Exception as original_error:
            for endpoints in per_slot:
                for endpoint in endpoints.values():
                    try:
                        endpoint.close()
                    except Exception as cleanup_error:  # noqa: BLE001
                        original_error.add_note(
                            "Aboleth endpoint rollback also failed: "
                            f"{cleanup_error}"
                        )
            raise
        self._endpoints = {
            peer: tuple(endpoints[peer] for endpoints in per_slot)
            for peer in peer_transports
        }
        self._endpoints_per_peer = endpoints_per_peer
        self._topology = topology
        unique = set(peer_transports.values())
        self._transport = next(iter(unique)) if len(unique) == 1 else Transport.AUTO

    def endpoint_registration(self, lanes: int | None = None):
        """Collective: every rank's platform, queue memory, lanes and link.

        The facts plan-endpoints matches profiles against (one canonical
        image in rank order). `lanes` is the endpoint budget each rank grants
        one program family; it defaults to the connected fan-out.
        """
        import json
        from pathlib import Path

        from .endpoint_profiles import (
            EndpointRegistration,
            endpoint_platform_key,
            registered_peer_link,
        )

        self._ensure_open()
        lanes = self._endpoints_per_peer if lanes is None else int(lanes)
        if not 1 <= lanes <= self._endpoints_per_peer:
            raise ValueError("lanes exceed the connected endpoints per peer")
        rdma = next(
            (
                self.endpoint(peer)
                for peer, kind in self.transport_geography
                if kind in ("rdma", "ionic")
            ),
            None,
        )
        key, memory = "", "none"
        if rdma is not None:
            nic = Path("/sys/class/infiniband") / self.hca
            key = endpoint_platform_key(
                gpu_name=torch.cuda.get_device_name(self.device),
                compute_capability=torch.cuda.get_device_capability(self.device),
                nic_vendor=(nic / "device/vendor").read_text().strip(),
                nic_device=(nic / "device/device").read_text().strip(),
                nic_firmware=(nic / "fw_ver").read_text().strip(),
                link_rate=(nic / f"ports/{self.runtime.rdma_port}/rate")
                .read_text()
                .strip(),
                submission=rdma.submit_mode,
            )
            memory = "gpu" if rdma.gpu_resident_queues else "host"
        hip = torch.version.hip is not None
        amd_registration = None
        if hip and all(kind == "p2p" for _, kind in self.transport_geography):
            # AMD peer memory: the platform is the KFD device registration.
            from .amd_topology import registered_amd_device
            amd_registration = registered_amd_device(self.device)
            if amd_registration is not None:
                key = amd_registration["platform"]
        identity = str(
            getattr(torch.cuda.get_device_properties(self.device), "uuid", "")
        )
        local_facts = [key, memory, lanes, identity]
        if hip:
            local_facts.append(amd_registration)
        image = self.bootstrap.all_gather(
            f"aboleth.world.endpoint-registration.{lanes}",
            json.dumps(local_facts).encode(),
        )
        registered = tuple(json.loads(image[peer]) for peer in range(self.world_size))
        if hip:
            from .amd_topology import registered_amd_peer_link
            link = registered_amd_peer_link(
                tuple(facts[4] for facts in registered))
        else:
            link = registered_peer_link(
                identity,
                tuple(
                    registered[peer][3]
                    for peer, kind in self.transport_geography
                    if kind == "p2p"
                ),
            )
        links = self.bootstrap.all_gather(
            f"aboleth.world.endpoint-links.{lanes}", link.encode()
        )
        return EndpointRegistration(
            platform=key,
            queue_memory=memory,
            participants=tuple(tuple(facts[:3]) for facts in registered),
            peer_link=link,
            participant_links=tuple(
                links[peer].decode() for peer in range(self.world_size)
            ),
        )

    def endpoint_profiles(
        self,
        *,
        pin: Iterable[str] = (),
        calibrate: bool = True,
        refresh: bool = False,
        store=None,
    ):
        """Collective: the endpoint profiles this world plans fan-out with.

        Pinned, cached or seed entries that every rank agrees on are used as
        they are; otherwise the ranks probe their endpoints once, fit the
        profile, and cache it per platform (`aboleth.endpoint_profiles`).
        `pin` reproduces stored identities exactly; `calibrate=False` or a
        failed probe leaves one lane, and the result's record says why.
        """
        from .endpoint_profiles import resolve_endpoint_profiles

        self._ensure_open()
        pin = tuple(pin)
        if refresh or self._profiles is None or self._profiles[0] != (pin, calibrate):
            resolved = resolve_endpoint_profiles(
                self,
                self.endpoint_registration(),
                pin=pin,
                calibrate=calibrate,
                refresh=refresh,
                store=store,
            )
            self._profiles = ((pin, calibrate), resolved)
        return self._profiles[1]

    @property
    def resolved_endpoint_profiles(self):
        """The last `endpoint_profiles()` result, or None before it ran."""
        return None if self._profiles is None else self._profiles[1]

    def register(
        self,
        tensor: torch.Tensor,
    ) -> RegisteredBuffer:
        self._ensure_open()
        return self.runtime.register(tensor)

    def region(
        self,
        name: str,
        registration: RegisteredBuffer,
        tensor: torch.Tensor | None = None,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> LocalRegion:
        self._ensure_open()
        if not name:
            raise ValueError("Aboleth region name cannot be empty")
        self.runtime._validate_buffer(registration)
        if tensor is not None:
            if not tensor.is_cuda:
                raise ValueError("Aboleth region tensor must be on CUDA")
            if not tensor.is_contiguous():
                raise ValueError("Aboleth region tensor must be contiguous")
            if tensor.device.index != registration.device:
                raise ValueError(
                    "region tensor and registration use different CUDA devices"
                )
            address = tensor.data_ptr()
            available = tensor.numel() * tensor.element_size()
            if offset:
                address += offset
                available -= offset
        else:
            address = registration.address + offset
            available = registration.length - offset
        if length is None:
            length = available
        if (
            offset < 0
            or length < 0
            or address < registration.address
            or address + length > registration.address + registration.length
        ):
            raise ValueError(f"region {name!r} is outside its registration")
        return LocalRegion(
            name=name,
            address=address,
            length=length,
            lkey=registration.lkey,
            device=registration.device,
            generation=registration.generation,
            _owner=registration,
        )

    def exchange_regions(
        self,
        regions: Mapping[str, LocalRegion],
        *,
        tag: str = "default",
    ) -> PeerRegionTable:
        self._ensure_open()
        prior_table = self._region_tables.get(tag)
        if prior_table is not None and not prior_table.closed:
            raise RuntimeError(f"region exchange {tag!r} is still live")
        if not self._endpoints:
            raise RuntimeError("connect World before exchanging regions")
        local_regions = dict(regions)
        for name, region in local_regions.items():
            if name != region.name:
                raise ValueError("region dictionary key/name mismatch")
            if region.device != self.device:
                raise ValueError(f"region {name!r} is on the wrong GPU")
            self.runtime._validate_buffer(region._owner)
            if (
                region.device != region._owner.device
                or region.generation != region._owner.generation
                or region.lkey != region._owner.lkey
                or region.length < 0
                or region.address < region._owner.address
                or region.address + region.length
                > region._owner.address + region._owner.length
            ):
                raise ValueError(f"region {name!r} does not match its registration")
        encoded = _encode_regions(
            self.rank,
            local_regions,
            self.endpoints,
        )
        gathered = self.bootstrap.all_gather(
            self.region_tag(tag),
            encoded,
        )
        remote_regions = {}
        strictly_ordered = True
        expected_names = set(local_regions)
        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            remote_rank, handles = _decode_regions(
                gathered[peer],
                target_rank=self.rank,
            )
            if remote_rank != peer or set(handles) != expected_names:
                raise RuntimeError(f"region bundle mismatch from rank {peer}")
            strictly_ordered = strictly_ordered and all(
                handle.strictly_ordered
                for handle in handles.values()
                if handle.provider == AllocationProvider.IB_VERBS
            )
            peer_regions = {}
            try:
                for name, handle in handles.items():
                    if (
                        handle.provider == AllocationProvider.CUDA_IPC
                        and handle.device != self._topology[peer].device
                    ):
                        raise RuntimeError(
                            f"IPC handle device does not match rank {peer} topology"
                        )
                    peer_regions[name] = self.endpoint(peer).import_remote(handle)
            except Exception:
                for remote in peer_regions.values():
                    remote.close()
                raise
            remote_regions[peer] = peer_regions
        table = PeerRegionTable(
            self,
            local_regions,
            remote_regions,
            strictly_ordered,
        )
        self._region_tables[tag] = table
        return table

    def device_library(
        self,
        specialize: bool = False,
        *,
        generated: bool = True,
        exclusive: bool = False,
        gpu_resident_queues: bool | None = None,
    ):
        """The Tier-1 device library for this world's architecture.

        By default this returns the compiler-generated mixed RDMA+P2P library
        from the digest-keyed cache. It is geography-independent and can be
        requested before endpoints connect. `specialize=True` requires a
        connected world and bakes its complete peer-rank -> backend geography
        into the artifact. A uniform world naturally needs only one backend;
        a mixed world emits peer-specialized bindings over both backends.

        `generated=False` is the temporary handwritten RDMA retirement oracle.

        `gpu_resident_queues` (default: `self.gpu_resident_queues`, so False
        before connecting) builds the rdma doorbell as a GPU-scope release.
        """
        self._ensure_open()
        arch = normalize_arch(
            os.environ.get("ABOLETH_RDMA_ARCH"), self.device
        )
        if not generated:
            if specialize:
                raise ValueError(
                    "specialize=True requires the generated library"
                )
            if self._endpoints_per_peer != 1:
                raise ValueError(
                    "endpoint fan-out requires the generated library "
                    "(the handwritten header indexes one slot per peer)"
                )
            return _legacy_rdma_device_library(arch)

        from .device_library import DEFAULT_TRANSPORTS, cached_device_library

        if arch.startswith("gfx"):
            from .device_library import HIP_DEFAULT_TRANSPORTS
            DEFAULT_TRANSPORTS = HIP_DEFAULT_TRANSPORTS
            if any(ep.transport == Transport.IONIC for ep in self.endpoints.values()):
                DEFAULT_TRANSPORTS = (*HIP_DEFAULT_TRANSPORTS, "ionic")
        transports = DEFAULT_TRANSPORTS
        peer_transports = None
        if specialize:
            geography = self.transport_geography
            if not geography:
                raise RuntimeError(
                    "connect World before requesting a specialized library"
                )
            selected = {transport for _, transport in geography}
            unsupported = selected - set(DEFAULT_TRANSPORTS)
            if unsupported:
                raise NotImplementedError(
                    "generated Tier-1 specialization does not yet support "
                    + ", ".join(sorted(unsupported))
                )
            transports = tuple(
                name
                for name in DEFAULT_TRANSPORTS
                if name in selected
            )
            # A uniform geography has no remaining transport choice, so one
            # shared per-backend artifact is already fully specialized. Mixed
            # worlds carry the exact peer mapping in both code and cache key.
            if len(transports) > 1:
                peer_transports = geography
        if gpu_resident_queues is None:
            gpu_resident_queues = self.gpu_resident_queues
        elif gpu_resident_queues and not self.gpu_resident_queues:
            raise ValueError(
                "gpu_resident_queues=True needs every connected rdma "
                "endpoint's queues in GPU memory")
        return cached_device_library(
            arch,
            transports=transports,
            peer_transports=peer_transports,
            exclusive=exclusive,
            strict_registrations=self.strict_registrations,
            endpoints_per_peer=self._endpoints_per_peer,
            gpu_resident_queues=gpu_resident_queues,
        )

    def compilation(
        self,
        *,
        peers: Iterable[int] | None = None,
        completion_interval: int | None = None,
        pattern_plan=None,
        kernel: str | None = None,
    ):
        """Compile one backend-homogeneous peer partition.

        Uniform worlds need no `peers` argument. Mixed worlds require the
        caller/compiler to partition dynamic work first; the resulting scope
        lowers directly to that partition's backend with no transport branch.
        """
        self._ensure_open()
        if not self._endpoints:
            raise RuntimeError("connect World before compiling")
        selected_peers = (
            tuple(dict.fromkeys(int(peer) for peer in peers))
            if peers is not None
            else tuple(self._endpoints)
        )
        if not selected_peers:
            raise ValueError("compilation peer partition cannot be empty")
        selected_endpoints = tuple(
            self.endpoint(peer) for peer in selected_peers
        )
        selected_transports = {
            endpoint.transport for endpoint in selected_endpoints
        }
        if len(selected_transports) != 1:
            raise RuntimeError(
                "mixed-transport compilation requires a backend-homogeneous "
                "peer partition; split peers by World.transport_geography"
            )
        queue_capacity = min(
            endpoint.capacity
            for peer in selected_peers
            for endpoint in self._endpoints[peer]
        )
        return selected_endpoints[0].compilation(
            completion_interval=completion_interval,
            pattern_plan=pattern_plan,
            kernel=kernel,
            queue_capacity=queue_capacity,
            strict_registrations=self.strict_registrations,
            # One kernel may post to any endpoint of the partition, so the
            # doorbell narrows to GPU scope only when all of them allow it.
            gpu_resident_queues=all(
                endpoint.gpu_resident_queues
                for peer in selected_peers
                for endpoint in self._endpoints[peer]
            ),
            endpoint_profile=(
                None if self._profiles is None else self._profiles[1].identity
            ),
        )

    @property
    def strict_registrations(self) -> bool:
        """Every live region exchange imported only strictly ordered memory.

        Strong signals compile to a fenced write only when this holds, so a
        kernel must publish only into regions exchanged through this World.
        With no live exchange it is False and signals stay atomic.
        """
        tables = [
            table for table in self._region_tables.values() if not table.closed
        ]
        return bool(tables) and all(table.strictly_ordered for table in tables)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Aboleth World is closed")

    def close(self) -> None:
        if self._closed:
            return
        first_error = None
        for table in reversed(tuple(self._region_tables.values())):
            try:
                table.close()
            except Exception as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
        self._region_tables.clear()
        for slots in reversed(tuple(self._endpoints.values())):
            for endpoint in reversed(slots):
                try:
                    endpoint.close()
                except Exception as error:  # noqa: BLE001
                    if first_error is None:
                        first_error = error
        self._endpoints.clear()
        self._topology.clear()
        if self._owns_runtime:
            try:
                self.runtime.close()
            except Exception as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
        self._closed = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(f"Aboleth World cleanup also failed: {cleanup_error}")
