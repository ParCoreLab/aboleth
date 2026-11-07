from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass
from enum import Enum, IntEnum

import torch
if torch.version.hip:
    from .hip import driver as cuda_driver, runtime as cudart
else:
    from cuda.bindings import driver as cuda_driver
    from cuda.bindings import runtime as cudart

from .compiler import compilation as compiler_compilation


_CUDA_IPC_HANDLE_BYTES = 64


def _cuda_ipc_handle_bytes(handle) -> bytes:
    # CUDA Python 13 keeps IPC structs opaque; getPtr also works with the
    # older bindings. Copy the fixed CUDA ABI without accessing reserved.
    return ctypes.string_at(handle.getPtr(), _CUDA_IPC_HANDLE_BYTES)


def _cuda_ipc_handle_from_bytes(handle_type, data: bytes):
    if len(data) != _CUDA_IPC_HANDLE_BYTES:
        raise ValueError("CUDA IPC handles must contain exactly 64 bytes")
    handle = handle_type()
    ctypes.memmove(handle.getPtr(), data, _CUDA_IPC_HANDLE_BYTES)
    return handle


class Transport(str, Enum):
    AUTO = "auto"
    P2P = "p2p"
    PROXY = "proxy"
    RDMA = "rdma"
    EFA = "efa"
    IONIC = "ionic"


class Provider(IntEnum):
    """Transport providers a build can carry.

    A bit is set in :func:`provider_caps` only when the provider was compiled
    in, its shared library resolved at import time, and the machine presents a
    matching device. One binary therefore runs on hosts with different
    hardware; ask before choosing a transport instead of catching the failure.
    """

    VERBS = 1 << 0
    MLX5 = 1 << 1
    EFA = 1 << 2
    LIBFABRIC = 1 << 3
    CUDA = 1 << 4
    HIP = 1 << 5
    IONIC = 1 << 6


def provider_caps() -> int:
    from . import _runtime

    return _runtime.provider_caps()


def available_transports() -> tuple[Transport, ...]:
    """The transports this host can actually run, most capable first."""
    caps = provider_caps()
    available = [Transport.P2P]
    if caps & Provider.IONIC:
        available.append(Transport.IONIC)
    if caps & Provider.MLX5:
        available.append(Transport.RDMA)
    if caps & Provider.VERBS:
        available.append(Transport.PROXY)
    if caps & Provider.EFA:
        available.append(Transport.EFA)
    return tuple(available)


class AllocationProvider(IntEnum):
    IB_VERBS = 1
    CUDA_IPC = 2
    CUDA_VMM = 3


_CONNECTION_TRANSPORT_TO_NATIVE = {
    Transport.RDMA: 1,
    Transport.PROXY: 2,
    Transport.EFA: 3,
    Transport.IONIC: 4,
}
_CONNECTION_TRANSPORT_FROM_NATIVE = {
    value: key
    for key, value in _CONNECTION_TRANSPORT_TO_NATIVE.items()
}


def connection_info_from_bytes(data: bytes):
    """The endpoint connection info a peer's wire bytes describe.

    The bytes are the runtime's own `AbolethConnectionInfo`, so this is a
    size-checked copy rather than a parse: one layout, declared once in C,
    is what travels between ranks.
    """
    from . import _runtime

    return _runtime.NativeConnectionInfo.from_bytes(data)


def connection_transport(info) -> Transport:
    """The transport a connection info was opened for."""
    try:
        return _CONNECTION_TRANSPORT_FROM_NATIVE[info.transport]
    except KeyError as error:
        raise ValueError(
            f"unknown native endpoint transport {info.transport}"
        ) from error


@dataclass(frozen=True)
class RemoteHandle:
    address: int
    length: int
    rkey: int
    peer: int
    generation: int

    _FORMAT = "<QQIIQ"

    def to_bytes(self) -> bytes:
        return struct.pack(
            self._FORMAT,
            self.address,
            self.length,
            self.rkey,
            self.peer,
            self.generation,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "RemoteHandle":
        if len(data) != struct.calcsize(cls._FORMAT):
            raise ValueError("invalid Aboleth remote handle size")
        return cls(*struct.unpack(cls._FORMAT, data))


@dataclass(frozen=True)
class AllocationHandle:
    provider: AllocationProvider
    peer: int
    device: int
    generation: int
    address: int
    allocation_length: int
    allocation_size: int
    mapping_offset: int
    offset: int
    length: int
    rkey: int = 0
    ipc_handle: bytes = b""
    ref_counter_handle: bytes = b""
    ref_counter_offset: int = 0
    event_handle: bytes = b""
    event_sync_required: bool = False
    # The verbs registration carries no relaxed ordering, so a fenced write may
    # publish into it. Travels as bit 0 of the header's flags field, which
    # older encoders zero: an unknown registration reads as relaxed.
    strictly_ordered: bool = False
    abi_version: int = 2

    _HEADER = struct.Struct("<BBHIiQQQQQQQIHHHQB")
    _STRICTLY_ORDERED = 0x1

    def __post_init__(self) -> None:
        if self.abi_version != 2:
            raise ValueError(
                "unsupported Aboleth allocation-handle version"
            )
        if self.peer < 0 or self.device < 0:
            raise ValueError(
                "allocation handle requires non-negative peer/device"
            )
        if (
            self.allocation_length < 0
            or self.allocation_size < 0
            or self.mapping_offset < 0
            or self.offset < 0
            or self.length < 0
            or self.offset + self.length
            > self.allocation_length
        ):
            raise ValueError(
                "allocation handle region is outside its allocation"
            )
        if (
            self.mapping_offset + self.allocation_length
            > self.allocation_size
        ):
            raise ValueError(
                "registered buffer exceeds CUDA allocation range"
            )
        if self.provider == AllocationProvider.CUDA_IPC:
            if not self.ipc_handle:
                raise ValueError(
                    "CUDA IPC allocation requires a memory handle"
                )
            if self.rkey != 0:
                raise ValueError(
                    "CUDA IPC allocation cannot carry an rkey"
                )
        elif self.provider == AllocationProvider.IB_VERBS:
            if (
                self.ipc_handle
                or self.ref_counter_handle
                or self.event_handle
            ):
                raise ValueError(
                    "verbs allocation cannot carry a CUDA IPC handle"
                )

    def to_bytes(self) -> bytes:
        header = self._HEADER.pack(
            self.abi_version,
            int(self.provider),
            self._STRICTLY_ORDERED if self.strictly_ordered else 0,
            self.peer,
            self.device,
            self.generation,
            self.address,
            self.allocation_length,
            self.allocation_size,
            self.mapping_offset,
            self.offset,
            self.length,
            self.rkey,
            len(self.ipc_handle),
            len(self.ref_counter_handle),
            len(self.event_handle),
            self.ref_counter_offset,
            int(self.event_sync_required),
        )
        return (
            header
            + self.ipc_handle
            + self.ref_counter_handle
            + self.event_handle
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "AllocationHandle":
        if len(data) < cls._HEADER.size:
            raise ValueError(
                "invalid Aboleth allocation-handle size"
            )
        (
            abi_version,
            provider,
            flags,
            peer,
            device,
            generation,
            address,
            allocation_length,
            allocation_size,
            mapping_offset,
            offset,
            length,
            rkey,
            ipc_length,
            ref_length,
            event_length,
            ref_counter_offset,
            event_sync_required,
        ) = cls._HEADER.unpack_from(data)
        expected = (
            cls._HEADER.size
            + ipc_length
            + ref_length
            + event_length
        )
        if len(data) != expected:
            raise ValueError(
                "invalid Aboleth allocation payload size"
            )
        cursor = cls._HEADER.size
        ipc_handle = data[cursor:cursor + ipc_length]
        cursor += ipc_length
        ref_counter_handle = data[
            cursor:cursor + ref_length
        ]
        cursor += ref_length
        event_handle = data[cursor:cursor + event_length]
        resolved = AllocationProvider(provider)
        return cls(
            provider=resolved,
            peer=peer,
            device=device,
            generation=generation,
            address=address,
            allocation_length=allocation_length,
            allocation_size=allocation_size,
            mapping_offset=mapping_offset,
            offset=offset,
            length=length,
            rkey=rkey,
            ipc_handle=(
                ipc_handle
                if resolved == AllocationProvider.CUDA_IPC
                else b""
            ),
            ref_counter_handle=(
                ref_counter_handle
                if resolved == AllocationProvider.CUDA_IPC
                else b""
            ),
            ref_counter_offset=ref_counter_offset,
            event_handle=(
                event_handle
                if resolved == AllocationProvider.CUDA_IPC
                else b""
            ),
            event_sync_required=bool(event_sync_required),
            strictly_ordered=bool(flags & cls._STRICTLY_ORDERED),
            abi_version=abi_version,
        )


class RegisteredBuffer:
    def __init__(
        self,
        runtime: "Runtime",
        tensor: torch.Tensor,
        generation: int,
        register_network: bool,
    ):
        if not tensor.is_cuda:
            raise ValueError("Aboleth buffers must reside on a CUDA device")
        if not tensor.is_contiguous():
            raise ValueError("Aboleth buffers must be contiguous")
        self._runtime = runtime
        self._tensor = tensor
        self._generation = generation
        self._native = (
            runtime._ensure_native().register_buffer(
                self.device,
                self.address,
                self.length,
                generation,
            )
            if register_network
            else None
        )
        self._closed = False

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def address(self) -> int:
        return self._tensor.data_ptr()

    @property
    def length(self) -> int:
        return self._tensor.numel() * self._tensor.element_size()

    @property
    def device(self) -> int:
        index = self._tensor.device.index
        if index is None:
            raise RuntimeError("CUDA buffer has no device index")
        return index

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def lkey(self) -> int:
        self._ensure_open()
        return 0 if self._native is None else self._native.lkey

    @property
    def rkey(self) -> int:
        self._ensure_open()
        return 0 if self._native is None else self._native.rkey

    @property
    def strictly_ordered(self) -> bool:
        """The network registration carries no relaxed ordering."""
        self._ensure_open()
        return self._native is not None and self._native.strictly_ordered

    def export(self, peer: int) -> RemoteHandle:
        self._ensure_open()
        return RemoteHandle(
            address=self.address,
            length=self.length,
            rkey=self.rkey,
            peer=peer,
            generation=self.generation,
        )

    def export_allocation(
        self,
        peer: int,
        *,
        provider: AllocationProvider,
        offset: int = 0,
        length: int | None = None,
    ) -> AllocationHandle:
        self._ensure_open()
        if length is None:
            length = self.length - offset
        _check_slice(self.length, offset, length)
        if provider == AllocationProvider.IB_VERBS:
            return AllocationHandle(
                provider=provider,
                peer=int(peer),
                device=self.device,
                generation=self.generation,
                address=self.address,
                allocation_length=self.length,
                allocation_size=self.length,
                mapping_offset=0,
                offset=offset,
                length=length,
                rkey=self.rkey,
                strictly_ordered=self.strictly_ordered,
            )
        if provider != AllocationProvider.CUDA_IPC:
            raise NotImplementedError(
                f"allocation provider {provider.name} is unavailable"
            )
        _cuda_check(
            cudart.cudaSetDevice(self.device),
            f"cudaSetDevice({self.device})",
        )
        torch.cuda.synchronize(self.device)
        range_result = cuda_driver.cuMemGetAddressRange(
            self.address
        )
        _driver_check(
            range_result,
            "cuMemGetAddressRange",
        )
        allocation_base = int(range_result[1])
        allocation_size = int(range_result[2])
        if not (
            allocation_base
            <= self.address
            < allocation_base + allocation_size
        ):
            raise RuntimeError(
                "CUDA allocation range does not contain buffer"
            )
        ipc_result = cudart.cudaIpcGetMemHandle(
            allocation_base
        )
        _cuda_check(ipc_result, "cudaIpcGetMemHandle")
        return AllocationHandle(
            provider=provider,
            peer=int(peer),
            device=self.device,
            generation=self.generation,
            address=self.address,
            allocation_length=self.length,
            allocation_size=allocation_size,
            mapping_offset=self.address - allocation_base,
            offset=offset,
            length=length,
            ipc_handle=_cuda_ipc_handle_bytes(ipc_result[1]),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Aboleth registered buffer is closed")

    def close(self) -> None:
        if self._closed:
            return
        first_error = None
        try:
            torch.cuda.synchronize(self.device)
        except Exception as error:
            first_error = error
        if self._native is not None:
            try:
                self._native.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        self._closed = True
        self._runtime._discard_buffer(self)
        if first_error is not None:
            raise first_error


@dataclass(frozen=True)
class LocalBuffer:
    address: int
    length: int
    lkey: int
    device: int
    generation: int
    _owner: RegisteredBuffer

    def slice(self, offset: int, length: int) -> "LocalBuffer":
        _check_slice(self.length, offset, length)
        return LocalBuffer(
            address=self.address + offset,
            length=length,
            lkey=self.lkey,
            device=self.device,
            generation=self.generation,
            _owner=self._owner,
        )


@dataclass
class _IpcMappingEntry:
    mapping_base: int
    local_device: int
    exporter_device: int
    generations: dict[int, int]
    references: int = 0


class ImportedAllocation:
    def __init__(
        self,
        runtime: "Runtime",
        key: tuple[int, int, bytes],
        *,
        mapping_offset: int,
        peer: int,
        generation: int,
        address: int,
        length: int,
    ):
        self._runtime = runtime
        self._key = key
        self._mapping_offset = int(mapping_offset)
        self.peer = int(peer)
        self.generation = int(generation)
        self.address = int(address)
        self.length = int(length)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "Aboleth imported allocation is closed"
            )
        entry = self._runtime._ipc_mappings.get(self._key)
        if (
            entry is None
            or entry.generations.get(self._mapping_offset)
            != self.generation
        ):
            raise RuntimeError(
                "Aboleth imported allocation generation is stale"
            )

    @property
    def device(self) -> int:
        self.ensure_open()
        return self._runtime._ipc_mappings[
            self._key
        ].local_device

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime._release_import(self)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class RemoteBuffer:
    address: int
    length: int
    rkey: int
    peer: int
    generation: int
    _owner: RegisteredBuffer | ImportedAllocation | None = None

    def slice(self, offset: int, length: int) -> "RemoteBuffer":
        self.ensure_open()
        _check_slice(self.length, offset, length)
        return RemoteBuffer(
            address=self.address + offset,
            length=length,
            rkey=self.rkey,
            peer=self.peer,
            generation=self.generation,
            _owner=self._owner,
        )

    def export(self) -> RemoteHandle:
        self.ensure_open()
        return RemoteHandle(
            address=self.address,
            length=self.length,
            rkey=self.rkey,
            peer=self.peer,
            generation=self.generation,
        )

    def ensure_open(self) -> None:
        if isinstance(self._owner, ImportedAllocation):
            self._owner.ensure_open()

    def close(self) -> None:
        if isinstance(self._owner, ImportedAllocation):
            self._owner.close()



class Endpoint:
    def __init__(
        self,
        pair: "EndpointPair",
        device: int,
        peer: int,
        transport: Transport,
        descriptor: torch.Tensor,
    ):
        self._pair = pair
        self._device = device
        self._peer = peer
        self._transport = transport
        self._descriptor = descriptor

    @property
    def device(self) -> int:
        return self._device

    @property
    def peer(self) -> int:
        return self._peer

    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def device_handle(self) -> torch.Tensor:
        self._pair._ensure_open()
        return self._descriptor

    def local_buffer(
        self,
        buffer: RegisteredBuffer,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> LocalBuffer:
        self._pair._ensure_open()
        self._runtime_validate_buffer(buffer)
        if buffer.device != self._device:
            raise ValueError(
                f"local buffer is on device {buffer.device}, "
                f"not endpoint device {self._device}"
            )
        if length is None:
            length = buffer.length - offset
        _check_slice(buffer.length, offset, length)
        return LocalBuffer(
            address=buffer.address + offset,
            length=length,
            lkey=buffer.lkey,
            device=buffer.device,
            generation=buffer.generation,
            _owner=buffer,
        )

    def remote_buffer(
        self,
        buffer: RegisteredBuffer,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> RemoteBuffer:
        self._pair._ensure_open()
        self._runtime_validate_buffer(buffer)
        if buffer.device != self._peer:
            raise ValueError(
                f"remote buffer is on device {buffer.device}, "
                f"not endpoint peer {self._peer}"
            )
        if length is None:
            length = buffer.length - offset
        _check_slice(buffer.length, offset, length)
        return RemoteBuffer(
            address=buffer.address + offset,
            length=length,
            rkey=buffer.rkey,
            peer=self._peer,
            generation=buffer.generation,
            _owner=buffer,
        )

    def _runtime_validate_buffer(
        self,
        buffer: RegisteredBuffer,
    ) -> None:
        self._pair._runtime._validate_buffer(buffer)
        buffer._ensure_open()

    def import_remote(self, handle: RemoteHandle) -> RemoteBuffer:
        self._pair._ensure_open()
        return self._pair._runtime.import_remote_buffer(
            self._peer,
            handle,
        )

    def compilation(
        self,
        *,
        completion_interval: int | None = None,
        pattern_plan=None,
        kernel: str | None = None,
        queue_capacity: int | None = None,
        strict_registrations: bool = False,
    ):
        return compiler_compilation(
            self._transport.value,
            completion_interval=completion_interval,
            pattern_plan=pattern_plan,
            kernel=kernel,
            queue_capacity=queue_capacity,
            strict_registrations=strict_registrations,
        )


class LocalEndpoint:
    def __init__(
        self,
        runtime: "Runtime",
        device: int,
        transport: Transport,
        descriptor: torch.Tensor,
        native,
    ):
        self._runtime = runtime
        self._device = device
        self._transport = transport
        self._descriptor = descriptor
        self._native = native
        self._peer: int | None = None
        self._peer_device: int | None = None
        self._closed = False

    @property
    def device(self) -> int:
        return self._device

    @property
    def peer(self) -> int | None:
        return self._peer

    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def device_handle(self) -> torch.Tensor:
        self._ensure_open()
        return self._descriptor

    @property
    def connection_info(self):
        """This endpoint's address on the fabric, as the runtime holds it.

        `to_bytes()` is what a peer needs and `connection_info_from_bytes`
        is how it reads them back.
        """
        self._ensure_open()
        if self._native is None:
            raise RuntimeError(
                "P2P endpoints do not have verbs connection info"
            )
        return self._native.connection_info()

    def connect(
        self,
        remote,
        *,
        peer: int,
    ) -> None:
        from . import _runtime

        self._ensure_open()
        if not isinstance(remote, _runtime.NativeConnectionInfo):
            raise TypeError(
                "expected an Aboleth endpoint connection info"
            )
        if connection_transport(remote) != self._transport:
            raise ValueError(
                "remote endpoint transport does not match"
            )
        self._native.connect(remote)
        self._peer = int(peer)

    def connect_p2p(
        self,
        *,
        peer: int,
        peer_device: int,
    ) -> None:
        self._ensure_open()
        if self._transport != Transport.P2P:
            raise RuntimeError(
                "connect_p2p requires a P2P endpoint"
            )
        peer_device = int(peer_device)
        if not torch.cuda.can_device_access_peer(
            self._device,
            peer_device,
        ):
            raise RuntimeError(
                f"GPU {self._device} cannot access GPU "
                f"{peer_device}"
            )
        self._peer = int(peer)
        self._peer_device = peer_device

    def local_buffer(
        self,
        buffer: RegisteredBuffer,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> LocalBuffer:
        self._ensure_open()
        self._runtime._validate_buffer(buffer)
        if buffer.device != self._device:
            raise ValueError(
                f"local buffer is on device {buffer.device}, "
                f"not endpoint device {self._device}"
            )
        if length is None:
            length = buffer.length - offset
        _check_slice(buffer.length, offset, length)
        return LocalBuffer(
            address=buffer.address + offset,
            length=length,
            lkey=buffer.lkey,
            device=buffer.device,
            generation=buffer.generation,
            _owner=buffer,
        )

    def import_remote(
        self,
        handle: AllocationHandle | RemoteHandle,
    ) -> RemoteBuffer:
        self._ensure_open()
        if self._peer is None:
            raise RuntimeError(
                "local endpoint is not connected"
            )
        if isinstance(handle, AllocationHandle):
            if (
                self._transport == Transport.P2P
                and handle.provider
                != AllocationProvider.CUDA_IPC
            ):
                raise ValueError(
                    "P2P endpoint requires a CUDA IPC allocation"
                )
            if (
                self._transport == Transport.P2P
                and self._peer_device is not None
                and handle.device != self._peer_device
            ):
                raise ValueError(
                    "IPC handle device does not match connected peer"
                )
            if (
                self._transport != Transport.P2P
                and handle.provider
                != AllocationProvider.IB_VERBS
            ):
                raise ValueError(
                    "verbs endpoint requires an IB allocation"
                )
        return self._runtime.import_remote_buffer(
            self._peer,
            handle,
        )

    @property
    def capacity(self) -> int:
        self._ensure_open()
        if self._native is None:
            return (1 << 31) - 1
        return int(self._native.capacity)

    @property
    def max_inline_bytes(self) -> int:
        self._ensure_open()
        if self._native is None:
            return 0
        return int(self._native.max_inline_bytes)

    @property
    def submit_mode(self) -> str:
        """Who performs the final submission action, resolved at bring-up.

        ``"direct"``: the GPU submits to the adapter itself. ``"staged"``:
        the GPU builds and publishes work and a host doorbell agent performs
        the one MMIO store the platform withholds from it. ``"proxy"``: the
        host builds and posts the work from GPU-written commands. P2P
        endpoints report ``"direct"`` — the GPU stores payloads itself.
        """
        self._ensure_open()
        if self._native is None:
            return "direct"
        return str(self._native.submit_mode)

    @property
    def gpu_resident_queues(self) -> bool:
        """Every queue and doorbell record of this endpoint is in GPU memory.

        Then a GPU-scope release publishes a submission: the NIC reads GPU
        memory through the GPU's L2, where that release has already put the
        commands. P2P endpoints have no queue and report False.
        """
        self._ensure_open()
        return self._native is not None and self._native.gpu_resident_queues

    def state(self) -> tuple[int, int, int]:
        self._ensure_open()
        if self._native is None:
            return (0, 0, 0)
        torch.cuda.synchronize(self._device)
        return tuple(self._native.state())

    def compilation(
        self,
        *,
        completion_interval: int | None = None,
        pattern_plan=None,
        kernel: str | None = None,
        queue_capacity: int | None = None,
        strict_registrations: bool = False,
        gpu_resident_queues: bool | None = None,
        endpoint_profile: str | None = None,
    ):
        """`gpu_resident_queues` defaults to this endpoint's own placement."""
        if gpu_resident_queues is None:
            gpu_resident_queues = self.gpu_resident_queues
        return compiler_compilation(
            self._transport.value,
            completion_interval=completion_interval,
            pattern_plan=pattern_plan,
            kernel=kernel,
            queue_capacity=queue_capacity,
            strict_registrations=strict_registrations,
            gpu_resident_queues=gpu_resident_queues,
            endpoint_profile=endpoint_profile,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "Aboleth local endpoint is closed"
            )

    def close(self) -> None:
        if self._closed:
            return
        first_error = None
        try:
            torch.cuda.synchronize(self._device)
        except Exception as error:
            first_error = error
        if self._native is not None:
            try:
                self._native.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        self._closed = True
        self._runtime._discard_local_endpoint(self)
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "LocalEndpoint":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(
                "Aboleth local-endpoint cleanup also "
                f"failed: {cleanup_error}"
            )


class EndpointPair:
    def __init__(
        self,
        runtime: "Runtime",
        transport: Transport,
        device0: int,
        device1: int,
        descriptor0: torch.Tensor,
        descriptor1: torch.Tensor,
        native_pair,
        peer_access_enabled: tuple[bool, bool],
    ):
        self._runtime = runtime
        self._transport = transport
        self._native_pair = native_pair
        self._peer_access_enabled = peer_access_enabled
        self._closed = False
        self.endpoints = (
            Endpoint(
                self,
                device0,
                device1,
                transport,
                descriptor0,
            ),
            Endpoint(
                self,
                device1,
                device0,
                transport,
                descriptor1,
            ),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Aboleth endpoint pair is closed")

    def queue_state(self) -> tuple[int, int, int, int]:
        self._ensure_open()
        if self._native_pair is None:
            return (0, 0, 0, 0)
        return tuple(self._native_pair.queue_state())

    def status(self) -> tuple[int, int]:
        self._ensure_open()
        if self._native_pair is None:
            return (0, 0)
        torch.cuda.synchronize(self.endpoints[0].device)
        torch.cuda.synchronize(self.endpoints[1].device)
        return tuple(self._native_pair.status())

    def completion_state_addresses(self) -> tuple[int, int]:
        self._ensure_open()
        if self._native_pair is None:
            return (0, 0)
        return tuple(
            self._native_pair.completion_state_addresses()
        )

    def completion_state(self, endpoint_index: int) -> tuple[int, int, int, int]:
        self._ensure_open()
        if self._native_pair is None:
            return (0, 0, 0, 0)
        return tuple(
            self._native_pair.completion_state(endpoint_index)
        )

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.endpoints[0].device)
        torch.cuda.synchronize(self.endpoints[1].device)
        if self._native_pair is not None:
            self._native_pair.close()
        for source, destination, enabled in (
            (
                self.endpoints[0].device,
                self.endpoints[0].peer,
                self._peer_access_enabled[0],
            ),
            (
                self.endpoints[1].device,
                self.endpoints[1].peer,
                self._peer_access_enabled[1],
            ),
        ):
            if enabled:
                _disable_peer_access(source, destination)
        self._closed = True
        self._runtime._discard_pair(self)

    def __enter__(self) -> "EndpointPair":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(
                "Aboleth endpoint-pair cleanup also failed: "
                f"{cleanup_error}"
            )


class EndpointTable:
    def __init__(
        self,
        device: int,
        endpoints: dict[int, Endpoint],
    ):
        self._device = device
        self._endpoints = dict(endpoints)
        size = max((device, *endpoints.keys())) + 1
        self._device_handles = torch.zeros(
            size,
            dtype=torch.uint64,
            device=f"cuda:{device}",
        )
        for peer, endpoint in endpoints.items():
            self._device_handles[peer] = (
                endpoint.device_handle.data_ptr()
            )

    @property
    def device(self) -> int:
        return self._device

    @property
    def device_handles(self) -> torch.Tensor:
        return self._device_handles

    @property
    def peers(self) -> tuple[int, ...]:
        return tuple(sorted(self._endpoints))

    def endpoint(self, peer: int) -> Endpoint:
        try:
            return self._endpoints[peer]
        except KeyError as error:
            raise KeyError(
                f"device {self._device} has no endpoint for peer {peer}"
            ) from error


class EndpointMesh:
    def __init__(
        self,
        runtime: "Runtime",
        devices: tuple[int, ...],
        pairs: list[EndpointPair],
    ):
        self._runtime = runtime
        self._pairs = pairs
        endpoints: dict[int, dict[int, Endpoint]] = {
            device: {} for device in devices
        }
        for pair in pairs:
            forward, reverse = pair.endpoints
            endpoints[forward.device][forward.peer] = forward
            endpoints[reverse.device][reverse.peer] = reverse
        self.tables = {
            device: EndpointTable(
                device,
                endpoints[device],
            )
            for device in devices
        }
        self._closed = False

    @property
    def runtime(self) -> "Runtime":
        return self._runtime

    def endpoint(self, device: int, peer: int) -> Endpoint:
        if self._closed:
            raise RuntimeError("Aboleth endpoint mesh is closed")
        return self.tables[device].endpoint(peer)

    @property
    def pairs(self) -> tuple[EndpointPair, ...]:
        if self._closed:
            raise RuntimeError("Aboleth endpoint mesh is closed")
        return tuple(self._pairs)

    def close(self) -> None:
        if self._closed:
            return
        for pair in reversed(self._pairs):
            pair.close()
        self._closed = True

    def __enter__(self) -> "EndpointMesh":
        if self._closed:
            raise RuntimeError("Aboleth endpoint mesh is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(
                "Aboleth endpoint-mesh cleanup also failed: "
                f"{cleanup_error}"
            )


class Runtime:
    def __init__(
        self,
        *,
        device: int | None = None,
        hca: str | None = None,
    ):
        if device is None and hca is not None:
            raise ValueError(
                "an explicit HCA requires a rank-local device"
            )
        self._device = (
            None if device is None else int(device)
        )
        self._hca = hca
        self._native = None
        self._next_generation = 1
        self._buffers: list[RegisteredBuffer] = []
        self._local_endpoints: list[LocalEndpoint] = []
        self._pairs: list[EndpointPair] = []
        self._ipc_mappings: dict[
            tuple[int, int, bytes],
            _IpcMappingEntry,
        ] = {}
        self._ipc_generation_watermark: dict[
            tuple[int, int, bytes, int],
            int,
        ] = {}
        self._imports: list[ImportedAllocation] = []
        self._closed = False

    @property
    def device(self) -> int | None:
        return self._device

    @property
    def hca(self) -> str:
        self._ensure_open()
        return self._ensure_native().hca_name

    @property
    def rdma_port(self) -> int:
        self._ensure_open()
        return self._ensure_native().port

    @property
    def gid_index(self) -> int:
        self._ensure_open()
        return self._ensure_native().gid_index

    @property
    def lid(self) -> int:
        self._ensure_open()
        return self._ensure_native().lid

    @property
    def link_layer(self) -> str:
        self._ensure_open()
        return {
            1: "infiniband",
            2: "ethernet",
        }.get(
            self._ensure_native().link_layer,
            "unspecified",
        )

    def register(
        self,
        tensor: torch.Tensor,
        *,
        network: bool = True,
    ) -> RegisteredBuffer:
        self._ensure_open()
        buffer = RegisteredBuffer(
            self,
            tensor,
            self._next_generation,
            bool(network),
        )
        self._next_generation += 1
        self._buffers.append(buffer)
        return buffer

    def create_local_endpoint(
        self,
        *,
        transport: str | Transport,
        device: int | None = None,
        qp_depth: int = 0,
        max_inline: int = 0,
        hca: str | None = None,
        port: int = 0,
        gid_index: int = -1,
    ) -> LocalEndpoint:
        """The endpoint, and the queue pair behind it.

        `qp_depth` and `max_inline` are requests the direct (RDMA) transport
        passes to the adapter, which may grant more or less: read
        `capacity` and `max_inline_bytes` back. `hca`, `port` and
        `gid_index` state what this runtime already opened and are refused
        when they disagree with it. Defaults leave every choice to the
        runtime and its environment; see `docs/transport-control.md`.
        """
        self._ensure_open()
        if transport == "ibrc":
            resolved = Transport.PROXY
        else:
            resolved = Transport(transport)
        if resolved not in (
            Transport.P2P,
            Transport.RDMA,
            Transport.PROXY,
            Transport.EFA,
            Transport.IONIC,
        ):
            raise ValueError(
                "rank-local endpoints require p2p, rdma, proxy, or efa"
            )
        if device is None:
            if self._device is None:
                raise ValueError(
                    "device is required for a non-rank-local runtime"
                )
            device = self._device
        device = int(device)
        if self._device is not None and device != self._device:
            raise ValueError(
                f"endpoint device {device} does not match "
                f"runtime device {self._device}"
            )

        if resolved == Transport.P2P:
            # Peer memory has no queue pair and no fabric address, so a
            # request naming one is a mistake rather than a no-op.
            for name, value, default in (
                ("qp_depth", qp_depth, 0),
                ("max_inline", max_inline, 0),
                ("hca", hca if hca else None, None),
                ("port", port, 0),
                ("gid_index", gid_index, -1),
            ):
                if value != default:
                    raise ValueError(
                        f"{name}={value!r} has no meaning for a p2p "
                        "endpoint, which has no queue pair; see "
                        "docs/transport-control.md"
                    )
            descriptor = torch.empty(
                1,
                dtype=torch.uint8,
                device=f"cuda:{device}",
            )
            native = None
        else:
            descriptor_words = {
                Transport.RDMA: 12,
                Transport.IONIC: 12,
                Transport.PROXY: 5,
                # Nine 64-bit words; see EfaAbi.h.
                Transport.EFA: 9,
            }[resolved]
            descriptor = torch.empty(
                descriptor_words,
                dtype=torch.uint64,
                device=f"cuda:{device}",
            )
            native = self._ensure_native().create_local_endpoint(
                _CONNECTION_TRANSPORT_TO_NATIVE[resolved],
                device,
                descriptor.data_ptr(),
                descriptor.numel() * descriptor.element_size(),
                qp_depth=int(qp_depth),
                max_inline=int(max_inline),
                hca="" if hca is None else str(hca),
                port=int(port),
                gid_index=int(gid_index),
            )
        endpoint = LocalEndpoint(
            self,
            device,
            resolved,
            descriptor,
            native,
        )
        self._local_endpoints.append(endpoint)
        return endpoint

    def export_buffer(
        self,
        buffer: RegisteredBuffer,
        *,
        peer: int,
    ) -> RemoteHandle:
        self._validate_buffer(buffer)
        return buffer.export(peer)

    def export_allocation(
        self,
        buffer: RegisteredBuffer,
        *,
        peer: int,
        provider: AllocationProvider,
        offset: int = 0,
        length: int | None = None,
    ) -> AllocationHandle:
        self._validate_buffer(buffer)
        return buffer.export_allocation(
            peer,
            provider=provider,
            offset=offset,
            length=length,
        )

    def import_remote_buffer(
        self,
        peer: int,
        handle: AllocationHandle | RemoteHandle,
    ) -> RemoteBuffer:
        self._ensure_open()
        if isinstance(handle, RemoteHandle):
            handle = AllocationHandle(
                provider=AllocationProvider.IB_VERBS,
                peer=handle.peer,
                device=handle.peer,
                generation=handle.generation,
                address=handle.address,
                allocation_length=handle.length,
                allocation_size=handle.length,
                mapping_offset=0,
                offset=0,
                length=handle.length,
                rkey=handle.rkey,
            )
        if not isinstance(handle, AllocationHandle):
            raise TypeError(
                "expected an Aboleth AllocationHandle"
            )
        if handle.peer != peer:
            raise ValueError(
                f"remote handle belongs to peer {handle.peer}, "
                f"not requested peer {peer}"
            )
        if handle.provider == AllocationProvider.IB_VERBS:
            return RemoteBuffer(
                address=handle.address + handle.offset,
                length=handle.length,
                rkey=handle.rkey,
                peer=handle.peer,
                generation=handle.generation,
            )
        if handle.provider != AllocationProvider.CUDA_IPC:
            raise NotImplementedError(
                f"allocation provider {handle.provider.name} "
                "is unavailable"
            )
        if self._device is None:
            raise RuntimeError(
                "CUDA IPC import requires a rank-local runtime"
            )
        if not torch.cuda.can_device_access_peer(
            self._device,
            handle.device,
        ):
            raise RuntimeError(
                f"GPU {self._device} cannot import GPU "
                f"{handle.device}"
            )
        key = (
            self._device,
            handle.device,
            bytes(handle.ipc_handle[-64:]),
        )
        watermark_key = (*key, handle.mapping_offset)
        watermark = self._ipc_generation_watermark.get(
            watermark_key,
            0,
        )
        if handle.generation < watermark:
            raise RuntimeError(
                "stale CUDA IPC allocation generation"
            )
        entry = self._ipc_mappings.get(key)
        if entry is None:
            _cuda_check(
                cudart.cudaSetDevice(self._device),
                f"cudaSetDevice({self._device})",
            )
            native_handle = _cuda_ipc_handle_from_bytes(
                cudart.cudaIpcMemHandle_t, handle.ipc_handle[-64:]
            )
            result = cudart.cudaIpcOpenMemHandle(
                native_handle,
                cudart.cudaIpcMemLazyEnablePeerAccess,
            )
            _cuda_check(result, "cudaIpcOpenMemHandle")
            mapping_base = int(result[1])
            if (
                handle.event_sync_required
                and handle.event_handle
            ):
                native_event = _cuda_ipc_handle_from_bytes(
                    cudart.cudaIpcEventHandle_t, handle.event_handle
                )
                event_result = cudart.cudaIpcOpenEventHandle(
                    native_event
                )
                _cuda_check(
                    event_result,
                    "cudaIpcOpenEventHandle",
                )
                event = event_result[1]
                try:
                    _cuda_check(
                        cudart.cudaEventSynchronize(event),
                        "cudaEventSynchronize",
                    )
                finally:
                    _cuda_check(
                        cudart.cudaEventDestroy(event),
                        "cudaEventDestroy",
                    )
            entry = _IpcMappingEntry(
                mapping_base=mapping_base,
                local_device=self._device,
                exporter_device=handle.device,
                generations={},
            )
            self._ipc_mappings[key] = entry
        known_generation = entry.generations.get(
            handle.mapping_offset
        )
        if (
            known_generation is not None
            and known_generation != handle.generation
        ):
            raise RuntimeError(
                "stale CUDA IPC allocation generation"
            )
        entry.generations[
            handle.mapping_offset
        ] = handle.generation
        self._ipc_generation_watermark[watermark_key] = max(
            watermark,
            handle.generation,
        )
        entry.references += 1
        mapped_address = (
            entry.mapping_base + handle.mapping_offset
        )
        owner = ImportedAllocation(
            self,
            key,
            mapping_offset=handle.mapping_offset,
            peer=peer,
            generation=handle.generation,
            address=mapped_address,
            length=handle.allocation_length,
        )
        self._imports.append(owner)
        return RemoteBuffer(
            address=mapped_address + handle.offset,
            length=handle.length,
            rkey=0,
            peer=handle.peer,
            generation=handle.generation,
            _owner=owner,
        )

    def create_endpoint_pair(
        self,
        device0: int | RegisteredBuffer,
        device1: int | RegisteredBuffer,
        *,
        transport: str | Transport = Transport.AUTO,
    ) -> EndpointPair:
        self._ensure_open()
        if self._device is not None:
            raise RuntimeError(
                "rank-local runtimes create one endpoint at a time; "
                "use create_local_endpoint()"
            )
        if isinstance(device0, RegisteredBuffer):
            self._validate_buffer(device0)
            device0 = device0.device
        if isinstance(device1, RegisteredBuffer):
            self._validate_buffer(device1)
            device1 = device1.device
        device0 = int(device0)
        device1 = int(device1)
        if device0 == device1:
            raise ValueError("endpoint pair requires two distinct devices")

        if transport == "ibrc":
            requested = Transport.PROXY
        else:
            requested = Transport(transport)
        resolved = requested
        if requested == Transport.AUTO:
            can_access = (
                torch.cuda.can_device_access_peer(
                    device0,
                    device1,
                )
                and torch.cuda.can_device_access_peer(
                    device1,
                    device0,
                )
            )
            resolved = (
                Transport.P2P if can_access else Transport.RDMA
            )

        enabled = (False, False)
        if resolved == Transport.P2P:
            enabled = (
                _enable_peer_access(
                    device0,
                    device1,
                ),
                _enable_peer_access(
                    device1,
                    device0,
                ),
            )
            descriptor0 = torch.empty(
                1,
                dtype=torch.uint8,
                device=f"cuda:{device0}",
            )
            descriptor1 = torch.empty(
                1,
                dtype=torch.uint8,
                device=f"cuda:{device1}",
            )
            native_pair = None
        elif resolved == Transport.RDMA:
            can_access = (
                torch.cuda.can_device_access_peer(
                    device0,
                    device1,
                )
                and torch.cuda.can_device_access_peer(
                    device1,
                    device0,
                )
            )
            if can_access:
                enabled = (
                    _enable_peer_access(
                        device0,
                        device1,
                    ),
                    _enable_peer_access(
                        device1,
                        device0,
                    ),
                )
            (
                descriptor0,
                descriptor1,
                native_pair,
            ) = _create_rdma_pair(
                self._ensure_native(),
                device0,
                device1,
            )
        elif resolved == Transport.PROXY:
            can_access = (
                torch.cuda.can_device_access_peer(
                    device0,
                    device1,
                )
                and torch.cuda.can_device_access_peer(
                    device1,
                    device0,
                )
            )
            if can_access:
                enabled = (
                    _enable_peer_access(
                        device0,
                        device1,
                    ),
                    _enable_peer_access(
                        device1,
                        device0,
                    ),
                )
            (
                descriptor0,
                descriptor1,
                native_pair,
            ) = _create_proxy_pair(
                self._ensure_native(),
                device0,
                device1,
            )
        elif resolved == Transport.EFA:
            # SRD needs no CUDA peer access; the adapters move the bytes.
            (
                descriptor0,
                descriptor1,
                native_pair,
            ) = _create_efa_pair(
                self._ensure_native(),
                device0,
                device1,
            )
        else:
            raise ValueError(f"unsupported transport {resolved.value}")

        pair = EndpointPair(
            self,
            resolved,
            device0,
            device1,
            descriptor0,
            descriptor1,
            native_pair,
            enabled,
        )
        self._pairs.append(pair)
        return pair

    def create_endpoint_mesh(
        self,
        devices,
        *,
        transport: str | Transport = Transport.AUTO,
    ) -> EndpointMesh:
        self._ensure_open()
        if self._device is not None:
            raise RuntimeError(
                "rank-local runtimes do not create in-process meshes"
            )
        normalized = tuple(dict.fromkeys(int(device) for device in devices))
        if len(normalized) < 2:
            raise ValueError("endpoint mesh requires at least two devices")
        pairs = []
        try:
            for index, device in enumerate(normalized):
                for peer in normalized[index + 1:]:
                    pairs.append(
                        self.create_endpoint_pair(
                            device,
                            peer,
                            transport=transport,
                        )
                    )
        except Exception:
            for pair in reversed(pairs):
                pair.close()
            raise
        return EndpointMesh(self, normalized, pairs)

    def _validate_buffer(self, buffer: RegisteredBuffer) -> None:
        if not isinstance(buffer, RegisteredBuffer):
            raise TypeError("expected an Aboleth RegisteredBuffer")
        if buffer._runtime is not self:
            raise ValueError(
                "buffer belongs to a different Aboleth Runtime"
            )
        buffer._ensure_open()

    def _discard_buffer(self, buffer: RegisteredBuffer) -> None:
        if buffer in self._buffers:
            self._buffers.remove(buffer)

    def _discard_local_endpoint(
        self,
        endpoint: LocalEndpoint,
    ) -> None:
        if endpoint in self._local_endpoints:
            self._local_endpoints.remove(endpoint)

    def _discard_pair(self, pair: EndpointPair) -> None:
        if pair in self._pairs:
            self._pairs.remove(pair)

    def _release_import(
        self,
        imported: ImportedAllocation,
    ) -> None:
        if imported in self._imports:
            self._imports.remove(imported)
        entry = self._ipc_mappings.get(imported._key)
        if entry is None:
            return
        entry.references -= 1
        if entry.references > 0:
            return
        _cuda_check(
            cudart.cudaSetDevice(entry.local_device),
            f"cudaSetDevice({entry.local_device})",
        )
        _cuda_check(
            cudart.cudaIpcCloseMemHandle(
                entry.mapping_base
            ),
            "cudaIpcCloseMemHandle",
        )
        del self._ipc_mappings[imported._key]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Aboleth runtime is closed")

    def _ensure_native(self):
        self._ensure_open()
        if self._native is None:
            from . import _runtime

            self._native = _runtime.NativeRuntime(
                -1 if self._device is None else self._device,
                "" if self._hca is None else self._hca,
            )
        return self._native

    def close(self) -> None:
        if self._closed:
            return
        first_error = None
        for imported in list(reversed(self._imports)):
            try:
                imported.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        self._ipc_generation_watermark.clear()
        for pair in list(reversed(self._pairs)):
            try:
                pair.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        for endpoint in list(
            reversed(self._local_endpoints)
        ):
            try:
                endpoint.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        for buffer in list(reversed(self._buffers)):
            try:
                buffer.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if self._native is not None:
            try:
                self._native.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        self._closed = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "Runtime":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(
                "Aboleth runtime cleanup also failed: "
                f"{cleanup_error}"
            )


def _create_rdma_pair(
    native_runtime,
    device0: int,
    device1: int,
):
    descriptor0 = torch.empty(
        12,
        dtype=torch.uint64,
        device=f"cuda:{device0}",
    )
    descriptor1 = torch.empty(
        12,
        dtype=torch.uint64,
        device=f"cuda:{device1}",
    )
    native_pair = native_runtime.create_endpoint_pair(
        device0,
        descriptor0.data_ptr(),
        descriptor0.numel() * descriptor0.element_size(),
        device1,
        descriptor1.data_ptr(),
        descriptor1.numel() * descriptor1.element_size(),
    )
    return (
        descriptor0,
        descriptor1,
        native_pair,
    )


def _create_efa_pair(
    native_runtime,
    device0: int,
    device1: int,
):
    # Nine 64-bit words per endpoint descriptor; see EfaAbi.h.
    descriptor0 = torch.empty(
        9,
        dtype=torch.uint64,
        device=f"cuda:{device0}",
    )
    descriptor1 = torch.empty(
        9,
        dtype=torch.uint64,
        device=f"cuda:{device1}",
    )
    native_pair = native_runtime.create_efa_endpoint_pair(
        device0,
        descriptor0.data_ptr(),
        descriptor0.numel() * descriptor0.element_size(),
        device1,
        descriptor1.data_ptr(),
        descriptor1.numel() * descriptor1.element_size(),
    )
    return (
        descriptor0,
        descriptor1,
        native_pair,
    )


def _create_proxy_pair(
    native_runtime,
    device0: int,
    device1: int,
):
    descriptor0 = torch.empty(
        5,
        dtype=torch.uint64,
        device=f"cuda:{device0}",
    )
    descriptor1 = torch.empty(
        5,
        dtype=torch.uint64,
        device=f"cuda:{device1}",
    )
    native_pair = native_runtime.create_proxy_endpoint_pair(
        device0,
        descriptor0.data_ptr(),
        descriptor0.numel() * descriptor0.element_size(),
        device1,
        descriptor1.data_ptr(),
        descriptor1.numel() * descriptor1.element_size(),
    )
    return (
        descriptor0,
        descriptor1,
        native_pair,
    )


def _check_slice(total: int, offset: int, length: int) -> None:
    if offset < 0 or length < 0 or offset + length > total:
        raise ValueError(
            f"buffer slice [{offset}, {offset + length}) "
            f"is outside [0, {total})"
        )


def _cuda_check(result, operation: str) -> None:
    error = result[0]
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {error}")


def _driver_check(result, operation: str) -> None:
    error = result[0]
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with {error}")


def _enable_peer_access(source: int, destination: int) -> bool:
    _cuda_check(
        cudart.cudaSetDevice(source),
        f"cudaSetDevice({source})",
    )
    result = cudart.cudaDeviceEnablePeerAccess(destination, 0)
    if result[0] == cudart.cudaError_t.cudaSuccess:
        return True
    if (
        result[0]
        == cudart.cudaError_t.cudaErrorPeerAccessAlreadyEnabled
    ):
        cudart.cudaGetLastError()
        return False
    _cuda_check(result, "cudaDeviceEnablePeerAccess")
    return False


def _disable_peer_access(source: int, destination: int) -> None:
    _cuda_check(
        cudart.cudaSetDevice(source),
        f"cudaSetDevice({source})",
    )
    result = cudart.cudaDeviceDisablePeerAccess(destination)
    if (
        result[0]
        == cudart.cudaError_t.cudaErrorPeerAccessNotEnabled
    ):
        cudart.cudaGetLastError()
        return
    _cuda_check(result, "cudaDeviceDisablePeerAccess")
