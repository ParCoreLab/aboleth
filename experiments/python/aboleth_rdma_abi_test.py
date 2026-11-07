import ctypes
import os
import struct
from pathlib import Path

import torch
import triton
import triton.language as tl


ABI_VERSION = 3
DESCRIPTOR_BYTES = 96
DESCRIPTOR_WORDS = DESCRIPTOR_BYTES // 8


class RdmaEndpointDescriptor(ctypes.Structure):
    _fields_ = [
        ("send_queue", ctypes.c_uint64),
        ("send_dbrec", ctypes.c_uint64),
        ("blueflame", ctypes.c_uint64),
        ("completion_queue", ctypes.c_uint64),
        ("completion_dbrec", ctypes.c_uint64),
        ("producer_index_address", ctypes.c_uint64),
        ("completion_consumer_address", ctypes.c_uint64),
        ("completion_timeout_cycles", ctypes.c_uint64),
        ("send_stride", ctypes.c_uint32),
        ("send_wqe_count", ctypes.c_uint32),
        ("qpn", ctypes.c_uint32),
        ("completion_entry_size", ctypes.c_uint32),
        ("completion_entry_count", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("atomic_result_lkey", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
    ]


@triton.jit
def read_descriptor_kernel(
    descriptor,
    output,
    word_count: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    mask = offsets < word_count
    words = tl.load(descriptor + offsets, mask=mask, other=0)
    tl.store(output + offsets, words, mask=mask)


def load_runtime():
    build_dir = Path(
        os.environ.get(
            "ABOLETH_BUILD_DIR",
            Path(__file__).resolve().parents[2] / "build",
        )
    )
    library_path = build_dir / "lib" / "libaboleth_rdma_abi.so"
    library = ctypes.CDLL(str(library_path))
    library.aboleth_runtime_create.argtypes = []
    library.aboleth_runtime_create.restype = ctypes.c_void_p
    library.aboleth_runtime_destroy.argtypes = [ctypes.c_void_p]
    library.aboleth_runtime_destroy.restype = ctypes.c_int
    library.aboleth_runtime_create_endpoint_pair.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.aboleth_runtime_create_endpoint_pair.restype = ctypes.c_void_p
    library.aboleth_runtime_destroy_endpoint_pair.argtypes = [ctypes.c_void_p]
    library.aboleth_runtime_destroy_endpoint_pair.restype = ctypes.c_int
    library.aboleth_runtime_endpoint_pair_state.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.aboleth_runtime_endpoint_pair_state.restype = ctypes.c_int
    library.aboleth_runtime_endpoint_pair_max_inline.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.aboleth_runtime_endpoint_pair_max_inline.restype = ctypes.c_int
    library.aboleth_runtime_endpoint_pair_send_queue_snapshot.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.aboleth_runtime_endpoint_pair_send_queue_snapshot.restype = (
        ctypes.c_int
    )
    library.aboleth_runtime_register_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint64,
    ]
    library.aboleth_runtime_register_buffer.restype = ctypes.c_void_p
    library.aboleth_runtime_unregister_buffer.argtypes = [ctypes.c_void_p]
    library.aboleth_runtime_unregister_buffer.restype = ctypes.c_int
    library.aboleth_runtime_buffer_lkey.argtypes = [ctypes.c_void_p]
    library.aboleth_runtime_buffer_lkey.restype = ctypes.c_uint32
    library.aboleth_runtime_buffer_rkey.argtypes = [ctypes.c_void_p]
    library.aboleth_runtime_buffer_rkey.restype = ctypes.c_uint32
    library.aboleth_runtime_last_error.argtypes = []
    library.aboleth_runtime_last_error.restype = ctypes.c_char_p
    return library


def last_error(library) -> str:
    """The reason behind the failing call this thread just made."""
    return library.aboleth_runtime_last_error().decode()


def read_descriptor(descriptor) -> RdmaEndpointDescriptor:
    """The descriptor the runtime published, read back out of device memory.

    The runtime writes the descriptor where the kernel reads it and nowhere
    else, so this readback is the host's view of the same bytes.
    """
    words = struct.pack(
        f"<{DESCRIPTOR_WORDS}Q",
        *descriptor.cpu().tolist(),
    )
    value = RdmaEndpointDescriptor()
    ctypes.memmove(ctypes.byref(value), words, DESCRIPTOR_BYTES)
    return value


class EndpointPair:
    """A connected endpoint pair and the windows its kernels drive it over.

    Endpoint 0 sends out of `window0` into `window1` and endpoint 1 the other
    way round, so `keys(index)` is the (lkey, rkey) pair a kernel posting on
    endpoint `index` needs. A descriptor of None is allocated here, which is
    what a one-directional test wants for the half that only terminates
    transfers.

    The runtime opens whichever HCA ABOLETH_HCA or the GPU's own topology
    picks; the CUDA ordinals come from the windows, so CUDA_VISIBLE_DEVICES
    still decides which GPUs those are.
    """

    def __init__(self, library, descriptor0, window0, descriptor1, window1):
        self._library = library
        self._runtime = None
        self._pair = None
        self._buffers = [None, None]
        self._owned = []
        devices = (window0.get_device(), window1.get_device())
        descriptors = (
            descriptor0 if descriptor0 is not None
            else self._allocate_descriptor(devices[0]),
            descriptor1 if descriptor1 is not None
            else self._allocate_descriptor(devices[1]),
        )
        try:
            self._runtime = library.aboleth_runtime_create()
            if not self._runtime:
                raise RuntimeError(last_error(library))
            # The runtime echoes the host-built descriptors here before the
            # device copy, so the conformance comparison holds an expected
            # value independent of the device memory the kernel reads.
            self.expected_descriptors = (
                RdmaEndpointDescriptor(),
                RdmaEndpointDescriptor(),
            )
            self._pair = library.aboleth_runtime_create_endpoint_pair(
                self._runtime,
                devices[0],
                descriptors[0].data_ptr(),
                DESCRIPTOR_BYTES,
                devices[1],
                descriptors[1].data_ptr(),
                DESCRIPTOR_BYTES,
                ctypes.addressof(self.expected_descriptors[0]),
                ctypes.addressof(self.expected_descriptors[1]),
            )
            if not self._pair:
                raise RuntimeError(last_error(library))
            self._buffers = [
                self._register(device, window)
                for device, window in zip(devices, (window0, window1))
            ]
            self.descriptors = tuple(
                read_descriptor(value) for value in descriptors
            )
        except BaseException:
            self.close()
            raise

    def _allocate_descriptor(self, device):
        descriptor = torch.empty(
            DESCRIPTOR_WORDS,
            dtype=torch.uint64,
            device=f"cuda:{device}",
        )
        self._owned.append(descriptor)
        return descriptor

    def _register(self, device, window):
        handle = self._library.aboleth_runtime_register_buffer(
            self._runtime,
            device,
            window.data_ptr(),
            window.numel() * window.element_size(),
            0,
        )
        if not handle:
            raise RuntimeError(last_error(self._library))
        return handle

    @property
    def descriptor(self) -> RdmaEndpointDescriptor:
        return self.expected_descriptors[0]

    def keys(self, endpoint_index: int = 0) -> tuple[int, int]:
        """The local key of what endpoint `endpoint_index` sends from and the
        remote key of what it writes into."""
        return (
            self._library.aboleth_runtime_buffer_lkey(
                self._buffers[endpoint_index]
            ),
            self._library.aboleth_runtime_buffer_rkey(
                self._buffers[1 - endpoint_index]
            ),
        )

    def max_inline(self, endpoint_index: int = 0) -> int:
        """The inline payload capacity the adapter granted this endpoint."""
        granted = ctypes.c_uint32()
        status = self._library.aboleth_runtime_endpoint_pair_max_inline(
            self._pair,
            endpoint_index,
            ctypes.byref(granted),
        )
        if status != 0:
            raise RuntimeError(last_error(self._library))
        return granted.value

    def send_queue_snapshot(
        self,
        byte_count: int,
        endpoint_index: int = 0,
    ) -> tuple[bytes, int]:
        """The send queue's leading `byte_count` bytes and its doorbell head."""
        buffer = (ctypes.c_uint8 * byte_count)()
        dbrec_head = ctypes.c_uint32()
        status = (
            self._library.aboleth_runtime_endpoint_pair_send_queue_snapshot(
                self._pair,
                endpoint_index,
                buffer,
                byte_count,
                ctypes.byref(dbrec_head),
            )
        )
        if status != 0:
            raise RuntimeError(last_error(self._library))
        return bytes(buffer), dbrec_head.value

    def state(self) -> tuple[int, int, int, int]:
        """Both endpoints' producer and completion-consumer indices."""
        values = [ctypes.c_uint64() for _ in range(4)]
        status = self._library.aboleth_runtime_endpoint_pair_state(
            self._pair,
            *(ctypes.byref(value) for value in values),
        )
        if status != 0:
            raise RuntimeError(last_error(self._library))
        return tuple(value.value for value in values)

    def close(self) -> None:
        for index, handle in enumerate(self._buffers):
            self._buffers[index] = None
            if handle and self._library.aboleth_runtime_unregister_buffer(
                handle
            ) != 0:
                raise RuntimeError(last_error(self._library))
        pair, self._pair = self._pair, None
        if pair and self._library.aboleth_runtime_destroy_endpoint_pair(
            pair
        ) != 0:
            raise RuntimeError(last_error(self._library))
        runtime, self._runtime = self._runtime, None
        if runtime and self._library.aboleth_runtime_destroy(runtime) != 0:
            raise RuntimeError(last_error(self._library))
        self._owned.clear()


def create_endpoints(library, descriptor, source, destination) -> EndpointPair:
    """One endpoint a kernel drives and a peer that terminates its transfers.

    `source` is registered for the lkey the kernel sends with and `destination`
    for the rkey it writes to, which is the shape every one-directional test
    here wants.
    """
    return EndpointPair(library, descriptor, source, None, destination)


def main() -> None:
    assert ctypes.sizeof(RdmaEndpointDescriptor) == DESCRIPTOR_BYTES
    library = load_runtime()

    descriptor = torch.empty(
        DESCRIPTOR_WORDS,
        dtype=torch.uint64,
        device="cuda:0",
    )
    observed = torch.zeros_like(descriptor)
    source = torch.empty(4096, dtype=torch.uint8, device="cuda:0")
    destination = torch.empty_like(source)

    endpoints = create_endpoints(
        library,
        descriptor,
        source,
        destination,
    )

    try:
        expected = endpoints.descriptor
        source_lkey, destination_rkey = endpoints.keys()
        read_descriptor_kernel[(1,)](
            descriptor,
            observed,
            word_count=DESCRIPTOR_WORDS,
            block_size=16,
            num_warps=1,
        )
        torch.cuda.synchronize(0)

        expected_bytes = ctypes.string_at(
            ctypes.byref(expected),
            ctypes.sizeof(expected),
        )
        expected_words = list(
            struct.unpack("<12Q", expected_bytes)
        )
        observed_words = observed.cpu().tolist()
        assert observed_words == expected_words

        assert expected.send_queue != 0
        assert expected.send_dbrec != 0
        assert expected.blueflame != 0
        assert expected.completion_queue != 0
        assert expected.completion_dbrec != 0
        assert expected.producer_index_address != 0
        assert expected.completion_consumer_address != 0
        assert expected.completion_timeout_cycles > 0
        assert expected.send_stride == 64
        assert expected.send_wqe_count >= 256
        assert expected.qpn > 0
        assert source_lkey > 0
        assert destination_rkey > 0
        assert expected.completion_entry_size in (64, 128)
        assert expected.completion_entry_count >= 256
        assert expected.abi_version == ABI_VERSION
        assert expected.atomic_result_lkey != 0
        assert expected.reserved1 == 0

        for name, _ in expected._fields_:
            print(f"{name}={getattr(expected, name)}")
        print(f"source_lkey={source_lkey}")
        print(f"destination_rkey={destination_rkey}")
        print(f"descriptor_words_verified={DESCRIPTOR_WORDS}")
        print("aboleth_rdma_abi=PASS")
    finally:
        endpoints.close()


if __name__ == "__main__":
    main()
