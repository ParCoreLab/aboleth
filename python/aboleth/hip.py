"""HIP control-plane operations used by the shared Runtime/World layer.

The small adapters retain the call/tuple shape consumed by runtime.py; CUDA
bindings are never imported on a ROCm build. No CUDA ABI is used for HIP IPC.
"""
import ctypes as C
from enum import IntEnum
from types import SimpleNamespace


_lib = C.CDLL("libamdhip64.so.7")


class _IpcHandle(C.Structure):
    _fields_ = [("reserved", C.c_byte * 64)]

    def getPtr(self):
        return C.addressof(self)


class _Error(IntEnum):
    cudaSuccess = 0
    cudaErrorPeerAccessAlreadyEnabled = 704
    cudaErrorPeerAccessNotEnabled = 705


def _status(name, *types):
    function = getattr(_lib, name)
    function.restype = C.c_int
    function.argtypes = types
    return lambda *args: (function(*args),)


def _pointer_result(name, input_types, *args):
    result = C.c_void_p()
    function = getattr(_lib, name)
    function.restype = C.c_int
    function.argtypes = [C.POINTER(C.c_void_p), *input_types]
    status = function(C.byref(result), *args)
    return status, result.value


def _ipc_export(address):
    handle = _IpcHandle()
    function = _lib.hipIpcGetMemHandle
    function.argtypes = [C.POINTER(_IpcHandle), C.c_void_p]
    return function(C.byref(handle), address), handle


def _ipc_import(handle, flags):
    return _pointer_result("hipIpcOpenMemHandle", [_IpcHandle, C.c_uint], handle, flags)


def _address_range(address):
    base = C.c_void_p()
    length = C.c_size_t()
    function = _lib.hipMemGetAddressRange
    function.argtypes = [C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p]
    return function(C.byref(base), C.byref(length), address), base.value, length.value


runtime = SimpleNamespace(
    cudaError_t=_Error,
    cudaIpcMemHandle_t=_IpcHandle,
    cudaIpcEventHandle_t=_IpcHandle,
    cudaIpcMemLazyEnablePeerAccess=1,
    cudaSetDevice=_status("hipSetDevice", C.c_int),
    cudaDeviceEnablePeerAccess=_status("hipDeviceEnablePeerAccess", C.c_int, C.c_uint),
    cudaDeviceDisablePeerAccess=_status("hipDeviceDisablePeerAccess", C.c_int),
    cudaGetLastError=_status("hipGetLastError"),
    cudaIpcGetMemHandle=_ipc_export,
    cudaIpcOpenMemHandle=_ipc_import,
    cudaIpcCloseMemHandle=_status("hipIpcCloseMemHandle", C.c_void_p),
    cudaIpcOpenEventHandle=lambda handle: _pointer_result("hipIpcOpenEventHandle", [_IpcHandle], handle),
    cudaEventSynchronize=_status("hipEventSynchronize", C.c_void_p),
    cudaEventDestroy=_status("hipEventDestroy", C.c_void_p),
)
driver = SimpleNamespace(CUresult=SimpleNamespace(CUDA_SUCCESS=0),
                         cuMemGetAddressRange=_address_range)


def empty(shape, *, dtype=None, device=0):
    """A device tensor backed by fine-grained memory for peer/NIC polling."""
    import math
    import torch
    from . import _runtime
    dtype = torch.uint8 if dtype is None else dtype
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    count = math.prod(shape)
    if count <= 0:
        raise ValueError("HIP coherent tensor must be nonempty")
    torch.cuda.set_device(int(device))
    element_size = torch.empty((), dtype=dtype).element_size()
    capsule = _runtime.hip_allocate(count * element_size, int(device))
    return torch.utils.dlpack.from_dlpack(capsule).view(dtype).view(shape)
