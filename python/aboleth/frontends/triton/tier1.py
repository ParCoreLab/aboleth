"""The generated Tier-1 device library in Triton kernels, through extern_libs.

The Triton counterpart of the generated `aboleth/device.cuh`, and the way a
Triton user links a communication library today (Triton-distributed links
NVSHMEM's device bitcode the same way): the library's per-op bitcode is linked
at the end of lowering, and each op is an opaque extern call. A Triton extern
call executes on every thread, so a kernel issues the protocol from one
thread under `if tid() == 0:` (Triton-distributed's idiom), as a CUDA kernel
does under `if (threadIdx.x == 0)`.

    import aboleth.frontends.triton.tier1 as ab
    ab.bind(world.device_library())          # once, before compiling

    @triton.jit
    def exchange(endpoints, peer, dst, dst_key, src, src_key, nbytes, ...):
        if ab.tid() == 0:
            ab.put_signal(endpoints, peer, dst, dst_key, src, src_key,
                          nbytes, flag, flag_key, epoch)
            ab.quiet(endpoints, peer)
            ab.wait_signal(my_flag, epoch)

    exchange[(1,)](..., num_warps=1, extern_libs=ab.extern_libs())

`endpoints` is a pointer to the rank's endpoint table
(`PeerRegionTable.endpoint_handles`); addresses are int64, keys int32.
Dispatch follows the header: one transport compiles to direct calls, a mixed
library switches on the handle's kind bits.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language import core

from ...emit.tier1_header import KIND_BITS, KIND_MASK as _KIND_MASK, op_slots
from ...tier1 import tier1_symbol

_TYPES = {"i1": core.int32, "i32": core.int32, "i64": core.int64}
KIND_MASK = tl.constexpr(_KIND_MASK)
HANDLE_MASK = tl.constexpr(~_KIND_MASK)
KIND_P2P = tl.constexpr(KIND_BITS["p2p"])

# The bound library, read at trace time. RDMA/P2P say which transports the
# library compiled in; FAN_OUT is its endpoints per peer.
HAS_RDMA = tl.constexpr(True)
HAS_P2P = tl.constexpr(True)
FAN_OUT = tl.constexpr(1)
_LIBRARY: dict = {}


def bind(library) -> None:
    """Bind a `GeneratedDeviceLibrary` for the kernels compiled next."""
    global HAS_RDMA, HAS_P2P, FAN_OUT
    if library.peer_transports is not None:
        raise ValueError("bind the generic library; the geography-specialized "
                         "build is header-only today")
    HAS_RDMA = tl.constexpr("rdma" in library.transports)
    HAS_P2P = tl.constexpr("p2p" in library.transports)
    FAN_OUT = tl.constexpr(library.endpoints_per_peer)
    _LIBRARY.clear()
    _LIBRARY["library"] = library


def extern_libs() -> dict:
    """The `extern_libs` launch option: every per-op bitcode of the library."""
    library = _LIBRARY["library"]
    return {f"aboleth_{transport}_{op}": str(path)
            for transport, ops in library.op_bitcodes.items()
            for op, path in ops.items()}


def _call(op: str, transport, args: list, _semantic):
    transport = core._unwrap_if_constexpr(transport)
    types = tuple(_TYPES[kind] for _, kind in op_slots(op))
    cast = [core.cast(value, kind, _semantic=_semantic)
            for value, kind in zip(args, types)]
    return core.extern_elementwise(
        "aboleth_tier1", "", cast,
        {types: (tier1_symbol(op, transport), core.int64)},
        is_pure=False, _semantic=_semantic)


@core.extern
def _put(transport, handle, dst, src, nbytes, src_key, dst_key,
         _semantic=None):
    return _call("put", transport,
                 [handle, dst, src, nbytes, src_key, dst_key, 1], _semantic)


@core.extern
def _put_warp(transport, handle, dst, src, nbytes, src_key, dst_key,
              _semantic=None):
    return _call("put_warp", transport,
                 [handle, dst, src, nbytes, src_key, dst_key, 1], _semantic)


@core.extern
def _put_signal(transport, handle, dst, src, nbytes, src_key, dst_key, flag,
                flag_key, epoch, _semantic=None):
    return _call("put_signal", transport,
                 [handle, dst, src, nbytes, src_key, dst_key, 1, flag,
                  flag_key, epoch], _semantic)


@core.extern
def _signal(transport, handle, flag, flag_key, epoch, _semantic=None):
    return _call("signal", transport, [handle, flag, flag_key, epoch],
                 _semantic)


@core.extern
def _put_inline(transport, handle, dst, dst_key, value, _semantic=None):
    return _call("put_inline", transport, [handle, dst, dst_key, value],
                 _semantic)


@core.extern
def _quiet(transport, handle, _semantic=None):
    return _call("quiet", transport, [handle], _semantic)


@core.extern
def _wait_signal(transport, flag, epoch, _semantic=None):
    return _call("wait_signal", transport, [flag, epoch], _semantic)


@core.extern
def tid(_semantic=None):
    """This thread's index in the CTA (x), for the one-issuer guard."""
    return core.extern_elementwise(
        "", "", [], {(): ("llvm.nvvm.read.ptx.sreg.tid.x", core.int32)},
        is_pure=True, _semantic=_semantic)


@triton.jit
def _tagged(endpoints, peer):
    # tagged_endpoint(): CTA b drives slot b % N of the peer's N endpoints.
    # The table is uint64; the handle's bits are read as int64.
    if FAN_OUT == 1:
        tagged = tl.load(endpoints + peer)
    else:
        tagged = tl.load(endpoints + peer * FAN_OUT
                         + tl.program_id(0) % FAN_OUT)
    return tagged.to(tl.int64, bitcast=True)


@triton.jit
def put(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    """Post one put; returns without waiting (quiet completes it)."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & HANDLE_MASK
    if not HAS_P2P:
        _put("rdma", handle, dst, src, nbytes, src_key, dst_key)
    elif not HAS_RDMA:
        _put("p2p", handle, dst, src, nbytes, src_key, dst_key)
    elif (tagged & KIND_MASK) == KIND_P2P:
        _put("p2p", handle, dst, src, nbytes, src_key, dst_key)
    else:
        _put("rdma", handle, dst, src, nbytes, src_key, dst_key)


@triton.jit
def put_warp(endpoints, peer, dst, dst_key, src, src_key, nbytes):
    """Warp-collective put, entered by every lane of one complete warp: on
    p2p the 32 lanes copy cooperatively behind a warp-scope fence; on rdma
    lane 0 posts one WQE (the header's put_warp)."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & HANDLE_MASK
    if not HAS_P2P:
        if tid() % 32 == 0:
            _put("rdma", handle, dst, src, nbytes, src_key, dst_key)
    elif not HAS_RDMA:
        _put_warp("p2p", handle, dst, src, nbytes, src_key, dst_key)
    elif (tagged & KIND_MASK) == KIND_P2P:
        _put_warp("p2p", handle, dst, src, nbytes, src_key, dst_key)
    else:
        if tid() % 32 == 0:
            _put("rdma", handle, dst, src, nbytes, src_key, dst_key)


@triton.jit
def signal(endpoints, peer, flag, flag_key, epoch):
    """Publish `epoch` to the peer's flag, ordered behind earlier puts."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & HANDLE_MASK
    if not HAS_P2P:
        _signal("rdma", handle, flag, flag_key, epoch)
    elif not HAS_RDMA:
        _signal("p2p", handle, flag, flag_key, epoch)
    elif (tagged & KIND_MASK) == KIND_P2P:
        _signal("p2p", handle, flag, flag_key, epoch)
    else:
        _signal("rdma", handle, flag, flag_key, epoch)


@triton.jit
def put_inline(endpoints, peer, dst, dst_key, value):
    """Write eight bytes from registers."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & HANDLE_MASK
    if not HAS_P2P:
        _put_inline("rdma", handle, dst, dst_key, value)
    elif not HAS_RDMA:
        _put_inline("p2p", handle, dst, dst_key, value)
    elif (tagged & KIND_MASK) == KIND_P2P:
        _put_inline("p2p", handle, dst, dst_key, value)
    else:
        _put_inline("rdma", handle, dst, dst_key, value)


@triton.jit
def put_signal(endpoints, peer, dst, dst_key, src, src_key, nbytes, flag,
               flag_key, epoch):
    """The payload and its epoch in one batch (one doorbell on rdma)."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & HANDLE_MASK
    if not HAS_P2P:
        _put_signal("rdma", handle, dst, src, nbytes, src_key, dst_key, flag,
                    flag_key, epoch)
    elif not HAS_RDMA:
        _put("p2p", handle, dst, src, nbytes, src_key, dst_key)
        _signal("p2p", handle, flag, flag_key, epoch)
    elif (tagged & KIND_MASK) == KIND_P2P:
        _put("p2p", handle, dst, src, nbytes, src_key, dst_key)
        _signal("p2p", handle, flag, flag_key, epoch)
    else:
        _put_signal("rdma", handle, dst, src, nbytes, src_key, dst_key, flag,
                    flag_key, epoch)


@triton.jit
def quiet(endpoints, peer):
    """Block until every op posted to `peer` so far has completed."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & HANDLE_MASK
    if not HAS_P2P:
        _quiet("rdma", handle)
    elif not HAS_RDMA:
        _quiet("p2p", handle)
    elif (tagged & KIND_MASK) == KIND_P2P:
        _quiet("p2p", handle)
    else:
        _quiet("rdma", handle)


@triton.jit
def wait_signal(flag, epoch):
    """Spin until the local flag at address `flag` holds at least `epoch`."""
    if HAS_RDMA:
        _wait_signal("rdma", flag, epoch)
    else:
        _wait_signal("p2p", flag, epoch)
