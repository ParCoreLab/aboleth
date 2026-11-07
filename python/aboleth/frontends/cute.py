"""The generated Tier-1 device library in CuTe DSL kernels.

The CuTe counterpart of the generated `aboleth/device.cuh`: the same
exported per-op symbols, linked as bitcode through `cute.extern` and fully
inlined, behind the header's op signatures. A kernel passes the rank's
endpoint table (`PeerRegionTable.endpoint_handles`) as a `cute.Tensor` and
calls the ops from any thread; like every Tier-1 op they have per-thread
semantics, so a kernel issues each protocol call from one thread, as a CUDA
kernel does.

    import aboleth.frontends.cute as ab
    ab.bind(world.device_library())          # once, before cute.compile

    @cute.kernel
    def exchange(endpoints: cute.Tensor, peer: Int32, ...):
        if cute.arch.thread_idx()[0] == 0:
            ab.put_signal(endpoints, peer, dst, dst_key, src, src_key,
                          nbytes, flag, flag_key, epoch)
            ab.quiet(endpoints, peer)
            ab.wait_signal(my_flag, epoch)

Dispatch follows the header: a single-transport library calls its backend
directly; a mixed one switches on the kind bits the endpoint handle carries.
The bound library is module state, so `bind` precedes compilation and a
compiled kernel keeps the library it was traced against.
"""

from __future__ import annotations

import inspect

import cutlass
import cutlass.cute as cute
from cutlass import Uint32, Uint64

from ..emit.tier1_header import KIND_BITS, KIND_MASK, op_slots
from ..tier1 import tier1_symbol

_TYPES = {"i1": Uint32, "i32": Uint32, "i64": Uint64}
_HANDLE_MASK = ~KIND_MASK & ((1 << 64) - 1)
_P2P = KIND_BITS["p2p"]

# The bound library: (op, transport) -> extern handler. Module globals, so the
# CuTe AST rewrite of the ops below resolves them at trace time.
_EXTERNS: dict[tuple[str, str], object] = {}
_TRANSPORTS: tuple[str, ...] = ()
_FAN_OUT = 1
_DIGEST = ""


def _extern(op: str, transport: str, bitcode) -> object:
    symbol = tier1_symbol(op, transport)

    def stub(*args):
        ...

    stub.__name__ = symbol
    stub.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY,
                           annotation=_TYPES[kind])
         for name, kind in op_slots(op)],
        return_annotation=Uint64)
    return cute.extern(stub, name=symbol, source=cute.BitCode(str(bitcode)),
                       inline=True, overloaded=False)


def bind(library) -> None:
    """Bind a `GeneratedDeviceLibrary` for the kernels compiled next."""
    global _TRANSPORTS, _FAN_OUT, _DIGEST
    if library.peer_transports is not None:
        raise ValueError(
            "aboleth.frontends.cute binds the generic library; the "
            "geography-specialized build is header-only today")
    _EXTERNS.clear()
    for transport, ops in library.op_bitcodes.items():
        for op, bitcode in ops.items():
            _EXTERNS[(op, transport)] = _extern(op, transport, bitcode)
    _TRANSPORTS = tuple(library.transports)
    _FAN_OUT = library.endpoints_per_peer
    _DIGEST = library.digest


def bound_digest() -> str:
    """The digest of the bound library: part of any compiled-kernel key."""
    return _DIGEST


@cute.jit
def _tagged(endpoints: cute.Tensor, peer):
    # tagged_endpoint(): CTA b drives slot b % N of the peer's N endpoints.
    if cutlass.const_expr(_FAN_OUT == 1):
        return Uint64(endpoints[peer])
    block, _, _ = cute.arch.block_idx()
    return Uint64(endpoints[peer * _FAN_OUT + block % _FAN_OUT])


@cute.jit
def _is_p2p(tagged):
    return (tagged & Uint64(KIND_MASK)) == Uint64(_P2P)


@cute.jit
def put(endpoints: cute.Tensor, peer, destination, destination_key, source,
        source_key, nbytes):
    """Post one put; returns without waiting (quiet completes it)."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & Uint64(_HANDLE_MASK)
    arguments = (handle, Uint64(destination), Uint64(source), Uint32(nbytes),
                 Uint32(source_key), Uint32(destination_key), Uint32(1))
    if cutlass.const_expr(len(_TRANSPORTS) == 1):
        _EXTERNS[("put", _TRANSPORTS[0])](*arguments)
    else:
        if _is_p2p(tagged):
            _EXTERNS[("put", "p2p")](*arguments)
        else:
            _EXTERNS[("put", "rdma")](*arguments)


@cute.jit
def put_warp(endpoints: cute.Tensor, peer, destination, destination_key,
             source, source_key, nbytes):
    """Warp-collective put, entered by every lane of one complete warp: on
    p2p the 32 lanes copy cooperatively behind a warp-scope fence; on rdma
    lane 0 posts one WQE (the header's put_warp)."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & Uint64(_HANDLE_MASK)
    arguments = (handle, Uint64(destination), Uint64(source), Uint32(nbytes),
                 Uint32(source_key), Uint32(destination_key), Uint32(1))
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(_TRANSPORTS == ("p2p",)):
        _EXTERNS[("put_warp", "p2p")](*arguments)
    elif cutlass.const_expr(_TRANSPORTS == ("rdma",)):
        if tidx % 32 == 0:
            _EXTERNS[("put", "rdma")](*arguments)
    else:
        if _is_p2p(tagged):
            _EXTERNS[("put_warp", "p2p")](*arguments)
        else:
            if tidx % 32 == 0:
                _EXTERNS[("put", "rdma")](*arguments)


@cute.jit
def signal(endpoints: cute.Tensor, peer, flag, flag_key, epoch):
    """Publish `epoch` to the peer's flag, ordered behind earlier puts."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & Uint64(_HANDLE_MASK)
    arguments = (handle, Uint64(flag), Uint32(flag_key), Uint64(epoch))
    if cutlass.const_expr(len(_TRANSPORTS) == 1):
        _EXTERNS[("signal", _TRANSPORTS[0])](*arguments)
    else:
        if _is_p2p(tagged):
            _EXTERNS[("signal", "p2p")](*arguments)
        else:
            _EXTERNS[("signal", "rdma")](*arguments)


@cute.jit
def put_inline(endpoints: cute.Tensor, peer, destination, destination_key,
               value):
    """Write eight bytes from registers."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & Uint64(_HANDLE_MASK)
    arguments = (handle, Uint64(destination), Uint32(destination_key),
                 Uint64(value))
    if cutlass.const_expr(len(_TRANSPORTS) == 1):
        _EXTERNS[("put_inline", _TRANSPORTS[0])](*arguments)
    else:
        if _is_p2p(tagged):
            _EXTERNS[("put_inline", "p2p")](*arguments)
        else:
            _EXTERNS[("put_inline", "rdma")](*arguments)


@cute.jit
def put_signal(endpoints: cute.Tensor, peer, destination, destination_key,
               source, source_key, nbytes, flag, flag_key, epoch):
    """The payload and its epoch in one batch (one doorbell on rdma)."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & Uint64(_HANDLE_MASK)
    put_arguments = (handle, Uint64(destination), Uint64(source),
                     Uint32(nbytes), Uint32(source_key),
                     Uint32(destination_key), Uint32(1))
    signal_arguments = (handle, Uint64(flag), Uint32(flag_key),
                        Uint64(epoch))
    if cutlass.const_expr(_TRANSPORTS == ("p2p",)):
        _EXTERNS[("put", "p2p")](*put_arguments)
        _EXTERNS[("signal", "p2p")](*signal_arguments)
    elif cutlass.const_expr(_TRANSPORTS == ("rdma",)):
        _EXTERNS[("put_signal", "rdma")](*put_arguments, *signal_arguments[1:])
    else:
        if _is_p2p(tagged):
            _EXTERNS[("put", "p2p")](*put_arguments)
            _EXTERNS[("signal", "p2p")](*signal_arguments)
        else:
            _EXTERNS[("put_signal", "rdma")](*put_arguments,
                                             *signal_arguments[1:])


@cute.jit
def quiet(endpoints: cute.Tensor, peer):
    """Block until every op posted to `peer` so far has completed."""
    tagged = _tagged(endpoints, peer)
    handle = tagged & Uint64(_HANDLE_MASK)
    if cutlass.const_expr(len(_TRANSPORTS) == 1):
        _EXTERNS[("quiet", _TRANSPORTS[0])](handle)
    else:
        if _is_p2p(tagged):
            _EXTERNS[("quiet", "p2p")](handle)
        else:
            _EXTERNS[("quiet", "rdma")](handle)


@cute.jit
def wait_signal(flag, epoch):
    """Spin until the local flag at address `flag` holds at least `epoch`."""
    _EXTERNS[("wait_signal", _TRANSPORTS[0])](Uint64(flag), Uint64(epoch))


__all__ = ["bind", "bound_digest", "put", "put_warp", "signal",
           "put_inline",
           "put_signal", "quiet", "wait_signal"]
