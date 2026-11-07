"""The generated Tier-1 device header.

Emits a `device.cuh`-shaped bindings header over the compiler-generated op
symbols: identical types and op signatures to the hand-written facade
(`include/aboleth/device.cuh`), so swapping the include path swaps the
implementation with zero caller diff. Each op is an `extern "C"` declaration
plus a `__forceinline__` wrapper; the implementation arrives by linking the
generated archive (nvcc `-rdc=true`) or bitcode (clang
`-mlink-builtin-bitcode`, Triton `extern_libs`, `cute.BitCode`).

New over the facade: `PutDesc`/`put_many` (N puts, one doorbell), the
`open(ctx, peer)` hoisted handle, `put_warp` (warp-collective put), the
`kKind*` constants naming the transport tag `PeerRegionTable.endpoint_handles`
carries in a handle's low three bits, and — in a multi-transport build — the
per-peer dispatch on those bits in a generic build, pruned to the transports
actually compiled in. A geography-specialized build instead bakes the exact
peer-rank -> transport mapping into the header: runtime peers select by rank,
and the `op<Peer>` overloads make a compile-time peer a direct backend call.
"""

from __future__ import annotations

ABI_VERSION = 2

# The transport tag in a handle's low bits (descriptors are >=8-aligned).
# Must match `kind_bits` in python/aboleth/world.py: rdma stays 0 so an
# untagged descriptor address is a valid rdma handle.
KIND_BITS = {"rdma": 0, "p2p": 1, "proxy": 2, "efa": 3, "ionic": 4}
KIND_MASK = 0x7
_KIND_CONSTANT = {"rdma": "kKindRdma", "p2p": "kKindP2P",
                  "proxy": "kKindProxy", "efa": "kKindEfa", "ionic": "kKindIonic"}

# op -> exported ABI slots, identical across transports (that uniformity is
# what makes the kind switch a bare re-target of the callee).
_OP_SLOTS = {
    "put": (
        ("endpoint", "i64"), ("destination", "i64"), ("source", "i64"),
        ("bytes", "i32"), ("source_key", "i32"),
        ("destination_key", "i32"), ("predicate", "i1"),
    ),
    "put_warp": (
        ("endpoint", "i64"), ("destination", "i64"), ("source", "i64"),
        ("bytes", "i32"), ("source_key", "i32"),
        ("destination_key", "i32"), ("predicate", "i1"),
    ),
    "put_signal": (
        ("endpoint", "i64"), ("destination", "i64"), ("source", "i64"),
        ("bytes", "i32"), ("source_key", "i32"),
        ("destination_key", "i32"), ("predicate", "i1"),
        ("flag", "i64"), ("flag_key", "i32"), ("epoch", "i64"),
    ),
    "put_inline": (
        ("endpoint", "i64"), ("destination", "i64"),
        ("destination_key", "i32"), ("value", "i64"),
    ),
    "put_inline16": (
        ("endpoint", "i64"), ("destination", "i64"),
        ("destination_key", "i32"), ("value0", "i64"), ("value1", "i64"),
    ),
    "put_inline24": (
        ("endpoint", "i64"), ("destination", "i64"),
        ("destination_key", "i32"), ("value0", "i64"), ("value1", "i64"),
        ("value2", "i64"),
    ),
    "signal": (
        ("endpoint", "i64"), ("destination", "i64"),
        ("destination_key", "i32"), ("value", "i64"),
    ),
    "quiet": (("endpoint", "i64"),),
    "wait_signal": (("signal", "i64"), ("expected", "i64")),
}

_PUT_MANY_SLOTS = (("endpoint", "i64"), ("descriptors", "i64"))
_PUT_MANY_SIGNAL_SLOTS = _PUT_MANY_SLOTS + (
    ("flag", "i64"), ("flag_key", "i32"), ("epoch", "i64"),
)


def op_slots(op: str) -> tuple[tuple[str, str], ...]:
    """The exported (name, type) slots of one Tier-1 op's symbol.

    Every symbol returns i64; i1 slots travel as i32. The DSL bindings
    (`aboleth.frontends`) declare their externs from this table, as this
    header does.
    """
    slots = _OP_SLOTS.get(op)
    if slots is None and op.startswith("put_many_"):
        slots = (_PUT_MANY_SIGNAL_SLOTS
                 if op.endswith("_signal") else _PUT_MANY_SLOTS)
    if slots is None:
        raise ValueError(f"op {op!r} has no declared Tier-1 ABI")
    return slots


def _extern_declaration(symbol: str, slots) -> str:
    types = {"i1": "std::uint32_t", "i32": "std::uint32_t",
             "i64": "std::uint64_t"}
    parameters = ",\n".join(
        f"    {types[mlir_type]} {name}" for name, mlir_type in slots
    )
    return (
        f"extern \"C\" __device__ std::uint64_t {symbol}(\n"
        f"{parameters});"
    )


def _declarations(transports, symbols) -> str:
    blocks = []
    for transport in transports:
        for op, symbol in symbols[transport].items():
            blocks.append(_extern_declaration(symbol, op_slots(op)))
    return "\n\n".join(blocks)


def _dispatch(
    transports,
    cases: dict,
    peer_transports=None,
    indent: str = "    ",
) -> str:
    """One generic-kind or specialized-peer dispatch.

    A single-transport build emits the lone case body directly — no switch,
    no dead branch. A generic mixed build switches on kind bits. A specialized
    mixed build switches on peer rank, so `op<Peer>` constant-folds it away.
    """
    if len(transports) == 1:
        return "\n".join(f"{indent}{line}" for line in cases[transports[0]])
    selector = "peer" if peer_transports is not None else "tagged & kKindMask"
    lines = [f"{indent}switch ({selector}) {{"]
    if peer_transports is None:
        groups = (
            ((_KIND_CONSTANT[transport],), transport)
            for transport in transports
        )
    else:
        groups = (
            (
                tuple(
                    str(peer)
                    for peer, selected in peer_transports
                    if selected == transport
                ),
                transport,
            )
            for transport in transports
        )
    for labels, transport in groups:
        for label in labels:
            lines.append(f"{indent}case {label}:")
        lines.append(f"{indent}{{")
        lines.extend(f"{indent}    {line}" for line in cases[transport])
        lines.append(f"{indent}    break;")
        lines.append(f"{indent}}}")
    lines.append(f"{indent}default:")
    lines.append(f"{indent}    __builtin_unreachable();")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def _open_body(transports, peer_transports) -> str:
    if len(transports) == 1:
        kind = _KIND_CONSTANT[transports[0]]
        return f"    return PeerRef{{tagged & ~kKindMask, {kind}}};"
    if peer_transports is None:
        return (
            "    return PeerRef{tagged & ~kKindMask, "
            "tagged & kKindMask};"
        )
    cases = {
        transport: [
            "return PeerRef{tagged & ~kKindMask, "
            f"{_KIND_CONSTANT[transport]}}};"
        ]
        for transport in transports
    }
    return _dispatch(transports, cases, peer_transports)


def _template_assert(peer_transports) -> str:
    if peer_transports is None:
        return ""
    valid = " || ".join(f"Peer == {peer}" for peer, _ in peer_transports)
    return (
        f'    static_assert({valid}, "Peer is absent from this specialized '
        'Aboleth world");\n'
    )


def _put_call(symbol: str) -> list[str]:
    return [
        f"tier1::{symbol}(",
        "    handle, destination.address, source.address, bytes,",
        "    source.key, destination.key, 1u);",
    ]


def _put_rdma(symbols) -> list[str]:
    """The rdma put case: small payloads ride inline in the command when the
    build opts in, so the NIC never gathers the source buffer."""
    lines = [
        "#if ABOLETH_TIER1_SMALL_INLINE",
        "if ((source.address & 7u) == 0u && bytes >= 8u && bytes <= 24u",
        "        && (bytes & 7u) == 0u) {",
        "    const std::uint64_t* words =",
        "        reinterpret_cast<const std::uint64_t*>(source.address);",
        "    if (bytes == 8u) {",
        f"        tier1::{symbols['put_inline']}(",
        "            handle, destination.address, destination.key,",
        "            words[0]);",
        "        return;",
        "    }",
        "    if (bytes == 16u) {",
        f"        tier1::{symbols['put_inline16']}(",
        "            handle, destination.address, destination.key,",
        "            words[0], words[1]);",
        "        return;",
        "    }",
        f"    tier1::{symbols['put_inline24']}(",
        "        handle, destination.address, destination.key,",
        "        words[0], words[1], words[2]);",
        "    return;",
        "}",
        "#endif",
    ]
    lines.extend(_put_call(symbols["put"]))
    return lines


def _put_signal_rdma(symbols) -> list[str]:
    return [
        f"tier1::{symbols['put_signal']}(",
        "    handle, destination.address, source.address, bytes,",
        "    source.key, destination.key, 1u,",
        "    flag.address, flag.key, epoch);",
    ]


def _put_signal_p2p(symbols) -> list[str]:
    # No queue to batch on: the put's stores plus the release-system epoch
    # store are already the minimal sequence.
    return [
        f"tier1::{symbols['put']}(",
        "    handle, destination.address, source.address, bytes,",
        "    source.key, destination.key, 1u);",
        f"tier1::{symbols['signal']}(",
        "    handle, flag.address, flag.key, epoch);",
    ]


def _value_call(symbol: str) -> list[str]:
    return [
        f"tier1::{symbol}(",
        "    handle, destination.address, destination.key, value);",
    ]


def _signal_call(symbol: str) -> list[str]:
    return [
        f"tier1::{symbol}(",
        "    handle, flag.address, flag.key, epoch);",
    ]


def _put_many_rdma(symbols, sizes) -> list[str]:
    lines = []
    for size in sizes:
        lines.extend((
            f"while (count >= {size}) {{",
            f"    tier1::{symbols[f'put_many_{size}']}(",
            "        handle,"
            " reinterpret_cast<std::uint64_t>(descriptors));",
            f"    descriptors += {size};",
            f"    count -= {size};",
            "}",
        ))
    lines.extend((
        "while (count > 0) {",
        f"    tier1::{symbols['put']}(",
        "        handle, descriptors->destination, descriptors->source,",
        "        descriptors->bytes, descriptors->source_key,",
        "        descriptors->destination_key, 1u);",
        "    ++descriptors;",
        "    --count;",
        "}",
    ))
    return lines


def _put_many_p2p(symbols) -> list[str]:
    # Peer-memory puts have no doorbell to amortize: the batch is just its
    # copies, so the composite degenerates to the per-descriptor loop.
    return [
        "while (count > 0) {",
        f"    tier1::{symbols['put']}(",
        "        handle, descriptors->destination, descriptors->source,",
        "        descriptors->bytes, descriptors->source_key,",
        "        descriptors->destination_key, 1u);",
        "    ++descriptors;",
        "    --count;",
        "}",
    ]


def _put_many_fused_rdma(symbols, sizes, kind: str) -> list[str]:
    """put_many with the trailing signal or flush fused into the final batch.

    Full-size chunks drain through the plain variants; the LAST batch is the
    fused `put_many_N_signal`/`put_many_N_flush` symbol, so the epoch (or the
    signalled NOP and its poll) never pays a posting sequence of its own.
    """
    lines = []
    for size in sizes:
        lines.extend((
            f"while (count > {size}) {{",
            f"    tier1::{symbols[f'put_many_{size}']}(",
            "        handle,"
            " reinterpret_cast<std::uint64_t>(descriptors));",
            f"    descriptors += {size};",
            f"    count -= {size};",
            "}",
            f"if (count == {size}) {{",
            f"    tier1::{symbols[f'put_many_{size}_{kind}']}(",
            "        handle,"
            " reinterpret_cast<std::uint64_t>(descriptors)"
            + (",\n        flag.address, flag.key, epoch);"
               if kind == "signal" else ");"),
            "    return;",
            "}",
        ))
    lines.extend((
        "while (count > 1) {",
        f"    tier1::{symbols['put']}(",
        "        handle, descriptors->destination, descriptors->source,",
        "        descriptors->bytes, descriptors->source_key,",
        "        descriptors->destination_key, 1u);",
        "    ++descriptors;",
        "    --count;",
        "}",
    ))
    if kind == "signal":
        lines.extend((
            "if (count == 1) {",
            f"    tier1::{symbols['put_signal']}(",
            "        handle, descriptors->destination,"
            " descriptors->source,",
            "        descriptors->bytes, descriptors->source_key,",
            "        descriptors->destination_key, 1u,",
            "        flag.address, flag.key, epoch);",
            "} else {",
            f"    tier1::{symbols['signal']}(",
            "        handle, flag.address, flag.key, epoch);",
            "}",
        ))
    else:
        lines.extend((
            "if (count == 1) {",
            f"    tier1::{symbols['put']}(",
            "        handle, descriptors->destination,"
            " descriptors->source,",
            "        descriptors->bytes, descriptors->source_key,",
            "        descriptors->destination_key, 1u);",
            "}",
            f"tier1::{symbols['quiet']}(handle);",
        ))
    return lines


def _put_many_signal_p2p(symbols) -> list[str]:
    return _put_many_p2p(symbols) + [
        f"tier1::{symbols['signal']}(",
        "    handle, flag.address, flag.key, epoch);",
    ]


def _put_warp_rdma(symbols, *, hip=False) -> list[str]:
    # One WQE per warp: lane 0 posts, the warp leaves together. The lane
    # election mirrors the wave-uniform contract p2p's cooperative copy
    # imposes anyway: the warp must enter complete.
    return [
        "{",
        *( ["    unsigned lane = threadIdx.x % warpSize;"] if hip else [
            "    unsigned lane;",
            '    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));']),
        "    if (lane == 0u) {",
        f"        tier1::{symbols['put']}(",
        "            handle, destination.address, source.address,",
        "            bytes, source.key, destination.key, 1u);",
        "    }",
        "    __syncwarp();",
        "}",
    ]


def emit_tier1_header(
    *,
    arch: str,
    transports: tuple[str, ...],
    symbols: dict,
    put_many_sizes: tuple[int, ...],
    library_digest: str,
    peer_transports: tuple[tuple[int, str], ...] | None = None,
    endpoints_per_peer: int = 1,
) -> str:
    """The header text for one generated library build.

    `symbols` maps transport -> op -> exported symbol; `put_many_sizes` are
    the descending batch variants the rdma `put_many` composes from.
    `peer_transports` is absent for the generic kind-tagged library and is the
    complete local rank's peer geography for a specialized mixed library.
    `endpoints_per_peer` matches `World.connect_all(endpoints_per_peer=N)`:
    the endpoint table is peer-major [world x N] and each op selects slot
    `blockIdx.x % N`, giving concurrent CTAs distinct endpoints.
    """
    if endpoints_per_peer < 1:
        raise ValueError("endpoints_per_peer must be >= 1")
    unsupported = [t for t in transports if t not in _KIND_CONSTANT]
    if unsupported:
        raise ValueError(
            f"transports {unsupported} have no generated Tier-1 binding"
        )
    if peer_transports is not None:
        if not peer_transports or len({p for p, _ in peer_transports}) != len(
            peer_transports
        ):
            raise ValueError("peer_transports must name unique peer ranks")
        if {transport for _, transport in peer_transports} != set(transports):
            raise ValueError(
                "peer_transports must use every and only emitted transport"
            )
    sizes = tuple(sorted(put_many_sizes, reverse=True))
    has_rdma = "rdma" in transports
    has_p2p = "p2p" in transports

    put_cases = {
        t: (_put_rdma(symbols[t]) if t == "rdma"
            else _put_call(symbols[t]["put"]))
        for t in transports
    }
    inline_cases = {
        t: _value_call(symbols[t]["put_inline"]) for t in transports
    }
    signal_cases = {
        t: _signal_call(symbols[t]["signal"]) for t in transports
    }
    quiet_cases = {
        t: [f"tier1::{symbols[t]['quiet']}(handle);"] for t in transports
    }
    put_signal_cases = {
        t: (_put_signal_rdma(symbols[t]) if t in ("rdma", "proxy", "ionic")
            else _put_signal_p2p(symbols[t]))
        for t in transports
    }
    put_many_cases = {}
    put_many_signal_cases = {}
    put_many_flush_cases = {}
    put_warp_cases = {}
    for t in transports:
        if t in ("rdma", "proxy", "ionic"):
            put_many_cases[t] = _put_many_rdma(symbols[t], sizes)
            put_many_signal_cases[t] = _put_many_fused_rdma(
                symbols[t], sizes, "signal")
            put_many_flush_cases[t] = _put_many_fused_rdma(
                symbols[t], sizes, "flush")
            put_warp_cases[t] = (_put_warp_rdma(symbols[t], hip=arch.startswith("gfx")) if t == "rdma" else [
                "if (threadIdx.x % warpSize == 0) {",
                *_put_call(symbols[t]["put"]), "}", "__syncwarp();"])

        else:
            put_many_cases[t] = _put_many_p2p(symbols[t])
            put_many_signal_cases[t] = _put_many_signal_p2p(symbols[t])
            # Peer-memory quiet is vacuous: the flush is just the copies.
            put_many_flush_cases[t] = _put_many_p2p(symbols[t])
            put_warp_cases[t] = _put_call(
                symbols[t].get("put_warp", symbols[t]["put"])
            )
    # wait_signal takes no peer, so there is no kind to dispatch on — every
    # transport's variant is the same local acquire-load spin; bind one.
    wait_transport = "rdma" if has_rdma else transports[0]
    wait_symbol = symbols[wait_transport]["wait_signal"]
    open_body = _open_body(transports, peer_transports)
    template_assert = _template_assert(peer_transports)
    specialized = peer_transports is not None or len(transports) == 1

    transport_list = ", ".join(transports)
    geography = (
        ", ".join(
            f"{peer}:{transport}" for peer, transport in peer_transports
        )
        if peer_transports is not None
        else (
            f"uniform {transports[0]}"
            if len(transports) == 1
            else "generic kind-tag dispatch"
        )
    )
    return f"""#pragma once

// Aboleth Tier 1: one-sided transport from arbitrary CUDA code.
//
//   DeviceContext ctx{{world_endpoints, world_doorbell_locks}};
//   aboleth::put(ctx, peer, remote_row, local_row, bytes);
//   aboleth::put_inline(ctx, peer, remote_flag, header_word);
//   aboleth::signal(ctx, peer, remote_flag, epoch);
//   aboleth::put_signal(ctx, peer, dst, src, bytes, flag, epoch);  // fused
//   aboleth::put_many(ctx, peer, descriptors, n);
//   aboleth::put_many_signal(ctx, peer, descriptors, n, flag, epoch);
//   aboleth::put_many_flush(ctx, peer, descriptors, n);  // == +quiet
//   aboleth::put_warp(ctx, peer, remote_row, local_row, bytes);  // full warp
//   aboleth::quiet(ctx, peer);
//   aboleth::wait_signal(ctx, local_flag, epoch);
//
// COMPILER-GENERATED for {arch} ({transport_list}); geography: {geography};
// library digest
// {library_digest}. Every op is a one-op transport program emitted by the
// same lowering the in-kernel (Tier 3) path uses — this header only binds
// the exported symbols. Link exactly one of:
//
//   nvcc  -rdc=true client.cu libaboleth_tier1_{arch}.a
//   clang++ -x cuda -fcuda-short-ptr \\
//       -Xclang -mlink-builtin-bitcode -Xclang aboleth_tier1_{arch}.bc
//   Triton: extern_libs={{"aboleth_tier1": "aboleth_tier1_{arch}.bc"}}
//   CuTe DSL: cute.extern(name=..., source=cute.BitCode(...), inline=True)
//
// Semantics match the hand-written facade this header replaces: put,
// put_inline, signal, and put_many post and doorbell without waiting; quiet
// is the only blocking call and retires everything posted to that peer so
// far; publication is ordered per endpoint without doorbell locks (the
// generated batch publishes in reservation order). quiet remains
// single-consumer per endpoint: give each concurrent consumer its own
// endpoint — connect_all(endpoints_per_peer=N) fans the table out so CTA b
// owns slot b % N of every peer (kEndpointsPerPeer below); ordering across
// slots to one peer is unordered, per-slot it stays FIFO.
// On peer-memory (p2p) peers a put completes when its stores
// retire and signal is a release-system store, so quiet is vacuous there.
// put_warp must be entered by a complete warp: on p2p all 32 lanes copy
// cooperatively behind a warp-scope fence; on rdma lane 0 posts one WQE.
//
// Generated by aboleth.device_library; do not edit.

#include <cstdint>

#define ABOLETH_TIER1_ABI {ABI_VERSION}
#define ABOLETH_TIER1_GENERATED 1
#define ABOLETH_TIER1_PUT_MANY 1
#define ABOLETH_TIER1_PUT_SIGNAL 1
#define ABOLETH_TIER1_PUT_MANY_FLUSH 1
#define ABOLETH_TIER1_SPECIALIZED {1 if specialized else 0}
#define ABOLETH_TIER1_HAS_RDMA {1 if has_rdma else 0}
#define ABOLETH_TIER1_HAS_P2P {1 if has_p2p else 0}
#define ABOLETH_TIER1_ENDPOINTS_PER_PEER {endpoints_per_peer}

// Opt-in: puts of exactly 8/16/24 aligned bytes ride inline in the command
// (no NIC gather of the source buffer). Off by default so the wire form of
// `put` stays byte-stable unless a build asks for the trade.
#ifndef ABOLETH_TIER1_SMALL_INLINE
#define ABOLETH_TIER1_SMALL_INLINE 0
#endif

namespace aboleth {{

struct LocalRegion {{
    std::uint64_t address;
    std::uint32_t key;
}};

struct RemoteRegion {{
    std::uint64_t address;
    std::uint32_t key;
}};

// One put_many descriptor. 32 bytes, 8-aligned; mirrors PUT_DESC_* in
// python/aboleth/tier1.py.
struct PutDesc {{
    std::uint64_t destination;
    std::uint64_t source;
    std::uint32_t bytes;
    std::uint32_t destination_key;
    std::uint32_t source_key;
    std::uint32_t _pad;
}};
static_assert(sizeof(PutDesc) == 32, "put_many descriptor layout");

// The transport tag PeerRegionTable.endpoint_handles carries in a handle's
// low three bits; descriptors are >=8-aligned so the bits are free, and rdma
// is zero so an untagged descriptor address is a valid rdma handle.
inline constexpr std::uint64_t kKindMask = {KIND_MASK:#x};
inline constexpr std::uint64_t kKindRdma = {KIND_BITS["rdma"]};
inline constexpr std::uint64_t kKindP2P = {KIND_BITS["p2p"]};
inline constexpr std::uint64_t kKindProxy = {KIND_BITS["proxy"]};
inline constexpr std::uint64_t kKindEfa = {KIND_BITS["efa"]};
{('inline constexpr std::uint64_t kKindIonic = 4;' + chr(10)) if 'ionic' in transports else ''}
// Kernel-argument view of one rank's endpoints: the per-peer table of
// endpoint handles (aboleth.PeerRegionTable.endpoint_handles). The generated
// ops publish in reservation order, so `doorbell_locks` is accepted for
// source compatibility and never read.
struct DeviceContext {{
    const std::uint64_t* endpoints;
    std::uint32_t* doorbell_locks;
}};

// A hoisted per-peer handle: the kind bits are resolved once, outside the
// hot path, never per op.
struct PeerRef {{
    std::uint64_t handle;
    std::uint64_t kind;
}};

// Endpoint fan-out (aboleth.World.connect_all(endpoints_per_peer=N)): the
// endpoint table is peer-major [world x N] and CTA b drives slot b % N, so
// CTAs on distinct slots never share an endpoint — quiet's single-consumer
// rule then holds per CTA for grids up to N blocks per peer. N == 1 folds
// to the plain per-peer read this library always emitted.
inline constexpr int kEndpointsPerPeer = {endpoints_per_peer};

__device__ __forceinline__ std::uint64_t tagged_endpoint(
    const DeviceContext& context, int peer) {{
    if constexpr (kEndpointsPerPeer == 1)
        return context.endpoints[peer];
    return context.endpoints[peer * kEndpointsPerPeer
                             + static_cast<int>(blockIdx.x)
                                   % kEndpointsPerPeer];
}}

namespace tier1 {{

{_declarations(transports, symbols)}

}} // namespace tier1

__device__ __forceinline__ PeerRef open(
    const DeviceContext& context, int peer) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
{open_body}
}}

__device__ __forceinline__ void put(
    const DeviceContext& context, int peer,
    const RemoteRegion& destination, const LocalRegion& source,
    std::uint32_t bytes) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
    const std::uint64_t handle = tagged & ~kKindMask;
{_dispatch(transports, put_cases, peer_transports)}
}}

// Eight bytes carried in the descriptor itself, straight out of registers.
__device__ __forceinline__ void put_inline(
    const DeviceContext& context, int peer,
    const RemoteRegion& destination, std::uint64_t value) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
    const std::uint64_t handle = tagged & ~kKindMask;
{_dispatch(transports, inline_cases, peer_transports)}
}}

// Publication: a strong epoch publication ordered behind every payload write
// posted before it, so a peer that observes the epoch observes the payload.
__device__ __forceinline__ void signal(
    const DeviceContext& context, int peer,
    const RemoteRegion& flag, std::uint64_t epoch) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
    const std::uint64_t handle = tagged & ~kKindMask;
{_dispatch(transports, signal_cases, peer_transports)}
}}

// The payload and its publication in ONE batch: one reservation, one
// doorbell, with the epoch still ordered behind the payload on the wire.
// Semantically identical to put(...) then signal(...), minus a full posting
// sequence. (On p2p peers it is exactly put + release-store signal.)
__device__ __forceinline__ void put_signal(
    const DeviceContext& context, int peer,
    const RemoteRegion& destination, const LocalRegion& source,
    std::uint32_t bytes, const RemoteRegion& flag, std::uint64_t epoch) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
    const std::uint64_t handle = tagged & ~kKindMask;
{_dispatch(transports, put_signal_cases, peer_transports)}
}}

// `count` descriptors to one peer as batched submissions: each generated
// variant is one reservation, N WQEs, and ONE doorbell — the per-call
// doorbell cost the one-put facade op cannot avoid. (Peer-memory puts have
// no doorbell, so a p2p peer just copies per descriptor.)
__device__ __forceinline__ void put_many(
    const DeviceContext& context, int peer,
    const PutDesc* descriptors, std::uint32_t count) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
    const std::uint64_t handle = tagged & ~kKindMask;
{_dispatch(transports, put_many_cases, peer_transports)}
}}

// put_many with the publication fused into the final batch: the epoch never
// pays a posting sequence of its own. Equivalent to put_many + signal.
__device__ __forceinline__ void put_many_signal(
    const DeviceContext& context, int peer,
    const PutDesc* descriptors, std::uint32_t count,
    const RemoteRegion& flag, std::uint64_t epoch) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
    const std::uint64_t handle = tagged & ~kKindMask;
{_dispatch(transports, put_many_signal_cases, peer_transports)}
}}

// put_many whose final batch carries its own signalled NOP and poll: the
// whole window is one reservation, one doorbell, one CQE, and returns with
// everything outstanding on the endpoint retired (quiet's contract, quiet's
// single-consumer rule). Equivalent to put_many + quiet.
__device__ __forceinline__ void put_many_flush(
    const DeviceContext& context, int peer,
    const PutDesc* descriptors, std::uint32_t count) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
    const std::uint64_t handle = tagged & ~kKindMask;
{_dispatch(transports, put_many_flush_cases, peer_transports)}
}}

// Warp-collective put; the calling warp must enter complete. On p2p all 32
// lanes copy cooperatively behind a warp-scope fence; on rdma lane 0 posts
// the one WQE and the warp leaves together.
__device__ __forceinline__ void put_warp(
    const DeviceContext& context, int peer,
    const RemoteRegion& destination, const LocalRegion& source,
    std::uint32_t bytes) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
    const std::uint64_t handle = tagged & ~kKindMask;
{_dispatch(transports, put_warp_cases, peer_transports)}
}}

// Sender-side completion up to here, which is also what reclaims the queue.
// Single-consumer per endpoint. Vacuous on p2p peers.
__device__ __forceinline__ void quiet(
    const DeviceContext& context, int peer) {{
    const std::uint64_t tagged = tagged_endpoint(context, peer);
    const std::uint64_t handle = tagged & ~kKindMask;
{_dispatch(transports, quiet_cases, peer_transports)}
}}

// Epochs are monotonic, so a later epoch satisfies an earlier wait.
__device__ __forceinline__ void wait_signal(
    const DeviceContext&, const std::uint64_t* flag, std::uint64_t epoch) {{
    tier1::{wait_symbol}(
        reinterpret_cast<std::uint64_t>(flag), epoch);
}}

// Compile-time-peer overloads. In a geography-specialized mixed library the
// forced-inline runtime wrapper sees a constant `peer`, folds the peer switch,
// and leaves only that peer's backend call.
template<int Peer>
__device__ __forceinline__ PeerRef open(const DeviceContext& context) {{
{template_assert}    return open(context, Peer);
}}

template<int Peer>
__device__ __forceinline__ void put(
    const DeviceContext& context, const RemoteRegion& destination,
    const LocalRegion& source, std::uint32_t bytes) {{
{template_assert}    put(context, Peer, destination, source, bytes);
}}

template<int Peer>
__device__ __forceinline__ void put_inline(
    const DeviceContext& context, const RemoteRegion& destination,
    std::uint64_t value) {{
{template_assert}    put_inline(context, Peer, destination, value);
}}

template<int Peer>
__device__ __forceinline__ void signal(
    const DeviceContext& context, const RemoteRegion& flag,
    std::uint64_t epoch) {{
{template_assert}    signal(context, Peer, flag, epoch);
}}

template<int Peer>
__device__ __forceinline__ void put_signal(
    const DeviceContext& context, const RemoteRegion& destination,
    const LocalRegion& source, std::uint32_t bytes,
    const RemoteRegion& flag, std::uint64_t epoch) {{
{template_assert}    put_signal(context, Peer, destination, source, bytes,
               flag, epoch);
}}

template<int Peer>
__device__ __forceinline__ void put_many(
    const DeviceContext& context, const PutDesc* descriptors,
    std::uint32_t count) {{
{template_assert}    put_many(context, Peer, descriptors, count);
}}

template<int Peer>
__device__ __forceinline__ void put_many_signal(
    const DeviceContext& context, const PutDesc* descriptors,
    std::uint32_t count, const RemoteRegion& flag, std::uint64_t epoch) {{
{template_assert}    put_many_signal(context, Peer, descriptors, count, flag, epoch);
}}

template<int Peer>
__device__ __forceinline__ void put_many_flush(
    const DeviceContext& context, const PutDesc* descriptors,
    std::uint32_t count) {{
{template_assert}    put_many_flush(context, Peer, descriptors, count);
}}

template<int Peer>
__device__ __forceinline__ void put_warp(
    const DeviceContext& context, const RemoteRegion& destination,
    const LocalRegion& source, std::uint32_t bytes) {{
{template_assert}    put_warp(context, Peer, destination, source, bytes);
}}

template<int Peer>
__device__ __forceinline__ void quiet(const DeviceContext& context) {{
{template_assert}    quiet(context, Peer);
}}

}} // namespace aboleth
"""
