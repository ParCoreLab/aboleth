"""The compiler-generated Tier-1 op programs.

Each Tier-1 device op (`put`, `put_inline`, `signal`, `quiet`, `wait_signal`,
`put_many`) is a one-op transport program compiled by the fragment pipeline
with detached inline emission (`device_bitcode=None`): the WQE construction,
publication ordering, and doorbell come from the same lowering Tier 3 trusts,
not from hand-written CUDA. `experiments/python/aboleth_tier1_differential.py`
holds the generated ops to bitwise WQE parity with the hand-written facade in
`include/aboleth/device.cuh`.

Facade parity, per op:

- `put` — reserve (capacity wait) -> one RDMA-write WQE (no CQE) -> ordered
  publication. `aboleth.submit {deferCompletion}` with no external publication
  is exactly that shape; a later `quiet` owns retirement.
- `put_inline` — the same batch around one `aboleth.post_inline`. The facade
  carries exactly 8 bytes; the direct emitter accepts up to the WQE ceiling,
  so wider variants are a generator option rather than new code.
- `signal` — a strong `aboleth.post_signal` of the epoch as a register value.
  Publication planning picks its mechanism: a responder compare-and-swap,
  like the facade, unless the library is built for strictly ordered
  registrations, where it is a fenced inline write. No `planned_wait`
  follows, matching the facade's fire-and-forget contract.
- `quiet` — `planned_drain_program`, unchanged: one signalled NOP, one CQ
  poll, single-consumer per endpoint.
- `wait_signal` — the standalone `aboleth.wait_signal` acquire-load spin.
- `put_many` — N descriptor-driven puts in ONE batch: one reservation, N
  WQEs, one doorbell. This is the composite the per-call facade cannot
  express, and the answer to its measured ~5.5 us/put doorbell cost.
- `put_signal` / `put_many_N_signal` — the payload put(s) and the strong
  epoch in the SAME batch: one reservation, one doorbell where the facade's
  put-then-signal pays two full posting sequences back to back.
- `put_many_N_flush` — the batch carries its own signalled NOP and a
  coalesced poll: the whole window is one reservation, one doorbell, one
  CQE, replacing the separate quiet's second posting sequence. The poll
  retires everything outstanding on the endpoint through this batch, so it
  keeps quiet's single-consumer contract.
- wider `put_inline_{16,24}` — multi-word register payloads (the direct
  emitter takes up to the WQE ceiling); embedding a small put's payload in
  the command removes the NIC's gather DMA of the source buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from .pattern import KernelTransportArgument, KernelTransportProgram
from .planned_phase import planned_drain_program

TIER1_MAX_REGISTERS = 64

# One `put_many` descriptor as the generated header lays it out:
#   struct PutDesc {  // 32 bytes, 8-aligned
#     uint64 destination;      // word 0: remote VA
#     uint64 source;           // word 1: local VA
#     uint32 bytes;            // word 2 low
#     uint32 destination_key;  // word 2 high
#     uint32 source_key;       // word 3 low
#     uint32 _pad;             // word 3 high
#   };
PUT_DESC_BYTES = 32
PUT_DESC_WORDS = 4

# The N variants the generator emits by default; the header's `put_many`
# composes them (largest variants plus a single-put remainder).
DEFAULT_PUT_MANY_SIZES = (8, 32)


def tier1_symbol(op: str, transport: str = "rdma") -> str:
    """The exported symbol for one op: aboleth_tier1_<op>_<transport>.

    Stable `extern "C"` names, scalar-only arguments — one symbol binds from
    nvcc -rdc, Triton extern_libs, and cute.extern alike. The architecture
    lives in file names, never in symbols.
    """
    return f"aboleth_tier1_{op}_{transport}"


@dataclass(frozen=True)
class RawTransportProgram:
    """A transport program authored as a raw fragment body.

    `KernelTransportProgram.to_mlir` narrates a fixed set of program shapes,
    and its exactly-one-terminal-contract validator cannot spell the Tier-1
    ops (a self-publishing put, a fire-and-forget strong epoch, a standalone
    wait). The materialization pass clones fragment bodies without inspecting
    them, so the body rides as text: the dialect verifier gates it at parse
    time and the WQE differential is the behavioural gate.

    Duck-types the surface `compile_fragment_artifact` consumes:
    `arguments`/`transfers`/`signal`/`reductions` feed the invoke manifest and
    the binding signature (which additionally hashes `body`), and
    `pointer_pointees` states the slot classification `_slots` would otherwise
    derive from transfers. `block_types` overrides a fragment block argument's
    spelled type: bind_* consumers take the default i64 placeholder, but a
    slot the body loads through must be a real !tt.ptr type.
    """

    name: str
    arguments: tuple[KernelTransportArgument, ...]
    body: str
    pointer_pointees: Mapping[str, str] = field(default_factory=dict)
    block_types: Mapping[str, str] = field(default_factory=dict)
    issuer_scope: str = "thread"
    program_id: int = 0
    artifact: str = "bindable_epilogue"
    transfers: tuple = ()
    signal: None = None
    reductions: tuple = ()
    external_publication: bool = False
    system_fence: bool = False
    drain_endpoint: bool = False
    publish_head: bool = False
    planned_reservation: bool = False
    reservation_scope: str = "gpu"
    # P2P only: the copy emission the lowering uses for this program's puts
    # ('', 'scalar', 'wide', 'warp'; see AbolethLowerP2PPass).
    p2p_copy_mode: str = ""

    def _block_type(self, argument: KernelTransportArgument) -> str:
        override = self.block_types.get(argument.name)
        if override is not None:
            return override
        if argument.name in self.pointer_pointees:
            # bind_* operands keep the i64 spelling the materialization pass
            # permits as a pointer placeholder.
            return "i64"
        return argument.mlir_type

    def to_mlir(self) -> str:
        names = ", ".join(
            f'"{argument.name}"' for argument in self.arguments
        )
        block = ", ".join(
            f"%arg{index}: {self._block_type(argument)}"
            for index, argument in enumerate(self.arguments)
        )
        return "\n".join((
            '    "aboleth.transport_fragment"() <{',
            f"      programId = {self.program_id} : i32,",
            f'      name = "{self.name}",',
            f"      argumentNames = [{names}]",
            "    }> ({",
            f"    ^bb0({block}):",
            self.body,
            "      ub.unreachable",
            "    }) : () -> ()",
        ))


def _arguments(*slots: tuple[str, str]) -> tuple[KernelTransportArgument, ...]:
    return tuple(
        KernelTransportArgument(name, mlir_type)
        for name, mlir_type in slots
    )


_SELF_PUBLISHING_SUBMIT = (
    '{{batchSize = {size} : i32, deferCompletion, issuerScope = "thread"}}'
)


def put_program(*, name: str = "tier1.put",
                p2p_copy_mode: str = "") -> RawTransportProgram:
    """One self-publishing put.

    Transport-neutral Queue IR: through lower-rdma it is reserve -> one WQE
    -> ordered doorbell; through lower-p2p the same batch is a peer-memory
    copy whose emission `p2p_copy_mode` selects.
    """
    submit = _SELF_PUBLISHING_SUBMIT.format(size=1)
    return RawTransportProgram(
        name=name,
        p2p_copy_mode=p2p_copy_mode,
        arguments=_arguments(
            ("endpoint", "i64"),
            ("destination", "i64"),
            ("source", "i64"),
            ("bytes", "i32"),
            ("source_key", "i32"),
            ("destination_key", "i32"),
            ("predicate", "i1"),
        ),
        pointer_pointees={
            "endpoint": "i64",
            "destination": "i8",
            "source": "i8",
        },
        body="\n".join((
            "      %endpoint = aboleth.bind_endpoint %arg0"
            " : i64 -> !aboleth.endpoint",
            "      %destination = aboleth.bind_remote_region %arg1, %arg5"
            " : (i64, i32) -> !aboleth.remote_region",
            "      %source = aboleth.bind_local_region %arg2, %arg4"
            " : (i64, i32) -> !aboleth.local_region",
            "      %put = aboleth.post_put"
            " %endpoint, %destination, %source, %arg3, %arg6"
            " : (!aboleth.endpoint, !aboleth.remote_region,"
            " !aboleth.local_region, i32, i1) -> !aboleth.token",
            f"      %submission = aboleth.submit %endpoint, %put {submit}"
            " : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token",
        )),
    )


def _inline_program(name: str, words: int = 1) -> RawTransportProgram:
    submit = _SELF_PUBLISHING_SUBMIT.format(size=1)
    value_slots = tuple(
        (f"value{index}" if words > 1 else "value", "i64")
        for index in range(words)
    )
    values = ", ".join(f"%arg{3 + index}" for index in range(words))
    value_types = ", ".join("i64" for _ in range(words))
    return RawTransportProgram(
        name=name,
        arguments=_arguments(
            ("endpoint", "i64"),
            ("destination", "i64"),
            ("destination_key", "i32"),
            *value_slots,
        ),
        pointer_pointees={
            "endpoint": "i64",
            "destination": "i8",
        },
        body="\n".join((
            "      %true = arith.constant true",
            "      %endpoint = aboleth.bind_endpoint %arg0"
            " : i64 -> !aboleth.endpoint",
            "      %destination = aboleth.bind_remote_region %arg1, %arg2"
            " : (i64, i32) -> !aboleth.remote_region",
            "      %put = aboleth.post_inline"
            f" %endpoint, %destination, [{values}], %true"
            f" : (!aboleth.endpoint, !aboleth.remote_region, {value_types},"
            " i1) -> !aboleth.token",
            f"      %submission = aboleth.submit %endpoint, %put {submit}"
            " : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token",
        )),
    )


def put_inline_program(
    *, name: str = "tier1.put_inline"
) -> RawTransportProgram:
    """One 8-byte register-valued inline write, self-publishing."""
    return _inline_program(name)


def put_inline_wide_program(
    words: int, *, name: str | None = None
) -> RawTransportProgram:
    """A multi-word register-valued inline write (8*words bytes).

    The payload rides in the command, so the NIC never gathers the source
    buffer: for small puts this trades one remote DMA read for a slightly
    larger WQE. 24 bytes is the largest payload that still fits the ctrl +
    remote-address segments in one 64-byte WQEBB alongside the inline header.
    """
    if words < 2:
        raise ValueError("wide inline variants start at two words; "
                         "use put_inline")
    return _inline_program(
        name or f"tier1.put_inline_{words * 8}", words=words)


def signal_program(*, name: str = "tier1.signal") -> RawTransportProgram:
    """The facade's signal: a strong publication of the epoch.

    Fire-and-forget — no planned wait follows. Strong publication orders the
    epoch after every earlier command on the queue, which is what lets a
    consumer's `wait_signal` observe it as publication.
    """
    submit = _SELF_PUBLISHING_SUBMIT.format(size=1)
    return RawTransportProgram(
        name=name,
        arguments=_arguments(
            ("endpoint", "i64"),
            ("destination", "i64"),
            ("destination_key", "i32"),
            ("value", "i64"),
        ),
        pointer_pointees={
            "endpoint": "i64",
            "destination": "i8",
        },
        body="\n".join((
            "      %endpoint = aboleth.bind_endpoint %arg0"
            " : i64 -> !aboleth.endpoint",
            "      %destination = aboleth.bind_remote_region %arg1, %arg2"
            " : (i64, i32) -> !aboleth.remote_region",
            "      %signal = aboleth.post_signal"
            " %endpoint, %destination, %arg3"
            " : (!aboleth.endpoint, !aboleth.remote_region, i64)"
            " -> !aboleth.token",
            f"      %submission = aboleth.submit %endpoint, %signal {submit}"
            " : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token",
        )),
    )


def put_signal_program(
    *, name: str = "tier1.put_signal"
) -> RawTransportProgram:
    """The payload put and the strong epoch in ONE batch.

    The facade's put-then-signal runs the full posting protocol twice — two
    reservations, two ordered publications, two doorbells — and staged
    submission prices every extra doorbell at ~5.5 us. Here both commands
    share one reservation and one doorbell; the epoch is still a strong
    publication ordered behind the payload, so the receiver contract is
    unchanged.
    """
    submit = _SELF_PUBLISHING_SUBMIT.format(size=2)
    return RawTransportProgram(
        name=name,
        arguments=_arguments(
            ("endpoint", "i64"),
            ("destination", "i64"),
            ("source", "i64"),
            ("bytes", "i32"),
            ("source_key", "i32"),
            ("destination_key", "i32"),
            ("predicate", "i1"),
            ("flag", "i64"),
            ("flag_key", "i32"),
            ("epoch", "i64"),
        ),
        pointer_pointees={
            "endpoint": "i64",
            "destination": "i8",
            "source": "i8",
            "flag": "i8",
        },
        body="\n".join((
            "      %endpoint = aboleth.bind_endpoint %arg0"
            " : i64 -> !aboleth.endpoint",
            "      %destination = aboleth.bind_remote_region %arg1, %arg5"
            " : (i64, i32) -> !aboleth.remote_region",
            "      %source = aboleth.bind_local_region %arg2, %arg4"
            " : (i64, i32) -> !aboleth.local_region",
            "      %flag = aboleth.bind_remote_region %arg7, %arg8"
            " : (i64, i32) -> !aboleth.remote_region",
            "      %put = aboleth.post_put"
            " %endpoint, %destination, %source, %arg3, %arg6"
            " : (!aboleth.endpoint, !aboleth.remote_region,"
            " !aboleth.local_region, i32, i1) -> !aboleth.token",
            "      %signal = aboleth.post_signal"
            " %endpoint, %flag, %arg9, %put"
            " : (!aboleth.endpoint, !aboleth.remote_region, i64,"
            " !aboleth.token) -> !aboleth.token",
            "      %submission = aboleth.submit %endpoint, %put, %signal"
            f" {submit}"
            " : (!aboleth.endpoint, !aboleth.token, !aboleth.token)"
            " -> !aboleth.token",
        )),
    )


def quiet_program(*, name: str = "tier1.quiet") -> KernelTransportProgram:
    """Capacity reclamation: one signalled NOP, one CQ poll.

    Single-consumer per endpoint, exactly like the facade's `quiet`; the
    multi-CTA answer is per-CTA endpoint fan-out, not a shared drain.
    """
    return planned_drain_program(name=name)


def wait_signal_program(
    *, name: str = "tier1.wait_signal"
) -> RawTransportProgram:
    """The standalone acquire-load spin on a local epoch cell."""
    return RawTransportProgram(
        name=name,
        arguments=_arguments(
            ("signal", "i64"),
            ("expected", "i64"),
        ),
        pointer_pointees={"signal": "i64"},
        body="\n".join((
            "      %zero = arith.constant 0 : i32",
            "      %signal = aboleth.bind_local_region %arg0, %zero"
            " : (i64, i32) -> !aboleth.local_region",
            "      aboleth.wait_signal %signal, %arg1"
            " : (!aboleth.local_region, i64) -> ()",
        )),
    )


def _put_many_lines(count: int) -> list[str]:
    """The unrolled descriptor-put body shared by every put_many variant."""
    if count < 2:
        raise ValueError("put_many needs at least two puts; use put")
    lines = [
        "      %true = arith.constant true",
        "      %c32 = arith.constant 32 : i64",
        "      %endpoint = aboleth.bind_endpoint %arg0"
        " : i64 -> !aboleth.endpoint",
    ]
    for index in range(count):
        base = index * PUT_DESC_WORDS
        chain = f", %put{index - 1}" if index else ""
        chain_type = ", !aboleth.token" if index else ""
        lines.extend((
            f"      %od{index} = arith.constant {base} : i32",
            f"      %pd{index} = tt.addptr %arg1, %od{index}"
            " : !tt.ptr<i64>, i32",
            f"      %wd{index} = tt.load %pd{index} : !tt.ptr<i64>",
            f"      %os{index} = arith.constant {base + 1} : i32",
            f"      %ps{index} = tt.addptr %arg1, %os{index}"
            " : !tt.ptr<i64>, i32",
            f"      %ws{index} = tt.load %ps{index} : !tt.ptr<i64>",
            f"      %om{index} = arith.constant {base + 2} : i32",
            f"      %pm{index} = tt.addptr %arg1, %om{index}"
            " : !tt.ptr<i64>, i32",
            f"      %wm{index} = tt.load %pm{index} : !tt.ptr<i64>",
            f"      %ok{index} = arith.constant {base + 3} : i32",
            f"      %pk{index} = tt.addptr %arg1, %ok{index}"
            " : !tt.ptr<i64>, i32",
            f"      %wk{index} = tt.load %pk{index} : !tt.ptr<i64>",
            f"      %bytes{index} = arith.trunci %wm{index} : i64 to i32",
            f"      %dkw{index} = arith.shrui %wm{index}, %c32 : i64",
            f"      %dkey{index} = arith.trunci %dkw{index} : i64 to i32",
            f"      %skey{index} = arith.trunci %wk{index} : i64 to i32",
            f"      %dp{index} = tt.int_to_ptr %wd{index}"
            " : i64 -> !tt.ptr<i8>",
            f"      %sp{index} = tt.int_to_ptr %ws{index}"
            " : i64 -> !tt.ptr<i8>",
            f"      %dst{index} = aboleth.bind_remote_region"
            f" %dp{index}, %dkey{index}"
            " : (!tt.ptr<i8>, i32) -> !aboleth.remote_region",
            f"      %src{index} = aboleth.bind_local_region"
            f" %sp{index}, %skey{index}"
            " : (!tt.ptr<i8>, i32) -> !aboleth.local_region",
            f"      %put{index} = aboleth.post_put"
            f" %endpoint, %dst{index}, %src{index}, %bytes{index},"
            f" %true{chain}"
            " : (!aboleth.endpoint, !aboleth.remote_region,"
            f" !aboleth.local_region, i32, i1{chain_type})"
            " -> !aboleth.token",
        ))
    return lines


def _put_many_raw(
    name: str,
    lines: list[str],
    extra_arguments: tuple[tuple[str, str], ...] = (),
    extra_pointees: Mapping[str, str] | None = None,
) -> RawTransportProgram:
    pointees = {"endpoint": "i64", "descriptors": "i64"}
    pointees.update(extra_pointees or {})
    return RawTransportProgram(
        name=name,
        arguments=_arguments(
            ("endpoint", "i64"),
            ("descriptors", "i64"),
            *extra_arguments,
        ),
        pointer_pointees=pointees,
        # The body loads through the descriptor pointer, so its block
        # argument is a real pointer rather than the bind placeholder.
        block_types={"descriptors": "!tt.ptr<i64>"},
        body="\n".join(lines),
    )


def put_many_program(
    count: int, *, name: str | None = None
) -> RawTransportProgram:
    """`count` descriptor-driven puts in one batch: N WQEs, one doorbell.

    The descriptor array is `count` PutDesc records (layout above). The loads
    are unrolled — the batch's WQEBB cost and each command's offset are
    compile-time facts, which is what lets form-batches' single reservation
    and single publication stand for the whole batch.
    """
    lines = _put_many_lines(count)
    tokens = ", ".join(f"%put{index}" for index in range(count))
    token_types = ", ".join("!aboleth.token" for _ in range(count))
    submit = _SELF_PUBLISHING_SUBMIT.format(size=count)
    lines.append(
        f"      %submission = aboleth.submit %endpoint, {tokens} {submit}"
        f" : (!aboleth.endpoint, {token_types}) -> !aboleth.token"
    )
    return _put_many_raw(name or f"tier1.put_many_{count}", lines)


def put_many_signal_program(
    count: int, *, name: str | None = None
) -> RawTransportProgram:
    """`count` puts and the strong epoch in one batch: N+1 WQEs, one
    doorbell. The batched analogue of `put_signal`."""
    lines = _put_many_lines(count)
    lines.extend((
        "      %flag = aboleth.bind_remote_region %arg2, %arg3"
        " : (i64, i32) -> !aboleth.remote_region",
        "      %signal = aboleth.post_signal"
        f" %endpoint, %flag, %arg4, %put{count - 1}"
        " : (!aboleth.endpoint, !aboleth.remote_region, i64,"
        " !aboleth.token) -> !aboleth.token",
    ))
    tokens = ", ".join(f"%put{index}" for index in range(count))
    token_types = ", ".join("!aboleth.token" for _ in range(count))
    submit = _SELF_PUBLISHING_SUBMIT.format(size=count + 1)
    lines.append(
        f"      %submission = aboleth.submit %endpoint, {tokens}, %signal"
        f" {submit}"
        f" : (!aboleth.endpoint, {token_types}, !aboleth.token)"
        " -> !aboleth.token"
    )
    return _put_many_raw(
        name or f"tier1.put_many_{count}_signal", lines,
        extra_arguments=(
            ("flag", "i64"), ("flag_key", "i32"), ("epoch", "i64"),
        ),
        extra_pointees={"flag": "i8"},
    )


def put_many_flush_program(
    count: int, *, name: str | None = None
) -> RawTransportProgram:
    """`count` puts, a signalled NOP, and a coalesced poll in one batch.

    One reservation, one doorbell, exactly one CQE for the whole window —
    where put_many + quiet pays a second full posting sequence for the NOP.
    The coalesced poll retires everything outstanding on the endpoint through
    this batch, so the op keeps quiet's contract (single consumer per
    endpoint) and a caller needs no separate quiet for work it flushed here.
    """
    lines = _put_many_lines(count)
    lines.append(
        f"      %nop = aboleth.post_nop %endpoint, %put{count - 1}"
        " : (!aboleth.endpoint, !aboleth.token) -> !aboleth.token"
    )
    tokens = ", ".join(f"%put{index}" for index in range(count))
    token_types = ", ".join("!aboleth.token" for _ in range(count))
    submit = (
        f'{{batchSize = {count + 1} : i32, issuerScope = "thread"}}'
    )
    lines.extend((
        f"      %submission = aboleth.submit %endpoint, {tokens}, %nop"
        f" {submit}"
        f" : (!aboleth.endpoint, {token_types}, !aboleth.token)"
        " -> !aboleth.token",
        "      aboleth.poll %submission {coalesced} : !aboleth.token",
    ))
    return _put_many_raw(name or f"tier1.put_many_{count}_flush", lines)


def _asm_store_program(
    name: str, *, release: bool, hip_device_library: str | None = None
) -> RawTransportProgram:
    """An 8-byte register-valued store to a peer-mapped address.

    P2P's value ops need no queue and no dialect vocabulary: put_inline is a
    plain global store through the IPC mapping, and signal is the same store
    with release-system ordering (the publication a p2p wait_signal's
    acquire load pairs with). Emitted as unpredicated inline asm because a
    detached caller's lane id must not decide whether the store lands.
    """
    if hip_device_library is not None:
        import json
        return RawTransportProgram(
            name=name,
            arguments=_arguments(("endpoint", "i64"), ("destination", "i64"),
                                 ("destination_key", "i32"), ("value", "i64")),
            pointer_pointees={"destination": "i64"},
            block_types={"destination": "!tt.ptr<i64>"},
            body=(
                '      %address = tt.ptr_to_int %arg1 : !tt.ptr<i64> -> i64\n'
                '      %zero = arith.constant 0 : i32\n'
                '      %stored = tt.extern_elementwise %address, %arg3, %zero '
                '{libname = "aboleth_hip", libpath = '
                + json.dumps(hip_device_library)
                + ', symbol = "aboleth_hip_release64", pure = false}'
                ' : (i64, i64, i32) -> i64'
            ),
        )
    ordering = "st.release.sys.global.b64" if release else "st.global.b64"
    asm = "{ mov.b64 $0, $2; " + ordering + " [$1], $2; }"
    return RawTransportProgram(
        name=name,
        arguments=_arguments(
            ("endpoint", "i64"),
            ("destination", "i64"),
            ("destination_key", "i32"),
            ("value", "i64"),
        ),
        pointer_pointees={"destination": "i64"},
        block_types={"destination": "!tt.ptr<i64>"},
        body=(
            f'      %stored = tt.elementwise_inline_asm "{asm}"'
            ' {constraints = "=l,l,l", packed_element = 1 : i32,'
            ' pure = false} %arg1, %arg3 : !tt.ptr<i64>, i64 -> i64'
        ),
    )


def p2p_quiet_program(*, name: str = "tier1.p2p.quiet"):
    """Vacuous on peer memory: a put is complete when its stores retire
    and the signal's release ordering already publishes them. The empty body
    keeps the ABI uniform with the queue-backed transports."""
    return RawTransportProgram(
        name=name,
        arguments=_arguments(("endpoint", "i64"),),
        pointer_pointees={},
        body="      %noop = arith.constant 0 : i32",
    )


def tier1_programs(
    transport: str = "rdma",
    *,
    put_many_sizes: tuple[int, ...] = DEFAULT_PUT_MANY_SIZES,
    hip_device_library: str | None = None,
):
    """op name -> program, for one transport's library build."""
    if transport == "ionic":
        return {op: replace(program, name=f"tier1.ionic.{op}")
                for op, program in tier1_programs(
                    "proxy", put_many_sizes=put_many_sizes,
                    hip_device_library=hip_device_library).items()}
    if transport == "proxy":
        programs = {
            "put": put_program(name="tier1.proxy.put"),
            "put_signal": put_signal_program(name="tier1.proxy.put_signal"),
            "put_inline": put_inline_program(name="tier1.proxy.put_inline"),
            "signal": signal_program(name="tier1.proxy.signal"),
            "quiet": quiet_program(name="tier1.proxy.quiet"),
            "wait_signal": wait_signal_program(name="tier1.proxy.wait_signal"),
        }
        for count in put_many_sizes:
            programs[f"put_many_{count}"] = put_many_program(count)
            programs[f"put_many_{count}_signal"] = put_many_signal_program(count)
            programs[f"put_many_{count}_flush"] = put_many_flush_program(count)
        return programs
    if transport == "rdma":
        programs = {
            "put": put_program(),
            "put_signal": put_signal_program(),
            "put_inline": put_inline_program(),
            "put_inline16": put_inline_wide_program(2),
            "put_inline24": put_inline_wide_program(3),
            "signal": signal_program(),
            "quiet": quiet_program(),
            "wait_signal": wait_signal_program(),
        }
        for count in put_many_sizes:
            programs[f"put_many_{count}"] = put_many_program(count)
            programs[f"put_many_{count}_signal"] = (
                put_many_signal_program(count))
            programs[f"put_many_{count}_flush"] = (
                put_many_flush_program(count))
        return programs
    if transport == "p2p":
        return {
            "put": put_program(
                name="tier1.p2p.put", p2p_copy_mode="wide"),
            "put_warp": put_program(
                name="tier1.p2p.put_warp", p2p_copy_mode="warp"),
            "put_inline": _asm_store_program(
                "tier1.p2p.put_inline", release=False,
                hip_device_library=hip_device_library),
            "signal": _asm_store_program(
                "tier1.p2p.signal", release=True,
                hip_device_library=hip_device_library),
            "quiet": p2p_quiet_program(),
            "wait_signal": wait_signal_program(
                name="tier1.p2p.wait_signal"),
        }
    raise ValueError(
        f"transport {transport!r} has no Tier-1 program set yet"
    )
