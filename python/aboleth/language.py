import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate
from triton.language.core import base_type, base_value, builtin


@tl.core.extern
def globaltimer(_semantic=None):
    """Device wall time in nanoseconds on CUDA and the qualified MI355X."""
    arch = str(getattr(_semantic.builder.options, "arch", ""))
    if arch.startswith("gfx"):
        if arch != "gfx950":
            raise ValueError("HIP wall-clock frequency is qualified only for gfx950")
        from pathlib import Path
        path = str(Path(__file__).with_name("aboleth_hip.bc"))
        ticks = tl.core.extern_elementwise(
            "aboleth_hip", path, [], {(): ("aboleth_hip_clock", tl.int64)},
            is_pure=False, _semantic=_semantic)
        return ticks.__mul__(10, _semantic=_semantic)
    return tl.extra.cuda.globaltimer(_semantic=_semantic)


class token_type(base_type):
    """Frontend type for ``!aboleth.token``."""

    def __eq__(self, other):
        return isinstance(other, token_type)

    def __hash__(self):
        return hash(token_type)

    def mangle(self):
        return "AbolethToken"

    def _flatten_ir_types(self, builder, out):
        raise TypeError("Aboleth tokens cannot cross Triton function boundaries")

    def _unflatten_ir(self, handles, cursor):
        return token(handles[cursor]), cursor + 1


class token(base_value):
    """Compiler handle for an ordered Aboleth transfer."""

    def __init__(self, handle):
        self.handle = handle
        self.type = token_type()

    def _flatten_ir(self, handles):
        handles.append(self.handle)


class group_type(base_type):
    """Frontend type for ``!aboleth.group``."""

    def __eq__(self, other):
        return isinstance(other, group_type)

    def __hash__(self):
        return hash(group_type)

    def mangle(self):
        return "AbolethGroup"

    def _flatten_ir_types(self, builder, out):
        raise TypeError(
            "Aboleth completion groups cannot cross Triton function boundaries"
        )

    def _unflatten_ir(self, handles, cursor):
        return completion_group(handles[cursor]), cursor + 1


class completion_group(base_value):
    """Compiler handle for endpoint submissions completed by one wait."""

    def __init__(self, handle):
        self.handle = handle
        self.type = group_type()

    def _flatten_ir(self, handles):
        handles.append(self.handle)


class endpoint_type(base_type):
    """Frontend type for ``!aboleth.endpoint``."""

    def __eq__(self, other):
        return isinstance(other, endpoint_type)

    def __hash__(self):
        return hash(endpoint_type)

    def mangle(self):
        return "AbolethEndpoint"

    def _flatten_ir_types(self, builder, out):
        raise TypeError(
            "Aboleth endpoints cannot cross Triton function boundaries"
        )

    def _unflatten_ir(self, handles, cursor):
        return selected_endpoint(handles[cursor]), cursor + 1


class selected_endpoint(base_value):
    """An endpoint selected from a World's endpoint table."""

    def __init__(self, handle):
        self.handle = handle
        self.type = endpoint_type()

    def _flatten_ir(self, handles):
        handles.append(self.handle)


@aggregate
class region:
    """A registered buffer: an address and the key of its registration."""

    pointer: tl.tensor
    key: tl.tensor

    @triton.jit
    def at(self, offset):
        """The same registration, `offset` elements further on."""
        return region(self.pointer + offset, self.key)

    @builtin
    def valid(self, count, capacity, item_bytes, _semantic=None):
        """Declare valid rows followed by dead padding within this capacity."""
        capacity, item_bytes = _constexpr_value(capacity), _constexpr_value(item_bytes)
        if type(capacity) is not int or type(item_bytes) is not int or capacity < 1 or item_bytes < 1:
            raise ValueError("valid rows require positive constexpr capacity and item_bytes")
        if capacity * item_bytes >= 2**31:
            raise ValueError("valid row capacity must fit positive i32 transfer bytes")
        count = _semantic.cast(_semantic.to_tensor(count), tl.int64)
        bound = _semantic.to_tensor(capacity)
        normalized = _semantic.builder.create_aboleth_valid_extent([count.handle, bound.handle])
        zero = _semantic.cast(_semantic.to_tensor(0), tl.int64)
        return _valid_region(self.pointer, self.key, tl.tensor(normalized, tl.int64),
                             zero, tl.constexpr(capacity), tl.constexpr(item_bytes))


@aggregate
class _valid_region:
    pointer: tl.tensor
    key: tl.tensor
    count: tl.tensor
    byte_offset: tl.tensor
    capacity: tl.constexpr
    item_bytes: tl.constexpr

    @triton.jit
    def at(self, offset):
        pointer = self.pointer + offset
        displacement = pointer.to(tl.int64) - self.pointer.to(tl.int64)
        return _valid_region(pointer, self.key, self.count, self.byte_offset + displacement,
                             self.capacity, self.item_bytes)


@triton.jit
def remote(table, peer, dtype: tl.constexpr = tl.uint8):
    """The region `peer` exposes, from a World region table.

    The table holds an (address, key) pair per rank, as
    `PeerRegionTable.regions` builds it; the local rank's entry is its own
    buffer.
    """
    entry = table + 2 * peer
    address = tl.load(entry)
    key = tl.load(entry + 1).to(tl.uint32)
    return region(address.to(tl.pointer_type(dtype)), key)


def _bind(value, key, semantic):
    """A region's pointer and key, or a pointer and the key passed with it."""
    if isinstance(value, region):
        value, key = value.pointer, value.key
    return (
        semantic.to_tensor(value),
        semantic.cast(semantic.to_tensor(key), tl.uint32),
    )


def _endpoint(value, semantic):
    """A selected endpoint, or a descriptor pointer the plugin binds."""
    if isinstance(value, selected_endpoint):
        return value
    return semantic.to_tensor(value)


def _constexpr_value(value):
    return value.value if isinstance(value, tl.constexpr) else value


def _tuple_values(value):
    if isinstance(value, (tuple, list, tl.tuple)):
        return tuple(value)
    raise TypeError("transport bindings must be tuples")


@builtin
def invoke_transport_program(
    manifest,
    endpoint,
    transfers,
    publication,
    _semantic=None,
):
    """Invoke a bound program using typed transfer and publication tuples."""

    manifest = _constexpr_value(manifest)
    transfer_values = _tuple_values(transfers)
    if len(transfer_values) != len(manifest.transfers):
        raise ValueError(
            "transport invocation transfer count does not match manifest"
        )
    values = {"endpoint": endpoint}
    transfer_fields = (
        "destination",
        "source",
        "byte_count",
        "source_key",
        "destination_key",
        "predicate",
    )
    for transfer, bound in zip(
        manifest.transfers,
        transfer_values,
    ):
        bound = _tuple_values(bound)
        if len(bound) != len(transfer_fields):
            raise ValueError(
                "transport transfer binding requires destination, "
                "source, byte count, source key, destination key, "
                "and predicate"
            )
        for field, value in zip(transfer_fields, bound):
            values[getattr(transfer, field)] = value
    publication = _tuple_values(publication)
    publication_fields = (
        "destination",
        "source",
        "source_key",
        "destination_key",
        "iteration",
        "iterations",
    )
    if manifest.signal is None and publication:
        raise ValueError("a signal-free transport fragment has no publication operands")
    if manifest.signal is not None and len(publication) != len(publication_fields):
        raise ValueError(
            "transport publication binding requires destination, "
            "source, source key, destination key, iteration, "
            "and iterations"
        )
    for field, value in zip(publication_fields, publication):
        values[getattr(manifest.signal, field)] = value
    missing = [
        argument.name
        for argument in manifest.arguments
        if argument.name not in values
    ]
    if missing:
        raise ValueError(
            "transport invocation has unbound arguments: "
            + ", ".join(missing)
        )
    program_id = _semantic.cast(
        _semantic.to_tensor(manifest.program_id),
        tl.int32,
    )
    operands = [program_id.handle]
    operands.extend(
        (values[argument.name].handle
         if isinstance(values[argument.name], selected_endpoint)
         else _semantic.to_tensor(values[argument.name]).handle)
        for argument in manifest.arguments
    )
    _semantic.builder.create_aboleth_invoke_transport_phase(
        operands
    )


@builtin
def endpoint(table, index, _semantic=None):
    """Select the endpoint at `index` of a World's endpoint table.

    The World builds the table with one endpoint per index, so programs that
    select by distinct indices never share one. When the index is an injective
    function of the program id, the compiler gives each endpoint a single
    issuer: reserving its queue needs no atomic.
    """
    table = _semantic.to_tensor(table)
    index = _semantic.cast(_semantic.to_tensor(index), tl.int32)
    handle = _semantic.builder.create_aboleth_endpoint(
        [table.handle, index.handle]
    )
    return selected_endpoint(handle)


@builtin
def begin_group(endpoint, _semantic=None):
    """Begin a completion group on one ordered endpoint."""
    endpoint = _endpoint(endpoint, _semantic)
    handle = _semantic.builder.create_aboleth_begin_group([endpoint.handle])
    return completion_group(handle)


@builtin
def put(
    endpoint,
    destination,
    source,
    byte_count,
    source_key=0,
    destination_key=0,
    predicate=True,
    after=None,
    _semantic=None,
):
    """Schedule an ordered remote-memory write."""
    endpoint = _endpoint(endpoint, _semantic)
    destination, destination_key = _bind(
        destination, destination_key, _semantic
    )
    if isinstance(source, _valid_region):
        source_key = _semantic.cast(source.key, tl.uint32)
        capacity = _semantic.to_tensor(source.capacity)
        item_bytes = _semantic.to_tensor(source.item_bytes)
        source_handle = _semantic.builder.create_aboleth_valid_region([
            source.pointer.handle, source_key.handle, source.count.handle,
            source.byte_offset.handle, capacity.handle, item_bytes.handle])
    else:
        source, source_key = _bind(source, source_key, _semantic)
        source_handle = source.handle
    byte_count = _semantic.to_tensor(byte_count)
    predicate = _semantic.cast(
        _semantic.to_tensor(predicate),
        tl.int1,
    )
    arguments = [
        endpoint.handle,
        destination.handle,
        source_handle,
        byte_count.handle,
        source_key.handle,
        destination_key.handle,
        predicate.handle,
    ]
    if after is not None:
        if not isinstance(after, token):
            raise TypeError("after must be an Aboleth token")
        arguments.append(after.handle)
    handle = _semantic.builder.create_aboleth_put(arguments)
    return token(handle)


@builtin
def put_inline(
    endpoint,
    destination,
    *values,
    predicate=True,
    after=None,
    destination_key=0,
    _semantic=None,
):
    """Schedule an ordered remote write of register-resident values.

    The payload is the concatenation of the values' little-endian
    representations, so the transferred byte count is the sum of their widths.
    No local region, no registration key, and no staging buffer take part.

    Values are byte-sized integers. A float payload is sent by bitcasting it to
    an integer of the same width, since the transport copies bytes and never
    interprets them.
    """
    if not values:
        raise ValueError("put_inline requires at least one payload value")
    endpoint = _endpoint(endpoint, _semantic)
    destination, destination_key = _bind(
        destination, destination_key, _semantic
    )
    predicate = _semantic.cast(
        _semantic.to_tensor(predicate),
        tl.int1,
    )
    arguments = [
        endpoint.handle,
        destination.handle,
        destination_key.handle,
        predicate.handle,
    ]
    for value in values:
        value = _semantic.to_tensor(value)
        scalar = value.type.scalar
        if not scalar.is_int() or scalar.primitive_bitwidth % 8:
            raise TypeError(
                "put_inline payload values must be byte-sized integers, "
                f"not {scalar}"
            )
        arguments.append(value.handle)
    if after is not None:
        if not isinstance(after, token):
            raise TypeError("after must be an Aboleth token")
        arguments.append(after.handle)
    handle = _semantic.builder.create_aboleth_put_inline(arguments)
    return token(handle)


@builtin
def signal(
    endpoint,
    destination,
    source,
    after=None,
    source_key=0,
    destination_key=0,
    group=None,
    _semantic=None,
):
    """Publish epoch `source` to the remote flag `destination`.

    `source` is the epoch itself, an integer the transport carries inside the
    command where it can, or a pointer to a registered local cell holding it.
    Ordered after transfers (`after`), the signal makes them visible to a
    receiver that observes the epoch.
    """
    endpoint = _endpoint(endpoint, _semantic)
    destination, destination_key = _bind(
        destination, destination_key, _semantic
    )
    source, source_key = _bind(source, source_key, _semantic)
    if not source.type.is_ptr():
        source = _semantic.cast(source, tl.uint64)
    arguments = [
        endpoint.handle,
        destination.handle,
        source.handle,
        source_key.handle,
        destination_key.handle,
    ]
    if after is not None:
        if not isinstance(after, token):
            raise TypeError("after must be an Aboleth token")
        arguments.append(after.handle)
    if group is None:
        handle = _semantic.builder.create_aboleth_signal(arguments)
    else:
        if not isinstance(group, completion_group):
            raise TypeError("group must be an Aboleth completion group")
        arguments.append(group.handle)
        handle = _semantic.builder.create_aboleth_group_signal(arguments)
    return token(handle)


@builtin
def wait(transfer, predicate=None, iteration=None, reuse=None, _semantic=None):
    """Wait for sender-side completion of `transfer` and everything before it.

    A kernel that issues `transfer` once per `iteration` and overwrites its
    sources only `reuse` iterations later declares that instead: the compiler
    then completes only every k-th iteration, for the largest power of two k
    within the reuse distance, the transport's preferred interval, and the
    iterations the endpoint queue holds. The kernel drains its endpoints after
    the last iteration.

    With `predicate`, a false value defers the completion to a later wait
    whose predicate is true, which completes every submission before it on
    the endpoint; the kernel must end with one. The deferral is the kernel's
    own, so it may outrun the endpoint queue: on RDMA, when one program
    issues on each endpoint, the compiler then retires completed submissions
    whenever the queue is full (`self-reclaim`).
    """
    if not isinstance(transfer, token):
        raise TypeError("transfer must be an Aboleth token")
    if reuse is not None:
        reuse = _constexpr_value(reuse)
        if predicate is not None or iteration is None:
            raise ValueError("wait(reuse=) takes an iteration and no predicate")
        if not isinstance(reuse, int) or reuse < 1:
            raise ValueError("reuse must be a positive constexpr integer")
        iteration = _semantic.to_tensor(iteration)
        _semantic.builder.create_aboleth_reuse_wait(
            [transfer.handle, iteration.handle,
             _semantic.to_tensor(reuse).handle]
        )
        return
    if predicate is None:
        _semantic.builder.create_aboleth_wait(
            [transfer.handle]
        )
        return
    predicate = _semantic.cast(
        _semantic.to_tensor(predicate),
        tl.int1,
    )
    _semantic.builder.create_aboleth_wait_if(
        [transfer.handle, predicate.handle]
    )


@builtin
def planned_wait(
    transfer,
    iteration,
    iterations,
    _semantic=None,
):
    """Materialize sender completion from the active transport plan."""
    if not isinstance(transfer, token):
        raise TypeError("transfer must be an Aboleth token")
    iteration = _semantic.to_tensor(iteration)
    iterations = _semantic.cast(
        _semantic.to_tensor(iterations),
        iteration.type.scalar,
    )
    _semantic.builder.create_aboleth_planned_wait(
        [transfer.handle, iteration.handle, iterations.handle]
    )


@builtin
def wait_group(group, _semantic=None):
    """Wait once for every submission in a completion group."""
    if not isinstance(group, completion_group):
        raise TypeError("group must be an Aboleth completion group")
    _semantic.builder.create_aboleth_wait_group([group.handle])


@builtin
def system_fence(
    predicate=True,
    _semantic=None,
):
    predicate = _semantic.cast(
        _semantic.to_tensor(predicate),
        tl.int1,
    )
    (
        _semantic.builder
        .create_aboleth_system_fence(
            [predicate.handle]
        )
    )


@builtin
def drain(endpoint, _semantic=None):
    """Complete all outstanding sender work on an endpoint."""
    endpoint = _endpoint(endpoint, _semantic)
    _semantic.builder.create_aboleth_drain([endpoint.handle])


@builtin
def wait_signal(signal_pointer, expected, _semantic=None):
    """Wait until a local signal reaches the expected value."""
    if isinstance(signal_pointer, region):
        signal_pointer = signal_pointer.pointer
    signal_pointer = _semantic.to_tensor(signal_pointer)
    expected = _semantic.cast(_semantic.to_tensor(expected), tl.uint64)
    _semantic.builder.create_aboleth_wait_signal(
        [signal_pointer.handle, expected.handle]
    )
