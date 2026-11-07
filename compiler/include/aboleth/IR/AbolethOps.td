#ifndef ABOLETH_OPS
#define ABOLETH_OPS

include "aboleth/IR/AbolethDialect.td"
include "aboleth/IR/AbolethTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AbolethEndpointState
    : Resource<"::mlir::aboleth::EndpointState">;

class Aboleth_Op<string mnemonic, list<Trait> traits = []>
    : Op<Aboleth_Dialect, mnemonic, traits>;

def Aboleth_BindEndpointOp
    : Aboleth_Op<"bind_endpoint", [Pure]> {
  let summary = "Bind a frontend endpoint value to an Aboleth endpoint";
  let arguments = (ins AnyType:$value);
  let results = (outs Aboleth_EndpointType:$endpoint);
  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($endpoint)
  }];
}

def Aboleth_EndpointOp : Aboleth_Op<"endpoint", [Pure]> {
  let summary = "Select an endpoint from a per-index table";
  let description = [{
    `table` holds one 64-bit endpoint handle per index, as a World builds it,
    so distinct indices name distinct endpoints. The low two bits of a handle
    may tag its transport and are not part of the descriptor address.

    `exclusiveAxis` records that the index is an injective function of the
    program id along that axis, so each endpoint has one issuing program as
    long as the grid is one-dimensional along it; the lowering checks that.
    `exclusiveAxes` generalizes this proof to a linearized two-dimensional
    grid. `exclusiveGridShape` contains three launch constraints: zero leaves
    a selected axis unconstrained, and a positive value requires that extent.
    Unselected axes must have extent one. These are proof results, not caller
    requests to omit synchronization.
  }];
  let arguments = (ins
    AnyType:$table,
    I32:$index,
    OptionalAttr<I32Attr>:$exclusiveAxis,
    OptionalAttr<DenseI32ArrayAttr>:$exclusiveAxes,
    OptionalAttr<DenseI64ArrayAttr>:$exclusiveGridShape
  );
  let results = (outs Aboleth_EndpointType:$endpoint);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $table `[` $index `]` attr-dict `:` type($table) `->` type($endpoint)
  }];
}

def Aboleth_BindLocalRegionOp
    : Aboleth_Op<"bind_local_region", [Pure]> {
  let summary = "Bind a frontend pointer and key to a local region";
  let arguments = (ins AnyType:$pointer, I32:$key);
  let results = (outs Aboleth_LocalRegionType:$region);
  let assemblyFormat = [{
    $pointer `,` $key attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_ValidExtentOp : Aboleth_Op<"valid_extent", [Pure]> {
  let summary = "Normalize a dynamic valid count within its declared capacity";
  let arguments = (ins I64:$count, I64Attr:$capacity);
  let results = (outs I64:$normalized);
  let hasVerifier = 1;
  let assemblyFormat = "$count attr-dict `:` type($count)";
}

def Aboleth_BindValidLocalRegionOp
    : Aboleth_Op<"bind_valid_local_region", [Pure]> {
  let summary = "Bind a source region with dynamic valid rows and dead padding";
  let arguments = (ins AnyType:$pointer, I32:$key, I64:$count,
      I64:$byteOffset, I64Attr:$capacity, I64Attr:$itemBytes);
  let results = (outs Aboleth_LocalRegionType:$region);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $pointer `,` $key `,` $count `,` $byteOffset attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_BindRemoteRegionOp
    : Aboleth_Op<"bind_remote_region", [Pure]> {
  let summary = "Bind a frontend pointer and key to a remote region";
  let arguments = (ins AnyType:$pointer, I32:$key);
  let results = (outs Aboleth_RemoteRegionType:$region);
  let assemblyFormat = [{
    $pointer `,` $key attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_BeginGroupOp : Aboleth_Op<"begin_group"> {
  let summary = "Begin a backend-neutral completion group on one endpoint";
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint
  );
  let results = (outs Aboleth_GroupType:$group);
  let assemblyFormat = [{
    $endpoint attr-dict `:` type($endpoint) `->` type($group)
  }];
}

def Aboleth_PutOp : Aboleth_Op<"put"> {
  let summary = "Schedule an ordered remote-memory write";
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    Arg<Aboleth_RemoteRegionType, "", [
      MemWrite<DefaultResource>
    ]>:$destination,
    Arg<Aboleth_LocalRegionType, "", [
      MemRead<DefaultResource>
    ]>:$source,
    I32:$bytes,
    I1:$predicate,
    Optional<Aboleth_TokenType>:$after
  );
  let results = (outs Aboleth_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $endpoint `,` $destination `,` $source `,` $bytes
    `,` $predicate
    (`,` $after^)?
    attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_PutInlineOp
    : Aboleth_Op<"put_inline", [AttrSizedOperandSegments]> {
  let summary = "Schedule an ordered remote write of register-resident values";
  let description = [{
    Carries its payload as operands rather than through a local region. The
    values are written to the destination in operand order, each in its own
    little-endian byte representation, so the payload byte count is the sum of
    the operand widths. No local region, no registration key, and no staging
    store take part: the transport embeds the bytes in the command it builds.
  }];
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    Arg<Aboleth_RemoteRegionType, "", [
      MemWrite<DefaultResource>
    ]>:$destination,
    Variadic<AnyInteger>:$values,
    I1:$predicate,
    Optional<Aboleth_TokenType>:$after
  );
  let results = (outs Aboleth_TokenType:$token);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // Total payload bytes, the sum of the operand widths.
    uint64_t getPayloadBytes();
  }];
  let assemblyFormat = [{
    $endpoint `,` $destination `,` `[` $values `]` `,` $predicate
    (`,` $after^)?
    attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_SignalOp
    : Aboleth_Op<"signal", [AttrSizedOperandSegments]> {
  let summary = "Publish a remote signal after an ordered transfer";
  let description = [{
    Publishes the epoch `source` to the remote flag `destination`. The epoch is
    an i64 value, which travels inside the command where the transport
    allows, or a local region holding it. Publication planning decides how a
    strong signal is ordered: `fencedSignal` selects a fenced write over the
    transport's default ordering point. `completedPrefix` selects sender
    completion of the preceding payload before an unchecked epoch write.
  }];
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    Arg<Aboleth_RemoteRegionType, "", [
      MemWrite<DefaultResource>
    ]>:$destination,
    Arg<AnyTypeOf<[Aboleth_LocalRegionType, I64]>, "", [
      MemRead<DefaultResource>
    ]>:$source,
    Optional<Aboleth_TokenType>:$after,
    Optional<Aboleth_GroupType>:$group,
    OptionalAttr<UnitAttr>:$weakPublication,
    OptionalAttr<UnitAttr>:$fencedSignal,
    OptionalAttr<UnitAttr>:$completedPrefix
  );
  let results = (outs Aboleth_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $endpoint `,` $destination `,` $source
    (`,` $after^)?
    (`group` $group^)?
    attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_WaitOp : Aboleth_Op<"wait", [
  MemoryEffects<[
    MemRead<AbolethEndpointState>,
    MemWrite<AbolethEndpointState>,
    MemRead<DefaultResource>,
    MemWrite<DefaultResource>
  ]>
]> {
  let summary = "Wait for sender-side completion of a transfer";
  let arguments = (ins
    Aboleth_TokenType:$transfer,
    Optional<I1>:$predicate,
    // The predicate is the kernel's own (`wait(predicate=)`), so nothing
    // bounds how many submissions it defers by the queue's depth. A backend
    // with a bounded queue reclaims completed slots while it waits for
    // capacity instead of relying on a poll that may not come in time. Set by
    // the frontend; requires a predicate.
    OptionalAttr<UnitAttr>:$selfReclaim
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $transfer (`,` $predicate^)? attr-dict `:` type($transfer)
  }];
}

def Aboleth_PlannedWaitOp : Aboleth_Op<"planned_wait", [
  MemoryEffects<[
    MemRead<AbolethEndpointState>,
    MemWrite<AbolethEndpointState>,
    MemRead<DefaultResource>,
    MemWrite<DefaultResource>
  ]>
]> {
  let summary = "Materialize sender completion from a transport plan";
  let description = [{
    Sender completion of a transfer issued once per `iteration`. With
    `reuseDistance`, the kernel declares that the transfer's sources are next
    overwritten that many iterations later, and planning derives how often a
    completion is due from the transport and its queue. Without it, an
    external plan supplies the interval and `iterations` names the final
    iteration, which always completes.
  }];
  let arguments = (ins
    Aboleth_TokenType:$transfer,
    AnyInteger:$iteration,
    Optional<AnyInteger>:$iterations,
    OptionalAttr<I64Attr>:$reuseDistance,
    OptionalAttr<I64Attr>:$completionInterval,
    OptionalAttr<I64Attr>:$queueCapacity
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $transfer `,` $iteration (`,` $iterations^)? attr-dict
    `:` type($transfer) `,` type($iteration) (`,` type($iterations)^)?
  }];
}

def Aboleth_WaitGroupOp : Aboleth_Op<"wait_group", [
  MemoryEffects<[
    MemRead<AbolethEndpointState>,
    MemWrite<AbolethEndpointState>,
    MemRead<DefaultResource>,
    MemWrite<DefaultResource>
  ]>
]> {
  let summary = "Wait once for all submissions in a completion group";
  let arguments = (ins Aboleth_GroupType:$group);
  let hasVerifier = 1;
  let assemblyFormat = "$group attr-dict `:` type($group)";
}

def Aboleth_WaitSignalOp : Aboleth_Op<"wait_signal", [
  MemoryEffects<[
    MemRead<DefaultResource>,
    MemWrite<DefaultResource>
  ]>
]> {
  let summary = "Wait until a local signal reaches an expected value";
  let arguments = (ins
    Aboleth_LocalRegionType:$signal,
    I64:$expected
  );
  let assemblyFormat = [{
    $signal `,` $expected attr-dict `:` functional-type(operands, results)
  }];
}

def Aboleth_PostPutOp : Aboleth_Op<"post_put"> {
  let summary = "Construct a remote-write command without submitting it";
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    Arg<Aboleth_RemoteRegionType, "", [
      MemWrite<DefaultResource>
    ]>:$destination,
    Arg<Aboleth_LocalRegionType, "", [
      MemRead<DefaultResource>
    ]>:$source,
    I32:$bytes,
    I1:$predicate,
    Optional<Aboleth_TokenType>:$after
  );
  let results = (outs Aboleth_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $endpoint `,` $destination `,` $source `,` $bytes
    `,` $predicate
    (`,` $after^)?
    attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_PostInlineOp
    : Aboleth_Op<"post_inline", [AttrSizedOperandSegments]> {
  let summary = "Construct a remote-write command carrying its payload inline";
  let description = [{
    The Queue IR form of `aboleth.put_inline`: the same command as
    `aboleth.post_put` without a source region or local key, with the payload
    supplied as operands instead.
  }];
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    Arg<Aboleth_RemoteRegionType, "", [
      MemWrite<DefaultResource>
    ]>:$destination,
    Variadic<AnyInteger>:$values,
    I1:$predicate,
    Optional<Aboleth_TokenType>:$after
  );
  let results = (outs Aboleth_TokenType:$token);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // Total payload bytes, the sum of the operand widths.
    uint64_t getPayloadBytes();
  }];
  let assemblyFormat = [{
    $endpoint `,` $destination `,` `[` $values `]` `,` $predicate
    (`,` $after^)?
    attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_PostSignalOp
    : Aboleth_Op<"post_signal", [AttrSizedOperandSegments]> {
  let summary = "Construct an ordered remote-signal command";
  let description = [{
    With a `predicate`, the signal publishes only where it holds; elsewhere the
    command keeps its slot as an ordered no-op that publishes and orders
    nothing. Set by aboleth-elide-epochs on publications whose epochs a later
    one supersedes. A checked publication (the mlx5 responder compare-and-swap)
    replaces only the value it expects, `epoch - 1` unless `expect` names it;
    an elided checked signal carries the epoch last published. A
    `completedPrefix` publication instead retires its payload on the sender
    before writing the epoch; a false predicate skips both that ordering
    point and the epoch. It never compares a predecessor.
  }];
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    Arg<Aboleth_RemoteRegionType, "", [
      MemWrite<DefaultResource>
    ]>:$destination,
    Arg<AnyTypeOf<[Aboleth_LocalRegionType, I64]>, "", [
      MemRead<DefaultResource>
    ]>:$source,
    Optional<Aboleth_TokenType>:$after,
    Optional<Aboleth_GroupType>:$group,
    Optional<I1>:$predicate,
    Optional<I64>:$expected,
    OptionalAttr<UnitAttr>:$weakPublication,
    OptionalAttr<UnitAttr>:$fencedSignal,
    OptionalAttr<UnitAttr>:$completedPrefix
  );
  let results = (outs Aboleth_TokenType:$token);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // Whether a strong signal just before it on the endpoint orders it, so a
    // plain write publishes it: one published together with it or not at all.
    bool isOrderedByPrevious();
    // Whether its mechanism checks the value it replaces: a strong signal
    // that is neither fenced, completed-prefix, nor ordered by a previous one.
    bool isChecked();
  }];
  let assemblyFormat = [{
    $endpoint `,` $destination `,` $source
    (`,` $after^)?
    (`group` $group^)?
    (`if` $predicate^)?
    (`expect` $expected^)?
    attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_PostNopOp : Aboleth_Op<"post_nop"> {
  let summary = "Construct an ordered no-op command that carries a completion";
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    Optional<Aboleth_TokenType>:$after
  );
  let results = (outs Aboleth_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $endpoint (`,` $after^)? attr-dict
    `:` functional-type(operands, results)
  }];
}

// TODO: submit fuses reservation, announcement (doorbell), and
// completion request. Investigate splitting them, or allowing several
// signal-terminated segments per submission, so doorbell coalescing and
// completion placement become rewrites of single operations.
def Aboleth_SubmitOp : Aboleth_Op<"submit", [
  AttrSizedOperandSegments,
  MemoryEffects<[
    MemRead<DefaultResource>,
    MemWrite<DefaultResource>
  ]>
]> {
  let summary = "Submit a group of posted commands with one notification";
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    Variadic<Aboleth_TokenType>:$transfers,
    Optional<Aboleth_GroupType>:$group,
    I32Attr:$batchSize,
    OptionalAttr<UnitAttr>:$deferCompletion,
    OptionalAttr<UnitAttr>:$headPublication,
    // Capacity for the complete phase was proven before this reservation.
    // The lowering advances the producer without reloading completion state.
    OptionalAttr<UnitAttr>:$plannedReservation,
    // "gpu" (default) or "cta"; verified in SubmitOp::verify.
    OptionalAttr<StrAttr>:$reservationScope,
    // "cta" (default) issues once from the block's first thread; "thread"
    // issues once per caller for per-token integration hooks.
    OptionalAttr<StrAttr>:$issuerScope,
    // The enclosing integration owns publication and completion. This
    // submission reserves and writes WQEs but emits no doorbell or poll.
    OptionalAttr<UnitAttr>:$externalPublication,
    // Other threads of the block may have written a region-sourced payload
    // since the last barrier, so a block barrier precedes the submission.
    // Set by aboleth-order-producers.
    OptionalAttr<UnitAttr>:$producerBarrier,
    // One program issues on this endpoint, so the reservation needs no atomic
    // and publication no reservation-order wait. Set by
    // aboleth-analyze-issuers.
    OptionalAttr<UnitAttr>:$exclusiveIssuer,
    // Announce only when this submission's poll is due. The commands are
    // still published to the endpoint's ready head, so the next announcement
    // on the endpoint (another submission's or a drain's) covers them. Set by
    // aboleth-defer-announcement.
    OptionalAttr<UnitAttr>:$deferredAnnouncement
  );
  let results = (outs Aboleth_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $endpoint `,` $transfers (`group` $group^)? attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_PollOp : Aboleth_Op<"poll", [
  MemoryEffects<[
    MemRead<AbolethEndpointState>,
    MemWrite<AbolethEndpointState>,
    MemRead<DefaultResource>,
    MemWrite<DefaultResource>
  ]>
]> {
  let summary = "Poll sender-side completion of a submitted group";
  let arguments = (ins
    Aboleth_TokenType:$submission,
    Optional<I1>:$predicate,
    OptionalAttr<UnitAttr>:$coalesced,
    // Carried from the wait (see aboleth.wait): the submission's completion
    // may be deferred past the queue's depth.
    OptionalAttr<UnitAttr>:$selfReclaim
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $submission (`,` $predicate^)? attr-dict `:` type($submission)
  }];
}

def Aboleth_DrainOp : Aboleth_Op<"drain", [
  MemoryEffects<[
    MemRead<AbolethEndpointState>,
    MemWrite<AbolethEndpointState>,
    MemRead<DefaultResource>,
    MemWrite<DefaultResource>
  ]>
]> {
  let summary = "Complete all outstanding sender work on an endpoint";
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    DefaultValuedStrAttr<StrAttr, "cta">:$issuerScope,
    OptionalAttr<UnitAttr>:$exclusiveIssuer
  );
  let assemblyFormat = "$endpoint attr-dict `:` type($endpoint)";
}

def Aboleth_SystemFenceOp : Aboleth_Op<"system_fence", [
  MemoryEffects<[
    MemRead<DefaultResource>,
    MemWrite<DefaultResource>
  ]>
]> {
  let summary = "Fence producer writes once at a phase effect boundary";
  let arguments = (ins I1:$predicate);
  let assemblyFormat = "$predicate attr-dict `:` type($predicate)";
}

def Aboleth_PublishHeadOp : Aboleth_Op<"publish_head", [
  MemoryEffects<[
    MemRead<AbolethEndpointState>,
    MemWrite<AbolethEndpointState>,
    MemRead<DefaultResource>,
    MemWrite<DefaultResource>
  ]>
]> {
  let summary = "Publish externally constructed WQEs up to the queue head";
  let arguments = (ins
    Arg<Aboleth_EndpointType, "", [
      MemRead<AbolethEndpointState>,
      MemWrite<AbolethEndpointState>
    ]>:$endpoint,
    I1:$onlyIfAdvanced
  );
  let assemblyFormat = [{
    $endpoint `,` $onlyIfAdvanced attr-dict
    `:` type($endpoint) `,` type($onlyIfAdvanced)
  }];
}

def Aboleth_ShapeFamilyOp : Aboleth_Op<"shape_family", [Pure]> {
  let summary = "Declare a uniform runtime bound and private in-flight slots";
  let description = [{
    The client supplies bound and stream/slot facts, not a decode heuristic.
    Specializations are mutually exclusive within a slot. Their publication
    state remains private to each variant and slot. Sharing payload storage
    still requires a separate last-reader and transport-retirement proof.
    Optional iterationBound describes a repeated-use domain, not every call.
    The compiler retains an overall-maximum fallback for larger initialization
    or mixed phases; it does not assume that those inputs cannot occur.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$minimumBound,
    I64Attr:$maximumBound, I64Attr:$slots, BoolAttr:$uniformSelector,
    OptionalAttr<I64Attr>:$iterationBound,
    OptionalAttr<DenseI64ArrayAttr>:$variantBounds,
    OptionalAttr<I64Attr>:$controlDomains);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

def Aboleth_EndpointWorkOp : Aboleth_Op<"endpoint_work", [Pure]> {
  let summary = "Bound unordered record chains and their registered endpoints";
  let description = [{
    Each indivisible record has the same fields on every lane. The client
    declares that separate records are unordered. The compiler partitions the
    record domain, assigns the count to lane zero, and requires a strong signal
    and receive wait on every lane, including lanes with zero valid records.
    Partitions are homogeneous launch facts from plan-transport. Profiles are
    once-measured platform facts, matched exactly to the registered platform
    and queue memory. Unknown platforms retain one lane.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$payloadBytes,
    I64Attr:$maxItems, ArrayAttr:$partitions, ArrayAttr:$profiles,
    OptionalAttr<I64Attr>:$rankCount,
    OptionalAttr<BoolAttr>:$groupedReceiverSupported,
    OptionalAttr<ArrayAttr>:$launchPlans);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

def Aboleth_TransportProgramOp : Aboleth_Op<"transport_program"> {
  let summary = "Contain one fixed-order transport effect trace";
  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<DenseI32ArrayAttr>:$peerRanks,
    OptionalAttr<ArrayAttr>:$peerTransports,
    OptionalAttr<ArrayAttr>:$launchPartitions
  );
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = "$body attr-dict";
}

def Aboleth_TransportFragmentOp
    : Aboleth_Op<"transport_fragment", [
        HasParent<"TransportProgramOp">,
        SingleBlock
      ]> {
  let summary = "Define a typed inline transport fragment";
  let description = [{
    A fragment is cloned into the kernel that invokes it, out of a plan module
    that is parsed, read, and destroyed. Nothing in its body may therefore
    name a value defined outside it: such a value would dangle the moment the
    plan module goes away. Every input arrives as a block argument, named by
    `argumentNames`, which is what the invocation binds.

    That is exactly the IsolatedFromAbove contract, but the trait cannot be
    used: MLIR's own verification asks `hasTrait<IsIsolatedFromAbove>()`, and
    a trait's TypeID is not shared across the plugin boundary, so the check
    fires an assertion instead of a diagnostic. The verifier below walks the
    body and proves the same property directly.
  }];
  let arguments = (ins
    I32Attr:$programId,
    StrAttr:$name,
    ArrayAttr:$argumentNames
  );
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def Aboleth_InvokeTransportPhaseOp
    : Aboleth_Op<"invoke_transport_phase"> {
  let summary = "Invoke an external inline transport fragment";
  let arguments = (ins
    I32:$programId,
    Variadic<AnyType>:$arguments
  );
  let hasVerifier = 1;
}

def Aboleth_TransportKernelOp
    : Aboleth_Op<"transport_kernel", [
        Pure,
        HasParent<"TransportProgramOp">
      ]> {
  let summary = "Describe effects of one opaque kernel";
  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<StrAttr>:$boundKernel,
    OptionalAttr<StrAttr>:$artifactKind,
    OptionalAttr<StrAttr>:$insertionPoint,
    OptionalAttr<BoolAttr>:$asynchronous,
    OptionalAttr<I32Attr>:$maxInflight,
    OptionalAttr<BoolAttr>:$eventOrdered,
    // True states that the client compiles this kernel outside Aboleth, so no
    // transform may place work inside it. Absent leaves placement to whether
    // the kernel's function is available.
    OptionalAttr<BoolAttr>:$opaque,
    ArrayAttr:$reads,
    ArrayAttr:$writes
  );
  let hasVerifier = 1;
}

def Aboleth_TransportPhaseOp
    : Aboleth_Op<"transport_phase", [
        Pure,
        HasParent<"TransportProgramOp">
      ]> {
  let summary = "Describe one communication phase and its effects";
  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<I32Attr>:$programId,
    OptionalAttr<StrAttr>:$boundKernel,
    OptionalAttr<StrAttr>:$artifactKind,
    OptionalAttr<StrAttr>:$insertionPoint,
    I32Attr:$commandsPerIteration,
    I32Attr:$iterationCount,
    I64Attr:$payloadBytes,
    OptionalAttr<I32Attr>:$completionInterval,
    OptionalAttr<I32Attr>:$resolvedCompletionInterval,
    OptionalAttr<BoolAttr>:$requiresFinalDrain,
    OptionalAttr<I32Attr>:$reuseDistance,
    OptionalAttr<StrAttr>:$completionBefore,
    OptionalAttr<I32Attr>:$completionIterationDelta,
    OptionalAttr<StrAttr>:$publicationBefore,
    OptionalAttr<I32Attr>:$publicationIterationDelta,
    ArrayAttr:$peerTransports,
    DenseI32ArrayAttr:$peerCapacities,
    OptionalAttr<I64Attr>:$policyTinyPayloadBytes,
    OptionalAttr<DenseI32ArrayAttr>:$peerCompletionIntervals,
    ArrayAttr:$reads,
    ArrayAttr:$writes
  );
  let hasVerifier = 1;
}

// Frontend-neutral communication dataflow. These operations describe logical
// values and destinations only; executable registered regions and Queue IR
// remain represented by the operations above.

def Aboleth_CommunicationDataflowOp
    : Aboleth_Op<"communication_dataflow"> {
  let summary = "Contain frontend-neutral communication dataflow";
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = "$name $body attr-dict";
}

def Aboleth_RegionOp
    : Aboleth_Op<"region", [
        Pure,
        HasParent<"CommunicationDataflowOp">
      ]> {
  let summary = "Declare logical communication storage";
  let description = [{
    Declares a frontend-neutral logical storage value. The index map maps
    logical indices to the declared physical layout. Visibility is one of
    private, local, peer, or global.
  }];
  let arguments = (ins
    StrAttr:$name,
    TypeAttr:$elementType,
    DenseI64ArrayAttr:$shape,
    DenseI64ArrayAttr:$strides,
    AffineMapAttr:$indexMap,
    StrAttr:$visibility,
    BoolAttr:$immutable,
    BoolAttr:$escapes,
    OptionalAttr<BoolAttr>:$registered,
    OptionalAttr<BoolAttr>:$donated,
    OptionalAttr<I64Attr>:$reuseInterval,
    OptionalAttr<I64Attr>:$poolDepth,
    OptionalAttr<StrAttr>:$validCount,
    OptionalAttr<BoolAttr>:$deadPadding
  );
  let results = (outs Aboleth_LogicalRegionType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Aboleth_ViewOp
    : Aboleth_Op<"view", [
        Pure,
        HasParent<"CommunicationDataflowOp">
      ]> {
  let summary = "Map a logical view of another region";
  let description = [{
    Defines an aliasing view. Its index map maps result-view indices to source
    indices. Visibility, immutability, and escape metadata must preserve the
    source constraints as checked by the verifier.
  }];
  let arguments = (ins
    Aboleth_LogicalRegionType:$source,
    StrAttr:$name,
    TypeAttr:$elementType,
    DenseI64ArrayAttr:$shape,
    DenseI64ArrayAttr:$strides,
    AffineMapAttr:$indexMap,
    StrAttr:$visibility,
    BoolAttr:$immutable,
    BoolAttr:$escapes,
    OptionalAttr<BoolAttr>:$registered,
    OptionalAttr<BoolAttr>:$donated,
    OptionalAttr<I64Attr>:$reuseInterval,
    OptionalAttr<I64Attr>:$poolDepth
  );
  let results = (outs Aboleth_LogicalRegionType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def Aboleth_IndexedViewOp
    : Aboleth_Op<"indexed_view", [
        Pure,
        HasParent<"CommunicationDataflowOp">
      ]> {
  let summary = "Map an injective indexed view of logical storage";
  let description = [{
    Defines a destination view whose first logical dimension is selected by
    an immutable integer index region. Explicit lower/upper bounds and an
    injectivity proof make indexed scatter legality visible in dataflow.
  }];
  let arguments = (ins
    Aboleth_LogicalRegionType:$source,
    Aboleth_LogicalRegionType:$indices,
    StrAttr:$name,
    TypeAttr:$elementType,
    DenseI64ArrayAttr:$shape,
    DenseI64ArrayAttr:$strides,
    AffineMapAttr:$indexMap,
    StrAttr:$visibility,
    BoolAttr:$immutable,
    BoolAttr:$escapes,
    BoolAttr:$injective,
    DenseI64ArrayAttr:$indexLowerBounds,
    DenseI64ArrayAttr:$indexUpperBounds,
    OptionalAttr<BoolAttr>:$registered,
    OptionalAttr<BoolAttr>:$donated,
    OptionalAttr<I64Attr>:$reuseInterval,
    OptionalAttr<I64Attr>:$poolDepth
  );
  let results = (outs Aboleth_LogicalRegionType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `using` $indices attr-dict
    `:` type($source) `,` type($indices) `->` type($result)
  }];
}

def Aboleth_WorksetOp
    : Aboleth_Op<"workset", [
        AttrSizedOperandSegments,
        HasParent<"CommunicationDataflowOp">
      ]> {
  let summary = "Declare bounded runtime communication work";
  let arguments = (ins
    Variadic<Aboleth_LogicalRegionType>:$fields,
    Optional<Aboleth_LogicalRegionType>:$count,
    StrAttr:$id,
    StrAttr:$name,
    ArrayAttr:$fieldNames,
    DenseI64ArrayAttr:$fieldLowerBounds,
    DenseI64ArrayAttr:$fieldUpperBounds,
    I64Attr:$maxItems,
    BoolAttr:$uniqueDestinations,
    BoolAttr:$ordered
  );
  let results = (outs Aboleth_WorksetType:$result);
  let hasVerifier = 1;
}

class Aboleth_DestinationOp<string mnemonic>
    : Aboleth_Op<mnemonic, [
        SameTypeOperands,
        HasParent<"CommunicationDataflowOp">
      ]> {
  let description = [{
    Takes an explicit destination and returns the post-write SSA state of that
    same logical destination. Consumers must use the result to observe the
    write, following destination-passing style without conflating logical
    storage with executable local or remote region handles.
  }];
  let arguments = (ins
    Aboleth_LogicalRegionType:$source,
    Aboleth_LogicalRegionType:$destination
  );
  let results = (outs Aboleth_LogicalRegionType:$result);
}

def Aboleth_TransferOp
    : Aboleth_DestinationOp<"transfer"> {
  let summary = "Transfer a logical value into an explicit destination";
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `into` $destination attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_TransferEachOp
    : Aboleth_Op<"transfer_each", [
        HasParent<"CommunicationDataflowOp">
      ]> {
  let summary = "Transfer rows selected by a runtime workset";
  let arguments = (ins
    Aboleth_WorksetType:$workset,
    Aboleth_LogicalRegionType:$source,
    Aboleth_LogicalRegionType:$destination,
    StrAttr:$id,
    StrAttr:$outputId,
    StrAttr:$sourceIndexField,
    StrAttr:$destinationPeerField,
    StrAttr:$destinationIndexField,
    OptionalAttr<StrAttr>:$bytesField,
    OptionalAttr<StrAttr>:$predicateField
  );
  let results = (outs Aboleth_LogicalRegionType:$result);
  let hasVerifier = 1;
}

def Aboleth_CopyEachOp
    : Aboleth_Op<"copy_each", [
        HasParent<"CommunicationDataflowOp">
      ]> {
  let summary = "Copy rows selected by a runtime workset";
  let arguments = (ins
    Aboleth_WorksetType:$workset,
    Aboleth_LogicalRegionType:$source,
    Aboleth_LogicalRegionType:$destination,
    StrAttr:$id,
    StrAttr:$outputId,
    StrAttr:$sourceIndexField,
    StrAttr:$destinationIndexField
  );
  let results = (outs Aboleth_LogicalRegionType:$result);
  let hasVerifier = 1;
}

def Aboleth_ReductionOp
    : Aboleth_DestinationOp<"reduction"> {
  let summary = "Reduce contributions into an explicit destination";
  let arguments = (ins
    Aboleth_LogicalRegionType:$source,
    Aboleth_LogicalRegionType:$destination,
    I64Attr:$cardinality,
    StrAttr:$operatorName
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `into` $destination attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_CopyOp
    : Aboleth_DestinationOp<"copy"> {
  let summary = "Copy a logical value into an explicit destination layout";
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `into` $destination attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_TransformOp
    : Aboleth_DestinationOp<"transform"> {
  let summary = "Transform a logical value into an explicit destination";
  let arguments = (ins
    Aboleth_LogicalRegionType:$source,
    Aboleth_LogicalRegionType:$destination,
    StrAttr:$transform,
    AffineMapAttr:$indexMap
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `into` $destination attr-dict
    `:` functional-type(operands, results)
  }];
}

def Aboleth_UseOp
    : Aboleth_Op<"use", [
        HasParent<"CommunicationDataflowOp">
      ]> {
  let summary = "Represent an opaque consumer of logical values";
  let arguments = (ins
    Variadic<Aboleth_LogicalRegionType>:$inputs,
    StrAttr:$name
  );
  let hasVerifier = 1;
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

def Aboleth_AsyncUseOp
    : Aboleth_Op<"async_use", [
        HasParent<"CommunicationDataflowOp">
      ]> {
  let summary = "Represent an event-ordered asynchronous consumer";
  let description = [{
    Consumes logical values only after their producer completion event. The
    maximum in-flight count states how many generations may remain live before
    storage reuse.
  }];
  let arguments = (ins
    Variadic<Aboleth_LogicalRegionType>:$inputs,
    StrAttr:$name,
    I64Attr:$maxInflight,
    BoolAttr:$eventOrdered
  );
  let hasVerifier = 1;
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

def Aboleth_OpaqueStageOp
    : Aboleth_Op<"opaque_stage", [
        AttrSizedOperandSegments,
        HasParent<"CommunicationDataflowOp">
      ]> {
  let summary = "Represent opaque compute with explicit region effects";
  let description = [{
    Models a compute stage whose implementation remains outside Aboleth.
    Read operands are consumed without mutation. Write operands are
    destination regions, with one post-write SSA result per destination.
    Asynchronous stages state their maximum in-flight count and whether an
    external event orders storage reuse.
  }];
  let arguments = (ins
    Variadic<Aboleth_LogicalRegionType>:$reads,
    Variadic<Aboleth_LogicalRegionType>:$writes,
    StrAttr:$name,
    ArrayAttr:$outputIds,
    BoolAttr:$asynchronous,
    I64Attr:$maxInflight,
    BoolAttr:$eventOrdered,
    OptionalAttr<BoolAttr>:$bindableOutputs,
    OptionalAttr<ArrayAttr>:$prefixReadCounts
  );
  let results = (outs Variadic<Aboleth_LogicalRegionType>:$results);
  let hasVerifier = 1;
}

#endif // ABOLETH_OPS
