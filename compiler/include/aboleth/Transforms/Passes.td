#ifndef ABOLETH_PASSES
#define ABOLETH_PASSES

include "mlir/Pass/PassBase.td"

def AbolethSimplifyCommunicationDataflowPass
    : Pass<"aboleth-simplify-communication-dataflow", "mlir::ModuleOp"> {
  let summary = "Simplify frontend-neutral communication dataflow";
  let description = [{
    Rewrites logical communication semantics directly: singleton reductions
    become transfers, legal staging copies are folded into their producers,
    immutable non-escaping copies become SSA reuse, and dead logical writes
    are removed. Unsupported or partially optimizable dataflow remains
    explicit as residual copy, transform, or reduction operations.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
  let options = [
    Option<"storageAliases", "storage-aliases", "bool", "true",
           "Retarget explicitly bindable opaque outputs to donated final storage">,
    Option<"deadCommunication", "dead-communication", "bool", "true",
           "Remove unobserved communication and its unused logical storage">
  ];
}

def AbolethPlanShapesPass
    : Pass<"aboleth-plan-shapes", "mlir::ModuleOp"> {
  let summary = "Plan bounded shape families and a rank-uniform selector";
  let dependentDialects = ["mlir::aboleth::AbolethDialect"];
  let options = [
    Option<"specialize", "specialize", "bool", "true",
           "Emit bounded power-of-two variants when the selector is uniform">,
    Option<"emitPlanJson", "emit-plan-json", "std::string", "\"\"",
           "Write variant capacities and private control domains as JSON">
  ];
}

def AbolethPlanEndpointsPass
    : Pass<"aboleth-plan-endpoints", "mlir::ModuleOp"> {
  let summary = "Split unordered bounded record chains across endpoint lanes";
  let dependentDialects = ["mlir::aboleth::AbolethDialect"];
  let options = [
    Option<"fanout", "fanout", "bool", "true",
           "Use qualified platform fan-out; false retains one endpoint">,
    Option<"coalesceReceives", "coalesce-receives", "bool", "true",
           "Join a peer's independent lane signals in one supported receiver program">,
    Option<"emitPlanJson", "emit-plan-json", "std::string", "\"\"",
           "Write lane, record-slice and readiness facts as JSON">
  ];
}

def AbolethPlanStoragePass
    : Pass<"aboleth-plan-storage", "mlir::ModuleOp"> {
  let summary = "Derive storage extents from bounded communication effects";
  let description = [{
    Bounds the physical storage reached by every read and write of a logical
    region. Indexed views and worksets contribute their proven row bounds;
    unknown, escaping and full-region uses retain the declared extent.
    Logical shapes remain unchanged. This pass does not schedule computation
    or introduce aliases between allocations.
  }];
  let dependentDialects = ["mlir::aboleth::AbolethDialect"];
  let options = [
    Option<"optimizeBounds", "optimize-bounds", "bool", "true",
           "Derive physical extents; false retains declared storage">,
    Option<"emitPlanJson", "emit-plan-json", "std::string", "\"\"",
           "Write the allocation facts as JSON">
  ];
}

def AbolethMaterializeValidExtentsPass
    : Pass<"aboleth-materialize-valid-extents", "mlir::ModuleOp"> {
  let summary = "Normalize valid counts and eliminate declared dead padding";
  let dependentDialects = ["mlir::aboleth::AbolethDialect", "mlir::arith::ArithDialect"];
  let options = [
    Option<"trimDeadPadding", "trim-dead-padding", "bool", "true",
           "Trim transfers to dynamic valid rows; false retains full capacity">
  ];
}

def AbolethForwardInlinePayloadsPass
    : Pass<"aboleth-forward-inline-payloads", "mlir::ModuleOp"> {
  let summary = "Forward staged scalar payloads into inline transfers";
  let description = [{
    Rewrites an `aboleth.put` whose source region is written entirely by
    dominating scalar stores into an `aboleth.put_inline` carrying those values
    as operands, and removes the staging stores. The payload then travels from
    registers into the transport's command with no staging store, no
    registration key, and no device read of the source buffer.

    A put is forwarded only when every fact is proven: the byte count is a
    constant no greater than `inline-bytes`, the source address is a constant
    offset from an identifiable base, the stores exactly tile the transferred
    range, they precede the put in its block, nothing between the last store and
    the put can touch that range, and no other operation in the kernel can reach
    it. Anything unprovable is left as a region-sourced put.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
  let options = [
    Option<"inlineBytes", "inline-bytes", "int64_t",
           /*default=*/"28",
           "Largest payload forwarded into an inline transfer. The default is "
           "the largest mlx5 inline payload that still fits one WQEBB; zero "
           "forwards nothing.">
  ];
}

def AbolethFormBatchesPass
    : Pass<"aboleth-form-batches", "mlir::ModuleOp"> {
  let summary = "Form explicit Aboleth submission batches";
  let description = [{
    Rewrites straight-line put/signal/wait sequences into post operations,
    one submit operation, and one completion poll.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
}

def AbolethCoalesceSubmissionsPass
    : Pass<"aboleth-coalesce-submissions", "mlir::ModuleOp"> {
  let summary = "Merge independent same-endpoint submissions";
  let description = [{
    Merges a submission into the next one on an equivalent endpoint in the same
    block when nothing between them can block or synchronize. The merged batch
    keeps each chain as a segment closed by its own signal and has one
    reservation, one announcement, and one completion; the earlier chain's
    announcement moves to the later submission. Runs where the transport has a
    GPU-built ordered queue (TransportCaps), after batch formation.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
  let options = [
    Option<"transport", "transport", "std::string", /*default=*/"\"\"",
           "Transport the kernel lowers to; empty disables the pass">
  ];
}

def AbolethDeferAnnouncementPass
    : Pass<"aboleth-defer-announcement", "mlir::ModuleOp"> {
  let summary = "Announce planned submissions only when their poll is due";
  let description = [{
    Marks a submission whose completion is planned (a predicated poll) to ring
    its doorbell only on the iterations where that poll is due. Its commands
    still reach the endpoint's ready head, so the next announcement on the
    endpoint, at the latest the drain, covers them. Delaying an announcement
    delays delivery, so a kernel that waits on remote progress (wait_signal)
    is left alone: the peer's reply could depend on an unannounced command.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
  let options = [
    Option<"transport", "transport", "std::string", /*default=*/"\"\"",
           "Transport the kernel lowers to; empty disables the pass">
  ];
}

def AbolethOrderPublicationsPass
    : Pass<"aboleth-order-publications", "mlir::ModuleOp"> {
  let summary = "Order a batch's payload before its signals";
  let description = [{
    In a submission with several strong signals (a coalesced batch), moves
    every payload command ahead of the signals and chains the strong signals
    one after another. The first strong signal then orders all of the batch's
    payload, and the lowering publishes each later one with a plain write,
    which the queue executes after it: one ordering point per batch instead of
    one per signal. Runs where the transport has a GPU-built ordered queue.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
  let options = [
    Option<"transport", "transport", "std::string", /*default=*/"\"\"",
           "Transport the kernel lowers to; empty disables the pass">
  ];
}

def AbolethElideEpochsPass
    : Pass<"aboleth-elide-epochs", "mlir::ModuleOp"> {
  let summary = "Publish planned loops' epochs only when completion is due";
  let description = [{
    In a counted loop whose submissions complete only every k-th iteration,
    predicates each signal on that completion's due iterations and the loop's
    last: a later epoch in the same cell supersedes the skipped ones. A checked
    signal (compare-and-swap) gets the epoch last published as its expected
    value, carried by the loop. Kernels that wait on a peer are left alone.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
    "mlir::arith::ArithDialect",
  ];
  let options = [
    Option<"transport", "transport", "std::string", /*default=*/"\"\"",
           "Transport the kernel lowers to; empty disables the pass">
  ];
}

def AbolethPlanTransportPass
    : Pass<"aboleth-plan-transport", "mlir::ModuleOp"> {
  let summary = "Plan completion for communication phases";
  let description = [{
    Resolves source-reuse and endpoint-capacity constraints into transport
    completion boundaries. This pass does not reorder program operations,
    assign streams, or construct an execution graph.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
  let options = [
    Option<"emitPlanJson", "emit-plan-json", "std::string",
           /*default=*/"\"\"",
           "Path to write the planned facts as one JSON document. The "
           "printed module is unaffected.">
  ];
}

def AbolethPlanPublicationPass
    : Pass<"aboleth-plan-publication", "mlir::ModuleOp"> {
  let summary = "Select strong or weak remote publication and its mechanism";
  let description = [{
    Keeps payload-bearing signals strongly ordered and marks standalone,
    ungrouped credit notifications as weak. Unknown or grouped publication
    remains strong. For every strong signal, semantic or already posted, it
    then selects the transport's ordering point from its capabilities and the
    receiver registrations' ordering: a fenced write where that is legal,
    otherwise the transport's default (a responder atomic on mlx5).
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
  let options = [
    Option<"transport", "transport", "std::string", /*default=*/"\"\"",
           "Transport the kernel lowers to; empty keeps the default ordering "
           "point">,
    Option<"strictRegistrations", "strict-registrations", "bool",
           /*default=*/"false",
           "Every receiver registration is strictly ordered (no relaxed "
           "ordering)">
  ];
}

def AbolethMaterializeTransportPhasePass
    : Pass<"aboleth-materialize-transport-phase", "mlir::ModuleOp"> {
  let summary = "Materialize external inline transport fragments";
  let description = [{
    Replaces transport phase invocations with typed semantic operations cloned
    from an external transport program.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
    "mlir::arith::ArithDialect",
  ];
  let options = [
    Option<"planPath", "plan-path", "std::string",
           /*default=*/"\"\"",
           "Path to a transport_program module containing inline fragments.">
  ];
}

def AbolethMaterializeCompletionPass
    : Pass<"aboleth-materialize-completion", "mlir::ModuleOp"> {
  let summary = "Materialize compiler-planned completion predicates";
  let description = [{
    Rewrites planned waits into unconditional or predicated semantic waits
    using the completion interval selected by transport planning.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
    "mlir::arith::ArithDialect",
  ];
  let options = [
    Option<"completionInterval", "completion-interval", "int64_t",
           /*default*/"1", "planned sender-completion interval">,
    Option<"transport", "transport", "std::string", /*default=*/"\"\"",
           "Transport the kernel lowers to; bounds reuse-planned waits">,
    Option<"queueCapacity", "queue-capacity", "int64_t", /*default=*/"0",
           "Smallest send-queue capacity of the kernel's endpoints, in slots; "
           "zero when unknown">,
    Option<"planReuse", "plan-reuse", "bool", /*default=*/"true",
           "Plan completion from declared reuse distances; false waits every "
           "iteration (the ablation baseline)">
  ];
}

def AbolethOrderProducersPass
    : Pass<"aboleth-order-producers", "mlir::ModuleOp"> {
  let summary = "Order in-kernel payload producers before submissions";
  let description = [{
    A submission is built and announced by one thread, while the payload it
    references may have been written by every thread of the block. Marks each
    submission that carries a region-sourced command and may follow such a
    write with `producerBarrier`, so the backend places one block barrier
    before it. Writes are tracked from the last barrier, so submissions that
    follow one another without new writes share it, and batches whose payload
    travels inline need none. A scalar store is issued by the same thread that
    announces the submission and needs no barrier.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
}

def AbolethAnalyzeIssuersPass
    : Pass<"aboleth-analyze-issuers", "mlir::ModuleOp"> {
  let summary = "Derive single-issuer endpoints from how a kernel indexes them";
  let description = [{
    Aboleth operations issue once per program. When every endpoint a kernel
    selects from a table uses one index, and that index is an injective
    function of the program id along one axis, no two programs share an
    endpoint. The selections then record the axis, and every submission and
    drain on them is marked `exclusiveIssuer`: reserving queue slots needs no
    atomic and announcing them needs no wait for other producers. The
    lowering checks once that the grid is one-dimensional along that axis.
  }];
  let dependentDialects = [
    "mlir::aboleth::AbolethDialect",
  ];
}

#endif // ABOLETH_PASSES
