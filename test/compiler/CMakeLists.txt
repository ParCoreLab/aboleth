# One ctest per pipeline/module pair. `ctest -L compiler-ir` replaces the old
# `make aboleth-ir-test`, and a failure now names the single case that drifted.

if(NOT ABOLETH_HAS_TRITON_PLUGIN)
  return()
endif()

if(NOT ABOLETH_FILECHECK)
  message(WARNING "[Aboleth] FileCheck not found; the IR tests are not registered")
  return()
endif()

# --- dataflow simplification ------------------------------------------------

aboleth_add_ir_test(ir.communication-dataflow
  FILE communication-dataflow.mlir
  PASSES simplify-communication-dataflow
  IDEMPOTENT)
aboleth_add_ir_test(ir.communication-dataflow-async
  FILE communication-dataflow-async.mlir
  PASSES simplify-communication-dataflow)
aboleth_add_ir_test(ir.communication-dataflow-opaque
  FILE communication-dataflow-opaque.mlir
  PASSES simplify-communication-dataflow)
aboleth_add_ir_test(ir.communication-dataflow-opaque-alias
  FILE communication-dataflow-opaque-alias.mlir
  PASSES simplify-communication-dataflow)
aboleth_add_ir_test(ir.communication-dataflow-workset
  FILE communication-dataflow-workset.mlir
  PASSES simplify-communication-dataflow)

foreach(case layout cardinality destination visibility escape containment
             view-map view-stride indexed-view id opaque workset)
  aboleth_add_ir_test(ir.communication-dataflow-${case}-invalid
    FILE communication-dataflow-${case}-invalid.mlir
    EXPECT_FAIL)
endforeach()

aboleth_add_ir_test(ir.opaque-output-binding
  FILE opaque-output-binding.mlir PASSES simplify-communication-dataflow IDEMPOTENT)
aboleth_add_ir_test(ir.opaque-output-binding.off
  FILE opaque-output-binding.mlir PASSES "simplify-communication-dataflow=unoptimized" CHECK_PREFIX GENERIC)

aboleth_add_ir_test(ir.dead-communication
  FILE dead-communication.mlir PASSES simplify-communication-dataflow IDEMPOTENT)
aboleth_add_ir_test(ir.dead-communication.off
  FILE dead-communication.mlir PASSES "simplify-communication-dataflow=optimized,keep-dead" CHECK_PREFIX GENERIC)

# --- transport planning -----------------------------------------------------

aboleth_add_ir_test(ir.shape-family
  FILE shape-family.mlir PASSES plan-shapes IDEMPOTENT)
aboleth_add_ir_test(ir.shape-family.off
  FILE shape-family.mlir PASSES "plan-shapes=,unoptimized" CHECK_PREFIX GENERIC)
aboleth_add_ir_test(ir.shape-family-invalid
  FILE shape-family-invalid.mlir EXPECT_FAIL)
aboleth_add_ir_test(ir.shape-iteration-bound
  FILE shape-iteration-bound.mlir PASSES plan-shapes IDEMPOTENT)
aboleth_add_ir_test(ir.shape-iteration-bound.off
  FILE shape-iteration-bound.mlir PASSES "plan-shapes=,unoptimized" CHECK_PREFIX GENERIC)
aboleth_add_ir_test(ir.shape-iteration-invalid
  FILE shape-iteration-invalid.mlir EXPECT_FAIL)

aboleth_add_ir_test(ir.analyze-issuer-helpers
  FILE analyze-issuer-helpers.mlir PASSES analyze-issuers IDEMPOTENT)

aboleth_add_ir_test(ir.endpoint-world
  FILE endpoint-world.mlir PASSES plan-endpoints IDEMPOTENT)
aboleth_add_ir_test(ir.endpoint-world.off
  FILE endpoint-world.mlir PASSES "plan-endpoints=,unoptimized" CHECK_PREFIX GENERIC)
aboleth_add_ir_test(ir.endpoint-world-invalid
  FILE endpoint-world-invalid.mlir EXPECT_FAIL)

aboleth_add_ir_test(ir.endpoint-work
  FILE endpoint-work.mlir PASSES plan-endpoints IDEMPOTENT)
aboleth_add_ir_test(ir.endpoint-work.off
  FILE endpoint-work.mlir PASSES "plan-endpoints=,unoptimized" CHECK_PREFIX GENERIC)
aboleth_add_ir_test(ir.endpoint-work-invalid
  FILE endpoint-work-invalid.mlir EXPECT_FAIL)

aboleth_add_ir_test(ir.endpoint-peer-link
  FILE endpoint-peer-link.mlir PASSES plan-endpoints IDEMPOTENT)
aboleth_add_ir_test(ir.endpoint-peer-link.off
  FILE endpoint-peer-link.mlir PASSES "plan-endpoints=,unoptimized" CHECK_PREFIX GENERIC)
aboleth_add_ir_test(ir.endpoint-peer-link.ungrouped
  FILE endpoint-peer-link.mlir PASSES "plan-endpoints=,,ungrouped" CHECK_PREFIX UNGROUPED)
aboleth_add_ir_test(ir.endpoint-peer-link-invalid
  FILE endpoint-peer-link-invalid.mlir EXPECT_FAIL)
aboleth_add_ir_test(ir.endpoint-geometry-invalid
  FILE endpoint-geometry-invalid.mlir EXPECT_FAIL)

aboleth_add_ir_test(ir.prefix-storage
  FILE prefix-storage.mlir PASSES plan-storage IDEMPOTENT)
aboleth_add_ir_test(ir.prefix-storage.off
  FILE prefix-storage.mlir PASSES "plan-storage=,unoptimized" CHECK_PREFIX GENERIC)
aboleth_add_ir_test(ir.prefix-storage-invalid
  FILE prefix-storage-invalid.mlir EXPECT_FAIL)

aboleth_add_ir_test(ir.prefix-storage-input-invalid
  FILE prefix-storage-input-invalid.mlir EXPECT_FAIL)

aboleth_add_ir_test(ir.plan-storage
  FILE plan-storage.mlir PASSES plan-storage IDEMPOTENT)
aboleth_add_ir_test(ir.plan-storage.off
  FILE plan-storage.mlir PASSES "plan-storage=,unoptimized" CHECK_PREFIX GENERIC)

aboleth_add_ir_test(ir.transport-plan
  FILE transport-plan.mlir PASSES plan-transport)
aboleth_add_ir_test(ir.transport-partitions
  FILE transport-partitions.mlir PASSES plan-transport IDEMPOTENT)
aboleth_add_ir_test(ir.transport-partitions-invalid
  FILE transport-partitions-invalid.mlir EXPECT_FAIL)
# emit-plan-json writes a side file, so the same module and the same CHECK lines
# must still hold: the JSON may never change the printed IR. The document itself
# is checked in test/python/test_plan_json.py, which is what consumes it.
aboleth_add_ir_test(ir.transport-plan.json
  FILE transport-plan.mlir
  PASSES "plan-transport=${CMAKE_CURRENT_BINARY_DIR}/transport-plan.json")
aboleth_add_ir_test(ir.transport-region-effects
  FILE transport-region-effects.mlir PASSES plan-transport)
aboleth_add_ir_test(ir.transport-caps
  FILE transport-caps.mlir PASSES plan-transport)
aboleth_add_ir_test(ir.transport-caps-invalid
  FILE transport-caps-invalid.mlir EXPECT_FAIL)
aboleth_add_ir_test(ir.phase-program-effects
  FILE phase-program-effects.mlir PASSES plan-transport)
aboleth_add_ir_test(ir.transport-plan-invalid
  FILE transport-plan-invalid.mlir PASSES plan-transport EXPECT_FAIL)
aboleth_add_ir_test(ir.transport-plan-async-opaque
  FILE transport-plan-async-opaque.mlir PASSES plan-transport)

# --- publication ------------------------------------------------------------

aboleth_add_ir_test(ir.publication-plan
  FILE publication-plan.mlir PASSES plan-publication form-batches)
aboleth_add_ir_test(ir.publication-plan.p2p
  FILE publication-plan.mlir PASSES plan-publication form-batches lower-p2p
  CHECK_PREFIX P2P)
aboleth_add_ir_test(ir.publication-plan.rdma
  FILE publication-plan.mlir PASSES plan-publication form-batches lower-rdma
  CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.publication-plan.proxy
  FILE publication-plan.mlir PASSES plan-publication form-batches lower-proxy
  CHECK_PREFIX LOWERED)
aboleth_add_ir_test(ir.publication-invalid
  FILE publication-invalid.mlir EXPECT_FAIL)
aboleth_add_ir_test(ir.coalesce-submissions
  FILE coalesce-submissions.mlir
  PASSES form-batches "coalesce-submissions=rdma" CHECK_PREFIX MERGED)
aboleth_add_ir_test(ir.coalesce-submissions.p2p
  FILE coalesce-submissions.mlir
  PASSES form-batches "coalesce-submissions=p2p" CHECK_PREFIX UNMERGED)
aboleth_add_ir_test(ir.coalesce-submissions.ordered
  FILE coalesce-submissions.mlir
  PASSES form-batches "coalesce-submissions=rdma" "order-publications=rdma"
  CHECK_PREFIX ORDERED)
aboleth_add_ir_test(ir.coalesce-submissions.ordered-rdma
  FILE coalesce-submissions.mlir
  PASSES form-batches "coalesce-submissions=rdma" "order-publications=rdma"
  lower-rdma CHECK_PREFIX ORDERED-LOWERED)
aboleth_add_ir_test(ir.coalesce-submissions.rdma
  FILE coalesce-submissions.mlir
  PASSES form-batches "coalesce-submissions=rdma" lower-rdma
  CHECK_PREFIX LOWERED)
aboleth_add_ir_test(ir.elide-epochs
  FILE elide-epochs.mlir
  PASSES form-batches "elide-epochs=rdma" CHECK_PREFIX MARKED)
aboleth_add_ir_test(ir.elide-epochs.rdma
  FILE elide-epochs.mlir
  PASSES form-batches "elide-epochs=rdma" lower-rdma CHECK_PREFIX LOWERED)
aboleth_add_ir_test(ir.elide-epochs-checked-invalid
  FILE elide-epochs-checked-invalid.mlir EXPECT_FAIL)
aboleth_add_ir_test(ir.defer-announcement
  FILE defer-announcement.mlir
  PASSES form-batches "defer-announcement=rdma" CHECK_PREFIX MARKED)
aboleth_add_ir_test(ir.defer-announcement.rdma
  FILE defer-announcement.mlir
  PASSES form-batches "defer-announcement=rdma" lower-rdma
  CHECK_PREFIX LOWERED)
aboleth_add_ir_test(ir.publication-fenced-invalid
  FILE publication-fenced-invalid.mlir EXPECT_FAIL)
foreach(case strict relaxed relaxed-proxy lowered)
  if(case STREQUAL "strict")
    set(_passes "plan-publication=rdma,strict" form-batches)
    set(_prefix STRICT)
  elseif(case STREQUAL "relaxed")
    set(_passes "plan-publication=rdma" form-batches)
    set(_prefix RELAXED)
  elseif(case STREQUAL "relaxed-proxy")
    set(_passes "plan-publication=proxy,strict" form-batches)
    set(_prefix RELAXED)
  else()
    set(_passes "plan-publication=rdma,strict" form-batches lower-rdma)
    set(_prefix LOWERED)
  endif()
  aboleth_add_ir_test(ir.publication-mechanism.${case}
    FILE publication-mechanism.mlir PASSES ${_passes}
    CHECK_PREFIX ${_prefix})
endforeach()

# --- completion -------------------------------------------------------------

aboleth_add_ir_test(ir.valid-extents
  FILE valid-extents.mlir PASSES materialize-valid-extents IDEMPOTENT)
aboleth_add_ir_test(ir.valid-extents.off
  FILE valid-extents.mlir PASSES "materialize-valid-extents=unoptimized")
aboleth_add_ir_test(ir.valid-extents-invalid
  FILE valid-extents-invalid.mlir EXPECT_FAIL)

aboleth_add_ir_test(ir.materialize-completion.interval
  FILE materialize-completion.mlir PASSES materialize-completion=8
  CHECK_PREFIX INTERVAL)
aboleth_add_ir_test(ir.materialize-completion.eager
  FILE materialize-completion.mlir PASSES materialize-completion=1
  CHECK_PREFIX EAGER)
aboleth_add_ir_test(ir.materialize-completion.queue
  FILE materialize-completion.mlir PASSES materialize-completion=8 form-batches
  CHECK_PREFIX QUEUE)

aboleth_add_ir_test(ir.materialize-completion.late-bounds
  FILE materialize-completion-late-bounds.mlir PASSES materialize-completion=8)

# --- transport-phase materialization ----------------------------------------

aboleth_add_ir_test(ir.materialize-transport-phase
  FILE materialize-transport-phase.mlir
  PASSES
    "materialize-transport-phase=${CMAKE_CURRENT_SOURCE_DIR}/external-transport-fragment.mlir")
aboleth_add_ir_test(ir.materialize-semantic-endpoint
  FILE materialize-semantic-endpoint.mlir PASSES
    "materialize-transport-phase=${CMAKE_CURRENT_SOURCE_DIR}/external-retirement-fragment.mlir")

# --- batching ---------------------------------------------------------------

foreach(case core-ir form-batches form-batches-payload
        form-batches-signal-store signal-only)
  aboleth_add_ir_test(ir.${case} FILE ${case}.mlir PASSES form-batches)
endforeach()
aboleth_add_ir_test(ir.form-batches-invalid
  FILE form-batches-invalid.mlir PASSES form-batches EXPECT_FAIL)
aboleth_add_ir_test(ir.form-batches-effects-invalid
  FILE form-batches-effects-invalid.mlir PASSES form-batches EXPECT_FAIL)
foreach(backend rdma p2p proxy)
  string(TOUPPER "${backend}" _prefix)
  aboleth_add_ir_test(ir.signal-value.${backend}
    FILE signal-value.mlir
    PASSES plan-publication form-batches lower-${backend}
    CHECK_PREFIX ${_prefix})
endforeach()
aboleth_add_ir_test(ir.signal-value.rdma-gpu-doorbell
  FILE signal-value.mlir
  PASSES plan-publication form-batches lower-rdma=gpu-scope-doorbell
  CHECK_PREFIX GPUDB)
foreach(case preferred queue reuse unplanned)
  string(TOUPPER "${case}" _prefix)
  if(case STREQUAL "preferred")
    set(_pass "materialize-completion=,rdma,1024")
  elseif(case STREQUAL "unplanned")
    set(_pass "materialize-completion=,rdma,1024,unplanned")
  elseif(case STREQUAL "queue")
    set(_pass "materialize-completion=,rdma,7")
  else()
    set(_pass "materialize-completion=,p2p")
  endif()
  aboleth_add_ir_test(ir.materialize-completion-reuse.${case}
    FILE materialize-completion-reuse.mlir PASSES "${_pass}"
    CHECK_PREFIX ${_prefix})
endforeach()
aboleth_add_ir_test(ir.materialize-completion-shared-queue
  FILE materialize-completion-shared-queue.mlir
  PASSES "materialize-completion=,rdma,7")
aboleth_add_ir_test(ir.materialize-completion-device-epoch
  FILE materialize-completion-device-epoch.mlir
  PASSES "materialize-completion=,rdma,1024")
aboleth_add_ir_test(ir.materialize-completion-device-epoch.off
  FILE materialize-completion-device-epoch.mlir
  PASSES "materialize-completion=,rdma,1024,unplanned" CHECK_PREFIX EAGER)
aboleth_add_ir_test(ir.materialize-completion-late-iteration
  FILE materialize-completion-late-iteration.mlir
  PASSES "materialize-completion=,rdma,1024")
aboleth_add_ir_test(ir.materialize-completion-reuse-invalid
  FILE materialize-completion-reuse-invalid.mlir
  PASSES "materialize-completion=,rdma" EXPECT_FAIL)
aboleth_add_ir_test(ir.analyze-issuers
  FILE analyze-issuers.mlir PASSES form-batches analyze-issuers)
aboleth_add_ir_test(ir.analyze-issuers.rdma
  FILE analyze-issuers.mlir PASSES form-batches analyze-issuers lower-rdma
  CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.analyze-issuer-domains
  FILE analyze-issuer-domains.mlir PASSES analyze-issuers IDEMPOTENT)
aboleth_add_ir_test(ir.analyze-issuer-domains.grid
  FILE analyze-issuer-domains.mlir PASSES analyze-issuers lower-rdma CHECK_PREFIX GRID)
aboleth_add_ir_test(ir.order-producers
  FILE order-producers.mlir PASSES form-batches order-producers)
aboleth_add_ir_test(ir.order-producers.rdma
  FILE order-producers.mlir PASSES form-batches order-producers lower-rdma
  CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.form-batches-payload-effects-invalid
  FILE form-batches-payload-effects-invalid.mlir PASSES form-batches
  EXPECT_FAIL)

# --- register-direct inline payloads ----------------------------------------

aboleth_add_ir_test(ir.inline-payload.core
  FILE inline-payload.mlir PASSES form-batches CHECK_PREFIX CORE)
aboleth_add_ir_test(ir.inline-payload.rdma
  FILE inline-payload.mlir PASSES form-batches lower-rdma CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.inline-payload.device
  FILE inline-payload-device.mlir PASSES lower-rdma CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.inline-payload.device-library
  FILE inline-payload-device.mlir PASSES lower-rdma=bitcode-path=test/rdma.bc
  CHECK_PREFIX DEVICE)
aboleth_add_ir_test(ir.inline-payload.device-invalid
  FILE inline-payload-device-invalid.mlir PASSES lower-rdma=bitcode-path=test/rdma.bc
  EXPECT_FAIL)
aboleth_add_ir_test(ir.inline-payload-invalid
  FILE inline-payload-invalid.mlir EXPECT_FAIL)
# The wire's own bounds: what one work queue entry can carry, and the widths
# the payload packer defines a placement for.
aboleth_add_ir_test(ir.inline-payload-oversize-invalid
  FILE inline-payload-oversize-invalid.mlir EXPECT_FAIL)
aboleth_add_ir_test(ir.inline-payload-width-invalid
  FILE inline-payload-width-invalid.mlir EXPECT_FAIL)
# Planning counts commands where the queue counts blocks, so a deferred
# completion over a multi-block inline command is refused.
aboleth_add_ir_test(ir.inline-deferred-capacity-invalid
  FILE inline-deferred-capacity-invalid.mlir
  PASSES plan-transport EXPECT_FAIL)

# The cap is what decides, so the same module runs at the default and at a cap
# that rejects it.
aboleth_add_ir_test(ir.forward-inline-payloads
  FILE forward-inline-payloads.mlir PASSES forward-inline-payloads
  CHECK_PREFIX FORWARD IDEMPOTENT)
aboleth_add_ir_test(ir.forward-inline-payloads.capped
  FILE forward-inline-payloads.mlir PASSES forward-inline-payloads=16
  CHECK_PREFIX CAP16)
aboleth_add_ir_test(ir.forward-inline-payloads.rdma
  FILE forward-inline-payloads.mlir
  PASSES forward-inline-payloads form-batches lower-rdma CHECK_PREFIX RDMA)

# --- backend lowerings ------------------------------------------------------

foreach(backend p2p proxy rdma)
  aboleth_add_ir_test(ir.lower-${backend}
    FILE lower-${backend}.mlir PASSES form-batches lower-${backend})
endforeach()

# Descriptor loads and RDMA-write WQE stores, both emission styles.
aboleth_add_ir_test(ir.lower-rdma.scalar
  FILE lower-rdma.mlir PASSES form-batches lower-rdma CHECK_PREFIX SCALAR)
aboleth_add_ir_test(ir.lower-rdma.invariant-vector
  FILE lower-rdma.mlir
  PASSES form-batches "lower-rdma=invariant-descriptors,vector-wqe-stores"
  CHECK_PREFIX WQE)

# Detached fragments run the direct emitter with no CTA context: per-thread
# inline-asm reservation, unpredicated publishing stores, groups refused.
aboleth_add_ir_test(ir.lower-rdma.detached
  FILE lower-rdma-detached.mlir PASSES form-batches lower-rdma=detached)
aboleth_add_ir_test(ir.lower-rdma.detached-exclusive
  FILE lower-rdma-detached.mlir
  PASSES form-batches "lower-rdma=detached,exclusive"
  CHECK_PREFIX EXCL)
aboleth_add_ir_test(ir.lower-rdma.detached-groups-invalid
  FILE completion-groups.mlir PASSES form-batches lower-rdma=detached
  CHECK_PREFIX DETACHED EXPECT_FAIL)

# Detached P2P: per-thread copy stores and signal atomics, no block barrier,
# drain vacuous. Empty copy mode falls back to 'wide'; 'warp' keeps the
# cooperative tensor copy behind a warp-scope fence.
aboleth_add_ir_test(ir.lower-p2p.inline
  FILE lower-p2p-inline.mlir PASSES form-batches lower-p2p)
aboleth_add_ir_test(ir.lower-p2p.detached
  FILE lower-p2p-detached.mlir PASSES form-batches "lower-p2p=,detached"
  CHECK_PREFIX WIDE)
aboleth_add_ir_test(ir.lower-p2p.detached-scalar
  FILE lower-p2p-detached.mlir PASSES form-batches "lower-p2p=scalar,detached"
  CHECK_PREFIX SCALAR)
aboleth_add_ir_test(ir.lower-p2p.detached-warp
  FILE lower-p2p-detached.mlir PASSES form-batches "lower-p2p=warp,detached"
  CHECK_PREFIX WARP)

# The planned-protocol mechanisms are selected per submission, so each arm is a
# separate pipeline over one module.
aboleth_add_ir_test(ir.rdma-planned-protocol
  FILE rdma-planned-protocol.mlir PASSES lower-rdma CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.rdma-planned-protocol.device
  FILE rdma-planned-protocol.mlir PASSES lower-rdma=bitcode-path=test/rdma.bc
  CHECK_PREFIX DEVICE)
aboleth_add_ir_test(ir.rdma-planned-protocol.inline-signals
  FILE rdma-planned-protocol.mlir PASSES lower-rdma CHECK_PREFIX INLINE)
aboleth_add_ir_test(ir.rdma-planned-protocol-invalid
  FILE rdma-planned-protocol-invalid.mlir EXPECT_FAIL)
aboleth_add_ir_test(ir.rdma-external-planned-put
  FILE rdma-external-planned-put.mlir PASSES lower-rdma CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.rdma-external-planned-put.device
  FILE rdma-external-planned-put.mlir PASSES lower-rdma=bitcode-path=test/rdma.bc
  CHECK_PREFIX DEVICE)
aboleth_add_ir_test(ir.external-planned-put-invalid
  FILE external-planned-put-invalid.mlir PASSES lower-rdma EXPECT_FAIL)
aboleth_add_ir_test(ir.rdma-publish-head
  FILE rdma-publish-head.mlir PASSES lower-rdma CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.rdma-publish-head.device
  FILE rdma-publish-head.mlir PASSES lower-rdma=bitcode-path=test/rdma.bc
  CHECK_PREFIX DEVICE)
aboleth_add_ir_test(ir.rdma-system-fence
  FILE rdma-system-fence.mlir PASSES lower-rdma)
aboleth_add_ir_test(ir.rdma-drain
  FILE rdma-drain.mlir PASSES lower-rdma CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.rdma-drain.device
  FILE rdma-drain.mlir PASSES lower-rdma=bitcode-path=test/rdma.bc
  CHECK_PREFIX DEVICE)
aboleth_add_ir_test(ir.rdma-drain-thread
  FILE rdma-drain-thread.mlir PASSES lower-rdma CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.rdma-drain-thread.device
  FILE rdma-drain-thread.mlir PASSES lower-rdma=bitcode-path=test/rdma.bc
  CHECK_PREFIX DEVICE)
aboleth_add_ir_test(ir.rdma-drain-invalid
  FILE rdma-drain-invalid.mlir PASSES lower-rdma EXPECT_FAIL)

# The EFA lowering bakes the device-library path into the IR, so the bitcode has
# to exist before it will run.
if(ABOLETH_HAS_EFA)
  list(GET ABOLETH_EFA_BITCODE_ARCHS 0 _efa_arch)
  aboleth_add_ir_test(ir.lower-efa
    FILE lower-efa.mlir PASSES form-batches lower-efa
    ENV "ABOLETH_EFA_BITCODE=${ABOLETH_LIB_DIR}/aboleth_efa_${_efa_arch}.bc")
endif()

# --- completion groups ------------------------------------------------------

aboleth_add_ir_test(ir.completion-groups.core
  FILE completion-groups.mlir PASSES form-batches CHECK_PREFIX CORE)
foreach(backend p2p proxy rdma)
  aboleth_add_ir_test(ir.completion-groups.${backend}
    FILE completion-groups.mlir PASSES form-batches lower-${backend}
    CHECK_PREFIX LOWERED)
endforeach()
aboleth_add_ir_test(ir.completion-group-invalid
  FILE completion-group-invalid.mlir EXPECT_FAIL)

aboleth_add_ir_test(ir.conditional-completion.core
  FILE conditional-completion.mlir PASSES form-batches CHECK_PREFIX CORE)
foreach(backend proxy rdma)
  aboleth_add_ir_test(ir.conditional-completion.${backend}
    FILE conditional-completion.mlir PASSES form-batches lower-${backend}
    CHECK_PREFIX LOWERED)
endforeach()
aboleth_add_ir_test(ir.conditional-completion.rdma-inline-signals
  FILE conditional-completion.mlir
  PASSES plan-publication form-batches lower-rdma
  CHECK_PREFIX INLINE)
aboleth_add_ir_test(ir.self-reclaim.core
  FILE self-reclaim.mlir PASSES form-batches CHECK_PREFIX CORE)
aboleth_add_ir_test(ir.self-reclaim.rdma
  FILE self-reclaim.mlir PASSES form-batches "lower-rdma=exclusive,self-reclaim"
  CHECK_PREFIX RECLAIM)
# Off, and on but with endpoints the module cannot prove single-issuer.
aboleth_add_ir_test(ir.self-reclaim.rdma-off
  FILE self-reclaim.mlir PASSES form-batches "lower-rdma=exclusive"
  CHECK_PREFIX PLAIN)
aboleth_add_ir_test(ir.self-reclaim.rdma-shared
  FILE self-reclaim.mlir PASSES form-batches "lower-rdma=self-reclaim"
  CHECK_PREFIX PLAIN)

# --- dynamic endpoints ------------------------------------------------------

aboleth_add_ir_test(ir.dynamic-endpoint.rdma
  FILE dynamic-endpoint.mlir PASSES form-batches lower-rdma CHECK_PREFIX RDMA)
aboleth_add_ir_test(ir.dynamic-endpoint.proxy
  FILE dynamic-endpoint.mlir PASSES form-batches lower-proxy CHECK_PREFIX PROXY)

# --- verifier rejections ----------------------------------------------------

foreach(case ordering submit submit-multiple poll-multiple
             transport-fragment-capture)
  aboleth_add_ir_test(ir.${case}-invalid FILE ${case}-invalid.mlir EXPECT_FAIL)
endforeach()

# --- goldens ----------------------------------------------------------------

# Every module through every pipeline that accepts it, compared byte for byte.
# Writing is a build target rather than a test, so a golden can never be
# refreshed as a side effect of running the suite.
if(EXISTS "${ABOLETH_GOLDEN_DIR}")
  add_test(NAME goldens.check
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/goldens.py"
            --check "${ABOLETH_GOLDEN_DIR}")
  set_tests_properties(goldens.check PROPERTIES
    LABELS "goldens"
    ENVIRONMENT "${ABOLETH_TEST_ENV}")
else()
  message(STATUS
    "[Aboleth] compiler goldens are not initialized; run "
    "aboleth-goldens-write before enabling goldens.check")
endif()

add_custom_target(aboleth-goldens-write
  COMMAND "${CMAKE_COMMAND}" -E env ${ABOLETH_TEST_ENV}
          "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/goldens.py"
          --write "${ABOLETH_GOLDEN_DIR}"
  DEPENDS AbolethPlugin
  COMMENT "Rewriting the compiler goldens in ${ABOLETH_GOLDEN_DIR}"
  VERBATIM)

# Empty routes retain an ordered NOP instead of a zero-length mlx5 data SGE.
aboleth_add_ir_test(ir.rdma-empty-put FILE rdma-empty-put.mlir
  PASSES form-batches lower-rdma)



aboleth_add_ir_test(ir.rdma-ring-index
  FILE rdma-ring-index.mlir PASSES form-batches lower-rdma)
