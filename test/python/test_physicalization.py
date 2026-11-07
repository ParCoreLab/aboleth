import os
import pathlib
import tempfile
import unittest
from dataclasses import replace

from aboleth import (
    Artifact,
    KVCacheMigrationSpec,
    MoEPatternSpec,
    OpaqueCapabilities,
    P2PRealizationProfile,
    Pattern,
    PhaseBinding,
    PhaseResources,
    RegionEffect,
    RuntimeBindings,
    RuntimeBufferBinding,
    build_moe_pattern,
    device_library,
)
from aboleth.artifact import (
    realize_bindable_epilogue,
    realize_standalone_kernel,
)
from aboleth.pattern import CommunicationStage
from aboleth.patterns.kv_cache_transport import (
    physicalize_kv_cache_pattern,
)
from aboleth.patterns.moe_transport import (
    moe_phase_bindings,
    physicalize_moe_pattern,
)
from aboleth.physicalization import (
    physicalize_pattern,
)
from aboleth.transport_binding import (
    BoundScalar,
    PublicationBinding,
    RegionBinding,
    RegionSide,
    TransportProgramBinding,
)

ARCH = "sm_80"
MOE_P2P_REALIZATION = P2PRealizationProfile(
    direct_destination_kernel="direct_dispatch_p2p_kernel",
    direct_publication_kernels=(
        ("packed_doorbell", "p2p_dispatch_doorbell_wait_kernel"),
        ("coalesced_kernel", "p2p_dispatch_publish_wait_kernel"),
        ("separate_kernel", "p2p_dispatch_publish_kernel"),
    ),
    parallel_copy_kernel="p2p_return_copy_kernel",
    parallel_publication_kernels=(
        ("consumer_prologue", None),
        ("coalesced_kernel", "p2p_return_publish_kernel"),
    ),
)


def built_device_library():
    """The device library this build emitted, whatever architecture that is."""
    root = pathlib.Path(__file__).resolve().parents[2]
    build = pathlib.Path(
        os.environ.get("ABOLETH_BUILD_DIR", root / "build")
    )
    built = sorted((build / "lib").glob("aboleth_rdma_*.bc"))
    if not built:
        return None
    return device_library(
        built[0].stem.removeprefix("aboleth_rdma_"), generated=False
    )


LIBRARY = built_device_library()

# Effect-only: the first capability level, which permits no rewrite at all.
EFFECT_ONLY = OpaqueCapabilities()
DESTINATION_BINDABLE = OpaqueCapabilities(
    accepts_destination_rewrite=True,
)
INDEXED_PRODUCER = OpaqueCapabilities(
    accepts_destination_rewrite=True,
    accepts_indexed_scatter=True,
)


def scatter_pattern(
    capabilities: OpaqueCapabilities,
    *,
    unique: bool = True,
):
    """One per-item transfer landing in storage the consumer declares.

    With `unique` the plan retargets the transfer straight into that storage
    and drops the copy; without it the local `copy_each` survives, which is
    the shape KV-cache migration produces for duplicate destinations.
    """
    pattern = Pattern("scatter")
    rows = pattern.region(
        "rows",
        element_type="i64",
        shape=(4,),
        strides=(1,),
        immutable=True,
    )
    peers = pattern.region(
        "peers",
        element_type="i32",
        shape=(4,),
        strides=(1,),
        immutable=True,
    )
    slots = pattern.region(
        "slots",
        element_type="i64",
        shape=(4,),
        strides=(1,),
        immutable=True,
    )
    source = pattern.region(
        "source",
        element_type="f32",
        shape=(4, 8),
        strides=(8, 1),
        immutable=True,
        registered=True,
    )
    staging = pattern.region(
        "staging",
        element_type="f32",
        shape=(4, 8),
        strides=(8, 1),
        visibility="peer",
    )
    storage = pattern.region(
        "storage",
        element_type="f32",
        shape=(4, 8),
        strides=(8, 1),
        visibility="peer",
        escapes=True,
        registered=True,
        donated=True,
        reuse_interval=1,
        pool_depth=2,
    )
    work = pattern.workset(
        "work",
        fields={
            "source_row": rows,
            "destination_peer": peers,
            "destination_row": slots,
        },
        bounds={
            "source_row": (0, 4),
            "destination_peer": (0, 2),
            "destination_row": (0, 4),
        },
        max_items=4,
        unique_destinations=unique,
    )
    moved = pattern.transfer_each(
        "payload", work, source, staging
    )
    landed = pattern.copy_each(
        "payload.materialize",
        work,
        moved,
        storage,
        source_index="destination_row",
        destination_index="destination_row",
        output="landed",
    )
    pattern.opaque_compute(
        "consumer",
        reads=(landed,),
        capabilities=capabilities,
    )
    return pattern.compile()


def reduce_pattern(*, indexed: bool = False):
    """One reduction of `cardinality` contributions ahead of a consumer."""
    pattern = Pattern("reduce")
    source = pattern.region(
        "source",
        element_type="f32",
        shape=(4, 8),
        strides=(8, 1),
        immutable=True,
        registered=True,
    )
    staging = pattern.region(
        "staging",
        element_type="f32",
        shape=(2, 8) if indexed else (4, 8),
        strides=(8, 1),
        visibility="peer",
    )
    if indexed:
        pattern.region(
            "rows",
            element_type="i64",
            shape=(2,),
            strides=(1,),
            immutable=True,
        )
        source = pattern.indexed_view(
            "source.live",
            source,
            "rows",
            shape=(2, 8),
            strides=(8, 1),
            index_lower_bounds=(0,),
            index_upper_bounds=(4,),
            injective=True,
        )
    reduced = pattern.reduction(
        "combine",
        source,
        staging,
        cardinality=2,
        operator="add",
        output="combined",
    )
    pattern.opaque_compute(
        "consumer",
        reads=(reduced,),
        capabilities=EFFECT_ONLY,
    )
    return pattern.compile()


def mangled_pattern():
    """Two transfers whose ids differ only where slot mangling erases it."""
    pattern = Pattern("mangled")
    source = pattern.region(
        "source",
        element_type="f32",
        shape=(4, 8),
        strides=(8, 1),
        immutable=True,
        registered=True,
    )
    first = pattern.transfer(
        "payload.first",
        source,
        pattern.region(
            "staging.first",
            element_type="f32",
            shape=(4, 8),
            strides=(8, 1),
            visibility="peer",
        ),
    )
    second = pattern.transfer(
        "payload_first",
        source,
        pattern.region(
            "staging.second",
            element_type="f32",
            shape=(4, 8),
            strides=(8, 1),
            visibility="peer",
        ),
    )
    pattern.opaque_compute(
        "consumer",
        reads=(first, second),
        capabilities=EFFECT_ONLY,
    )
    return pattern.compile()


def with_capabilities(compiled, stage: str, **overrides):
    """`compiled` with one opaque stage's declared capabilities amended."""
    return replace(compiled, opaque_capabilities=tuple(
        (
            name,
            replace(declared, **overrides)
            if name == stage
            else declared,
        )
        for name, declared in compiled.opaque_capabilities
    ))


def side(pointer: str, key: str) -> RegionSide:
    return RegionSide(
        pointer=BoundScalar(pointer, "i64"),
        key=BoundScalar(key, "i32"),
    )


def binding(**overrides) -> TransportProgramBinding:
    return replace(
        TransportProgramBinding(
            name="probe",
            operations=("payload",),
            kernel="probe_kernel",
            insertion_point="payload",
            endpoint=BoundScalar("endpoint", "i64"),
            regions=(RegionBinding(
                operations=("payload",),
                remote=side(
                    "payload_destination",
                    "payload_destination_key",
                ),
                local=side(
                    "payload_source",
                    "payload_source_key",
                ),
                byte_count=BoundScalar("payload_bytes", "i32"),
                predicate=BoundScalar("payload_predicate", "i1"),
                max_bytes=4096,
            ),),
            publication=PublicationBinding(
                remote=side(
                    "ready_destination",
                    "ready_destination_key",
                ),
                local=side("ready_source", "ready_source_key"),
                iteration=BoundScalar("iteration", "i32"),
                iterations=BoundScalar("iterations", "i32"),
            ),
        ),
        **overrides,
    )


def attachment(**overrides) -> PhaseBinding:
    return replace(
        PhaseBinding(
            producer=None,
            consumer="consumer",
            name="probe",
            kernel="probe_kernel",
            insertion_point="payload",
            peer_segments=2,
            publication="probe.ready",
        ),
        **overrides,
    )


def transport_bindings(compiled, *, num_peers=2):
    return PhaseResources(
        regions={
            region.id: RegionEffect(
                allocation=index,
                generation=1,
                offset=0,
                length=4096,
                stride=4096,
                count=1,
                period=4096,
            )
            for index, region in enumerate(
                compiled.analysis.regions
            )
        },
        peer_transports=("p2p",),
        peer_capacities=(2**31 - 1,),
        num_peers=num_peers,
        recycled_buffers=("send_counts",),
        advanced_values=("epoch_base",),
        p2p_realization=MOE_P2P_REALIZATION,
    )


class CommunicationPhysicalizationTest(unittest.TestCase):
    def setUp(self):
        self.compiled = build_moe_pattern(MoEPatternSpec(
            max_tokens=64,
            hidden=128,
            top_k=2,
            num_ranks=4,
        )).compile()

    def test_derives_operations_sizes_slots_and_manifest(self):
        result = physicalize_pattern(
            self.compiled,
            (
                PhaseBinding(
                    producer="route",
                    consumer="experts",
                    name="moe.dispatch",
                    kernel="dispatch_kernel",
                    insertion_point="payload",
                    peer_segments=4,
                    items_per_transfer=32,
                    publication="dispatch.ready",
                ),
            ),
        )
        binding = result.binding("moe.dispatch")
        manifest = result.manifest("moe.dispatch")

        self.assertEqual(
            binding.operations,
            (
                "dispatch.values",
                "dispatch.expert_ids",
                "dispatch.count",
            ),
        )
        self.assertEqual(
            tuple(
                region.max_bytes
                for region in binding.regions
            ),
            (
                32 * 128 * 2,
                32 * 4,
                32 * 128 * 2,
                32 * 4,
                32 * 128 * 2,
                32 * 4,
                32 * 128 * 2,
                32 * 4,
                4,
            ),
        )
        self.assertEqual(manifest.program_id, 0)
        self.assertEqual(
            len(manifest.transfers),
            len(binding.regions),
        )
        self.assertEqual(
            result.compiled.binding(
                "dispatch.values"
            ).kernel,
            "dispatch_kernel",
        )

    def test_compiler_selects_parallel_p2p_for_phi_shape(self):
        spec = MoEPatternSpec(
            max_tokens=128,
            hidden=2560,
            top_k=2,
            num_ranks=2,
            slots=1,
        )
        physicalized = physicalize_moe_pattern(spec)
        plan = physicalized.compiled.compile_plan(
            replace(
                transport_bindings(
                    physicalized.compiled,
                    num_peers=2,
                ),
                consumer_grid_size=128,
            ),
            cyclic=True,
        )

        dispatch = plan.phase_schedule("moe.dispatch")
        owner_return = plan.phase_schedule("moe.return")
        self.assertEqual(
            dispatch.strategy,
            "direct_destination",
        )
        self.assertEqual(
            owner_return.strategy,
            "parallel_copy",
        )
        self.assertEqual(
            dispatch.publication_strategy,
            "coalesced_kernel",
        )
        self.assertEqual(
            owner_return.publication_strategy,
            "consumer_prologue",
        )
        self.assertEqual(
            dispatch.publication_kernel,
            "p2p_dispatch_publish_wait_kernel",
        )
        self.assertEqual(
            dispatch.recycled_buffers,
            ("send_counts",),
        )
        self.assertEqual(
            dispatch.advanced_values,
            ("epoch_base",),
        )
        self.assertIsNone(owner_return.publication_kernel)
        self.assertEqual(
            dispatch.kernel,
            "direct_dispatch_p2p_kernel",
        )
        self.assertEqual(
            owner_return.grid_axes,
            ("peers", "payload_blocks"),
        )
        self.assertEqual(owner_return.block_elements, 1024)
        self.assertEqual(owner_return.buffer_depth, 2)
        dispatch_program = plan.transport_program_for_kernel(
            "pack_dispatch_alltoall_kernel"
        )
        self.assertEqual(len(dispatch_program.transfers), 1)
        self.assertEqual(
            dispatch_program.transfers[0].operations,
            ("dispatch.count",),
        )
        self.assertEqual(
            plan.transport_program_for_kernel(
                "return_alltoall_kernel"
            ).transfers,
            (),
        )
        self.assertNotIn("None", plan.transport_mlir)

        separate = physicalized.compiled.compile_plan(
            replace(
                transport_bindings(
                    physicalized.compiled,
                    num_peers=2,
                ),
                coalesce_publication_wait=False,
                coalesce_return_with_consumer=False,
                recycled_buffers=(),
                advanced_values=(),
                consumer_grid_size=128,
            ),
            cyclic=True,
        )
        self.assertEqual(
            separate.phase_schedule(
                "moe.dispatch"
            ).publication_kernel,
            "p2p_dispatch_publish_kernel",
        )
        self.assertEqual(
            separate.phase_schedule(
                "moe.return"
            ).publication_kernel,
            "p2p_return_publish_kernel",
        )
        self.assertEqual(
            separate.phase_schedule(
                "moe.dispatch"
            ).recycled_buffers,
            (),
        )
        self.assertEqual(
            separate.phase_schedule(
                "moe.dispatch"
            ).advanced_values,
            (),
        )

        generic_lifecycle = physicalized.compiled.compile_plan(
            replace(
                transport_bindings(
                    physicalized.compiled,
                    num_peers=2,
                ),
                recycled_buffers=("client_scratch",),
                advanced_values=("client_generation",),
                consumer_grid_size=128,
            ),
            cyclic=True,
        )
        generic_dispatch = generic_lifecycle.phase_schedule(
            "moe.dispatch"
        )
        self.assertEqual(
            generic_dispatch.recycled_buffers,
            ("client_scratch",),
        )
        self.assertEqual(
            generic_dispatch.advanced_values,
            ("client_generation",),
        )

        custom_profile = replace(
            MOE_P2P_REALIZATION,
            direct_destination_kernel="client_direct_kernel",
            direct_publication_kernels=(
                ("packed_doorbell", "client_doorbell_kernel"),
                ("coalesced_kernel", "client_coalesced_kernel"),
                ("separate_kernel", "client_publication_kernel"),
            ),
        )
        custom_plan = physicalized.compiled.compile_plan(
            replace(
                transport_bindings(
                    physicalized.compiled,
                    num_peers=2,
                ),
                p2p_realization=custom_profile,
                consumer_grid_size=128,
            ),
            cyclic=True,
        )
        custom_dispatch = custom_plan.phase_schedule(
            "moe.dispatch"
        )
        self.assertEqual(
            custom_dispatch.kernel,
            "client_direct_kernel",
        )
        self.assertEqual(
            custom_dispatch.publication_kernel,
            "client_coalesced_kernel",
        )

        packed_doorbell = (
            physicalized.compiled.compile_plan(
                replace(
                    transport_bindings(
                        physicalized.compiled,
                        num_peers=2,
                    ),
                    p2p_packed_doorbell=True,
                    consumer_grid_size=128,
                ),
                cyclic=True,
            )
        )
        self.assertEqual(
            packed_doorbell.phase_schedule(
                "moe.dispatch"
            ).publication_strategy,
            "packed_doorbell",
        )
        self.assertEqual(
            packed_doorbell.transport_program_for_kernel(
                "pack_dispatch_alltoall_kernel"
            ).transfers,
            (),
        )

    def test_compiler_keeps_small_p2p_inline(self):
        spec = MoEPatternSpec(
            max_tokens=2,
            hidden=64,
            top_k=1,
            num_ranks=2,
            slots=1,
        )
        physicalized = physicalize_moe_pattern(spec)
        plan = physicalized.compiled.compile_plan(
            transport_bindings(
                physicalized.compiled,
                num_peers=2,
            ),
            cyclic=True,
        )

        self.assertEqual(
            plan.phase_schedule("moe.dispatch").strategy,
            "inline_transport",
        )
        self.assertEqual(
            plan.phase_schedule("moe.return").strategy,
            "inline_transport",
        )

    def test_runtime_bindings_resolve_aliases_once(self):
        compiled = build_moe_pattern(MoEPatternSpec(
            max_tokens=2,
            hidden=4,
            top_k=1,
            num_ranks=2,
        )).compile()
        bindings = RuntimeBindings(tuple(
            RuntimeBufferBinding(
                f"buffer_{index}",
                (region.id,),
            )
            for index, region in enumerate(
                compiled.analysis.regions
            )
        ))

        effects = bindings.effects(
            compiled,
            lambda name: RegionEffect(
                allocation=int(name.split("_")[-1]),
                generation=1,
                offset=0,
                length=64,
                stride=64,
                count=1,
                period=64,
            ),
        )

        self.assertEqual(
            set(effects),
            {
                region.id
                for region in compiled.analysis.regions
            },
        )

    def test_runtime_bindings_reject_missing_regions(self):
        with self.assertRaisesRegex(
            KeyError,
            "no runtime buffer binding",
        ):
            RuntimeBindings((
                RuntimeBufferBinding(
                    "inputs",
                    ("input.tokens",),
                ),
            )).effects(
                self.compiled,
                lambda _: None,
            )

    def test_runtime_bindings_reject_unknown_regions(self):
        bindings = tuple(
            RuntimeBufferBinding(
                f"buffer_{index}",
                (region.id,),
            )
            for index, region in enumerate(
                self.compiled.analysis.regions
            )
        ) + (
            RuntimeBufferBinding(
                "typo",
                ("dispatch.typo",),
            ),
        )
        with self.assertRaisesRegex(
            KeyError,
            "unknown logical regions",
        ):
            RuntimeBindings(bindings).effects(
                self.compiled,
                lambda _: RegionEffect(
                    allocation=0,
                    generation=1,
                    offset=0,
                    length=64,
                    stride=64,
                    count=1,
                    period=64,
                ),
            )


class AttachmentValidationTest(unittest.TestCase):
    def setUp(self):
        self.compiled = scatter_pattern(INDEXED_PRODUCER)

    def test_unknown_producer_names_the_declared_stages(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                self.compiled,
                (attachment(producer="pack"),),
            )
        message = str(raised.exception)
        self.assertIn("unknown producer stage 'pack'", message)
        self.assertIn("the pattern declares consumer", message)

    def test_unknown_consumer_names_the_declared_stages(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                self.compiled,
                (attachment(consumer="expert"),),
            )
        message = str(raised.exception)
        self.assertIn("unknown consumer stage 'expert'", message)
        self.assertIn("consumer", message)

    def test_a_declared_but_wrong_boundary_lists_the_boundaries(self):
        spec = MoEPatternSpec(
            max_tokens=64,
            hidden=128,
            top_k=2,
            num_ranks=4,
        )
        dispatch, _ = moe_phase_bindings(spec)
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                build_moe_pattern(spec).compile(),
                (replace(dispatch, producer="combine"),),
            )
        message = str(raised.exception)
        self.assertIn("matched 0 stages", message)
        self.assertIn(
            "route -> communication.0 -> experts",
            message,
        )
        self.assertIn(
            "experts -> communication.1 -> combine",
            message,
        )

    def test_artifact_must_be_one_of_the_three_kinds(self):
        with self.assertRaises(ValueError) as raised:
            attachment(artifact="bindable_epilogue")
        message = str(raised.exception)
        self.assertIn("must be an Artifact", message)
        self.assertIn("Artifact.BINDABLE_EPILOGUE", message)

    def test_two_attachments_cannot_share_one_insertion_point(self):
        spec = MoEPatternSpec(
            max_tokens=64,
            hidden=128,
            top_k=2,
            num_ranks=4,
        )
        first, second = moe_phase_bindings(spec)
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                build_moe_pattern(spec).compile(),
                (
                    first,
                    replace(second, kernel=first.kernel),
                ),
            )
        message = str(raised.exception)
        self.assertIn("'moe.return'", message)
        self.assertIn(first.kernel, message)
        self.assertIn("attachment 'moe.dispatch'", message)


class ArtifactCapabilityTest(unittest.TestCase):
    def test_inline_fragment_asks_for_no_permission(self):
        result = physicalize_pattern(
            scatter_pattern(EFFECT_ONLY),
            (attachment(),),
        )
        self.assertEqual(
            result.program("probe").artifact,
            "inline_fragment",
        )

    def test_epilogue_requires_destination_rewrite(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                scatter_pattern(EFFECT_ONLY),
                (attachment(
                    artifact=Artifact.BINDABLE_EPILOGUE,
                ),),
            )
        message = str(raised.exception)
        self.assertIn("cannot be a bindable_epilogue", message)
        self.assertIn("retargeted into 'storage'", message)
        self.assertIn(
            "accepts_destination_rewrite",
            message,
        )

    def test_epilogue_requires_indexed_scatter_per_item(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                scatter_pattern(DESTINATION_BINDABLE),
                (attachment(
                    artifact=Artifact.BINDABLE_EPILOGUE,
                ),),
            )
        message = str(raised.exception)
        self.assertIn("cannot be a bindable_epilogue", message)
        self.assertIn("workset index map", message)
        self.assertIn("accepts_indexed_scatter", message)

    def test_detached_artifact_requires_global_addresses(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                scatter_pattern(replace(
                    INDEXED_PRODUCER,
                    accepted_address_spaces=("shared",),
                )),
                (attachment(
                    artifact=Artifact.BINDABLE_EPILOGUE,
                ),),
            )
        message = str(raised.exception)
        self.assertIn("address spaces shared", message)
        self.assertIn("not global", message)

    def test_standalone_requires_a_current_stream_contract(self):
        spec = MoEPatternSpec(
            max_tokens=64,
            hidden=128,
            top_k=2,
            num_ranks=4,
        )
        dispatch, _ = moe_phase_bindings(spec)
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                build_moe_pattern(spec).compile(),
                (replace(
                    dispatch,
                    artifact=Artifact.STANDALONE_KERNEL,
                ),),
            )
        message = str(raised.exception)
        self.assertIn("cannot be a standalone_kernel", message)
        self.assertIn("'external_event'", message)
        self.assertIn("'current_stream'", message)

    def test_permitted_kinds_reach_the_binding_and_the_plan(self):
        for kind in (
            Artifact.BINDABLE_EPILOGUE,
            Artifact.STANDALONE_KERNEL,
        ):
            with self.subTest(kind=kind):
                compiled = physicalize_pattern(
                    scatter_pattern(INDEXED_PRODUCER),
                    (attachment(artifact=kind),),
                ).compiled
                self.assertEqual(
                    compiled.binding("payload").artifact,
                    kind.value,
                )
                plan = compiled.compile_plan(
                    transport_bindings(compiled)
                )
                self.assertIn(
                    f'artifactKind = "{kind.value}"',
                    plan.transport_mlir,
                )
                self.assertEqual(
                    plan.to_dict()[
                        "kernel_transport_programs"
                    ][0]["artifact"],
                    kind.value,
                )


class PhysicalNeighbourGateTest(unittest.TestCase):
    """The gates ask the stages the communication physically sits between.

    An attachment may leave one boundary out to match on the other, but the
    stage it left out still runs beside the artifact and still has to permit
    it, so the names the attachment supplied do not select who is asked.
    """

    def setUp(self):
        self.spec = MoEPatternSpec(
            max_tokens=64,
            hidden=128,
            top_k=2,
            num_ranks=4,
        )
        self.compiled = build_moe_pattern(self.spec).compile()
        self.dispatch, self.owner_return = moe_phase_bindings(
            self.spec
        )

    def test_an_omitted_consumer_gates_the_completion_contract(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                self.compiled,
                (replace(
                    self.dispatch,
                    consumer=None,
                    artifact=Artifact.STANDALONE_KERNEL,
                ),),
            )
        message = str(raised.exception)
        self.assertIn("stage 'experts'", message)
        self.assertIn("'external_event'", message)
        self.assertIn("'current_stream'", message)

    def test_an_omitted_consumer_gates_the_destination_rewrite(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                self.compiled,
                (replace(
                    self.owner_return,
                    consumer=None,
                    artifact=Artifact.BINDABLE_EPILOGUE,
                ),),
            )
        message = str(raised.exception)
        self.assertIn(
            "retargeted into 'return.values.storage'",
            message,
        )
        self.assertIn("stage 'combine'", message)
        self.assertIn("accepts_destination_rewrite", message)

    def test_an_omitted_consumer_gates_the_address_space(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                with_capabilities(
                    self.compiled,
                    "experts",
                    accepted_address_spaces=("shared",),
                ),
                (replace(
                    self.dispatch,
                    consumer=None,
                    artifact=Artifact.BINDABLE_EPILOGUE,
                ),),
            )
        message = str(raised.exception)
        self.assertIn("stage 'experts'", message)
        self.assertIn("address spaces shared", message)
        self.assertIn("not global", message)

    def test_a_schedule_edge_has_no_neighbour_to_ask(self):
        # scatter's communication is the first stage, so nothing precedes it:
        # the one consumer beside it is the whole gate.
        compiled = scatter_pattern(INDEXED_PRODUCER)
        self.assertIsInstance(
            compiled.stages[0],
            CommunicationStage,
        )
        result = physicalize_pattern(
            compiled,
            (attachment(artifact=Artifact.BINDABLE_EPILOGUE),),
        )

        self.assertEqual(
            result.program("probe").artifact,
            "bindable_epilogue",
        )

    def test_an_unresolvable_declared_effect_is_refused(self):
        # A stage whose declared effect resolves to no region cannot answer
        # the capability question at all; answering it "permitted" is how the
        # gate used to be skipped.
        compiled = scatter_pattern(EFFECT_ONLY)
        consumer = compiled.opaque_stage("consumer")
        shadowed = replace(compiled, stages=tuple(
            replace(stage, reads=("landed.shadow",))
            if stage is consumer
            else stage
            for stage in compiled.stages
        ))
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                shadowed,
                (attachment(
                    artifact=Artifact.BINDABLE_EPILOGUE,
                ),),
            )
        message = str(raised.exception)
        self.assertIn("stage 'consumer'", message)
        self.assertIn("'landed.shadow'", message)


class DataflowIdTest(unittest.TestCase):
    """An ID has to survive both a quoted attribute and an identifier."""

    def region(self, pattern, name):
        return pattern.region(
            name,
            element_type="f32",
            shape=(4, 8),
            strides=(8, 1),
            immutable=True,
            registered=True,
        )

    def test_a_leading_digit_is_refused(self):
        # It would reach the shim header as a parameter name.
        with self.assertRaises(ValueError) as raised:
            self.region(Pattern("ids"), "0payload")
        message = str(raised.exception)
        self.assertIn("region ID '0payload'", message)
        self.assertIn("begins with a letter", message)

    def test_a_quote_is_refused(self):
        pattern = Pattern("ids")
        source = self.region(pattern, "source")
        staging = pattern.region(
            "staging",
            element_type="f32",
            shape=(4, 8),
            strides=(8, 1),
            visibility="peer",
        )
        with self.assertRaises(ValueError) as raised:
            pattern.transfer(
                'payload", evil = "',
                source,
                staging,
            )
        self.assertIn("operation ID", str(raised.exception))

    def test_dotted_and_hyphenated_names_stay_legal(self):
        # The two forms every shipped pattern uses: dotted namespacing on the
        # ids, a hyphenated label on the dataflow itself.
        pattern = Pattern("kv-cache-migration")
        source = self.region(pattern, "source.live.window")
        staging = pattern.region(
            "staging",
            element_type="f32",
            shape=(4, 8),
            strides=(8, 1),
            visibility="peer",
        )
        moved = pattern.transfer(
            "migration.transfer", source, staging
        )
        pattern.opaque_compute(
            "consumer",
            reads=(moved,),
            capabilities=EFFECT_ONLY,
        )

        self.assertEqual(
            physicalize_pattern(
                pattern.compile(),
                (attachment(),),
            ).program("probe").operations,
            ("migration.transfer",),
        )


class SlotCollisionTest(unittest.TestCase):
    """Two slot sources that mangle alike would name each other's storage."""

    def test_a_publication_cannot_take_a_transfers_stem(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                scatter_pattern(INDEXED_PRODUCER),
                (attachment(publication="payload"),),
            )
        message = str(raised.exception)
        self.assertIn("slot stem 'payload'", message)
        self.assertIn("operation 'payload'", message)
        self.assertIn("publication 'payload'", message)

    def test_two_operation_ids_cannot_mangle_alike(self):
        with self.assertRaises(ValueError) as raised:
            physicalize_pattern(
                mangled_pattern(),
                (attachment(),),
            )
        message = str(raised.exception)
        self.assertIn("slot stem 'payload_first'", message)
        self.assertIn("operation 'payload.first'", message)
        self.assertIn("operation 'payload_first'", message)

    def test_a_binding_refuses_one_name_for_two_roles(self):
        colliding = binding(publication=PublicationBinding(
            remote=side(
                "payload_destination",
                "payload_destination_key",
            ),
            local=side("ready_source", "ready_source_key"),
            iteration=BoundScalar("iteration", "i32"),
            iterations=BoundScalar("iterations", "i32"),
        ))
        with self.assertRaises(ValueError) as raised:
            colliding.slots()
        message = str(raised.exception)
        self.assertIn("'payload_destination'", message)
        self.assertIn("region 0 remote pointer", message)
        self.assertIn("publication remote pointer", message)

    def test_distinct_names_still_reach_the_program(self):
        self.assertEqual(
            tuple(slot.name for slot in binding().slots()),
            (
                "endpoint",
                "payload_destination",
                "payload_source",
                "payload_bytes",
                "payload_source_key",
                "payload_destination_key",
                "payload_predicate",
                "ready_destination",
                "ready_source",
                "ready_source_key",
                "ready_destination_key",
                "iteration",
                "iterations",
            ),
        )


class ReductionPhysicalizationTest(unittest.TestCase):
    def test_reduction_binds_like_a_transfer_plus_a_contract(self):
        result = physicalize_pattern(
            reduce_pattern(),
            (attachment(),),
        )
        program = result.program("probe")

        self.assertEqual(program.operations, ("combine",))
        # Half of a (4, 8) f32 source per peer segment.
        self.assertEqual(
            tuple(
                transfer.max_bytes
                for transfer in program.transfers
            ),
            (2 * 8 * 4,),
        )
        (reduction,) = program.reductions
        self.assertEqual(reduction.operator, "add")
        self.assertEqual(reduction.cardinality, 2)
        self.assertIsNone(reduction.cardinality_source)

    def test_indexed_source_supplies_the_cardinality_source(self):
        result = physicalize_pattern(
            reduce_pattern(indexed=True),
            (attachment(),),
        )
        (reduction,) = result.program("probe").reductions

        self.assertEqual(
            reduction.cardinality_source,
            "rows",
        )

    def test_the_contract_reaches_the_plan(self):
        compiled = physicalize_pattern(
            reduce_pattern(),
            (attachment(),),
        ).compiled
        plan = compiled.compile_plan(
            transport_bindings(compiled)
        )

        self.assertEqual(
            plan.to_dict()[
                "kernel_transport_programs"
            ][0]["reductions"],
            [{
                "operation": "combine",
                "operator": "add",
                "cardinality": 2,
                "cardinality_source": None,
            }],
        )


class CopyPhysicalizationTest(unittest.TestCase):
    def test_per_item_copy_binds_the_worksets_items(self):
        compiled = scatter_pattern(
            INDEXED_PRODUCER,
            unique=False,
        )
        program = physicalize_pattern(
            compiled,
            (attachment(),),
        ).program("probe")

        self.assertEqual(
            program.operations,
            ("payload", "payload.materialize"),
        )
        self.assertEqual(
            tuple(
                transfer.max_bytes
                for transfer in program.transfers
            ),
            # Both are workset-driven, so both bound all four items.
            (4 * 8 * 4, 4 * 8 * 4),
        )

    def test_kv_cache_duplicate_pages_physicalize(self):
        spec = KVCacheMigrationSpec(
            max_pages=8,
            page_bytes=4096,
            num_ranks=2,
            unique_destinations=False,
        )
        program = physicalize_kv_cache_pattern(
            spec
        ).program("kv_cache.migrate")

        self.assertEqual(
            program.operations,
            ("migration.transfer", "migration.materialize"),
        )
        # items_per_transfer=1 chunks both operations into eight pages each,
        # plus the publication.
        self.assertEqual(program.command_count, 17)

    def test_moe_staged_return_copies_the_whole_extent(self):
        spec = MoEPatternSpec(
            max_tokens=64,
            hidden=128,
            top_k=2,
            num_ranks=4,
            preserve_return_staging=True,
        )
        result = physicalize_pattern(
            build_moe_pattern(spec).compile(),
            moe_phase_bindings(spec),
        )
        program = result.program("moe.return")

        self.assertEqual(
            program.operations,
            ("return.values", "return.values.materialize"),
        )
        rows = 64 * 2 * 4
        self.assertEqual(
            tuple(
                transfer.max_bytes
                for transfer in program.transfers
            ),
            (rows // 4 * 128 * 2, rows * 128 * 2),
        )

    def test_a_bound_copy_stays_inside_its_phase(self):
        compiled = physicalize_pattern(
            scatter_pattern(INDEXED_PRODUCER, unique=False),
            (attachment(),),
        ).compiled
        plan = compiled.compile_plan(
            transport_bindings(compiled)
        )

        self.assertEqual(
            tuple(
                operation.name
                for operation in plan.transport_operations
            ),
            ("communication.0", "consumer"),
        )
        # Two commands and the publication, counted once.
        self.assertEqual(
            plan.transport_operations[
                0
            ].commands_per_iteration,
            3,
        )

    def _transform_pattern(self, transform_name: str) -> "Pattern":
        pattern = Pattern("transformed")
        source = pattern.region(
            "source",
            element_type="f32",
            shape=(4, 8),
            strides=(8, 1),
            immutable=True,
            registered=True,
        )
        staging = pattern.region(
            "staging",
            element_type="f32",
            shape=(4, 8),
            strides=(8, 1),
            visibility="peer",
        )
        moved = pattern.transfer("payload", source, staging)
        final = pattern.region(
            "final",
            element_type="f8E4M3FN",
            shape=(4, 8),
            strides=(8, 1),
            visibility="peer",
            escapes=True,
        )
        scaled = pattern.transform(
            "payload.scale",
            moved,
            final,
            transform=transform_name,
            index_map="affine_map<(d0, d1) -> (d0, d1)>",
        )
        pattern.opaque_compute(
            "consumer",
            reads=(scaled,),
            capabilities=EFFECT_ONLY,
        )
        return pattern

    def test_an_unknown_transform_is_rejected_by_name(self):
        with self.assertRaises(NotImplementedError) as raised:
            physicalize_pattern(
                self._transform_pattern("scale").compile(),
                (attachment(),),
            )
        message = str(raised.exception)
        self.assertIn("'scale'", message)
        self.assertIn("payload.scale", message)
        self.assertIn("quantize.e4m3.g128", message)

    def test_a_known_transform_becomes_a_kernel_obligation(self):
        physicalized = physicalize_pattern(
            self._transform_pattern(
                "quantize.e4m3.g128.values"
            ).compile(),
            (attachment(),),
        )
        program = physicalized.binding("probe").to_program()
        self.assertEqual(len(program.transforms), 1)
        contract = program.transforms[0]
        self.assertEqual(contract.operation, "payload.scale")
        self.assertEqual(
            contract.transform, "quantize.e4m3.g128"
        )
        self.assertEqual(contract.source_type, "f32")
        self.assertEqual(contract.destinations, ("final",))
        self.assertEqual(
            contract.destination_types, ("f8E4M3FN",)
        )
        rendered = replace(
            program, program_id=0
        ).to_mlir()
        self.assertIn("transforms = [{", rendered)
        self.assertIn('transform = "quantize.e4m3.g128"', rendered)
        plain = physicalize_pattern(
            scatter_pattern(INDEXED_PRODUCER),
            (attachment(),),
        ).binding("probe").to_program()
        self.assertNotIn(
            "transforms",
            replace(plain, program_id=0).to_mlir(),
        )


@unittest.skipIf(
    LIBRARY is None,
    "no aboleth_rdma_<arch>.bc in the build; configure with CUDA",
)
class ArtifactRealizationTest(unittest.TestCase):
    def program(self, kind: Artifact):
        return physicalize_pattern(
            scatter_pattern(INDEXED_PRODUCER),
            (attachment(artifact=kind),),
        ).program("probe")

    def test_bindable_epilogue_realizes_a_fragment_artifact(self):
        program = self.program(Artifact.BINDABLE_EPILOGUE)
        with tempfile.TemporaryDirectory() as directory:
            realized = realize_bindable_epilogue(
                program,
                arch=ARCH,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
                completion_interval=4,
            )

            self.assertEqual(
                realized.symbol,
                "aboleth_fragment_probe",
            )
            for path in realized.files:
                self.assertTrue(path.is_file(), path)
            self.assertIn(
                realized.binding_signature,
                realized.shim_header.read_text(),
            )

    def test_standalone_kernel_returns_launch_metadata(self):
        program = self.program(Artifact.STANDALONE_KERNEL)
        with tempfile.TemporaryDirectory() as directory:
            realized = realize_standalone_kernel(
                program,
                arch=ARCH,
                device_bitcode=LIBRARY.bitcode,
                output_dir=directory,
                completion_interval=4,
            )

            self.assertEqual(
                realized.symbol,
                "aboleth_kernel_probe",
            )
            for path in realized.files:
                self.assertTrue(path.is_file(), path)
            self.assertEqual(realized.num_warps, 1)
            self.assertGreaterEqual(realized.shared_bytes, 0)
            # Every manifest slot, in order, with the i1 predicate widened.
            manifest = physicalize_pattern(
                scatter_pattern(INDEXED_PRODUCER),
                (attachment(
                    artifact=Artifact.STANDALONE_KERNEL,
                ),),
            ).manifest("probe")
            self.assertEqual(
                tuple(
                    name for name, _ in realized.arguments
                ),
                manifest.slot_names,
            )
            self.assertNotIn(
                "i1",
                tuple(
                    kind for _, kind in realized.arguments
                ),
            )
            self.assertGreater(
                realized.scratch_parameters,
                0,
            )
            self.assertIn(
                f".target {ARCH}",
                realized.ptx.read_text(),
            )

    def test_realization_rejects_a_mismatched_kind(self):
        with self.assertRaises(ValueError) as raised:
            realize_standalone_kernel(
                self.program(Artifact.BINDABLE_EPILOGUE),
                arch=ARCH,
                device_bitcode=LIBRARY.bitcode,
                output_dir=tempfile.gettempdir(),
                completion_interval=4,
            )
        message = str(raised.exception)
        self.assertIn("'bindable_epilogue'", message)
        self.assertIn("Artifact.STANDALONE_KERNEL", message)


if __name__ == "__main__":
    unittest.main()
