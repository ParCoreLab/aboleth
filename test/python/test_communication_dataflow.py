import unittest

from aboleth.planning import CommunicationDataflow


def add_region(
    flow,
    name,
    *,
    visibility,
    immutable=False,
    escapes=False,
    strides=(16, 1),
):
    return flow.region(
        name,
        element_type="bf16",
        shape=(8, 16),
        strides=strides,
        visibility=visibility,
        immutable=immutable,
        escapes=escapes,
        registered=(
            visibility == "peer" and escapes
        ),
        donated=(
            visibility == "peer" and escapes
        ),
        reuse_interval=1 if escapes else 0,
        pool_depth=1 if escapes else 0,
    )


class CommunicationDataflowTest(unittest.TestCase):
    def test_retargets_transfer_and_preserves_output(self):
        flow = CommunicationDataflow("direct")
        source = add_region(flow, "source", visibility="local")
        staging = add_region(flow, "staging", visibility="peer")
        final = add_region(
            flow, "final", visibility="peer", escapes=True
        )
        received = flow.transfer(
            "send", source, staging, output="received"
        )
        output = flow.copy(
            "finish", received, final, output="output"
        )
        flow.use("consumer", output)

        graph = flow.simplify()
        transfer = graph.operation("send")
        self.assertEqual(transfer.kind, "transfer")
        self.assertEqual(transfer.destination, "final")
        self.assertEqual(transfer.output_id, "output")
        self.assertIsNone(graph.operation("finish"))
        self.assertEqual(
            graph.retargeted_destinations,
            {"send": ("staging", "final")},
        )

    def test_singleton_reduction_becomes_transfer(self):
        flow = CommunicationDataflow("singleton")
        source = add_region(flow, "source", visibility="local")
        staging = add_region(flow, "staging", visibility="peer")
        final = add_region(
            flow, "final", visibility="peer", escapes=True
        )
        reduced = flow.reduction(
            "reduce",
            source,
            staging,
            cardinality=1,
            operator="add",
            output="reduced",
        )
        output = flow.copy(
            "finish", reduced, final, output="output"
        )
        flow.use("consumer", output)

        graph = flow.simplify()
        operation = graph.operation("reduce")
        self.assertEqual(operation.kind, "transfer")
        self.assertEqual(operation.destination, "final")
        self.assertEqual(operation.output_id, "output")
        self.assertFalse(graph.operations_of_kind("reduction"))

    def test_observer_preserves_staged_materialization(self):
        flow = CommunicationDataflow("observed-staging")
        source = add_region(flow, "source", visibility="local")
        staging = add_region(flow, "staging", visibility="peer")
        final = add_region(
            flow, "final", visibility="peer", escapes=True
        )
        received = flow.transfer(
            "send", source, staging, output="received"
        )
        output = flow.copy(
            "finish", received, final, output="output"
        )
        flow.use("host-observer", received)
        flow.use("consumer", output)

        graph = flow.simplify()
        self.assertEqual(
            graph.operation("send").destination,
            "staging",
        )
        self.assertIsNotNone(graph.operation("finish"))

    def test_async_consumer_requires_event_and_pool_depth(self):
        def build(*, event_ordered, pool_depth, max_inflight):
            flow = CommunicationDataflow(
                f"async-{event_ordered}-{pool_depth}-{max_inflight}"
            )
            source = add_region(
                flow, "source", visibility="local"
            )
            staging = add_region(
                flow, "staging", visibility="peer"
            )
            final = flow.region(
                "final",
                element_type="bf16",
                shape=(8, 16),
                strides=(16, 1),
                visibility="peer",
                escapes=True,
                registered=True,
                donated=True,
                reuse_interval=1,
                pool_depth=pool_depth,
            )
            received = flow.transfer(
                "send", source, staging, output="received"
            )
            output = flow.copy(
                "finish", received, final, output="output"
            )
            flow.async_use(
                "consumer",
                output,
                max_inflight=max_inflight,
                event_ordered=event_ordered,
            )
            return flow.simplify()

        direct = build(
            event_ordered=True,
            pool_depth=2,
            max_inflight=2,
        )
        self.assertIsNone(direct.operation("finish"))
        no_event = build(
            event_ordered=False,
            pool_depth=2,
            max_inflight=2,
        )
        self.assertIsNotNone(no_event.operation("finish"))
        too_shallow = build(
            event_ordered=True,
            pool_depth=1,
            max_inflight=2,
        )
        self.assertIsNotNone(too_shallow.operation("finish"))

    def test_opaque_stage_carries_read_and_write_effects(self):
        flow = CommunicationDataflow("opaque-experts")
        source = add_region(flow, "source", visibility="local")
        staging = add_region(flow, "staging", visibility="peer")
        expert_input = flow.region(
            "expert.input",
            element_type="bf16",
            shape=(8, 16),
            strides=(16, 1),
            visibility="peer",
            escapes=True,
            registered=True,
            donated=True,
            reuse_interval=1,
            pool_depth=2,
        )
        expert_output = flow.region(
            "expert.output.storage",
            element_type="bf16",
            shape=(8, 16),
            strides=(16, 1),
            visibility="local",
            escapes=True,
        )
        received = flow.transfer(
            "dispatch", source, staging, output="received"
        )
        routed = flow.copy(
            "dispatch.copy",
            received,
            expert_input,
            output="routed",
        )
        (output,) = flow.opaque_stage(
            "experts",
            reads=(routed,),
            writes=(expert_output,),
            outputs=("expert.output",),
            asynchronous=True,
            max_inflight=2,
            event_ordered=True,
        )
        flow.use("combine", output)

        graph = flow.simplify()
        self.assertIsNone(graph.operation("dispatch.copy"))
        self.assertEqual(
            graph.operation("dispatch").destination,
            "expert.input",
        )
        experts = graph.operation("experts")
        self.assertEqual(experts.inputs, ("routed",))
        self.assertEqual(
            experts.destinations,
            ("expert.output.storage",),
        )
        self.assertEqual(
            experts.output_ids,
            ("expert.output",),
        )
        self.assertEqual(
            graph.use_inputs("combine"),
            ("expert.output",),
        )

    def test_opaque_stage_blocks_unsafe_async_donation(self):
        flow = CommunicationDataflow("opaque-unsafe")
        source = add_region(flow, "source", visibility="local")
        staging = add_region(flow, "staging", visibility="peer")
        final = flow.region(
            "final",
            element_type="bf16",
            shape=(8, 16),
            strides=(16, 1),
            visibility="peer",
            escapes=True,
            registered=True,
            donated=True,
            reuse_interval=1,
            pool_depth=1,
        )
        received = flow.transfer("send", source, staging)
        output = flow.copy("copy", received, final)
        flow.opaque_stage(
            "experts",
            reads=(output,),
            asynchronous=True,
            max_inflight=2,
            event_ordered=True,
        )

        graph = flow.simplify()
        self.assertIsNotNone(graph.operation("copy"))

    def test_dynamic_workset_retargets_transfer_each(self):
        flow = CommunicationDataflow("workset")
        source = add_region(flow, "source", visibility="local")
        staging = add_region(flow, "staging", visibility="peer")
        final = add_region(
            flow, "final", visibility="peer", escapes=True
        )
        fields = {}
        for name in (
            "source_row",
            "destination_peer",
            "destination_row",
        ):
            fields[name] = flow.region(
                f"work.{name}",
                element_type="i32",
                shape=(8,),
                strides=(1,),
                immutable=True,
            )
        count = flow.region(
            "work.count",
            element_type="i32",
            shape=(1,),
            strides=(1,),
            immutable=True,
        )
        workset = flow.workset(
            "work",
            fields=fields,
            bounds={
                "source_row": (0, 8),
                "destination_peer": (0, 2),
                "destination_row": (0, 8),
            },
            count=count,
            max_items=8,
            unique_destinations=True,
        )
        received = flow.transfer_each(
            "send",
            workset,
            source,
            staging,
        )
        output = flow.copy_each(
            "copy",
            workset,
            received,
            final,
            output="output",
        )
        flow.use("consumer", output)

        graph = flow.simplify()
        transfer = graph.operation("send")
        self.assertEqual(
            transfer.kind, "transfer_each"
        )
        self.assertEqual(transfer.workset, "work")
        self.assertEqual(
            transfer.destination, "final"
        )
        self.assertIsNone(graph.operation("copy"))

    def test_nonunique_workset_keeps_staging_copy(self):
        flow = CommunicationDataflow("nonunique-workset")
        source = add_region(flow, "source", visibility="local")
        staging = add_region(flow, "staging", visibility="peer")
        final = add_region(
            flow, "final", visibility="peer", escapes=True
        )
        fields = {
            name: flow.region(
                f"work.{name}",
                element_type="i32",
                shape=(8,),
                strides=(1,),
                immutable=True,
            )
            for name in (
                "source_row",
                "destination_peer",
                "destination_row",
            )
        }
        workset = flow.workset(
            "work",
            fields=fields,
            bounds={
                "source_row": (0, 8),
                "destination_peer": (0, 2),
                "destination_row": (0, 8),
            },
            max_items=8,
            unique_destinations=False,
        )
        received = flow.transfer_each(
            "send", workset, source, staging
        )
        output = flow.copy_each(
            "copy", workset, received, final
        )
        flow.use("consumer", output)

        graph = flow.simplify()
        self.assertIsNotNone(graph.operation("copy"))
        self.assertEqual(
            graph.operation("send").destination,
            "staging",
        )

    def test_partial_workset_does_not_replace_full_copy(self):
        flow = CommunicationDataflow("partial-workset")
        source = flow.region(
            "source",
            element_type="bf16",
            shape=(8, 16),
            strides=(16, 1),
        )
        staging = flow.region(
            "staging",
            element_type="bf16",
            shape=(8, 16),
            strides=(16, 1),
            visibility="peer",
        )
        final = flow.region(
            "final",
            element_type="bf16",
            shape=(8, 16),
            strides=(16, 1),
            visibility="peer",
            escapes=True,
            registered=True,
            donated=True,
            reuse_interval=1,
            pool_depth=1,
        )
        fields = {
            name: flow.region(
                f"work.{name}",
                element_type="i32",
                shape=(4,),
                strides=(1,),
                immutable=True,
            )
            for name in (
                "source_row",
                "destination_peer",
                "destination_row",
            )
        }
        workset = flow.workset(
            "work",
            fields=fields,
            bounds={
                "source_row": (0, 4),
                "destination_peer": (0, 2),
                "destination_row": (0, 8),
            },
            max_items=4,
            unique_destinations=True,
        )
        received = flow.transfer_each(
            "send", workset, source, staging
        )
        output = flow.copy("full.copy", received, final)
        flow.use("consumer", output)

        graph = flow.simplify()
        self.assertIsNotNone(
            graph.operation("full.copy")
        )
        self.assertEqual(
            graph.operation("send").destination,
            "staging",
        )

    def test_residual_reduction_and_partial_copy_remain(self):
        flow = CommunicationDataflow("partial")
        direct_source = add_region(
            flow, "direct.source", visibility="local"
        )
        direct_staging = add_region(
            flow, "direct.staging", visibility="peer"
        )
        direct_final = add_region(
            flow,
            "direct.final",
            visibility="peer",
            escapes=True,
        )
        residual_source = add_region(
            flow, "residual.source", visibility="local"
        )
        residual_staging = add_region(
            flow, "residual.staging", visibility="peer"
        )
        residual_final = add_region(
            flow,
            "residual.final",
            visibility="local",
            escapes=True,
        )
        direct = flow.transfer(
            "direct.transfer",
            direct_source,
            direct_staging,
        )
        direct_output = flow.copy(
            "direct.copy", direct, direct_final
        )
        reduced = flow.reduction(
            "residual.reduction",
            residual_source,
            residual_staging,
            cardinality=2,
            operator="add",
        )
        residual_output = flow.copy(
            "residual.copy", reduced, residual_final
        )
        flow.use("consumer", direct_output, residual_output)

        graph = flow.simplify()
        self.assertIsNone(graph.operation("direct.copy"))
        self.assertEqual(
            graph.operation("direct.transfer").destination,
            "direct.final",
        )
        self.assertEqual(
            graph.operation("residual.reduction").kind,
            "reduction",
        )
        self.assertIsNotNone(graph.operation("residual.copy"))

    def test_aliases_immutable_value_and_removes_dead_write(self):
        flow = CommunicationDataflow("alias-and-dead")
        metadata = add_region(
            flow,
            "metadata",
            visibility="local",
            immutable=True,
        )
        duplicate = add_region(
            flow, "duplicate", visibility="local"
        )
        copied = flow.copy(
            "metadata.copy",
            metadata,
            duplicate,
            output="metadata.output",
        )
        flow.use("metadata.consumer", copied)
        route_source = add_region(
            flow,
            "route.source",
            visibility="local",
            immutable=True,
        )
        route_output = add_region(
            flow, "route.output", visibility="peer"
        )
        flow.transfer(
            "route.transfer",
            route_source,
            route_output,
        )

        graph = flow.simplify()
        self.assertEqual(
            graph.use_inputs("metadata.consumer"),
            ("metadata",),
        )
        self.assertEqual(
            graph.use_provenance["metadata.consumer"],
            (("metadata.output", "metadata"),),
        )
        self.assertIn(
            "route.transfer",
            graph.removed_operation_ids,
        )
        self.assertIn("route.output", graph.removed_region_ids)

    def test_preserves_explicit_view_transform(self):
        flow = CommunicationDataflow("view-transform")
        source = flow.region(
            "source",
            element_type="f32",
            shape=(2, 4),
            strides=(4, 1),
            visibility="local",
        )
        storage = flow.region(
            "storage",
            element_type="f32",
            shape=(4, 2),
            strides=(2, 1),
            visibility="local",
            escapes=True,
        )
        view = flow.view(
            "transposed.view",
            storage,
            shape=(4, 2),
            strides=(1, 4),
        )
        output = flow.transform(
            "transpose",
            source,
            view,
            transform="transpose",
            index_map=(
                "affine_map<(d0, d1) -> (d1, d0)>"
            ),
            output="transposed",
        )
        flow.use("consumer", output)

        graph = flow.simplify()
        operation = graph.operation("transpose")
        self.assertEqual(operation.kind, "transform")
        self.assertEqual(operation.destination, "transposed.view")
        self.assertEqual(
            operation.index_map,
            "affine_map<(d0, d1) -> (d1, d0)>",
        )

    def test_view_copy_is_not_aliased(self):
        flow = CommunicationDataflow("view-alias")
        base = flow.region(
            "base",
            element_type="i32",
            shape=(8,),
            strides=(1,),
            visibility="local",
            immutable=True,
        )
        view = flow.view(
            "view",
            base,
            shape=(8,),
            strides=(1,),
        )
        destination = flow.region(
            "destination",
            element_type="i32",
            shape=(8,),
            strides=(1,),
            visibility="local",
        )
        copied = flow.copy(
            "copy",
            view,
            destination,
            output="copied",
        )
        flow.use("consumer", copied)

        graph = flow.simplify()
        self.assertIsNotNone(graph.operation("copy"))
        self.assertEqual(
            graph.use_inputs("consumer"),
            ("copied",),
        )

    def test_rejects_globally_duplicate_identifiers(self):
        flow = CommunicationDataflow("duplicate")
        source = flow.region(
            "source",
            element_type="i32",
            shape=(8,),
            strides=(1,),
        )
        destination = flow.region(
            "destination",
            element_type="i32",
            shape=(8,),
            strides=(1,),
            visibility="peer",
        )
        with self.assertRaisesRegex(
            ValueError, "duplicate dataflow identifier"
        ):
            flow.transfer(
                "source",
                source,
                destination,
                output="output",
            )

    def test_indexed_donation_requires_explicit_facts(self):
        def build(donated):
            flow = CommunicationDataflow(
                f"indexed-{donated}"
            )
            source = flow.region(
                "source",
                element_type="bf16",
                shape=(8, 16),
                strides=(16, 1),
            )
            staging = flow.region(
                "staging",
                element_type="bf16",
                shape=(8, 16),
                strides=(16, 1),
                visibility="peer",
            )
            storage = flow.region(
                "storage",
                element_type="bf16",
                shape=(16, 16),
                strides=(16, 1),
                visibility="peer",
                escapes=True,
                registered=True,
                donated=donated,
                reuse_interval=1,
                pool_depth=1,
            )
            indices = flow.region(
                "indices",
                element_type="i32",
                shape=(8,),
                strides=(1,),
                immutable=True,
            )
            final = flow.indexed_view(
                "final",
                storage,
                indices,
                shape=(8, 16),
                strides=(16, 1),
                index_lower_bounds=(0,),
                index_upper_bounds=(16,),
                injective=True,
            )
            received = flow.transfer(
                "transfer",
                source,
                staging,
                output="received",
            )
            output = flow.copy(
                "copy",
                received,
                final,
                output="output",
            )
            flow.use("consumer", output)
            return flow.simplify()

        staged = build(False)
        self.assertIsNotNone(staged.operation("copy"))
        direct = build(True)
        self.assertIsNone(direct.operation("copy"))
        self.assertEqual(
            direct.operation("transfer").destination,
            "final",
        )


if __name__ == "__main__":
    unittest.main()
