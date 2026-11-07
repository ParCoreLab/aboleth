"""Physical storage follows all declared accesses, including indexed reads."""

import os
import unittest
from unittest.mock import patch

from aboleth.planning import (
    CommunicationDataflow,
    _communication_dataflow_mlir,
    plan_storage_mlir,
)


def indexed_consumer(*, escaped=False):
    flow = CommunicationDataflow("bounded-reader")
    payload = flow.region(
        "payload",
        element_type="bf16",
        shape=(105, 11),
        strides=(11, 1),
        visibility="peer",
        registered=True,
        escapes=escaped,
        pool_depth=2,
        reuse_interval=1,
    )
    indices = flow.region(
        "indices",
        element_type="i32",
        shape=(105,),
        strides=(1,),
        immutable=True,
    )
    indexed = flow.indexed_view(
        "indexed",
        payload,
        indices,
        shape=(105, 11),
        strides=(11, 1),
        index_lower_bounds=(0,),
        index_upper_bounds=(15,),
        injective=False,
    )
    return flow, payload, indices, indexed


def plan(flow, enabled=True):
    with patch.dict(
        os.environ,
        {"ABOLETH_PASSES": "+plan-storage" if enabled else "-plan-storage"},
    ):
        return plan_storage_mlir(_communication_dataflow_mlir(flow))


class StoragePlanTest(unittest.TestCase):
    def test_indexed_payload_bound_is_derived_and_switchable(self):
        flow, _, _, indexed = indexed_consumer()
        flow.use("consumer", indexed)
        compact, generic = plan(flow), plan(flow, False)
        self.assertEqual(compact["payload"].logical_shape, (105, 11))
        self.assertEqual(compact["payload"].physical_shape, (15, 11))
        self.assertEqual(compact["payload"].byte_count, 15 * 11 * 2)
        self.assertEqual(generic["payload"].physical_shape, (105, 11))
        self.assertEqual(generic["payload"].byte_count, 105 * 11 * 2)
        self.assertEqual(compact["payload"].last_reader, "consumer")
        self.assertEqual(compact["indices"].physical_shape, (105,))

    def test_implicit_gather_reads_keep_the_entire_index_table(self):
        flow, _, indices, indexed = indexed_consumer()
        flow.workset(
            "short-metadata-use",
            fields={"row": indices},
            bounds={"row": (0, 15)},
            max_items=15,
        )
        flow.use("later-gather", indexed)
        result = plan(flow)
        self.assertEqual(result["payload"].physical_shape[0], 15)
        self.assertEqual(result["indices"].physical_shape, (105,))
        self.assertEqual(result["indices"].last_reader, "later-gather")

    def test_any_full_region_reader_restores_the_full_extent(self):
        flow, payload, _, indexed = indexed_consumer()
        flow.use("indexed-consumer", indexed)
        flow.use("unindexed-consumer", payload)
        self.assertEqual(plan(flow)["payload"].physical_shape, (105, 11))

    def test_escaping_storage_keeps_the_external_contract(self):
        flow, _, _, indexed = indexed_consumer(escaped=True)
        flow.use("consumer", indexed)
        self.assertEqual(plan(flow)["payload"].physical_shape, (105, 11))

    def test_async_reader_lifetime_is_not_invented(self):
        for ordered, inflight, expected in [
            (True, 2, True),
            (True, 3, False),
            (False, 2, False),
        ]:
            with self.subTest(ordered=ordered, inflight=inflight):
                flow, _, _, indexed = indexed_consumer()
                flow.async_use(
                    "async-consumer",
                    indexed,
                    max_inflight=inflight,
                    event_ordered=ordered,
                )
                result = plan(flow)
                self.assertEqual(
                    result["payload"].reader_lifetime_known, expected
                )
                self.assertEqual(
                    result["indices"].last_reader, "async-consumer"
                )
                self.assertEqual(result["indices"].max_inflight, inflight)

    def test_workset_read_and_write_bounds_both_reach_storage(self):
        flow = CommunicationDataflow("bounded-transfer")
        source = flow.region(
            "source", element_type="bf16", shape=(100, 13), strides=(13, 1)
        )
        destination = flow.region(
            "destination",
            element_type="bf16",
            shape=(100, 13),
            strides=(13, 1),
            visibility="peer",
            registered=True,
        )
        fields = {
            name: flow.region(
                name,
                element_type="i32",
                shape=(7,),
                strides=(1,),
                immutable=True,
            )
            for name in ["source_row", "destination_row", "destination_peer"]
        }
        work = flow.workset(
            "work",
            fields=fields,
            bounds={
                "source_row": (0, 9),
                "destination_row": (0, 17),
                "destination_peer": (0, 3),
            },
            max_items=7,
            unique_destinations=True,
        )
        flow.transfer_each("move", work, source, destination)
        result = plan(flow)
        self.assertEqual(result["source"].physical_shape, (9, 13))
        self.assertEqual(result["destination"].physical_shape, (17, 13))
        self.assertEqual(result["source_row"].last_reader, "move")
        self.assertFalse(result["source"].reader_lifetime_known)

    def test_nonidentity_views_retain_generic_allocation(self):
        flow = CommunicationDataflow("permuted-view")
        payload = flow.region(
            "payload", element_type="bf16", shape=(100, 13), strides=(13, 1)
        )
        view = flow.view(
            "transposed",
            payload,
            shape=(13, 100),
            strides=(1, 13),
            index_map="affine_map<(d0, d1) -> (d1, d0)>",
        )
        flow.use("consumer", view)
        self.assertEqual(plan(flow)["payload"].physical_shape, (100, 13))

    def test_bounded_writer_and_indexed_reader_share_the_updated_ssa_state(
        self,
    ):
        flow = CommunicationDataflow("bounded-producer-consumer")
        source = flow.region(
            "source", element_type="bf16", shape=(15, 11), strides=(11, 1)
        )
        destination = flow.region(
            "payload", element_type="bf16", shape=(105, 11), strides=(11, 1)
        )
        rows = flow.region(
            "rows",
            element_type="i32",
            shape=(15,),
            strides=(1,),
            immutable=True,
        )
        work = flow.workset(
            "copy-rows",
            fields={"source_row": rows, "destination_row": rows},
            bounds={"source_row": (0, 15), "destination_row": (0, 15)},
            max_items=15,
            unique_destinations=True,
        )
        updated = flow.copy_each("populate", work, source, destination)
        indices = flow.region(
            "indices",
            element_type="i32",
            shape=(105,),
            strides=(1,),
            immutable=True,
        )
        gathered = flow.indexed_view(
            "gathered",
            updated,
            indices,
            shape=(105, 11),
            strides=(11, 1),
            index_lower_bounds=(0,),
            index_upper_bounds=(15,),
            injective=False,
        )
        flow.use("consumer", gathered)
        result = plan(flow)
        self.assertEqual(result["payload"].physical_shape, (15, 11))
        self.assertEqual(result["payload"].last_reader, "consumer")
        self.assertEqual(result["indices"].physical_shape, (105,))
        self.assertEqual(result["rows"].last_reader, "populate")


if __name__ == "__main__":
    unittest.main()
