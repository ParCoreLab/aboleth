"""The C++ planner owns fanout; records, topology and profiles are facts."""

from dataclasses import asdict, replace
import os
import unittest
from unittest.mock import patch
from aboleth.planning import (
    EndpointPartitionFacts,
    TransportPartitionPlan,
    plan_endpoint_work,
    plan_transport_partitions,
)
from aboleth.endpoint_profiles import B200_NDR400_HOST as PROFILE
from aboleth.endpoint_profiles import B200_NV18, registered_peer_link


def plan(
    size=2097152,
    records=512,
    *,
    enabled=True,
    available=8,
    platform=None,
    memory="host",
    profiles=(PROFILE,),
):
    (partition,) = plan_transport_partitions(((1, "rdma"), (3, "rdma")))
    facts = EndpointPartitionFacts(
        partition,
        memory,
        PROFILE.platform if platform is None else platform,
        available,
    )
    with patch.dict(
        os.environ,
        {
            "ABOLETH_PASSES": (
                "+endpoint-fanout" if enabled else "-endpoint-fanout"
            )
        },
    ):
        return plan_endpoint_work(
            name="unordered-records",
            payload_bytes=size,
            max_items=records,
            partitions=(facts,),
            profiles=profiles,
        )[0]


class EndpointWorkTest(unittest.TestCase):
    def test_switch_changes_only_the_compiler_plan(self):
        on, off = plan(), plan(enabled=False)
        self.assertEqual((on.lanes, off.lanes), (4, 1))
        self.assertEqual(
            on.slices, ((0, 128), (128, 256), (256, 384), (384, 512))
        )
        self.assertEqual(on.sender_grid, (2, 4))
        self.assertEqual(on.receiver_grid, (2, 4))
        self.assertEqual((on.signal_slots, on.count_lane), (8, 0))
        self.assertEqual(on.profile_identity, PROFILE.identity)
        self.assertEqual(off.profile_identity, "")
        self.assertNotEqual(asdict(on), asdict(off))

    def test_record_bound_not_a_decode_heuristic(self):
        self.assertEqual(plan(2**34, 1).lanes, 1)
        self.assertEqual(plan(2**34, 3).lanes, 2)
        self.assertEqual(plan(2**34, 7).lanes, 4)
        self.assertEqual(plan(2**34, available=3).lanes, 2)

    def test_global_rule_and_boundary_sizes(self):
        for size, lanes in [
            (0, 1),
            (1, 1),
            (262144, 1),
            (262145, 2),
            (524288, 2),
            (524289, 4),
            (2**63 - 1, 4),
        ]:
            with self.subTest(size=size):
                self.assertEqual(plan(size).lanes, lanes)

    def test_unknown_or_mismatched_profile_is_conservative(self):
        self.assertEqual(plan(platform="unknown").lanes, 1)
        self.assertEqual(plan(memory="gpu").lanes, 1)
        self.assertEqual(plan(profiles=()).lanes, 1)
        self.assertEqual(
            plan(profiles=(replace(PROFILE, backend="p2p"),)).lanes, 1
        )

    def test_non_power_of_two_record_domains_are_tiled_exactly(self):
        for records in [2, 3, 7, 11, 513, 2**63 - 1]:
            with self.subTest(records=records):
                result = plan(2**63 - 1, records)
                self.assertEqual(result.slices[0][0], 0)
                self.assertEqual(result.slices[-1][1], records)
                self.assertTrue(all(a < b for a, b in result.slices))
                self.assertEqual(
                    [b for _, b in result.slices[:-1]],
                    [a for a, _ in result.slices[1:]],
                )

    def test_profile_is_part_of_artifact_identity(self):
        self.assertNotEqual(
            plan(), plan(profiles=(replace(PROFILE, identity="fresh-study"),))
        )

    def test_mixed_transports_share_rank_indexed_readiness(self):
        partitions = plan_transport_partitions(((0, "p2p"), (1, "rdma")))
        image = ((PROFILE.platform, "host", 4),) * 3
        facts = tuple(
            EndpointPartitionFacts(
                p,
                "none" if p.backend == "p2p" else "host",
                PROFILE.platform,
                4,
                image,
            )
            for p in partitions
        )
        with patch.dict(os.environ, {"ABOLETH_PASSES": "+endpoint-fanout"}):
            launches = plan_endpoint_work(
                name="mixed",
                payload_bytes=2097152,
                max_items=512,
                partitions=facts,
                profiles=(PROFILE,),
                rank_count=3,
            )
        self.assertEqual([p.lanes for p in launches], [1, 4])
        self.assertEqual([p.ready_stride for p in launches], [4, 4])
        self.assertEqual([p.signal_slots for p in launches], [12, 12])
        self.assertEqual({0 * 4 + lane for lane in range(4)} & {1 * 4}, set())
        with patch.dict(os.environ, {"ABOLETH_PASSES": "-endpoint-fanout"}):
            off = plan_endpoint_work(
                name="mixed",
                payload_bytes=2097152,
                max_items=512,
                partitions=facts,
                profiles=(PROFILE,),
                rank_count=3,
            )
        self.assertEqual(
            [(p.lanes, p.ready_stride, p.signal_slots) for p in off],
            [(1, 1, 3), (1, 1, 3)],
        )

    def test_one_world_image_prevents_mismatched_peer_fanout(self):
        partition = TransportPartitionPlan("rdma", (1,), 1)
        for memory, platform in [
            ("gpu", PROFILE.platform),
            ("host", "unknown"),
        ]:
            image = ((PROFILE.platform, "host", 4), (platform, memory, 4))
            facts = EndpointPartitionFacts(
                partition, "host", PROFILE.platform, 4, image
            )
            (launch,) = plan_endpoint_work(
                name="heterogeneous",
                payload_bytes=2097152,
                max_items=512,
                partitions=(facts,),
                profiles=(PROFILE,),
                rank_count=2,
            )
            self.assertEqual((launch.lanes, launch.ready_stride), (1, 4))
        image = ((PROFILE.platform, "host", 4), (PROFILE.platform, "host", 2))
        (launch,) = plan_endpoint_work(
            name="resource-bound",
            payload_bytes=2097152,
            max_items=512,
            partitions=(
                EndpointPartitionFacts(
                    partition, "host", PROFILE.platform, 4, image
                ),
            ),
            profiles=(PROFILE,),
            rank_count=2,
        )
        self.assertEqual((launch.lanes, launch.ready_stride), (2, 2))
        with self.assertRaises(RuntimeError):
            plan_endpoint_work(
                name="missing-image",
                payload_bytes=16,
                max_items=1,
                partitions=(
                    EndpointPartitionFacts(
                        partition, "host", PROFILE.platform, 4
                    ),
                ),
                rank_count=2,
            )
        with self.assertRaises(RuntimeError):
            plan_endpoint_work(
                name="insufficient",
                payload_bytes=2097152,
                max_items=512,
                partitions=(
                    EndpointPartitionFacts(
                        partition, "host", PROFILE.platform, 1, image
                    ),
                ),
                profiles=(PROFILE,),
                rank_count=2,
            )

    def test_bad_facts_are_rejected(self):
        for changes in [
            dict(size=-1),
            dict(records=0),
            dict(available=0),
            dict(available=2**31),
            dict(size=True),
            dict(records=2**63),
            dict(profiles=(replace(PROFILE, bytes_per_endpoint=0),)),
            dict(profiles=(PROFILE, PROFILE)),
        ]:
            with (
                self.subTest(changes=changes),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                plan(**changes)
        bad = EndpointPartitionFacts(
            TransportPartitionPlan("rdma", (1, 1), 2), "host"
        )
        with self.assertRaises(RuntimeError):
            plan_endpoint_work(
                name="bad", payload_bytes=16, max_items=1, partitions=(bad,)
            )
        with self.assertRaises(ValueError):
            bad = replace(
                bad, partition=TransportPartitionPlan("rdma", (1,), 2)
            )
            plan_endpoint_work(
                name="bad", payload_bytes=16, max_items=1, partitions=(bad,)
            )

    def test_peer_link_and_queue_facts_in_a_mixed_world(self):
        parts = plan_transport_partitions(((0, "p2p"), (1, "rdma")))
        image = ((PROFILE.platform, "host", 16),) * 3

        def compile(
            links=("NV18",) * 3, memory="host", grouped=True, switches=""
        ):
            participants = tuple(
                (platform, memory, lanes) for platform, _, lanes in image
            )
            facts = tuple(
                EndpointPartitionFacts(
                    p,
                    "none" if p.backend == "p2p" else memory,
                    PROFILE.platform,
                    16,
                    participants,
                    "NV18" if p.backend == "p2p" else "",
                    links,
                )
                for p in parts
            )
            with patch.dict(os.environ, {"ABOLETH_PASSES": switches}):
                return plan_endpoint_work(
                    name="mixed-links",
                    payload_bytes=2**24,
                    max_items=4096,
                    partitions=facts,
                    profiles=(PROFILE, replace(B200_NV18, sender_warps=8)),
                    rank_count=3,
                    grouped_receiver_supported=grouped,
                )

        on = compile()
        self.assertEqual([p.lanes for p in on], [16, 4])
        self.assertEqual([p.receiver_grid for p in on], [(1, 1), (1, 1)])
        self.assertEqual([p.receiver_lanes_per_program for p in on], [16, 4])
        self.assertEqual([p.ready_stride for p in on], [16, 16])
        self.assertEqual([p.sender_warps for p in on], [8, 1])
        self.assertEqual([p.receiver_warps for p in on], [1, 1])
        for links in [("NV18", "NV4", "NV18"), ("NV18", "", "NV18"), ()]:
            self.assertEqual([p.lanes for p in compile(links=links)], [1, 4])
        # P2P has no queue; this must not qualify the host-queue RDMA profile.
        self.assertEqual([p.lanes for p in compile(memory="gpu")], [16, 1])
        for results in [
            compile(grouped=False),
            compile(switches="-coalesce-receives"),
        ]:
            self.assertEqual(
                [p.receiver_grid for p in results], [(1, 16), (1, 4)]
            )
            self.assertEqual(
                [p.receiver_lanes_per_program for p in results], [1, 1]
            )
        self.assertEqual(
            [p.lanes for p in compile(switches="-endpoint-fanout")], [1, 1]
        )
        self.assertEqual(
            [p.sender_warps for p in compile(switches="-endpoint-fanout")],
            [1, 1],
        )
        with self.assertRaises(RuntimeError):
            compile(links=("NV18",))

    def test_invalid_geometry_facts_are_rejected(self):
        for warps in [0, 3, 16, -1]:
            with self.subTest(warps=warps), self.assertRaises(RuntimeError):
                plan(profiles=(replace(PROFILE, sender_warps=warps),))
        with self.assertRaises(ValueError):
            plan(profiles=(replace(PROFILE, sender_warps=True),))

    def test_uuid_mapping_ignores_masked_cuda_ordinals(self):
        uuids = [f"0000000{i}-0000-0000-0000-000000000000" for i in range(3)]
        report = "\n".join(
            f"{i}, GPU-{value}" for i, value in enumerate(uuids)
        )
        topology = "  GPU0 GPU1 GPU2 NIC0 CPU Affinity\nGPU0 X NV18 NV4 PIX 0-31\nGPU1 NV18 X NV18 PIX 0-31\nGPU2 NV4 NV18 X PIX 0-31"

        def link(local, peers, **reports):
            return registered_peer_link(
                local,
                peers,
                uuid_report=report,
                topology_report=reports.get("topology", topology),
            )

        self.assertEqual(link("GPU-" + uuids[2], [uuids[1]]), "NV18")
        self.assertEqual(
            link(
                uuids[2],
                [uuids[1]],
                topology="\x1b[4m" + topology.replace("\n", "\x1b[0m\n", 1),
            ),
            "NV18",
        )
        self.assertEqual(link(uuids[2], [uuids[0], uuids[1]]), "")
        self.assertEqual(link(uuids[1], [uuids[0], uuids[2]]), "NV18")
        for local, peers in [
            ("missing", [uuids[1]]),
            (uuids[0], []),
            (uuids[0], ["missing"]),
            (uuids[0], [uuids[0]]),
        ]:
            self.assertEqual(link(local, peers), "")
        self.assertEqual(link(uuids[0], [uuids[1]], topology="broken"), "")


if __name__ == "__main__":
    unittest.main()
