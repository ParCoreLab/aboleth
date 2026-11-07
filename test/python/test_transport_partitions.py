"""The compiler owns homogeneous launches and exact peer-set cache identity."""

from dataclasses import replace
import unittest

from aboleth.planning import plan_transport_partitions, TransportPartitionPlan
from aboleth import compile_phase
from test_phase import communication_pattern, phase_binding, resources


class TransportPartitionsTest(unittest.TestCase):
    def test_mixed_peers_appear_exactly_once(self):
        geography = ((0, "rdma"), (2, "p2p"), (3, "rdma"), (5, "p2p"))
        self.assertEqual(
            plan_transport_partitions(geography),
            (
                TransportPartitionPlan("p2p", (2, 5), 2),
                TransportPartitionPlan("rdma", (0, 3), 2),
            ),
        )

    def test_uniform_and_empty_geographies(self):
        self.assertEqual(plan_transport_partitions(()), ())
        self.assertEqual(
            plan_transport_partitions(((1, "rdma"), (3, "rdma"))),
            (TransportPartitionPlan("rdma", (1, 3), 2),),
        )

    def test_invalid_registration_facts_are_rejected(self):
        for geography in [
            ((1, "p2p"), (1, "rdma")),
            ((3, "rdma"), (1, "p2p")),
            ((0, "gin"),),
            ((-1, "rdma"),),
            ((2**31, "rdma"),),
            ((True, "rdma"),),
        ]:
            with (
                self.subTest(geography=geography),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                plan_transport_partitions(geography)

    def test_phase_exposes_its_launches_and_rejects_a_foreign_partition(self):
        pattern = communication_pattern()
        phase = compile_phase(
            pattern,
            phase_binding(),
            replace(
                resources(pattern),
                peer_transports=("p2p", "rdma"),
                peer_capacities=(2**31 - 1, 4096),
                transport_geography=((0, "p2p"), (1, "rdma")),
            ),
        )
        self.assertEqual(
            phase.launches,
            (
                TransportPartitionPlan("p2p", (0,), 1),
                TransportPartitionPlan("rdma", (1,), 1),
            ),
        )
        for launch in phase.launches:
            phase.compilation_launch(launch)
        with self.assertRaisesRegex(ValueError, "not a partition"):
            phase.compilation_launch(TransportPartitionPlan("rdma", (0, 1), 2))
        self.assertIn("launchPartitions", phase.plan.transport_mlir)
        self.assertEqual(len(phase.plan.to_dict()["transport_partitions"]), 2)


if __name__ == "__main__":
    unittest.main()
