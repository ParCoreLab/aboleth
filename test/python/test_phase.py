import unittest
from dataclasses import replace

from aboleth import (
    Artifact,
    OpaqueCapabilities,
    Pattern,
    PhaseBinding,
    PhaseResources,
    RegionEffect,
    compile_phase,
)


def communication_pattern():
    pattern = Pattern("single-phase")
    source = pattern.region(
        "source",
        element_type="f32",
        shape=(8, 4),
        strides=(4, 1),
        immutable=True,
        registered=True,
    )
    receive = pattern.region(
        "receive",
        element_type="f32",
        shape=(8, 4),
        strides=(4, 1),
        visibility="peer",
        registered=True,
    )
    moved = pattern.transfer("payload", source, receive)
    pattern.opaque_compute(
        "consumer",
        reads=(moved,),
        capabilities=OpaqueCapabilities(),
    )
    return pattern


def phase_binding(**overrides):
    values = {
        "producer": None,
        "consumer": "consumer",
        "name": "send",
        "kernel": "send_kernel",
        "insertion_point": "payload",
        "peer_segments": 2,
        "publication": "ready",
    }
    values.update(overrides)
    return PhaseBinding(**values)


def resources(pattern):
    compiled = pattern.compile()
    return PhaseResources(
        regions={
            region.id: RegionEffect(
                allocation=index,
                generation=1,
                offset=0,
                length=64,
                stride=64,
                count=4,
                period=256,
            )
            for index, region in enumerate(
                compiled.analysis.regions
            )
        },
        peer_transports=("rdma",),
        peer_capacities=(4096,),
        num_peers=2,
        iteration_count=4,
    )


class PhaseApiTest(unittest.TestCase):
    def test_compiles_one_phase_and_explains_the_result(self):
        pattern = communication_pattern()
        phase = compile_phase(
            pattern,
            phase_binding(),
            resources(pattern),
            cyclic=True,
        )

        self.assertEqual(phase.name, "send")
        self.assertEqual(phase.program.operations, ("payload",))
        self.assertEqual(phase.program.command_count, 2)
        self.assertEqual(phase.program.payload_bytes, 72)
        self.assertEqual(phase.completion_interval, 4)
        self.assertEqual(phase.manifest.program_id, 0)
        self.assertEqual(
            phase.describe(),
            {
                "name": "send",
                "artifact": "inline_fragment",
                "kernel": "send_kernel",
                "insertion_point": "payload",
                "command_count": 2,
                "payload_bytes": 72,
                "completion_interval": 4,
                "transport_geography": [],
                "strategy": "inline_transport",
                "scheduled_kernel": "send_kernel",
                "publication_kernel": None,
                "publication_strategy": "inline",
                "recycled_buffers": [],
                "advanced_values": [],
                "grid_axes": ["peers"],
                "block_elements": 0,
                "num_warps": 16,
                "buffer_depth": 1,
                "rationale": (
                    "inline transport is required by topology, "
                    "legality, or payload size"
                ),
                "slots": list(phase.manifest.slot_names),
            },
        )

    def test_full_geography_keys_higher_tier_compilation(self):
        pattern = communication_pattern()
        base = resources(pattern)
        first = compile_phase(
            pattern,
            phase_binding(),
            replace(
                base,
                peer_transports=("rdma", "p2p"),
                peer_capacities=(4096, 2**31 - 1),
                transport_geography=((0, "p2p"), (1, "rdma")),
            ),
        )
        second = compile_phase(
            pattern,
            phase_binding(),
            replace(
                base,
                peer_transports=("rdma", "p2p"),
                peer_capacities=(4096, 2**31 - 1),
                transport_geography=((0, "rdma"), (1, "p2p")),
            ),
        )
        self.assertNotEqual(
            first.plan.transport_mlir,
            second.plan.transport_mlir,
        )
        self.assertEqual(
            first.describe()["transport_geography"],
            [
                {"peer": 0, "transport": "p2p"},
                {"peer": 1, "transport": "rdma"},
            ],
        )
        with self.assertRaisesRegex(ValueError, "backend-homogeneous"):
            first.compilation("rdma")
        first.compilation("rdma", peers=(1,))
        with self.assertRaisesRegex(ValueError, "contains 0:p2p"):
            first.compilation("rdma", peers=(0,))

    def test_inline_phase_uses_compilation_not_detached_realization(self):
        pattern = communication_pattern()
        phase = compile_phase(
            pattern,
            phase_binding(),
            resources(pattern),
        )

        with self.assertRaisesRegex(
            ValueError,
            "compile its client kernel",
        ):
            phase.realize(
                arch="sm_100",
                device_bitcode="unused.bc",
                output_dir="unused",
            )

    def test_detached_phase_does_not_open_inline_compilation(self):
        pattern = communication_pattern()
        phase = compile_phase(
            pattern,
            phase_binding(artifact=Artifact.STANDALONE_KERNEL),
            resources(pattern),
        )

        with self.assertRaisesRegex(
            ValueError,
            "only Artifact.INLINE_FRAGMENT",
        ):
            phase.compilation("rdma")

        with self.assertRaisesRegex(
            ValueError,
            "apply only to bindable epilogues",
        ):
            phase.realize(
                arch="sm_100",
                device_bitcode="unused.bc",
                output_dir="unused",
                cuda_inline=True,
            )

if __name__ == "__main__":
    unittest.main()
