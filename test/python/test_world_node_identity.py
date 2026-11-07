import unittest
from unittest.mock import Mock, patch

from aboleth.world import TopologyInfo, Transport, World, _local_node_identity


class NodeIdentityTests(unittest.TestCase):
    FIRST = "f7464668-73f2-4f0f-b96a-123b02165046"
    SECOND = "5535ece4-f408-4f95-9d0c-6ba417e88557"

    def test_local_ranks_share_identity(self):
        with (
            patch(
                "aboleth.world.socket.gethostname", return_value="shared-name"
            ),
            patch(
                "aboleth.world.pathlib.Path.read_text",
                return_value=self.FIRST + "\n",
            ),
        ):
            self.assertEqual(_local_node_identity(), _local_node_identity())

    def test_duplicate_hostnames_on_different_kernels_are_distinct(self):
        with (
            patch(
                "aboleth.world.socket.gethostname", return_value="shared-name"
            ),
            patch(
                "aboleth.world.pathlib.Path.read_text",
                side_effect=[self.FIRST, self.SECOND],
            ),
        ):
            first, second = _local_node_identity(), _local_node_identity()
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("shared-name:"))

    def test_identity_roundtrip_does_not_turn_remote_peer_into_local(self):
        with (
            patch(
                "aboleth.world.socket.gethostname", return_value="shared-name"
            ),
            patch(
                "aboleth.world.pathlib.Path.read_text",
                side_effect=[self.FIRST, self.SECOND],
            ),
        ):
            local = TopologyInfo(
                rank=0, device=0, hostname=_local_node_identity()
            )
            peer = TopologyInfo(
                rank=8, device=0, hostname=_local_node_identity()
            )
        recovered = TopologyInfo.from_bytes(peer.to_bytes())
        self.assertEqual(recovered, peer)
        self.assertNotEqual(local.hostname, recovered.hostname)

    def test_unreadable_boot_id_keeps_portable_fallback(self):
        with (
            patch(
                "aboleth.world.socket.gethostname",
                return_value="portable-host",
            ),
            patch(
                "aboleth.world.pathlib.Path.read_text",
                side_effect=FileNotFoundError,
            ),
        ):
            self.assertEqual(_local_node_identity(), "portable-host")

    def test_invalid_boot_id_keeps_portable_fallback(self):
        with (
            patch(
                "aboleth.world.socket.gethostname",
                return_value="portable-host",
            ),
            patch(
                "aboleth.world.pathlib.Path.read_text",
                return_value="not-a-uuid",
            ),
        ):
            self.assertEqual(_local_node_identity(), "portable-host")

    def _connection_probe(self, peer_boot_id):
        world = object.__new__(World)
        world.rank, world.world_size, world.device = 0, 2, 0
        world._closed = False
        world._endpoints = {}
        world.runtime = Mock()
        world.runtime.create_local_endpoint.side_effect = RuntimeError(
            "endpoint creation reached"
        )
        world.bootstrap = Mock()

        def gather(tag, payload):
            self.assertEqual(tag, World.TOPOLOGY_TAG)
            local = TopologyInfo.from_bytes(payload)
            self.assertEqual(local.hostname, "shared-name:" + self.FIRST)
            peer = TopologyInfo(
                rank=1, device=1, hostname="shared-name:" + peer_boot_id
            )
            return {0: payload, 1: peer.to_bytes()}

        world.bootstrap.all_gather.side_effect = gather
        return world

    def test_auto_uses_rdma_for_duplicate_hostname_on_remote_kernel(self):
        world = self._connection_probe(self.SECOND)
        with (
            patch(
                "aboleth.world.socket.gethostname", return_value="shared-name"
            ),
            patch(
                "aboleth.world.pathlib.Path.read_text", return_value=self.FIRST
            ),
            patch("aboleth.world.torch.cuda.can_device_access_peer") as probe,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "endpoint creation reached"
            ):
                world.connect_all("auto", payload_bytes=8192)
        world.runtime.create_local_endpoint.assert_called_once_with(
            transport=Transport.RDMA, qp_depth=0
        )
        probe.assert_not_called()

    def test_forced_p2p_rejects_duplicate_hostname_on_remote_kernel(self):
        world = self._connection_probe(self.SECOND)
        with (
            patch(
                "aboleth.world.socket.gethostname", return_value="shared-name"
            ),
            patch(
                "aboleth.world.pathlib.Path.read_text", return_value=self.FIRST
            ),
            patch("aboleth.world.torch.cuda.can_device_access_peer") as probe,
        ):
            with self.assertRaisesRegex(ValueError, "not on the local node"):
                world.connect_all("p2p")
        world.runtime.create_local_endpoint.assert_not_called()
        probe.assert_not_called()

    def test_auto_preserves_local_p2p_with_shared_boot_identity(self):
        world = self._connection_probe(self.FIRST)
        with (
            patch(
                "aboleth.world.socket.gethostname", return_value="shared-name"
            ),
            patch(
                "aboleth.world.pathlib.Path.read_text", return_value=self.FIRST
            ),
            patch(
                "aboleth.world.torch.cuda.can_device_access_peer",
                return_value=True,
            ) as probe,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "endpoint creation reached"
            ):
                world.connect_all("auto", payload_bytes=8192)
        world.runtime.create_local_endpoint.assert_called_once_with(
            transport=Transport.P2P, qp_depth=0
        )
        self.assertEqual(probe.call_count, 2)


if __name__ == "__main__":
    unittest.main()
