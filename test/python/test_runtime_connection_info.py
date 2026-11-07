import unittest

from aboleth import _runtime
from aboleth.runtime import (
    LocalEndpoint,
    Transport,
    connection_info_from_bytes,
    connection_transport,
)


# What every rank reads and writes: the runtime's own AbolethConnectionInfo,
# 128 bytes with no format mirrored on the Python side.
FIELDS = {
    "abi_version": 2,
    "transport": _runtime.TRANSPORT_PROXY,
    "qpn": 0x123456,
    "psn": 0x654321,
    "gid_index": 3,
    "path_mtu": 4096,
    "lid": 0x4242,
    "port_num": 1,
    "link_layer": 2,
    "gid": bytes(range(16)),
    "provider": 2,
    "address_format": 12,
    "address": bytes(range(32)),
}


def connection_info(**overrides):
    return _runtime.NativeConnectionInfo(
        **{**FIELDS, **overrides}
    )


class ConnectionInfoWireTest(unittest.TestCase):
    def fields(self, info):
        return {
            name: getattr(info, name) for name in FIELDS
        }

    def test_wire_form_is_the_struct(self):
        encoded = connection_info().to_bytes()
        self.assertEqual(
            len(encoded),
            _runtime.CONNECTION_INFO_BYTES,
        )
        self.assertEqual(len(encoded), 128)

    def test_round_trip_keeps_every_field(self):
        info = connection_info()
        decoded = connection_info_from_bytes(info.to_bytes())
        self.assertEqual(self.fields(decoded), FIELDS)
        self.assertEqual(decoded.to_bytes(), info.to_bytes())

    def test_efa_transport_round_trip(self):
        # Transport.EFA (native code 3) reuses the FI_ADDR_EFA address
        # exchange the EFA proxy uses (provider 2).
        info = connection_info(
            transport=_runtime.TRANSPORT_EFA
        )
        decoded = connection_info_from_bytes(info.to_bytes())
        self.assertEqual(
            decoded.address,
            FIELDS["address"],
        )
        self.assertEqual(
            connection_transport(decoded),
            Transport.EFA,
        )

    def test_transport_names_the_python_transport(self):
        self.assertEqual(
            connection_transport(connection_info()),
            Transport.PROXY,
        )
        self.assertEqual(
            connection_transport(
                connection_info(
                    transport=_runtime.TRANSPORT_DIRECT
                )
            ),
            Transport.RDMA,
        )

    def test_unknown_transport_is_named(self):
        with self.assertRaisesRegex(
            ValueError,
            "unknown native endpoint transport 9",
        ):
            connection_transport(connection_info(transport=9))

    def test_a_short_payload_is_rejected(self):
        for payload in (b"", bytes(127), bytes(129)):
            with self.assertRaisesRegex(
                ValueError,
                "invalid Aboleth endpoint connection-info size",
            ):
                connection_info_from_bytes(payload)

    def test_efa_address_is_required(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires a provider address",
        ):
            connection_info(address=b"")

    def test_gid_must_be_sixteen_bytes(self):
        with self.assertRaisesRegex(
            ValueError,
            "exactly 16 bytes",
        ):
            connection_info(gid=bytes(15))

    def test_address_cannot_exceed_the_struct(self):
        with self.assertRaisesRegex(
            ValueError,
            "exceeds 64 bytes",
        ):
            connection_info(address=bytes(65))


class LocalEndpointInlineTest(unittest.TestCase):
    def test_max_inline_bytes_without_native_endpoint(self):
        # The query must answer without a NIC; an endpoint with no native
        # handle reports no runtime-granted inline capacity.
        endpoint = LocalEndpoint(
            runtime=None,
            device=0,
            transport=Transport.P2P,
            descriptor=None,
            native=None,
        )
        granted = endpoint.max_inline_bytes
        self.assertIsInstance(granted, int)
        self.assertGreaterEqual(granted, 0)


if __name__ == "__main__":
    unittest.main()
