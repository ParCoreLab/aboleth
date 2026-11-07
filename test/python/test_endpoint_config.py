"""The deferred-control surface: what an endpoint config asks for, what it is
refused for asking, and how a caller's struct_size is read.

The versioning arm needs no hardware — a config is checked before the runtime
handle it was passed with is even looked at, so a truncated or oversized struct
answers the same way on a machine with no NIC. Everything that has to reach an
adapter is skipped when one cannot be opened.
"""

import ctypes
import pathlib
import unittest

import aboleth
from aboleth import _runtime


LIBRARY_NAME = "libaboleth_rdma_abi.so"


class EndpointConfig(ctypes.Structure):
    """AbolethEndpointConfig, as a caller outside C++ has to spell it."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("qp_type", ctypes.c_uint32),
        ("qp_depth", ctypes.c_uint32),
        ("max_inline", ctypes.c_uint32),
        ("qps_per_peer", ctypes.c_uint32),
        ("sm_map_policy", ctypes.c_uint32),
        ("port", ctypes.c_int32),
        ("gid_index", ctypes.c_int32),
        ("hca", ctypes.c_char * 64),
    ]

    @classmethod
    def defaults(cls, **overrides):
        config = cls(
            struct_size=ctypes.sizeof(cls),
            gid_index=-1,
        )
        for name, value in overrides.items():
            setattr(config, name, value)
        return config


def load_library():
    """The runtime's C ABI, or None when this build has no runtime."""
    path = (
        pathlib.Path(aboleth.__file__).parent / LIBRARY_NAME
    )
    if not path.is_file():
        return None
    library = ctypes.CDLL(str(path))
    library.aboleth_runtime_last_error.restype = ctypes.c_char_p
    library.aboleth_runtime_create_for_device.restype = ctypes.c_void_p
    library.aboleth_runtime_create_for_device.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    library.aboleth_runtime_destroy.argtypes = [ctypes.c_void_p]
    library.aboleth_runtime_create_local_endpoint.restype = (
        ctypes.c_void_p
    )
    library.aboleth_runtime_create_local_endpoint.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(EndpointConfig),
    ]
    library.aboleth_runtime_local_endpoint_capacity.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.aboleth_runtime_local_endpoint_max_inline.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.aboleth_runtime_destroy_local_endpoint.argtypes = [
        ctypes.c_void_p
    ]
    return library


LIBRARY = load_library()


@unittest.skipIf(
    LIBRARY is None,
    f"no {LIBRARY_NAME} in the build; configure with the runtime",
)
class EndpointConfigVersioningTest(unittest.TestCase):
    """struct_size is the version, and it is checked before anything opens."""

    def create(self, config):
        """Creates against a null runtime: enough to reach the config check."""
        handle = LIBRARY.aboleth_runtime_create_local_endpoint(
            None,
            _runtime.TRANSPORT_DIRECT,
            0,
            None,
            0,
            None if config is None else ctypes.byref(config),
        )
        self.assertFalse(handle)
        return LIBRARY.aboleth_runtime_last_error().decode()

    def test_the_python_struct_is_the_c_struct(self):
        self.assertEqual(
            ctypes.sizeof(EndpointConfig),
            _runtime.ENDPOINT_CONFIG_BYTES,
        )

    def test_a_whole_struct_reaches_the_runtime(self):
        # The null handle is the next thing checked, so this message is the
        # proof that the config itself was accepted.
        self.assertEqual(
            self.create(EndpointConfig.defaults()),
            "null local-endpoint argument",
        )

    def test_no_config_at_all_is_every_default(self):
        self.assertEqual(
            self.create(None),
            "null local-endpoint argument",
        )

    def test_a_shorter_struct_is_a_whole_prefix(self):
        # An older caller's header stops after `qp_type`; the fields past it
        # take this runtime's defaults instead of the caller's bytes.
        for size in (4, 8, 12, _runtime.ENDPOINT_CONFIG_BYTES):
            with self.subTest(struct_size=size):
                self.assertEqual(
                    self.create(
                        EndpointConfig.defaults(struct_size=size)
                    ),
                    "null local-endpoint argument",
                )

    def test_a_longer_struct_names_the_mismatch(self):
        message = self.create(
            EndpointConfig.defaults(
                struct_size=_runtime.ENDPOINT_CONFIG_BYTES + 4
            )
        )
        self.assertIn("struct_size", message)
        self.assertIn(
            str(_runtime.ENDPOINT_CONFIG_BYTES + 4), message
        )
        self.assertIn("exceeds this runtime's", message)
        self.assertIn("docs/transport-control.md", message)

    def test_a_partial_field_is_refused(self):
        for size in (0, 2, 6, 95):
            with self.subTest(struct_size=size):
                message = self.create(
                    EndpointConfig.defaults(struct_size=size)
                )
                self.assertIn("struct_size", message)
                self.assertIn("whole prefix", message)
                self.assertIn(
                    str(_runtime.ENDPOINT_CONFIG_BYTES), message
                )


def open_runtime():
    """A rank-local runtime on GPU 0, or the reason there cannot be one.

    Creating a direct endpoint is the probe: it opens the queue pair and maps
    the adapter's user-access region into the GPU's address space, which is
    what an unprivileged process may not be allowed to do.
    """
    try:
        import torch

        from aboleth.runtime import Runtime

        if torch.cuda.device_count() < 1:
            return None, "no CUDA device is visible"
        runtime = Runtime(device=0)
    except Exception as error:  # noqa: BLE001 - the reason is the message
        return None, f"{type(error).__name__}: {error}"
    try:
        with runtime.create_local_endpoint(transport="rdma"):
            pass
    except Exception as error:  # noqa: BLE001
        runtime.close()
        return None, f"{type(error).__name__}: {error}"
    return runtime, ""


class NativeRuntime:
    """The C ABI directly, for the configs Python does not spell.

    A `None` config is only reachable here — every Python caller goes through
    the binding, which always fills one in.
    """

    def __init__(self, device):
        import torch

        self.device = device
        self.handle = LIBRARY.aboleth_runtime_create_for_device(
            device, None
        )
        if not self.handle:
            raise RuntimeError(
                LIBRARY.aboleth_runtime_last_error().decode()
            )
        self.descriptor = torch.empty(
            12, dtype=torch.uint64, device=f"cuda:{device}"
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        LIBRARY.aboleth_runtime_destroy(
            ctypes.c_void_p(self.handle)
        )
        self.handle = None
        return False

    def _create(self, config):
        return LIBRARY.aboleth_runtime_create_local_endpoint(
            ctypes.c_void_p(self.handle),
            _runtime.TRANSPORT_DIRECT,
            self.device,
            ctypes.c_void_p(self.descriptor.data_ptr()),
            self.descriptor.numel() * self.descriptor.element_size(),
            None if config is None else ctypes.byref(config),
        )

    def endpoint(self, config):
        """(capacity, granted inline bytes) for one endpoint, then closed."""
        handle = self._create(config)
        if not handle:
            raise RuntimeError(
                LIBRARY.aboleth_runtime_last_error().decode()
            )
        capacity = ctypes.c_uint32()
        inline = ctypes.c_uint32()
        LIBRARY.aboleth_runtime_local_endpoint_capacity(
            ctypes.c_void_p(handle), ctypes.byref(capacity)
        )
        LIBRARY.aboleth_runtime_local_endpoint_max_inline(
            ctypes.c_void_p(handle), ctypes.byref(inline)
        )
        LIBRARY.aboleth_runtime_destroy_local_endpoint(
            ctypes.c_void_p(handle)
        )
        return capacity.value, inline.value

    def refusal(self, config):
        """The error a config the runtime will not accept leaves behind."""
        handle = self._create(config)
        if handle:
            LIBRARY.aboleth_runtime_destroy_local_endpoint(
                ctypes.c_void_p(handle)
            )
            raise AssertionError("config was accepted")
        return LIBRARY.aboleth_runtime_last_error().decode()


@unittest.skipIf(
    LIBRARY is None,
    f"no {LIBRARY_NAME} in the build; configure with the runtime",
)
class EndpointConfigHardwareTest(unittest.TestCase):
    """What the adapter does with a request, and what it refuses to be asked."""

    @classmethod
    def setUpClass(cls):
        cls.runtime, reason = open_runtime()
        if cls.runtime is None:
            raise unittest.SkipTest(
                f"no usable RDMA endpoint here ({reason})"
            )

    @classmethod
    def tearDownClass(cls):
        if cls.runtime is not None:
            cls.runtime.close()

    def endpoint(self, **config):
        endpoint = self.runtime.create_local_endpoint(
            transport="rdma", **config
        )
        self.addCleanup(endpoint.close)
        return endpoint

    def test_defaults_match_no_config_at_all(self):
        # A zero-filled config and a null one are the same endpoint, which is
        # what lets every existing caller keep passing nothing.
        with NativeRuntime(self.runtime.device) as native:
            explicit = native.endpoint(
                EndpointConfig.defaults()
            )
            implicit = native.endpoint(None)
        self.assertEqual(explicit, implicit)

    def test_a_field_past_struct_size_is_not_read(self):
        # The bytes carry a queue depth, a refused qps_per_peer and a foreign
        # HCA; the struct_size stops before all three, so none of them apply.
        with NativeRuntime(self.runtime.device) as native:
            default = native.endpoint(EndpointConfig.defaults())
            truncated = native.endpoint(
                EndpointConfig.defaults(
                    struct_size=8,
                    qp_depth=4096,
                    qps_per_peer=17,
                    sm_map_policy=9,
                    port=1234,
                    gid_index=1234,
                    hca=b"mlx5_absent",
                )
            )
        self.assertEqual(truncated, default)

    def test_qp_depth_reaches_the_queue(self):
        capacities = {
            depth: self.endpoint(qp_depth=depth).capacity
            for depth in (64, 512, 2048)
        }
        for depth, capacity in capacities.items():
            # The provider rounds its ring up to whole 64-byte blocks, so the
            # granted queue holds at least the requests that were asked for.
            self.assertGreaterEqual(capacity, depth)
        self.assertLess(capacities[64], capacities[512])
        self.assertLess(capacities[512], capacities[2048])

    def test_a_depth_beyond_the_adapter_is_named(self):
        with self.assertRaises(RuntimeError) as raised:
            self.endpoint(qp_depth=1 << 24)
        message = str(raised.exception)
        self.assertIn("qp_depth", message)
        self.assertIn(str(1 << 24), message)
        self.assertIn("maximum send-queue depth", message)
        self.assertIn("docs/transport-control.md", message)

    def test_max_inline_is_requested_and_reported(self):
        # No adapter grants a megabyte inline, so this is the floor a refused
        # request falls back to.
        floor = self.endpoint(max_inline=1 << 20).max_inline_bytes
        granted = {
            request: self.endpoint(
                max_inline=request
            ).max_inline_bytes
            for request in (64, 128, 256)
        }
        for request, capacity in granted.items():
            with self.subTest(max_inline=request):
                self.assertTrue(
                    capacity >= request or capacity == floor,
                    f"{request} bytes granted {capacity}, "
                    f"neither honored nor the {floor}-byte floor",
                )
        self.assertLessEqual(granted[64], granted[128])
        self.assertLessEqual(granted[128], granted[256])
        # Two grant semantics exist: libmlx5 negotiation refuses the
        # oversized request outright (its fallback floor sits below the
        # honored grants), while a DEVX staged endpoint clamps it to the
        # direct emitter's 972-byte WQE ceiling.
        self.assertTrue(
            granted[256] > floor or floor == 972,
            f"grants {granted} do not order against the {floor}-byte floor",
        )

    def test_stating_the_runtimes_own_device_is_accepted(self):
        # Stating what the runtime already opened is allowed: it is an
        # assertion the caller can make, not a second selection.
        endpoint = self.endpoint(
            hca=self.runtime.hca,
            port=self.runtime.rdma_port,
            gid_index=self.runtime.gid_index,
        )
        self.assertGreater(endpoint.capacity, 0)

    def test_another_device_is_refused_by_name(self):
        cases = (
            ("hca", {"hca": "mlx5_absent"}, "aboleth_runtime_create_for_device"),
            ("port", {"port": self.runtime.rdma_port + 6}, "ABOLETH_PORT"),
            (
                "gid_index",
                {"gid_index": self.runtime.gid_index + 6},
                "ABOLETH_GID_INDEX",
            ),
        )
        for field, config, pointer in cases:
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError) as raised:
                    self.endpoint(**config)
                message = str(raised.exception)
                self.assertIn(f"AbolethEndpointConfig.{field}", message)
                self.assertIn(pointer, message)
                self.assertIn("docs/transport-control.md", message)

    def test_a_queue_request_a_transport_cannot_act_on_is_refused(self):
        for field, value in (("qp_depth", 512), ("max_inline", 64)):
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError) as raised:
                    self.runtime.create_local_endpoint(
                        transport="proxy", **{field: value}
                    )
                message = str(raised.exception)
                self.assertIn(f"AbolethEndpointConfig.{field}", message)
                self.assertIn("direct (mlx5) transport only", message)

    def test_p2p_has_no_queue_pair_to_configure(self):
        for field, value in (
            ("qp_depth", 512),
            ("max_inline", 64),
            ("hca", "mlx5_absent"),
            ("port", 2),
            ("gid_index", 3),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as raised:
                    self.runtime.create_local_endpoint(
                        transport="p2p", **{field: value}
                    )
                self.assertIn(field, str(raised.exception))

    def test_an_unimplemented_field_is_refused_by_name(self):
        # These three have no Python spelling, so the C struct is how a caller
        # reaches them — and how it learns they are deferred.
        expected = {
            "qp_type": "reliable connected",
            "qps_per_peer": "one endpoint is one queue pair",
            "sm_map_policy": "SM-to-queue-pair mapping",
        }
        with NativeRuntime(self.runtime.device) as native:
            for field, reason in expected.items():
                with self.subTest(field=field):
                    message = native.refusal(
                        EndpointConfig.defaults(**{field: 4})
                    )
                    self.assertIn(
                        f"AbolethEndpointConfig.{field}=4", message
                    )
                    self.assertIn(reason, message)
                    self.assertIn(
                        "docs/transport-control.md", message
                    )


if __name__ == "__main__":
    unittest.main()
