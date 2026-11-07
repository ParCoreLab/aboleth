"""The generated Tier-1 device library: files, caching, and linkability.

Generation runs without a GPU; the nvcc-dependent cases (archive, header
compile, device link) skip cleanly where nvcc is absent. The behavioural
gate for the ops themselves is the hardware WQE differential — here we pin
the packaging contract: every op symbol ships in the bitcode and archive,
the header binds them with the facade's signatures, and the digest-keyed
cache can never alias two different builds.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from aboleth.device_library import (
    DEFAULT_TRANSPORTS,
    _nvcc,
    cached_device_library,
    generate_device_library,
    library_identity,
)
from aboleth.tier1 import DEFAULT_PUT_MANY_SIZES, tier1_symbol
from aboleth.runtime import Transport
from aboleth.world import World, normalize_arch

ARCH = "sm_80"

CLIENT = r"""
#include <aboleth/device.cuh>

__global__ void tier1_client(
    const std::uint64_t* endpoints,
    aboleth::PutDesc* descriptors,
    std::uint64_t* flag) {
    aboleth::DeviceContext ctx{endpoints, nullptr};
    aboleth::put_warp(ctx, 1, {0x1000, 1}, {0x2000, 2}, 64);
    if (threadIdx.x == 0) {
        aboleth::put(ctx, 1, {0x1000, 1}, {0x2000, 2}, 64);
        aboleth::put_inline(ctx, 1, {0x1000, 1}, 7);
        aboleth::signal(ctx, 1, {0x1000, 1}, 1);
        aboleth::put_signal(ctx, 1, {0x1000, 1}, {0x2000, 2}, 64,
                            {0x3000, 3}, 1);
        aboleth::put_many(ctx, 1, descriptors, 40);
        aboleth::put_many_signal(ctx, 1, descriptors, 40, {0x3000, 3}, 2);
        aboleth::put_many_flush(ctx, 1, descriptors, 40);
        aboleth::quiet(ctx, 1);
        aboleth::wait_signal(ctx, flag, 1);
    }
}
"""


def generate(output_dir, transports=("rdma",), peer_transports=None):
    return generate_device_library(
        ARCH,
        output_dir=output_dir,
        transports=transports,
        peer_transports=peer_transports,
    )


class LibraryIdentityTargetPolicyTest(unittest.TestCase):
    SCALAR_SM100 = {100: {"invariant-descriptors": False}}

    def test_cache_identity_uses_effective_emission(self):
        options = dict(
            arch="sm_100a",
            transports=("rdma",),
            put_many_sizes=DEFAULT_PUT_MANY_SIZES,
            completion_interval=1,
            max_registers=64,
        )
        with mock.patch.dict("os.environ", {"ABOLETH_PASSES": ""}):
            default = library_identity(**options)
            with mock.patch.dict("aboleth.compiler.TARGET_OPTIMIZATIONS",
                                 self.SCALAR_SM100, clear=True):
                scalar = library_identity(**options)
        self.assertNotEqual(default, scalar)
        with mock.patch.dict(
            "os.environ", {"ABOLETH_PASSES": "-invariant-descriptors"}
        ):
            self.assertEqual(scalar, library_identity(**options))

    def test_detached_lowering_uses_the_compilation_target(self):
        from aboleth.artifact import _lower_arguments

        with mock.patch.dict("os.environ", {"ABOLETH_PASSES": ""}):
            for arch in ("sm_90a", "sm_100a"):
                self.assertIn(
                    "invariant-descriptors",
                    _lower_arguments(None, "rdma", None, arch=arch),
                )
            with mock.patch.dict("aboleth.compiler.TARGET_OPTIMIZATIONS",
                                 self.SCALAR_SM100, clear=True):
                self.assertNotIn(
                    "invariant-descriptors",
                    _lower_arguments(None, "rdma", None, arch="sm_100a"),
                )
                self.assertIn(
                    "invariant-descriptors",
                    _lower_arguments(None, "rdma", None, arch="sm_90a"),
                )


class GeneratedLibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-library-"
        )
        cls.library = generate(cls._directory.name)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_bitcode_carries_every_op_symbol(self):
        for transport, ops in self.library.ops.items():
            for op in ops:
                self.assertIn(
                    tier1_symbol(op, transport).encode(),
                    self.library.op_bitcodes[transport][op].read_bytes(),
                )
        if self.library.bitcode is not None:
            blob = self.library.bitcode.read_bytes()
            for transport, ops in self.library.ops.items():
                for op in ops:
                    self.assertIn(
                        tier1_symbol(op, transport).encode(), blob
                    )

    def test_header_binds_the_facade_surface(self):
        text = self.library.header.read_text()
        for expected in (
            "#define ABOLETH_TIER1_ABI 2",
            "#define ABOLETH_TIER1_GENERATED 1",
            "#define ABOLETH_TIER1_SPECIALIZED 1",
            "#define ABOLETH_TIER1_ENDPOINTS_PER_PEER 1",
            "kEndpointsPerPeer = 1",
            "struct DeviceContext",
            "struct PutDesc",
            "kKindMask",
            "PeerRef open(",
            "void put(",
            "void put_inline(",
            "void signal(",
            "void put_signal(",
            "void put_many(",
            "void put_many_signal(",
            "void put_many_flush(",
            "void put_warp(",
            "void quiet(",
            "void wait_signal(",
        ):
            self.assertIn(expected, text)
        # No hand-written implementation reachable from the generated header.
        self.assertNotIn("RdmaDevice.cuh", text)
        # A single-transport build folds to direct calls: no dispatch.
        self.assertNotIn("switch (tagged & kKindMask)", text)
        self.assertNotIn("__builtin_unreachable", text)

    def test_manifest_names_every_op(self):
        import json

        recorded = json.loads(self.library.manifest.read_text())
        expected = {"put", "put_signal", "put_inline", "put_inline16",
                    "put_inline24", "signal", "quiet", "wait_signal"}
        for size in DEFAULT_PUT_MANY_SIZES:
            expected.update((
                f"put_many_{size}",
                f"put_many_{size}_signal",
                f"put_many_{size}_flush",
            ))
        self.assertEqual(set(recorded["ops"]), {"rdma"})
        self.assertEqual(set(recorded["ops"]["rdma"]), expected)
        self.assertEqual(recorded["digest"], self.library.digest)

    def test_identity_is_stable(self):
        options = dict(
            arch=ARCH,
            transports=("rdma",),
            put_many_sizes=DEFAULT_PUT_MANY_SIZES,
            completion_interval=1,
            max_registers=64,
        )
        self.assertEqual(
            library_identity(**options), library_identity(**options)
        )
        self.assertNotEqual(
            library_identity(**options),
            library_identity(**{**options, "put_many_sizes": (4, 16)}),
        )
        # The transport set is a geography fingerprint: a mixed world's
        # library can never alias a uniform one.
        self.assertNotEqual(
            library_identity(**options),
            library_identity(
                **{**options, "transports": ("rdma", "p2p")}
            ),
        )
        # The fan-out stride changes the header's table indexing, so it can
        # never share a cache slot either.
        self.assertNotEqual(
            library_identity(**options),
            library_identity(**{**options, "endpoints_per_peer": 4}),
        )

    def test_fanout_header_strides_the_endpoint_table(self):
        from aboleth.emit.tier1_header import emit_tier1_header

        text = emit_tier1_header(
            arch=ARCH,
            transports=self.library.transports,
            symbols=self.library.ops,
            put_many_sizes=DEFAULT_PUT_MANY_SIZES,
            library_digest="0" * 12,
            endpoints_per_peer=4,
        )
        self.assertIn("#define ABOLETH_TIER1_ENDPOINTS_PER_PEER 4", text)
        self.assertIn("kEndpointsPerPeer = 4", text)
        self.assertIn("peer * kEndpointsPerPeer", text)
        self.assertIn("% kEndpointsPerPeer", text)
        # Every dispatching op routes through the fan-out helper — no direct
        # per-peer table read survives.
        self.assertNotIn("context.endpoints[peer]", text.replace(
            "return context.endpoints[peer];", "", 1))

    def test_cache_hits_without_regenerating(self):
        with tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-cache-"
        ) as cache:
            first = cached_device_library(ARCH, cache_root=cache)
            self.assertEqual(first.transports, DEFAULT_TRANSPORTS)
            stamp = first.manifest.stat().st_mtime_ns
            again = cached_device_library(ARCH, cache_root=cache)
            self.assertEqual(again.manifest, first.manifest)
            self.assertEqual(
                again.manifest.stat().st_mtime_ns, stamp,
                "a cache hit must not rewrite the manifest",
            )

    def test_cache_uses_matching_cmake_style_prebuilt(self):
        with tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-prebuilt-"
        ) as cache:
            prebuilt = generate_device_library(
                ARCH,
                output_dir=pathlib.Path(cache) / ARCH,
            )
            with mock.patch(
                "aboleth.device_library.generate_device_library",
                side_effect=AssertionError("matching prebuilt was rebuilt"),
            ):
                found = cached_device_library(ARCH, cache_root=cache)
            self.assertEqual(found.manifest, prebuilt.manifest)

    @unittest.skipIf(_nvcc() is None, "nvcc not found")
    def test_archive_holds_one_member_per_op(self):
        self.assertIsNotNone(self.library.archive)
        listing = subprocess.run(
            ("ar", "t", str(self.library.archive)),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        self.assertEqual(
            len(listing),
            sum(len(ops) for ops in self.library.ops.values()),
        )

    @unittest.skipIf(_nvcc() is None, "nvcc not found")
    def test_client_compiles_and_device_links(self):
        nvcc = _nvcc()
        virtual = ARCH.replace("sm_", "compute_")
        gencode = f"-gencode=arch={virtual},code={ARCH}"
        with tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-client-"
        ) as scratch:
            directory = pathlib.Path(scratch)
            client = directory / "client.cu"
            client.write_text(CLIENT)
            compiled = directory / "client.o"
            subprocess.run(
                (
                    nvcc, gencode, "-rdc=true", "-dc",
                    f"-I{self.library.include_dir}",
                    str(client), "-o", str(compiled),
                ),
                capture_output=True,
                text=True,
                check=True,
            )
            # The device link is where an unresolved tier1 symbol would
            # surface; linking against the archive must resolve them all.
            linked = directory / "linked.o"
            result = subprocess.run(
                (
                    nvcc, gencode, "-dlink",
                    str(compiled), str(self.library.archive),
                    "-o", str(linked),
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                result.stderr or result.stdout,
            )
            self.assertTrue(linked.is_file())


class MixedTransportLibraryTest(unittest.TestCase):
    """A (rdma, p2p) build: both program sets compiled, one header that
    dispatches per peer on the handle's kind bits, pruned to exactly these
    two transports."""

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-mixed-"
        )
        cls.library = generate(
            cls._directory.name, transports=("rdma", "p2p")
        )
        cls._specialized_directory = tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-geography-"
        )
        cls.specialized = generate(
            cls._specialized_directory.name,
            transports=("rdma", "p2p"),
            peer_transports=((1, "p2p"), (2, "rdma"), (3, "p2p")),
        )

    @classmethod
    def tearDownClass(cls):
        cls._specialized_directory.cleanup()
        cls._directory.cleanup()

    def test_both_transports_ship_their_ops(self):
        self.assertEqual(set(self.library.ops), {"rdma", "p2p"})
        self.assertIn("put_many_8", self.library.ops["rdma"])
        self.assertIn("put_warp", self.library.ops["p2p"])
        for transport, ops in self.library.ops.items():
            for op in ops:
                self.assertEqual(
                    self.library.ops[transport][op],
                    tier1_symbol(op, transport),
                )

    def test_header_dispatches_on_kind_bits(self):
        text = self.library.header.read_text()
        self.assertIsNone(self.library.peer_transports)
        self.assertIn("switch (tagged & kKindMask)", text)
        self.assertIn("case kKindRdma:", text)
        self.assertIn("case kKindP2P:", text)
        # Pruned dispatch: the absent transports fold away.
        self.assertNotIn("case kKindProxy:", text)
        self.assertIn("__builtin_unreachable();", text)
        self.assertIn("#define ABOLETH_TIER1_HAS_RDMA 1", text)
        self.assertIn("#define ABOLETH_TIER1_HAS_P2P 1", text)
        self.assertIn("#define ABOLETH_TIER1_SPECIALIZED 0", text)
        self.assertIn(tier1_symbol("put", "rdma"), text)
        self.assertIn(tier1_symbol("put", "p2p"), text)
        self.assertIn(tier1_symbol("put_warp", "p2p"), text)

    def test_specialized_header_bakes_full_peer_geography(self):
        text = self.specialized.header.read_text()
        self.assertEqual(
            self.specialized.peer_transports,
            ((1, "p2p"), (2, "rdma"), (3, "p2p")),
        )
        self.assertIn("switch (peer)", text)
        self.assertIn("case 1:", text)
        self.assertIn("case 2:", text)
        self.assertIn("case 3:", text)
        # The rdma put case opens with the (preprocessor-gated) small-inline
        # fast path; the direct call follows it inside the same case body.
        put_body = text.split("void put(", 1)[1].split(
            "void put_inline(", 1)[0]
        self.assertIn("case 2:", put_body)
        self.assertIn(tier1_symbol("put", "rdma"), put_body)
        self.assertIn(
            "case 1:\n    case 3:\n    {\n        tier1::"
            + tier1_symbol("put", "p2p"),
            text,
        )
        self.assertNotIn("switch (tagged & kKindMask)", text)
        self.assertIn("#define ABOLETH_TIER1_SPECIALIZED 1", text)
        self.assertIn("template<int Peer>", text)
        recorded = json.loads(self.specialized.manifest.read_text())
        self.assertEqual(
            recorded["peer_transports"],
            [[1, "p2p"], [2, "rdma"], [3, "p2p"]],
        )

    def test_geography_is_part_of_the_digest(self):
        options = dict(
            arch=ARCH,
            transports=("rdma", "p2p"),
            put_many_sizes=DEFAULT_PUT_MANY_SIZES,
            completion_interval=1,
            max_registers=64,
        )
        first = library_identity(
            **options,
            peer_transports=((1, "p2p"), (2, "rdma")),
        )
        second = library_identity(
            **options,
            peer_transports=((1, "rdma"), (2, "p2p")),
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, library_identity(**options))

    def test_digest_differs_from_the_uniform_build(self):
        with tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-uniform-"
        ) as scratch:
            uniform = generate_device_library(
                ARCH, output_dir=scratch, transports=("rdma",)
            )
            self.assertNotEqual(uniform.digest, self.library.digest)

    @unittest.skipIf(_nvcc() is None, "nvcc not found")
    def test_client_compiles_and_device_links(self):
        nvcc = _nvcc()
        virtual = ARCH.replace("sm_", "compute_")
        gencode = f"-gencode=arch={virtual},code={ARCH}"
        with tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-mixed-client-"
        ) as scratch:
            directory = pathlib.Path(scratch)
            client = directory / "client.cu"
            client.write_text(CLIENT)
            compiled = directory / "client.o"
            subprocess.run(
                (
                    nvcc, gencode, "-rdc=true", "-dc",
                    f"-I{self.library.include_dir}",
                    str(client), "-o", str(compiled),
                ),
                capture_output=True,
                text=True,
                check=True,
            )
            linked = directory / "linked.o"
            result = subprocess.run(
                (
                    nvcc, gencode, "-dlink",
                    str(compiled), str(self.library.archive),
                    "-o", str(linked),
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                result.stderr or result.stdout,
            )
            self.assertTrue(linked.is_file())


class WorldGeneratedLibrarySelectionTest(unittest.TestCase):
    @staticmethod
    def world(*transports: Transport) -> World:
        world = object.__new__(World)
        world._closed = False
        world.device = 0
        world.rank = 0
        world._endpoints_per_peer = 1
        world._region_tables = {}
        # peer -> slot tuple, the connect_all fan-out shape (slot 0 primary).
        world._endpoints = {
            peer: (SimpleNamespace(
                transport=transport,
                capacity=256 * peer,
                gpu_resident_queues=transport == Transport.RDMA,
                compilation=mock.Mock(return_value=(peer, transport)),
            ),)
            for peer, transport in enumerate(transports, start=1)
        }
        return world

    def selected_options(self, world: World, **options):
        sentinel = object()
        with (
            mock.patch("aboleth.world.normalize_arch", return_value=ARCH),
            mock.patch(
                "aboleth.device_library.cached_device_library",
                return_value=sentinel,
            ) as cached,
        ):
            self.assertIs(world.device_library(**options), sentinel)
        return cached.call_args.kwargs

    def test_default_library_is_mixed_without_connected_endpoints(self):
        world = self.world()
        selected = self.selected_options(world)
        self.assertEqual(selected["transports"], ("rdma", "p2p"))
        self.assertIsNone(selected["peer_transports"])

    def test_generated_alias_is_mixed_on_a_p2p_world(self):
        world = self.world(Transport.P2P)
        self.assertEqual(
            self.selected_options(world, generated=True)["transports"],
            ("rdma", "p2p"),
        )

    def test_specialized_p2p_world_selects_p2p_only(self):
        world = self.world(Transport.P2P)
        selected = self.selected_options(world, specialize=True)
        self.assertEqual(selected["transports"], ("p2p",))
        self.assertIsNone(selected["peer_transports"])

    def test_specialized_rdma_world_selects_rdma_only(self):
        world = self.world(Transport.RDMA)
        self.assertEqual(
            self.selected_options(world, specialize=True)["transports"],
            ("rdma",),
        )

    def test_specialized_mixed_world_keeps_generic_library(self):
        world = self.world(Transport.P2P, Transport.RDMA)
        selected = self.selected_options(world, specialize=True)
        self.assertEqual(selected["transports"], ("rdma", "p2p"))
        self.assertEqual(
            selected["peer_transports"],
            ((1, "p2p"), (2, "rdma")),
        )

    def test_gpu_resident_rdma_world_selects_gpu_scope_doorbell(self):
        # The fixture's rdma endpoints keep their queues in GPU memory.
        world = self.world(Transport.P2P, Transport.RDMA)
        self.assertTrue(world.gpu_resident_queues)
        self.assertTrue(self.selected_options(world)["gpu_resident_queues"])
        self.assertFalse(self.selected_options(
            world, gpu_resident_queues=False)["gpu_resident_queues"])

    def test_world_without_rdma_endpoints_keeps_system_scope(self):
        for world in (self.world(), self.world(Transport.P2P)):
            self.assertFalse(world.gpu_resident_queues)
            self.assertFalse(
                self.selected_options(world)["gpu_resident_queues"])

    def test_host_resident_queues_refuse_gpu_scope_doorbell(self):
        world = self.world(Transport.RDMA)
        world._endpoints[1][0].gpu_resident_queues = False
        self.assertFalse(self.selected_options(world)["gpu_resident_queues"])
        with self.assertRaises(ValueError):
            self.selected_options(world, gpu_resident_queues=True)

    def test_world_exposes_canonical_transport_geography(self):
        world = self.world(Transport.P2P, Transport.RDMA)
        self.assertEqual(
            world.transport_geography,
            ((1, "p2p"), (2, "rdma")),
        )

    def test_higher_tier_compilation_accepts_homogeneous_partition(self):
        world = self.world(Transport.P2P, Transport.RDMA, Transport.P2P)
        scope = world.compilation(peers=(1, 3), completion_interval=4)
        self.assertEqual(scope, (1, Transport.P2P))
        # The planner is bounded by the smallest queue in the partition.
        world._endpoints[1][0].compilation.assert_called_once_with(
            completion_interval=4,
            pattern_plan=None,
            kernel=None,
            queue_capacity=256,
            strict_registrations=False,
            gpu_resident_queues=False,
            endpoint_profile=None,
        )

    def test_strictly_ordered_exchanges_license_fenced_signals(self):
        world = self.world(Transport.RDMA)
        self.assertFalse(world.strict_registrations)
        world._region_tables = {
            "strict": SimpleNamespace(closed=False, strictly_ordered=True),
            "stale": SimpleNamespace(closed=True, strictly_ordered=False),
        }
        self.assertTrue(world.strict_registrations)
        self.assertTrue(self.selected_options(world)["strict_registrations"])
        world.compilation()
        self.assertTrue(
            world._endpoints[1][0].compilation.call_args.kwargs[
                "strict_registrations"]
        )
        # One live relaxed exchange is enough to keep every signal atomic.
        world._region_tables["relaxed"] = SimpleNamespace(
            closed=False, strictly_ordered=False)
        self.assertFalse(world.strict_registrations)

    def test_gpu_scope_doorbell_needs_every_endpoint_gpu_resident(self):
        world = self.world(Transport.RDMA, Transport.RDMA)
        world.compilation()
        self.assertTrue(
            world._endpoints[1][0].compilation.call_args.kwargs[
                "gpu_resident_queues"])
        world._endpoints[2][0].gpu_resident_queues = False
        world.compilation()
        self.assertFalse(
            world._endpoints[1][0].compilation.call_args.kwargs[
                "gpu_resident_queues"])

    def test_higher_tier_compilation_rejects_mixed_partition(self):
        world = self.world(Transport.P2P, Transport.RDMA)
        with self.assertRaisesRegex(RuntimeError, "backend-homogeneous"):
            world.compilation(peers=(1, 2))

    def test_specialized_library_requires_connected_endpoints(self):
        world = self.world()
        with (
            mock.patch("aboleth.world.normalize_arch", return_value=ARCH),
            self.assertRaisesRegex(RuntimeError, "connect World"),
        ):
            world.device_library(specialize=True)


class GeneratedArchitectureSelectionTest(unittest.TestCase):
    def test_numeric_prefixes_do_not_select_a_different_capability(self):
        with tempfile.TemporaryDirectory(prefix="aboleth-tier1-arch-") as build:
            for arch in ("sm_90a", "sm_100a", "sm_100f"):
                manifest = pathlib.Path(build) / "lib" / "tier1" / arch / "manifest.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps({"arch": arch}))
            with mock.patch.dict("os.environ", {"ABOLETH_BUILD_DIR": build}):
                for requested, expected in (
                    ("sm_1", "sm_1"), ("sm_10", "sm_10"),
                    ("sm_90", "sm_90a"), ("sm_100", "sm_100a"),
                    ("sm_100a", "sm_100a"), ("sm_100f", "sm_100f"),
                    ("sm_90b", "sm_90b"),
                ):
                    with self.subTest(requested=requested):
                        self.assertEqual(normalize_arch(requested), expected)

    def test_capability_uses_generated_suffixed_architecture(self):
        with tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-arch-"
        ) as build:
            manifest = (
                pathlib.Path(build)
                / "lib" / "tier1" / "sm_90a" / "manifest.json"
            )
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"arch": "sm_90a"}))
            with (
                mock.patch.dict(
                    "os.environ", {"ABOLETH_BUILD_DIR": build}
                ),
                mock.patch(
                    "aboleth.world.torch.cuda.get_device_capability",
                    return_value=(9, 0),
                ),
                mock.patch("aboleth.world.torch.version.hip", None),
            ):
                self.assertEqual(normalize_arch(), "sm_90a")


class GeneratedLibraryConcurrencyTest(unittest.TestCase):
    def test_cold_ranks_share_one_completed_generation(self):
        import threading
        import aboleth.device_library as implementation

        digest = "d" * 64
        entered = threading.Event()
        second_started = threading.Event()
        second_generation = threading.Event()
        generations, results, errors = [], [], []

        def load(path):
            if threading.current_thread().name == "second":
                second_started.set()
            return SimpleNamespace(digest=digest) if (path / "ready").exists() else None

        def generate(arch, *, output_dir, **options):
            generations.append(threading.current_thread().name)
            output_dir.mkdir(parents=True, exist_ok=True)
            if len(generations) == 1:
                entered.set()
                if not second_started.wait(2):
                    raise AssertionError("second request never reached the cache")
                # Without serialization the second cold writer enters here.
                # With it, that writer waits until this manifest is complete.
                second_generation.wait(.2)
            else:
                second_generation.set()
            (output_dir / "ready").write_text(digest)
            return SimpleNamespace(digest=digest)

        with tempfile.TemporaryDirectory(prefix="aboleth-tier1-race-") as cache:
            def request():
                try:
                    results.append(cached_device_library(ARCH, cache_root=cache))
                except BaseException as error:
                    errors.append(error)

            with (mock.patch.object(implementation, "library_identity", return_value=digest),
                  mock.patch.object(implementation, "_load", side_effect=load),
                  mock.patch.object(implementation, "generate_device_library", side_effect=generate)):
                first = threading.Thread(target=request, name="first")
                second = threading.Thread(target=request, name="second")
                first.start()
                self.assertTrue(entered.wait(2))
                second.start()
                first.join(5)
                second.join(5)
                self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual([value.digest for value in results], [digest, digest])
            self.assertEqual(generations, ["first"])


if __name__ == "__main__":
    unittest.main()
