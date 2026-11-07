"""The cache key the compiler hook gives Triton.

Triton's on-disk cache appends only the key string, so every input that
changes the generated code must change that string, not only the digest.
"""

import os
import unittest
from unittest import mock

from aboleth.compiler import (
    OPTIMIZATIONS, _pipeline_key, compilation, enabled_optimizations,
    rdma_lowering_options)


class PipelineKeyTest(unittest.TestCase):
    def keys(self, **options):
        with compilation("rdma", **options):
            return _pipeline_key()

    def test_the_key_string_carries_the_digest(self):
        key, digest = self.keys()
        self.assertTrue(key.endswith(digest))

    def test_every_optimization_switch_changes_the_key_string(self):
        default = self.keys()[0]
        for name, on in OPTIMIZATIONS.items():
            flipped = f"{'-' if on else '+'}{name}"
            with self.subTest(switch=flipped):
                with mock.patch.dict(os.environ, {"ABOLETH_PASSES": flipped}):
                    self.assertNotEqual(default, self.keys()[0])

    def test_unknown_switches_are_rejected(self):
        with mock.patch.dict(os.environ, {"ABOLETH_PASSES": "-cse"}):
            with self.assertRaises(ValueError):
                self.keys()

    def test_target_policy_revision_invalidates_rdma_cache(self):
        first = self.keys()
        with mock.patch("aboleth.compiler._RDMA_EMISSION_POLICY", "next"):
            second = self.keys()
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_explicit_target_override_cannot_alias_the_generic_default(self):
        with mock.patch.dict(os.environ, {"ABOLETH_PASSES": ""}):
            default = self.keys()
        with mock.patch.dict(
            os.environ, {"ABOLETH_PASSES": "+invariant-descriptors"}
        ):
            explicit = self.keys()
        self.assertNotEqual(default[0], explicit[0])
        self.assertNotEqual(default[1], explicit[1])

    def test_registration_ordering_changes_the_key_string(self):
        self.assertNotEqual(
            self.keys()[0], self.keys(strict_registrations=True)[0])

    def test_queue_placement_changes_the_key_string(self):
        self.assertNotEqual(
            self.keys()[0], self.keys(gpu_resident_queues=True)[0])

    def test_the_endpoint_profile_changes_the_key_string(self):
        measured = self.keys(endpoint_profile="rdma:measured:sha256:aa")[0]
        self.assertNotEqual(self.keys()[0], measured)
        self.assertNotEqual(
            measured, self.keys(endpoint_profile="rdma:measured:sha256:bb")[0])


class DoorbellScopeTest(unittest.TestCase):
    """Which RDMA lowerings publish the doorbell with a GPU-scope release."""

    @staticmethod
    def scoped(**options):
        with mock.patch(
                "aboleth.compiler._rdma_device_library_enabled",
                return_value=False):
            return "gpu-scope-doorbell" in rdma_lowering_options(
                enabled_optimizations(), **options)

    def test_kernels_follow_the_compilation_scope(self):
        self.assertFalse(self.scoped())
        with compilation("rdma", gpu_resident_queues=True):
            self.assertTrue(self.scoped())

    def test_detached_fragments_need_the_caller_to_state_it(self):
        # A Tier-1 op has no endpoint: the enclosing scope does not apply.
        with compilation("rdma", gpu_resident_queues=True):
            self.assertFalse(self.scoped(detached=True))
        self.assertTrue(self.scoped(detached=True, gpu_resident_queues=True))

    def test_the_switch_turns_it_off(self):
        with mock.patch.dict(os.environ, {"ABOLETH_PASSES": "-gpu-doorbell"}):
            self.assertFalse(
                self.scoped(detached=True, gpu_resident_queues=True))


class DoorbellScopeTest(unittest.TestCase):
    """Which RDMA lowerings publish the doorbell with a GPU-scope release."""

    @staticmethod
    def scoped(**options):
        with mock.patch(
                "aboleth.compiler._rdma_device_library_enabled",
                return_value=False):
            return "gpu-scope-doorbell" in rdma_lowering_options(
                enabled_optimizations(), **options)

    def test_kernels_follow_the_compilation_scope(self):
        self.assertFalse(self.scoped())
        with compilation("rdma", gpu_resident_queues=True):
            self.assertTrue(self.scoped())

    def test_detached_fragments_need_the_caller_to_state_it(self):
        # A Tier-1 op has no endpoint: the enclosing scope does not apply.
        with compilation("rdma", gpu_resident_queues=True):
            self.assertFalse(self.scoped(detached=True))
        self.assertTrue(self.scoped(detached=True, gpu_resident_queues=True))

    def test_the_switch_turns_it_off(self):
        with mock.patch.dict(os.environ, {"ABOLETH_PASSES": "-gpu-doorbell"}):
            self.assertFalse(
                self.scoped(detached=True, gpu_resident_queues=True))


if __name__ == "__main__":
    unittest.main()
