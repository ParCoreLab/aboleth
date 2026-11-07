import unittest

from aboleth import (
    GENERIC_KV_TARGET_PROFILE,
    H200_NVLINK_KV_TARGET_PROFILE,
    KVCacheLayoutContract,
    KVCacheMigrationSpec,
    KVRuntimeOverrides,
    compile_kv_cache_migration,
    select_kv_target_profile,
)
from aboleth.kv import (
    CompiledKVTransferManager,
    KVCompletionNotification,
    KVRequestLifecycle,
)


class FakeEvent:
    def __init__(self, done: bool = False):
        self.done = done

    def query(self) -> bool:
        return self.done


class FakeTicket:
    def __init__(
        self,
        completion: FakeEvent,
        layer_events: dict[str, FakeEvent],
    ):
        self.completion = completion
        self.layer_events = layer_events


class FakeRuntime:
    asynchronous_completion = True
    consumer_layer_pipeline = True

    def __init__(self):
        self.completion = FakeEvent()
        self.layer = FakeEvent()

    def start_layered_pull_blocks(self, **kwargs):
        del kwargs
        return FakeTicket(
            self.completion,
            {"layer.0": self.layer},
        )


class KVRuntimeTest(unittest.TestCase):
    def test_target_profile_is_hardware_specific(self):
        self.assertIs(
            select_kv_target_profile("NVIDIA H200"),
            H200_NVLINK_KV_TARGET_PROFILE,
        )
        self.assertIs(
            select_kv_target_profile("NVIDIA A100"),
            GENERIC_KV_TARGET_PROFILE,
        )
        generic = compile_kv_cache_migration(
            KVCacheMigrationSpec(
                max_pages=16,
                page_bytes=4096,
                max_consumers=4,
                consumer_asynchronous=True,
            )
        )
        self.assertEqual(
            generic.physicalization.target_profile,
            "generic-generated-p2p",
        )
        self.assertFalse(
            generic.physicalization.consumer_layer_pipeline
        )
        h200 = compile_kv_cache_migration(
            generic.spec,
            layout_contract=KVCacheLayoutContract(
                consumer_ordered=True,
            ),
            target_profile=H200_NVLINK_KV_TARGET_PROFILE,
        )
        self.assertTrue(
            h200.physicalization.consumer_layer_pipeline
        )
        self.assertEqual(
            h200.physicalization.consumers_per_readiness,
            4,
        )

    def test_runtime_overrides_parse_ablation_mapping(self):
        overrides = KVRuntimeOverrides.from_mapping({
            "aboleth_copy_workers": "32",
            "aboleth_copy_engine_threshold_bytes": 0,
            "aboleth_async_transfers": False,
            "aboleth_profile_transfers": True,
            "ignored": "value",
        })
        self.assertEqual(
            overrides.generated_copy_workers,
            32,
        )
        self.assertEqual(
            overrides.copy_engine_threshold_bytes,
            0,
        )
        self.assertFalse(overrides.asynchronous_completion)
        self.assertTrue(overrides.profile_transfers)

    def test_pipelined_request_lifecycle(self):
        lifecycle = KVRequestLifecycle()
        completion = FakeEvent()
        layer = FakeEvent()
        notification = KVCompletionNotification(
            target="producer",
            payload=b"request",
        )
        lifecycle.add(
            request_id="request",
            completion=completion,
            notification=notification,
            layer_events={"layer.0": layer},
            pipelined=True,
        )
        first = lifecycle.poll()
        self.assertEqual(
            first.ready_request_ids,
            frozenset({"request"}),
        )
        self.assertFalse(first.notifications)
        self.assertEqual(lifecycle.activate_admitted(), 1)
        waited = []
        lifecycle.wait_for_layer(
            "layer.0",
            waited.append,
        )
        self.assertEqual(waited, [layer])
        completion.done = True
        final = lifecycle.poll()
        self.assertEqual(
            final.notifications,
            (notification,),
        )
        self.assertEqual(lifecycle.pending_count, 0)

    def test_nonpipelined_request_waits_for_completion(self):
        lifecycle = KVRequestLifecycle()
        completion = FakeEvent()
        notification = KVCompletionNotification(
            target="producer",
            payload=b"request",
        )
        lifecycle.add(
            request_id="request",
            completion=completion,
            notification=notification,
            pipelined=False,
        )
        self.assertFalse(lifecycle.poll().ready_request_ids)
        completion.done = True
        final = lifecycle.poll()
        self.assertEqual(
            final.ready_request_ids,
            frozenset({"request"}),
        )
        self.assertEqual(
            final.notifications,
            (notification,),
        )
        self.assertEqual(lifecycle.pending_count, 0)

    def test_cancelled_request_keeps_source_lifetime(self):
        lifecycle = KVRequestLifecycle()
        completion = FakeEvent()
        notification = KVCompletionNotification(
            target="producer",
            payload=b"request",
        )
        lifecycle.add(
            request_id="request",
            completion=completion,
            notification=notification,
            layer_events={"layer.0": FakeEvent()},
            pipelined=True,
        )
        lifecycle.cancel({"request"})
        self.assertEqual(lifecycle.pending_count, 1)
        completion.done = True
        final = lifecycle.poll()
        self.assertFalse(final.ready_request_ids)
        self.assertEqual(
            final.notifications,
            (notification,),
        )
        self.assertEqual(lifecycle.pending_count, 0)

    def test_transfer_manager_owns_pipeline_state(self):
        runtime = FakeRuntime()
        manager = CompiledKVTransferManager(runtime)
        notification = KVCompletionNotification(
            target="producer",
            payload=b"request",
        )
        tracked = manager.start_transfer(
            request_id="request",
            local_block_ids=[0],
            remote_block_ids=[1],
            remote_rank=0,
            notification=notification,
        )
        self.assertTrue(tracked)
        self.assertEqual(
            manager.poll().ready_request_ids,
            frozenset({"request"}),
        )
        self.assertEqual(manager.begin_step(set()), 1)
        waited = []
        manager.wait_for_layer("layer.0", waited.append)
        self.assertEqual(waited, [runtime.layer])
        runtime.completion.done = True
        self.assertEqual(
            manager.poll().notifications,
            (notification,),
        )


if __name__ == "__main__":
    unittest.main()
