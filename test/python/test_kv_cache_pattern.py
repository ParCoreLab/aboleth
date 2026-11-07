import unittest

from aboleth import (
    CompiledKVCacheMigrationPlan,
    H200_NVLINK_KV_TARGET_PROFILE,
    KVCacheMigrationSpec,
    KVCacheLayoutContract,
    build_kv_cache_migration_pattern,
    compile_kv_cache_migration,
)
from aboleth.frontends.triton.kv_cache import (
    KVCacheMigrationResult,
    migrate_kv_cache_pages,
)
from aboleth.kv import canonicalize_block_runs
from aboleth.patterns.kv_cache_transport import (
    physicalize_kv_cache_pattern,
)
from aboleth.pattern import OpaqueStage


class KVCacheMigrationPatternTest(unittest.TestCase):
    def test_native_frontend_is_available(self):
        self.assertTrue(callable(migrate_kv_cache_pages))
        self.assertIn(
            "pages",
            KVCacheMigrationResult.__annotations__,
        )

    def test_transport_program_is_generated(self):
        spec = KVCacheMigrationSpec(
            max_pages=8,
            page_bytes=4096,
            num_ranks=2,
        )
        physicalized = physicalize_kv_cache_pattern(spec)
        program = physicalized.program("kv_cache.migrate")

        self.assertEqual(program.program_id, 0)
        self.assertEqual(program.command_count, 9)
        self.assertEqual(
            program.operations,
            ("migration.transfer",),
        )

    def test_unique_pages_migrate_directly_into_cache(self):
        compiled = build_kv_cache_migration_pattern(
            KVCacheMigrationSpec(
                max_pages=128,
                page_bytes=16 * 1024,
                num_ranks=8,
                slots=2,
            )
        ).compile()
        transfer = compiled.analysis.operation(
            "migration.transfer"
        )
        self.assertEqual(
            transfer.destination,
            "cache.destination.storage",
        )
        self.assertIsNone(
            compiled.analysis.operation(
                "migration.materialize"
            )
        )
        self.assertFalse(compiled.kernel_bindings)
        consumer = next(
            stage
            for stage in compiled.stages
            if isinstance(stage, OpaqueStage)
        )
        self.assertEqual(consumer.name, "cache.consumer")
        self.assertTrue(consumer.asynchronous)

    def test_duplicate_destinations_keep_materialization(self):
        compiled = build_kv_cache_migration_pattern(
            KVCacheMigrationSpec(
                max_pages=32,
                page_bytes=4096,
                num_ranks=4,
                unique_destinations=False,
            )
        ).compile()
        self.assertEqual(
            compiled.analysis.operation(
                "migration.transfer"
            ).destination,
            "cache.destination.staging",
        )
        self.assertIsNotNone(
            compiled.analysis.operation(
                "migration.materialize"
            )
        )

    def test_compiler_selects_batched_consumer_ready_plan(self):
        plan = compile_kv_cache_migration(
            KVCacheMigrationSpec(
                max_pages=32,
                page_bytes=16 * 1024,
                num_ranks=2,
                max_consumers=4,
                consumer_max_inflight=2,
            )
        )
        self.assertIsInstance(
            plan,
            CompiledKVCacheMigrationPlan,
        )
        self.assertTrue(
            plan.physicalization.direct_destination
        )
        self.assertEqual(
            plan.physicalization.pages_per_transfer,
            8,
        )
        self.assertEqual(
            plan.physicalization.readiness,
            "consumer",
        )
        self.assertEqual(
            plan.physicalization.slot_count,
            3,
        )
        self.assertEqual(plan.phase.program.command_count, 1)
        schedule = plan.phase.plan.phase_schedule(
            plan.phase.name
        )
        self.assertEqual(schedule.strategy, "parallel_copy")
        self.assertEqual(
            schedule.kernel,
            "kv_cache_page_batch_copy_kernel",
        )
        self.assertEqual(
            schedule.publication_strategy,
            "fused_kernel",
        )
        self.assertIsNone(schedule.publication_kernel)
        self.assertIsNotNone(plan.fallback_phase)
        self.assertEqual(
            plan.fallback_phase.program.command_count,
            1,
        )
        selected, pages = plan.select_phase(
            contiguous_page_runs=False,
            consumer_partitioned=True,
        )
        self.assertEqual(
            selected.name,
            "kv_cache.migrate.fallback",
        )
        self.assertEqual(pages, 1)

    def test_noncontiguous_contract_selects_page_writes(self):
        plan = compile_kv_cache_migration(
            KVCacheMigrationSpec(
                max_pages=16,
                page_bytes=4096,
                num_ranks=2,
            ),
            layout_contract=KVCacheLayoutContract(
                contiguous_page_runs=False,
            ),
        )
        self.assertEqual(
            plan.physicalization.pages_per_transfer,
            1,
        )
        self.assertIsNone(plan.fallback_phase)

    def test_compiler_selects_bulk_copy_engine_at_crossover(self):
        threshold = 64 << 20
        plan = compile_kv_cache_migration(
            KVCacheMigrationSpec(
                max_pages=256,
                page_bytes=512 * 1024,
                num_ranks=2,
                max_consumers=28,
                consumer_asynchronous=True,
            ),
            layout_contract=KVCacheLayoutContract(
                consumer_ordered=True,
            ),
            target_profile=H200_NVLINK_KV_TARGET_PROFILE,
            copy_engine_threshold_bytes=threshold,
        )
        self.assertEqual(
            plan.select_transfer_engine(
                transfer_bytes=threshold - 1,
                run_count=1,
            ),
            "generated_p2p",
        )
        self.assertEqual(
            plan.select_transfer_engine(
                transfer_bytes=threshold,
                run_count=1,
            ),
            "copy_engine",
        )
        self.assertEqual(
            plan.select_transfer_engine(
                transfer_bytes=threshold * 2,
                run_count=17,
            ),
            "generated_p2p",
        )
        self.assertTrue(
            plan.physicalization.asynchronous_completion
        )
        self.assertTrue(
            plan.physicalization.consumer_layer_pipeline
        )
        self.assertEqual(
            plan.physicalization.consumers_per_readiness,
            4,
        )
        with self.assertRaisesRegex(
            ValueError,
            "one ordered copy stream",
        ):
            compile_kv_cache_migration(
                plan.spec,
                copy_engine_streams=2,
                consumer_layer_pipeline=True,
            )
        unordered = compile_kv_cache_migration(
            plan.spec,
            layout_contract=KVCacheLayoutContract(
                consumer_ordered=False,
            ),
        )
        self.assertFalse(
            unordered.physicalization.consumer_layer_pipeline
        )
        self.assertEqual(
            plan.physicalization.copy_engine_streams,
            1,
        )

    def test_injective_page_pairs_canonicalize_across_requests(self):
        runs = canonicalize_block_runs(
            [2, 3, 0, 1],
            [12, 13, 10, 11],
        )
        self.assertEqual(runs.local_starts, (0,))
        self.assertEqual(runs.remote_starts, (10,))
        self.assertEqual(runs.lengths, (4,))

    def test_slots_are_derived_from_consumer_lifetime(self):
        spec = KVCacheMigrationSpec(
            max_pages=8,
            page_bytes=4096,
            consumer_asynchronous=True,
            consumer_max_inflight=3,
        )
        self.assertEqual(spec.slot_count, 4)
        with self.assertRaises(ValueError):
            KVCacheMigrationSpec(
                max_pages=8,
                page_bytes=4096,
                slots=3,
                consumer_max_inflight=3,
            )


if __name__ == "__main__":
    unittest.main()
