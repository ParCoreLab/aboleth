import unittest

from aboleth import (
    CapacityContract,
    ExecutionContract,
    ExecutionFallback,
    PaddingSemantics,
    PhaseResources,
    RankControl,
    RegionEffect,
    RuntimeGuard,
)


class ExecutionPlanTest(unittest.TestCase):
    def contract(
        self,
        rank_control: RankControl = RankControl.LOCKSTEP,
    ) -> ExecutionContract:
        return ExecutionContract(
            capacity=CapacityContract(
                max_items=16,
                dynamic_valid_count=True,
                padding=PaddingSemantics.DEAD,
                graph_stable_addresses=True,
            ),
            rank_control=rank_control,
            guards=(
                RuntimeGuard(
                    "actual_items_within_capacity",
                    ExecutionFallback.EXACT_AGREEMENT,
                ),
            ),
        )

    def test_fixed_capacity_legality_is_typed(self):
        self.assertTrue(self.contract().fixed_capacity_legal)
        self.assertFalse(
            self.contract(
                RankControl.DYNAMIC
            ).fixed_capacity_legal
        )
        fixed = self.contract().resolve(7)
        self.assertEqual(fixed.mode, "fixed")
        self.assertEqual(fixed.execution_items, 16)
        self.assertFalse(fixed.requires_rank_agreement)

        fallback = self.contract().resolve(
            7,
            rank_control=RankControl.DYNAMIC,
        )
        self.assertEqual(fallback.mode, "exact")
        self.assertTrue(fallback.requires_rank_agreement)
        self.assertTrue(fallback.fallback_used)

        with self.assertRaises(RuntimeError):
            self.contract().resolve(17)



if __name__ == "__main__":
    unittest.main()
