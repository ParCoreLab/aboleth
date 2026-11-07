"""Capacity/selector decisions are C++; control epochs are private per slot."""

import os
import unittest
from unittest.mock import patch
from aboleth.shapes import plan_shape_family


def plan(
    *,
    minimum=1,
    maximum=513,
    slots=2,
    uniform=True,
    enabled=True,
    iteration=None,
):
    with patch.dict(
        os.environ,
        {
            "ABOLETH_PASSES": (
                "+shape-specialization" if enabled else "-shape-specialization"
            )
        },
    ):
        return plan_shape_family(
            name="bounded-work",
            minimum_bound=minimum,
            maximum_bound=maximum,
            slots=slots,
            uniform_selector=uniform,
            iteration_bound=iteration,
        )


class ShapeFamilyTest(unittest.TestCase):
    def test_power_of_two_bounds_and_exact_selector(self):
        family = plan()
        self.assertEqual(
            tuple(v.capacity for v in family.variants),
            (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
        )
        for bound, capacity in [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 4),
            (256, 256),
            (257, 512),
            (513, 1024),
        ]:
            self.assertEqual(family.select(bound).capacity, capacity)
        self.assertTrue(family.uniform_required)
        self.assertEqual(family.private_control, "per_variant_slot")
        self.assertTrue(family.mutually_exclusive_within_slot)
        self.assertFalse(family.payload_aliases_proven)

    def test_no_uniform_fact_or_disabled_pass_uses_one_safe_capacity(self):
        for args in [dict(uniform=False), dict(enabled=False)]:
            family = plan(**args)
            self.assertEqual(len(family.variants), 1)
            self.assertEqual(family.select(1).capacity, 1024)
            self.assertFalse(family.uniform_required)

    def test_domains_do_not_overlap_between_variants_and_dbo_slots(self):
        family = plan()
        domains = [
            v.control_domain(slot)
            for v in family.variants
            for slot in range(2)
        ]
        self.assertEqual(domains, list(range(22)))
        order = [513, 1, 513, 3, 1, 513]
        self.assertEqual(
            [family.select(b).control_domain(0) for b in order],
            [20, 0, 20, 4, 0, 20],
        )
        self.assertEqual(
            [family.select(b).control_domain(1) for b in order],
            [21, 1, 21, 5, 1, 21],
        )

    def test_non_power_minimum_and_large_signed_bounds(self):
        self.assertEqual(
            tuple(v.capacity for v in plan(minimum=9, maximum=31).variants),
            (16, 32),
        )
        self.assertEqual(
            plan(minimum=2**62, maximum=2**62, slots=1).select(2**62).capacity,
            2**62,
        )

    def test_bad_facts_and_binding_rejected(self):
        for args in [
            dict(minimum=0),
            dict(minimum=514),
            dict(maximum=-1),
            dict(slots=0),
            dict(maximum=2**62 + 1),
            dict(slots=2**63 - 1),
            dict(uniform=1),
            dict(minimum=True),
        ]:
            with (
                self.subTest(args=args),
                self.assertRaises((RuntimeError, ValueError)),
            ):
                plan(**args)
        family = plan()
        for bound in [-1, 514, True, 1.0]:
            with self.assertRaises(ValueError):
                family.select(bound)
        for slot in [-1, 2, True, 1.0]:
            with self.assertRaises(ValueError):
                family.variants[0].control_domain(slot)

    def test_iteration_domain_keeps_a_safe_general_fallback(self):
        family = plan(maximum=4096, iteration=8)
        self.assertEqual(
            tuple(v.capacity for v in family.variants), (1, 2, 4, 8, 4096)
        )
        self.assertEqual(sum(v.capacity for v in family.variants), 4111)
        self.assertEqual(family.iteration_bound, 8)
        for bound, capacity in [
            (0, 1),
            (3, 4),
            (8, 8),
            (9, 4096),
            (2048, 4096),
            (4096, 4096),
        ]:
            self.assertEqual(family.select(bound).capacity, capacity)
        self.assertEqual(
            [v.control_domain(s) for v in family.variants for s in range(2)],
            list(range(10)),
        )
        self.assertEqual(
            sum(
                v.capacity for v in plan(maximum=4096, iteration=256).variants
            ),
            4607,
        )
        self.assertEqual(
            tuple(v.capacity for v in plan(iteration=2**63 - 1).variants),
            tuple(v.capacity for v in plan().variants),
        )
        self.assertEqual(
            tuple(
                v.capacity
                for v in plan(
                    maximum=4096, iteration=8, enabled=False
                ).variants
            ),
            (4096,),
        )
        self.assertEqual(
            tuple(
                v.capacity
                for v in plan(
                    maximum=4096, iteration=8, uniform=False
                ).variants
            ),
            (4096,),
        )
        for args in [
            dict(iteration=0),
            dict(iteration=-1),
            dict(minimum=4, iteration=2),
            dict(iteration=True),
            dict(iteration=1.5),
            dict(iteration=2**63),
        ]:
            with (
                self.subTest(args=args),
                self.assertRaises((RuntimeError, ValueError)),
            ):
                plan(**args)


if __name__ == "__main__":
    unittest.main()
