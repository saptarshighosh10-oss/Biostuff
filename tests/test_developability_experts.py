from __future__ import annotations

import unittest

from model.developability_experts import (
    candidate_combinations,
    choose_specialist,
    positional_identity,
    select_blend,
)


class TestDevelopabilityExperts(unittest.TestCase):
    def test_identity_requires_equal_nonempty_lengths(self) -> None:
        self.assertEqual(positional_identity("AAAA", "AAAA"), 1.0)
        self.assertEqual(positional_identity("AAAA", "AAAT"), 0.75)
        self.assertEqual(positional_identity("AAAA", "AAA"), 0.0)
        self.assertEqual(positional_identity("", ""), 0.0)

    def test_router_uses_identity_then_stable_name(self) -> None:
        references = {"zeta": ["AAAAC"], "alpha": ["AAAAT"]}
        self.assertIsNone(choose_specialist("AAAAG", references, threshold=0.90))
        self.assertEqual(
            choose_specialist("AAAAT", references, threshold=0.80),
            {"name": "alpha", "identity": 1.0},
        )
        self.assertEqual(
            choose_specialist("AAAAG", references, threshold=0.80),
            {"name": "alpha", "identity": 0.8},
        )

    def test_candidate_combinations_are_smallest_first_and_stable(self) -> None:
        self.assertEqual(
            candidate_combinations(["b", "a", "a"]),
            [("a",), ("b",), ("a", "b")],
        )

    def test_blend_selection_uses_unweighted_macro_sources(self) -> None:
        predictions = {
            "source_a": {
                "weak": {"a0": (0, 0.6), "a1": (1, 0.4)},
                "strong": {"a0": (0, 0.1), "a1": (1, 0.9)},
            },
            "source_b": {
                "weak": {"b0": (0, 0.7), "b1": (1, 0.3)},
                "strong": {"b0": (0, 0.2), "b1": (1, 0.8)},
            },
        }
        result = select_blend(predictions)
        self.assertEqual(result["members"], ["strong"])
        self.assertEqual(result["macro_source_auc"], 1.0)
        self.assertEqual(result["source_auc"], {"source_a": 1.0, "source_b": 1.0})

    def test_blend_selection_rejects_low_coverage_candidate(self) -> None:
        predictions = {
            "source": {
                "complete": {str(i): (i % 2, float(i % 2)) for i in range(10)},
                "partial": {str(i): (i % 2, float(i % 2)) for i in range(7)},
            }
        }
        result = select_blend(predictions, min_coverage=0.80)
        self.assertEqual(result["eligible_candidates"], ["complete"])


if __name__ == "__main__":
    unittest.main()
