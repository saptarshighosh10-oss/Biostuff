from __future__ import annotations

import math
import unittest

from model.metrics import (
    calibration_at_threshold,
    group_bootstrap_ci,
    mae,
    spearman_rho,
    topk_enrichment,
)


class TestSpearman(unittest.TestCase):
    def test_monotonic_increasing_is_one(self) -> None:
        self.assertAlmostEqual(spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), 1.0)

    def test_reversed_is_minus_one(self) -> None:
        self.assertAlmostEqual(spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]), -1.0)

    def test_tie_case_known_value(self) -> None:
        # y_pred has a tie: ranks -> [1, 2.5, 2.5, 4]; closed-form rho = 4.5/sqrt(22.5)
        rho = spearman_rho([1, 2, 3, 4], [1, 2, 2, 4])
        self.assertAlmostEqual(rho, 4.5 / math.sqrt(22.5))

    def test_degenerate_returns_zero(self) -> None:
        self.assertEqual(spearman_rho([1], [1]), 0.0)          # n < 2
        self.assertEqual(spearman_rho([5, 5, 5], [1, 2, 3]), 0.0)  # zero variance


class TestMae(unittest.TestCase):
    def test_known_pair(self) -> None:
        # |1-1.5| + |2-2| + |3-1| = 0.5 + 0 + 2 = 2.5 ; /3
        self.assertAlmostEqual(mae([1, 2, 3], [1.5, 2, 1]), 2.5 / 3)


class TestTopKEnrichment(unittest.TestCase):
    def test_all_positives_top_ranked_equals_inverse_base_rate(self) -> None:
        # 10 positives, 90 negatives; positives carry the highest scores.
        y = [1] * 10 + [0] * 90
        scores = [100 - i for i in range(100)]  # index 0..9 are the top scores
        base_rate = 10 / 100
        # k_fraction 0.10 -> top 10 captures all 10 positives -> recall 1.0
        self.assertAlmostEqual(topk_enrichment(y, scores, 0.10), 1.0 / base_rate)

    def test_no_positives_returns_zero(self) -> None:
        self.assertEqual(topk_enrichment([0, 0, 0], [3, 2, 1], 0.10), 0.0)


class TestGroupBootstrapCI(unittest.TestCase):
    def _data(self) -> dict[str, list[tuple[float, float]]]:
        return {
            "study_a": [(1.0, 1.2), (2.0, 1.8), (3.0, 3.1)],
            "study_b": [(4.0, 5.0), (5.0, 4.4), (6.0, 6.6)],
            "study_c": [(2.0, 2.5), (3.0, 2.9), (7.0, 6.0)],
            "study_d": [(1.0, 0.4), (8.0, 8.9), (2.0, 2.2)],
        }
    def test_ci_brackets_point_and_is_reproducible(self) -> None:
        point, lo, hi = group_bootstrap_ci(self._data(), mae, n_boot=500, seed=42)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)
        # deterministic: identical seed -> identical interval
        again = group_bootstrap_ci(self._data(), mae, n_boot=500, seed=42)
        self.assertEqual((point, lo, hi), again)


class TestCalibration(unittest.TestCase):
    def test_abstains_below_min_count(self) -> None:
        out = calibration_at_threshold([1, 0, 1], [0.9, 0.1, 0.8], 0.5, min_count=20)
        self.assertTrue(out.get("abstain"))
        self.assertIn("min_count", out["reason"])

    def test_reports_rates_above_min_count(self) -> None:
        y = [1, 0] * 15           # n = 30
        probs = [0.9, 0.1] * 15   # positives all score 0.9, negatives 0.1
        out = calibration_at_threshold(y, probs, 0.5, min_count=20)
        self.assertEqual(out["n"], 30)
        self.assertAlmostEqual(out["predicted_positive_rate"], 15 / 30)
        self.assertAlmostEqual(out["observed_positive_rate_in_predicted_pos"], 1.0)


if __name__ == "__main__":
    unittest.main()
