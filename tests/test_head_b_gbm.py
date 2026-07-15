"""Orchestration test for the Head B driver: proves rows flow through grouped
folds → stub regressor → per-assay metrics with no group leakage, stdlib-only
(no torch/lightgbm/sklearn). The metric math itself is covered by test_metrics."""
from __future__ import annotations

import unittest

from model.head_b_gbm import build_feature_vector, train_head_b, evaluate_assay

BASE = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYAD"


class SumRegressor:
    """Predicts the sum of the feature vector — deterministic, no deps."""
    def fit(self, X, y):  # noqa: D401
        return self
    def predict(self, X):
        return [sum(x) for x in X]


def _mk_rows(assay: str, n: int, groups: int) -> list[dict]:
    rows = []
    for i in range(n):
        # distinct sequence per row (swap one residue), spread across `groups`
        seq = BASE[:20] + "ACDEFGHIKLMNPQRSTVWY"[i % 20] + BASE[21:]
        row = {
            "variant_sequence": seq,
            "assay_metric": assay,
            "assay_family": "aggregation",
            "endpoint_direction": "higher_bad",
            "study_id": f"study{i % groups}",
            "molecule_id": f"mol{i}",
        }
        # engineer the label so a sum-predicting regressor is perfect → spearman 1.0
        row["endpoint_value"] = sum(build_feature_vector(row, use_plm=False))
        rows.append(row)
    return rows


class TestHeadBDriver(unittest.TestCase):
    def test_end_to_end_report_structure_and_signal(self) -> None:
        rows = _mk_rows("HIC", n=30, groups=6) + _mk_rows("ACSINS", n=30, groups=6)
        report = train_head_b(rows, use_plm=False, regressor_factory=lambda: SumRegressor(), n_splits=3)

        self.assertEqual(report["head"], "antibody_aggregation")
        self.assertIn("HIC", report["assays"])
        self.assertIn("ACSINS", report["assays"])
        for assay in ("HIC", "ACSINS"):
            a = report["assays"][assay]
            self.assertNotIn("abstain", a)
            self.assertGreaterEqual(a["n_groups"], 2)
            # engineered perfect signal → Spearman ~1.0
            self.assertGreater(a["spearman"], 0.99)
            self.assertEqual(a["features"], "biophysical_only")
            self.assertEqual(len(a["spearman_ci95"]), 2)

    def test_abstains_below_min_rows(self) -> None:
        rows = _mk_rows("HIC", n=5, groups=3)
        out = evaluate_assay(rows, use_plm=False, regressor_factory=lambda: SumRegressor())
        self.assertTrue(out.get("abstain"))

    def test_biophysical_fallback_without_plm_cache(self) -> None:
        # use_plm=True but no cached embeddings → degrades to biophysical, no crash.
        row = {"variant_sequence": BASE, "mutations": []}
        vec = build_feature_vector(row, use_plm=True)
        self.assertTrue(all(isinstance(x, (int, float)) for x in vec))
        self.assertGreater(len(vec), 0)

    def test_non_target_rows_are_excluded_from_head_b(self) -> None:
        rows = _mk_rows("HIC", n=30, groups=6)
        rows += [{**rows[0], "assay_metric": "Tm", "assay_family": "thermostability"}]
        report = train_head_b(rows, use_plm=False, regressor_factory=lambda: SumRegressor(), n_splits=3)
        self.assertEqual(report["n_rows"], 30)
        self.assertEqual(report["excluded_non_target_rows"], 1)


if __name__ == "__main__":
    unittest.main()
