"""Unit tests for AC-SINS ensemble assay-context helpers."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from model.head_b_acsins_ensemble import assay_context_features, run_ensemble


class TestAssayContextFeatures(unittest.TestCase):
    def test_assay_context_features(self) -> None:
        row = {"assay_metric": "AC-SINS pH6 His/Arg deltaLmax", "assay_id": "PBS"}
        self.assertEqual(
            assay_context_features(row),
            [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        )

        row = {"assay_metric": "", "assay_id": "pH7.4 screening PBS assay"}
        self.assertEqual(
            assay_context_features(row),
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )

    def test_run_ensemble_features_include_assay_context(self) -> None:
        with patch("model.head_b_acsins_ensemble._acsins_rows", return_value=[]):
            out = run_ensemble(add_assay_context=True)
            self.assertIn("assay_context", out["features"])

    def _mk_rows(self, studies: int = 2, rows_per_study: int = 60) -> list[dict]:
        rows: list[dict] = []
        for s in range(studies):
            for i in range(rows_per_study):
                rows.append({
                    "variant_sequence": f"SEQ_{s}_{i}",
                    "vh_sequence": f"VH_{s}_{i}",
                    "vl_sequence": f"VL_{s}_{i}",
                    "assay_id": "acsins",
                    "assay_metric": "acsins",
                    "assay_family": "aggregation",
                    "source": "gdpa",
                    "study_id": f"study{s}",
                    "endpoint_value": float(i),
                    "endpoint_direction": "higher_bad",
                })
        return rows

    def test_run_ensemble_paired_ablang_flag_is_passed_once_per_row(self) -> None:
        rows = self._mk_rows()
        with patch("model.head_b_acsins_ensemble._acsins_rows", return_value=rows), \
             patch("model.head_b_acsins_ensemble._ensemble_predict", side_effect=lambda *_args: [0.0] * len(_args[2])), \
             patch("model.head_b_acsins_ensemble.build_feature_vector", return_value=[0.1, 0.2]) as bv:
            out = run_ensemble(add_paired_ablang=True)

        self.assertIn("paired_ablang2", out["features"])
        self.assertEqual(len(rows), len(bv.call_args_list))
        for call in bv.call_args_list:
            self.assertIn("add_paired_ablang", call.kwargs)
            self.assertTrue(call.kwargs["add_paired_ablang"])


if __name__ == "__main__":
    unittest.main()
