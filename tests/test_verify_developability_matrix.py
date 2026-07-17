from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from data.run_manifest import file_sha256
from tools.verify_developability_matrix import verify


class TestVerifyDevelopabilityMatrix(unittest.TestCase):
    def test_recomputes_auc_from_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.csv"
            fields = ("experiment", "seed", "property", "feature_mode", "held_source",
                      "assay", "record_id", "label", "score")
            with predictions.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index, (label, score) in enumerate(((0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9))):
                    writer.writerow({"experiment": "e", "seed": 42, "property": "p",
                                     "feature_mode": "f", "held_source": "s", "assay": "a",
                                     "record_id": index, "label": label, "score": score})
            report = root / "report.json"
            report.write_text(json.dumps({
                "predictions_sha256": file_sha256(predictions),
                "experiments": [{
                    "experiment": "e", "seed": 42,
                    "macro_source_roc_auc": 1.0, "macro_source_pr_auc": 1.0,
                    "per_source": [{"source": "s", "roc_auc": 1.0, "pr_auc": 1.0,
                                    "per_assay": {"a": {"roc_auc": 1.0, "pr_auc": 1.0}}}],
                }],
            }), encoding="utf-8")
            result = verify(report, predictions)
            self.assertTrue(result["verified"])
            self.assertEqual(result["checked_assays"], 1)


if __name__ == "__main__":
    unittest.main()
