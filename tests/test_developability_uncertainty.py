from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from model.developability_uncertainty import consolidate


class TestDevelopabilityUncertainty(unittest.TestCase):
    def test_consolidates_seeds_and_abstains_out_of_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "predictions.csv"
            fields = ("experiment", "seed", "property", "feature_mode", "held_source",
                      "assay", "record_id", "label", "raw_score", "score", "ood_distance",
                      "applicability", "nearest_training_source")
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for seed, score in ((42, 0.2), (43, 0.4), (44, 0.6)):
                    writer.writerow({"experiment": "e", "seed": seed, "property": "p",
                                     "feature_mode": "f", "held_source": "s", "assay": "a",
                                     "record_id": "r", "label": 1, "raw_score": score,
                                     "score": score, "ood_distance": 2.0,
                                     "applicability": "out_of_domain",
                                     "nearest_training_source": "train"})
            output = root / "uncertainty.csv"
            report = consolidate(source, output)
            with output.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["seed_count"], "3")
            self.assertAlmostEqual(float(row["risk_score"]), 0.4)
            self.assertEqual(row["abstain"], "True")
            self.assertIn("out_of_domain", row["abstention_reason"])
            self.assertEqual(report["unique_prediction_records"], 1)


if __name__ == "__main__":
    unittest.main()
