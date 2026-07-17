from __future__ import annotations

import unittest

from tools.promotion_gate import evaluate


def policy(locked: bool) -> dict:
    properties = {}
    for name in ("polyreactivity", "aggregation", "hydrophobicity"):
        properties[name] = {
            "source_roles": {
                "development": [],
                "locked_validation": ([{"name": "locked", "prior_use": False}] if locked else []),
                "final": [{"name": "gdpa3", "prior_use": False}],
            },
            "baselines": [], "count_evidence": [],
        }
    return {"schema_version": "1", "metric_definitions": {}, "properties": properties,
            "frozen_artifacts": []}


class TestPromotionGate(unittest.TestCase):
    def test_requires_locked_validation_even_for_high_development_auc(self) -> None:
        experiment = "polyreactivity:best"
        source = {"roc_auc": 0.85, "enrichment_at_15pct": 2.5,
                  "calibration_method": "platt_logistic_train_only"}
        matrix = {
            "summaries": [{"experiment": experiment, "property": "polyreactivity",
                           "test_tier": "paired_fv_supervised", "mean_macro_source_auc": 0.85,
                           "source_mean_auc": {"a": 0.84, "b": 0.86},
                           "seed_macro_source_auc": [0.84, 0.85, 0.86], "seed_std_auc": 0.01,
                           "source_bootstrap_ci95": [0.84, 0.86]}],
            "experiments": [{"experiment": experiment, "per_source": [source, source, source]}],
        }
        report = evaluate(matrix, policy(locked=False))
        self.assertFalse(report["decisions"]["polyreactivity"]["promote"])
        self.assertFalse(report["decisions"]["polyreactivity"]["checks"]["locked_validation_source_assigned"])


if __name__ == "__main__":
    unittest.main()
