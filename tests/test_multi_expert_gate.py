from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.multi_expert_gate import evaluate, evaluate_specialists


class TestMultiExpertGate(unittest.TestCase):
    @staticmethod
    def summary(auc: float, source_b: float, enrichment: float = 2.0) -> dict:
        return {
            "mean_macro_source_auc": auc,
            "source_mean_auc": {"a": auc, "b": source_b},
            "mean_enrichment_at_15pct": enrichment,
            "seed_std_auc": 0.01,
            "audits_present": True,
        }

    @patch("tools.multi_expert_gate.evaluate_serving")
    def test_admits_development_gain_but_defers_serving_to_existing_gate(self, serving) -> None:
        serving.return_value = {
            "decisions": {
                "polyreactivity": {
                    "promote": False,
                    "checks": {"locked_validation_source_assigned": False},
                }
            }
        }
        result = evaluate(
            {"polyreactivity": self.summary(0.70, 0.70)},
            {"polyreactivity": self.summary(0.72, 0.71)},
            {"properties": {}},
            {"summaries": [], "experiments": []},
        )["polyreactivity"]
        self.assertTrue(result["experimental_admit"])
        self.assertFalse(result["serving_eligible"])
        self.assertIn("locked_validation_source_assigned", result["serving_reason"])
        serving.assert_called_once()

    @patch("tools.multi_expert_gate.evaluate_serving")
    def test_rejects_one_source_drop_over_three_points(self, serving) -> None:
        serving.return_value = {"decisions": {"polyreactivity": {"promote": False}}}
        result = evaluate(
            {"polyreactivity": self.summary(0.70, 0.70)},
            {"polyreactivity": self.summary(0.72, 0.66)},
            {"properties": {}},
            {"summaries": [], "experiments": []},
        )["polyreactivity"]
        self.assertFalse(result["experimental_admit"])
        self.assertFalse(result["checks"]["no_source_drop_over_0.030"])

    def test_specialist_gate_never_changes_generalist_metrics(self) -> None:
        decisions = evaluate_specialists(
            {"polyreactivity": {
                "lib": {"n": 100, "positives": 50, "mean_auc": 0.90}
            }},
            {"polyreactivity": {"lib": 0.904}},
        )
        self.assertTrue(decisions["polyreactivity"]["lib"]["experimental_admit"])
        self.assertAlmostEqual(decisions["polyreactivity"]["lib"]["delta_auc"], -0.004)


if __name__ == "__main__":
    unittest.main()
