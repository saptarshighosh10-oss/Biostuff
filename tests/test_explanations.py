from __future__ import annotations

import unittest

from model.explanations import explain_failure


class TestFailureExplanation(unittest.TestCase):
    def test_explanation_is_evidence_bounded(self) -> None:
        result = explain_failure(
            mutations=[(4, "A", "F")],
            features={
                "mutation_hits_hotspot": 1.0,
                "mean_hydrophobicity_delta": 1.0,
                "total_charge_delta": 0.0,
                "hydrophobic_patch_score": 3.0,
            },
            probability=0.91,
            confidence="high",
            top_features=[{"feature": "hydrophobic_patch_score"}],
        )
        self.assertEqual(result["certainty"], "inferred")
        self.assertIn("A5F", result["evidence"][0])
        self.assertIn("not a proven cause", result["description"])


if __name__ == "__main__":
    unittest.main()
