"""Phase 4 risk-output schema: no bare probabilities, threshold-conditioned,
orientation-correct, abstain-aware."""
from __future__ import annotations

import unittest

from model.risk import build_risk_output, to_risk_orientation


class TestRiskOutput(unittest.TestCase):
    def test_orientation(self) -> None:
        self.assertEqual(to_risk_orientation(3.0, "higher_bad"), 3.0)
        self.assertEqual(to_risk_orientation(3.0, "lower_bad"), -3.0)

    def test_higher_bad_flags_above_threshold(self) -> None:
        out = build_risk_output(assay_name="HIC", raw_prediction=12.0,
                                endpoint_direction="higher_bad", threshold=10.0)
        self.assertEqual(out["decision"], "flag_failure_risk")
        self.assertGreater(out["risk_decision_margin"], 0)
        self.assertIsNone(out["calibrated_decision_score"])  # no calibrator

    def test_lower_bad_flips_correctly(self) -> None:
        # SEC %monomer: LOW is bad. A low value must flag as risk.
        out = build_risk_output(assay_name="SEC", raw_prediction=80.0,
                                endpoint_direction="lower_bad", threshold=90.0)
        self.assertEqual(out["decision"], "flag_failure_risk")
        # and a high (good) value passes
        ok = build_risk_output(assay_name="SEC", raw_prediction=99.0,
                               endpoint_direction="lower_bad", threshold=90.0)
        self.assertEqual(ok["decision"], "pass")

    def test_calibrator_used_when_present(self) -> None:
        out = build_risk_output(assay_name="ACSINS", raw_prediction=5.0,
                                endpoint_direction="higher_bad", threshold=4.0,
                                calibrator=lambda x: 0.5 + 0.05 * x)
        self.assertAlmostEqual(out["calibrated_decision_score"], 0.75, places=6)

    def test_abstain_carries_provenance_and_no_score(self) -> None:
        out = build_risk_output(assay_name="HIC", raw_prediction=12.0,
                                endpoint_direction="higher_bad", threshold=10.0,
                                abstain=True, abstain_reason="VHH format shift",
                                pair_id="p1", identity_component_id="c1", split_group="s1")
        self.assertEqual(out["decision"], "abstain")
        self.assertEqual(out["decision_reason"], "VHH format shift")
        self.assertIsNone(out["calibrated_decision_score"])
        self.assertEqual(out["pair_id"], "p1")
        self.assertEqual(out["identity_component_id"], "c1")
        self.assertEqual(out["split_group"], "s1")

    def test_never_returns_bare_probability_key(self) -> None:
        out = build_risk_output(assay_name="HIC", raw_prediction=1.0,
                                endpoint_direction="higher_bad", threshold=0.0)
        self.assertNotIn("failure_probability", out)
        self.assertIn("risk_score_global", out)


if __name__ == "__main__":
    unittest.main()
