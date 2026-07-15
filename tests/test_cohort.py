"""Cohort assembler: FLAb loads, GDPa is skipped gracefully when absent, and
VH/VL pairing is attached deterministically."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data.cohort import build_antibody_cohort, _attach_pairing, is_head_b_aggregation_row


class TestCohort(unittest.TestCase):
    def test_pairing_attached_deterministically(self) -> None:
        rows = [
            {"vh_sequence": "EVQLVES", "vl_sequence": "DIQMTQS", "source": "gdpa"},
            {"variant_sequence": "EVQLVES", "source": "flab"},  # single chain
        ]
        _attach_pairing(rows)
        self.assertIsNotNone(rows[0]["pair_id"])
        self.assertIsNone(rows[1]["pair_id"])
        self.assertIn("single_chain_only", rows[1]["feature_flags"])
        # deterministic: same VH/VL → same pair_id
        again = [{"vh_sequence": "EVQLVES", "vl_sequence": "DIQMTQS"}]
        _attach_pairing(again)
        self.assertEqual(again[0]["pair_id"], rows[0]["pair_id"])

    def test_gdpa_absent_is_skipped_not_fatal(self) -> None:
        # point gdpa_dir at an empty tmp dir → load_gdpa raises FileNotFoundError,
        # cohort must still return FLAb rows.
        with tempfile.TemporaryDirectory() as d:
            rows = build_antibody_cohort(include_gdpa=True, gdpa_dir=d)
        self.assertGreater(len(rows), 0)
        self.assertTrue(all("pair_id" in r for r in rows))
        # FLAb rows are single-chain
        self.assertTrue(any("single_chain_only" in r.get("feature_flags", []) for r in rows))

    def test_head_b_scope_excludes_non_aggregation_endpoints(self) -> None:
        self.assertTrue(is_head_b_aggregation_row({"assay_family": "aggregation", "assay_metric": "SEC"}))
        self.assertTrue(is_head_b_aggregation_row({"assay_family": "expression", "assay_metric": "HIC"}))
        self.assertFalse(is_head_b_aggregation_row({"assay_family": "aggregation", "assay_metric": "pI"}))
        self.assertFalse(is_head_b_aggregation_row({"assay_family": "thermostability", "assay_metric": "Tm"}))


if __name__ == "__main__":
    unittest.main()
