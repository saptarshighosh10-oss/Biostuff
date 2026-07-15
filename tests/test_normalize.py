import unittest

from data.contract import NormalizedRow, canonical_sequence
from data.normalize import normalize_row, normalize_rows


FLAB_ROW = {
    "variant_sequence": "evqlv esggg",  # lowercase + spaces -> canonicalized
    "label": "confirmed_failure",
    "score": 0.12,
    "score_col": "titer",
    "assay_metric": "titer",
    "endpoint_value": 0.12,
    "endpoint_direction": "lower_bad",
    "dataset": "flab_titer.csv",
}

ABDEV_ROW = {
    "sequence": "QVQLVQSGAEVKK",
    "dataset": "abdev_pool",
}


class TestNormalize(unittest.TestCase):
    def test_flab_supervised_row(self):
        row = normalize_row(FLAB_ROW, "flab")
        self.assertIsInstance(row, NormalizedRow)
        self.assertEqual(row.supervision_status, "supervised")
        self.assertEqual(row.sequence, canonical_sequence(FLAB_ROW["variant_sequence"]))
        self.assertEqual(row.endpoint_direction, "lower_bad")
        self.assertEqual(row.binary_label, 1)
        # supervised requires these; adapter fills deterministic fallbacks
        self.assertIsInstance(row.endpoint_unit, str)
        self.assertTrue(row.endpoint_unit)
        self.assertIsInstance(row.label_threshold, float)
        self.assertIsInstance(row.endpoint_value, float)
        # sha256 hex
        self.assertEqual(len(row.sequence_hash), 64)

    def test_abdev_background_row(self):
        row = normalize_row(ABDEV_ROW, "abdev")
        self.assertIsInstance(row, NormalizedRow)
        self.assertEqual(row.supervision_status, "background_ood")
        self.assertEqual(row.source, "abdev")
        self.assertIsInstance(row.record_id, str)
        self.assertTrue(row.record_id)

    def test_sequence_hash_deterministic(self):
        a = normalize_row(FLAB_ROW, "flab")
        b = normalize_row(FLAB_ROW, "flab")
        self.assertEqual(a.sequence_hash, b.sequence_hash)
        self.assertEqual(a.record_id, b.record_id)

    def test_source_version_and_assay_conditions_survive_normalization(self):
        row = dict(FLAB_ROW)
        row.update({
            "source_version": "flab-snapshot-2026-07",
            "concentration": "10 mg/mL",
            "temperature": "25 C",
        })
        normalized = normalize_row(row, "flab")
        self.assertEqual(normalized.source_version, "flab-snapshot-2026-07")
        self.assertEqual(normalized.assay_conditions["temperature"], "25 C")

    def test_invalid_sequences_skipped(self):
        raws = [
            dict(ABDEV_ROW),
            {"sequence": ""},          # empty -> skip
            {"sequence": "XZ123"},     # non-amino-acid -> skip
            {"dataset": "no_seq"},     # missing sequence -> skip
            dict(ABDEV_ROW),
        ]
        rows, skipped = normalize_rows(raws, "abdev")
        self.assertEqual(len(rows), 2)
        self.assertEqual(skipped, 3)


if __name__ == "__main__":
    unittest.main()
