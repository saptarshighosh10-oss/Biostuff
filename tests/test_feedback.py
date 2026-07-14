from __future__ import annotations

import unittest

from model.feedback import merge_results


class TestFeedbackMerge(unittest.TestCase):
    def test_conflicting_duplicate_is_quarantined(self) -> None:
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        merged, added, conflicts = merge_results(
            [{"variant_sequence": sequence, "label": "working"}],
            [{"variant_sequence": sequence, "label": "confirmed_failure", "notes": "repeat"}],
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(added, 0)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["existing_label"], "working")

    def test_same_label_duplicate_is_ignored(self) -> None:
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        merged, added, conflicts = merge_results(
            [{"variant_sequence": sequence, "label": "working"}],
            [{"variant_sequence": sequence, "label": "working"}],
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(added, 0)
        self.assertEqual(conflicts, [])


if __name__ == "__main__":
    unittest.main()
