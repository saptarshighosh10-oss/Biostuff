from __future__ import annotations

import unittest
from pathlib import Path

from data.flab import load_all_developability_data, load_all_flab_data

FLAB_DIR = Path("data/external/flab")


class TestFlabMultiAssay(unittest.TestCase):
    def test_walks_multiple_assay_families(self) -> None:
        rows = load_all_developability_data(local_dir=FLAB_DIR)
        self.assertTrue(rows, "expected at least one developability row")

        families = {row["assay_family"] for row in rows}
        self.assertGreaterEqual(len(families), 3, f"only found families: {families}")

        for row in rows:
            self.assertTrue(row["variant_sequence"], "empty variant_sequence")
            self.assertIsInstance(row["endpoint_value"], float)
            self.assertIn(row["label"], ("confirmed_failure", "working"))

    def test_load_all_flab_data_still_aggregation_only(self) -> None:
        failures, working = load_all_flab_data()
        rows = failures + working
        self.assertTrue(rows, "expected aggregation rows")
        # legacy loader is scoped to the aggregation snapshot; it tags group_id
        # with the filename, and every file lives under aggregation/
        self.assertTrue(all(r["source"] == "flab" for r in rows))
        self.assertTrue(all(r["group_id"].endswith(".csv") for r in rows))


if __name__ == "__main__":
    unittest.main()
