from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data.gdpa import load_gdpa, load_gdpa3

VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQSIS"


def _write_gdpa1_csv(tmp: Path) -> None:
    # GDPa1-style header (verified names): id, VH/VL, 3 numeric assays + 1 blank.
    (tmp / "GDPa1_v1.2_test.csv").write_text(
        "antibody_name,vh_protein_sequence,vl_protein_sequence,AC-SINS_pH7.4,HIC,Tm2,Titer\n"
        f"ab-001,{VH},{VL},0.35,2.545,80.33,\n"      # Titer blank → skipped
        f"ab-002,{VH}K,{VL}K,0.10,2.705,85.03,114.75\n",
        encoding="utf-8",
    )


class TestGdpa(unittest.TestCase):
    def test_vh_vl_and_one_row_per_assay(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _write_gdpa1_csv(tmp)
            rows = load_gdpa("gdpa1", local_dir=tmp)

        # ab-001: 3 assays (Titer blank skipped); ab-002: 4 assays → 7 rows total.
        self.assertEqual(len(rows), 7)

        ab001 = [r for r in rows if r["group_id"] == "ab-001"]
        self.assertEqual(len(ab001), 3)  # one antibody expanded into N assay rows
        self.assertEqual({r["assay_metric"] for r in ab001}, {"AC-SINS_pH7.4", "HIC", "Tm2"})

        for r in rows:
            self.assertEqual(r["variant_sequence"], r["vh_sequence"])  # variant == VH
            self.assertTrue(r["vh_sequence"] and r["vl_sequence"])     # VH/VL populated
            self.assertEqual(r["source"], "gdpa")
            self.assertEqual(r["study_id"], "gdpa1")
            self.assertIsInstance(r["endpoint_value"], float)

        hic = next(r for r in ab001 if r["assay_metric"] == "HIC")
        self.assertEqual(hic["endpoint_value"], 2.545)
        self.assertEqual(hic["endpoint_direction"], "higher_bad")
        tm2 = next(r for r in ab001 if r["assay_metric"] == "Tm2")
        self.assertEqual(tm2["endpoint_direction"], "lower_bad")

    def test_gdpa3_is_hard_gated(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            load_gdpa3()
        self.assertIn("frozen", str(ctx.exception).lower())

    def test_missing_file_raises_helpful_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_gdpa("gdpa1", local_dir=d)
        msg = str(ctx.exception)
        self.assertIn("GDPa1", msg)
        self.assertIn("loader never downloads", msg)


if __name__ == "__main__":
    unittest.main()
