from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data.thera_sabdab import load_thera_sabdab

VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQSIS"


def _write_thera_csv(tmp: Path) -> None:
    # Real Thera-SAbDab header (verified): note the moving date suffix on
    # Highest_Clin_Trial and the bispec + Target/Year Proposed columns.
    (tmp / "TheraSAbDab_SeqStruc_OnlineDownload.csv").write_text(
        "Therapeutic,Format,CH1 Isotype,VD LC,Highest_Clin_Trial (Feb '25),"
        "Est. Status,Heavy Sequence,Light Sequence,"
        "Heavy Sequence (if bispec),Light Sequence (if bispec),Target,Year Proposed\n"
        # paired mAb
        f"adalimumab,Whole mAb,IgG1,Kappa,Approved,Active,{VH},{VL},,,TNF,2001\n"
        # VHH / single-domain: Light Sequence blank -> vl_sequence None
        f"caplacizumab,VHH,NA,NA,Approved,Active,{VH}K,,,,VWF,2016\n"
        # no released heavy sequence -> skipped
        "mysterymab,Whole mAb,IgG4,Lambda,Phase-2,Active,,,,,IL6,2020\n",
        encoding="utf-8",
    )


class TestTheraSabdab(unittest.TestCase):
    def test_loads_vh_vl_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _write_thera_csv(tmp)
            rows = load_thera_sabdab(local_dir=tmp)

        # 3 input rows; mysterymab (no VH) skipped -> 2 rows.
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["group_id"] for r in rows}, {"adalimumab", "caplacizumab"})

        for r in rows:
            self.assertEqual(r["variant_sequence"], r["vh_sequence"])  # variant == VH
            self.assertTrue(r["vh_sequence"])                          # VH populated
            self.assertEqual(r["source"], "thera_sabdab")
            self.assertEqual(r["supervision_status"], "background_ood")
            self.assertEqual(r["study_id"], "thera_sabdab")

        ada = next(r for r in rows if r["group_id"] == "adalimumab")
        self.assertEqual(ada["vl_sequence"], VL)                       # paired VL populated
        self.assertEqual(ada["clinical_phase"], "Approved")            # moving-date col matched
        self.assertEqual(ada["format"], "Whole mAb")
        self.assertEqual(ada["target"], "TNF")

        cap = next(r for r in rows if r["group_id"] == "caplacizumab")
        self.assertIsNone(cap["vl_sequence"])                          # VHH: no light chain
        self.assertEqual(cap["format"], "VHH")

    def test_max_sequences_caps(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _write_thera_csv(tmp)
            rows = load_thera_sabdab(local_dir=tmp, max_sequences=1)
        self.assertEqual(len(rows), 1)

    def test_missing_file_raises_helpful_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_thera_sabdab(local_dir=d)
        msg = str(ctx.exception)
        self.assertIn("TheraSAbDab", msg)
        self.assertIn("never downloads", msg)


if __name__ == "__main__":
    unittest.main()
