from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from data.oas import load_oas

# Realistic OAS-style unpaired heavy VH sequences (aa).
VH1 = "QIQLVQSGPELKKPGETVKISCKASGYTFTTYGMSWVKQAPGKGLKWMG"
VH2 = "QVQLQQSGAELARPGASVKLSCKASGYTFTSYGISWVKQRTGQGLEWIG"
VH3 = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS"


def _write_oas_gz(path: Path, seqs: list[str]) -> None:
    """Write a tiny OAS-style .csv.gz: JSON metadata line 1, AIRR header line 2."""
    metadata = json.dumps({"Author": "Chen_2020", "Species": "human", "Chain": "Heavy"})
    header = "sequence_alignment_aa,v_call,j_call,locus,Redundancy"
    lines = [metadata, header]
    for i, s in enumerate(seqs):
        lines.append(f"{s},IGHV1-8*01,IGHJ4*02,IGH,{i + 1}")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


class TestOas(unittest.TestCase):
    def test_parses_rows_and_tags_background_ood(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _write_oas_gz(tmp / "SRR123_Heavy_IGHG.csv.gz", [VH1, VH2, VH3])
            rows = load_oas(local_dir=tmp)

        self.assertEqual(len(rows), 3)  # 3 data rows; metadata line NOT counted
        seqs = {r["variant_sequence"] for r in rows}
        self.assertEqual(seqs, {VH1, VH2, VH3})
        for r in rows:
            self.assertEqual(r["source"], "oas")
            self.assertEqual(r["supervision_status"], "background_ood")
            self.assertEqual(r["variant_sequence"], r["vh_sequence"])  # heavy unit
            self.assertIsNone(r["vl_sequence"])                        # unpaired → None
            self.assertEqual(r["study_id"], "Chen_2020")               # from metadata
            self.assertEqual(r["group_id"], r["study_id"])
            self.assertEqual(r["v_call"], "IGHV1-8*01")
            self.assertEqual(r["j_call"], "IGHJ4*02")

    def test_metadata_line_not_treated_as_data(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _write_oas_gz(tmp / "unit.csv.gz", [VH1])
            rows = load_oas(local_dir=tmp)
        # If the JSON metadata line leaked in as a row, variant_sequence would
        # contain a brace or the header token instead of a clean aa sequence.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant_sequence"], VH1)

    def test_max_sequences_caps_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _write_oas_gz(tmp / "unit.csv.gz", [VH1, VH2, VH3])
            rows = load_oas(local_dir=tmp, max_sequences=2)
        self.assertEqual(len(rows), 2)

    def test_missing_dir_raises_filenotfound(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_oas(local_dir=Path(d) / "nope")
        self.assertIn("never downloads", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
