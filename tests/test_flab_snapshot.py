from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data.flab import download_csv, load_dataset


class TestFlabSnapshot(unittest.TestCase):
    def test_local_snapshot_is_checksum_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = "heavy,fitness\nACDEFGHIKLMNPQRSTVWY,1.0\n"
            (root / "sample.csv").write_text(payload, encoding="utf-8")
            digest = hashlib.sha256(payload.encode()).hexdigest()
            (root / "manifest.json").write_text(json.dumps({"files": [{"name": "sample.csv", "sha256": digest}]}), encoding="utf-8")
            self.assertEqual(download_csv("sample.csv", local_dir=root), payload)

    def test_local_snapshot_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.csv").write_text("tampered", encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"files": [{"name": "sample.csv", "sha256": "0" * 64}]}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                download_csv("sample.csv", local_dir=root)

    def test_paired_snapshot_preserves_both_chains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heavy = "ACDEFGHIKLMNPQRSTVWY"
            light = "YWVTSRQPNMLKIHGFEDCA"
            payload = "heavy,light,fitness\n" + "".join(
                f"{heavy},{light},{score}\n" for score in (1.0, 1.1, 1.2, 1.3, 1.4)
            )
            (root / "paired.csv").write_text(payload, encoding="utf-8")
            digest = hashlib.sha256(payload.encode()).hexdigest()
            (root / "manifest.json").write_text(json.dumps({"files": [{"name": "paired.csv", "sha256": digest}]}), encoding="utf-8")
            failures, working = load_dataset("paired.csv", max_rows=10, local_dir=root)
            row = failures[0] if failures else working[0]
            self.assertEqual(row["vh_sequence"], heavy)
            self.assertEqual(row["vl_sequence"], light)
            self.assertEqual(row["variant_sequence"], heavy + light)


if __name__ == "__main__":
    unittest.main()
