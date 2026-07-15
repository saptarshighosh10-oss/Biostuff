"""GDPa3 single-shot gating: fail-closed on missing file, hash mismatch, and
repeated completed attempts; happy path writes report + ledgers 'completed'."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from model import evaluate_gdpa3 as ev


def _fake_predict(rows):
    return [{"id": r.get("antibody_name", i), "risk_score_global": float(i)}
            for i, r in enumerate(rows)]


class TestGdpa3SingleShot(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.ledger = self.tmp / "gdpa3_attempts.jsonl"
        self.data = self.tmp / "GDPa3.csv"
        self.data.write_text("antibody_name,HIC\nab1,10\nab2,12\n")
        self.good_hash = hashlib.sha256(self.data.read_bytes()).hexdigest()
        self.out = self.tmp / "gdpa3_report.json"

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, expected_hash):
        return ev.run_once(gdpa3_file=self.data, expected_data_hash=expected_hash,
                           out_path=self.out, predict_fn=_fake_predict, ledger_path=self.ledger)

    def test_missing_file_fails_closed(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ev.run_once(gdpa3_file=self.tmp / "nope.csv", expected_data_hash="x",
                        out_path=self.out, predict_fn=_fake_predict, ledger_path=self.ledger)
        states = [e["state"] for e in ev._read_ledger(self.ledger)]
        self.assertIn("aborted", states)
        self.assertNotIn("completed", states)

    def test_hash_mismatch_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self._run("deadbeef")
        self.assertFalse(ev.has_completed_attempt(self.ledger))

    def test_happy_path_completes_and_writes(self) -> None:
        report = self._run(self.good_hash)
        self.assertEqual(report["external_set"], "gdpa3")
        self.assertEqual(report["n_rows"], 2)
        self.assertTrue(self.out.exists())
        self.assertTrue(ev.has_completed_attempt(self.ledger))

    def test_second_run_refused_after_completion(self) -> None:
        self._run(self.good_hash)
        with self.assertRaises(RuntimeError):
            self._run(self.good_hash)


if __name__ == "__main__":
    unittest.main()
