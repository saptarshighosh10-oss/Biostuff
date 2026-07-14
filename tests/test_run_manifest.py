from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.run_manifest import RunLedger, file_sha256


class TestRunLedger(unittest.TestCase):
    def test_run_id_is_stable_and_stage_hashes_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_file = root / "input.json"
            output_file = root / "output.json"
            manifest = root / "run_manifest.json"
            input_file.write_text('{"a": 1}\n', encoding="utf-8")

            first = RunLedger(manifest, {"mode": "quick", "seed": 42})
            first.start("phase1", inputs=[input_file], outputs=[output_file])
            output_file.write_text('{"ok": true}\n', encoding="utf-8")
            first.complete("phase1", outputs=[output_file])

            second = RunLedger(manifest, {"mode": "quick", "seed": 42})
            self.assertEqual(first.run_id, second.run_id)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["stages"]["phase1"]["status"], "complete")
            self.assertEqual(payload["stages"]["phase1"]["output_hashes"][str(output_file)], file_sha256(output_file))

    def test_changed_config_starts_a_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "run_manifest.json"
            first = RunLedger(manifest, {"mode": "quick"})
            second = RunLedger(manifest, {"mode": "full"})
            self.assertNotEqual(first.run_id, second.run_id)


if __name__ == "__main__":
    unittest.main()
