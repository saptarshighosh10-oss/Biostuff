from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model.local_explainer import explain_selected


class TestLocalExplainer(unittest.TestCase):
    @patch("model.local_explainer._generate", side_effect=["r0|Likely hydrophobic.\nr1|Likely stable.", "r0|PASS\nr1|PASS"])
    def test_batches_selected_rows(self, generate) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "in.json", root / "out.json"
            source.write_text(json.dumps({"results": [
                {"failure_probability": 0.9, "failure_explanation": {"evidence": ["x"]}},
                {"failure_probability": 0.2, "failure_explanation": {"evidence": ["y"]}},
            ]}))
            summary = explain_selected(str(source), str(target), top_n=1, batch_size=1)
            result = json.loads(target.read_text())["results"]
            self.assertEqual(summary["selected"], 1)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(result[0]["local_description_check"], "PASS")
            self.assertNotIn("local_description", result[1])

    @patch("model.local_explainer._generate", side_effect=["r0|one\nr1|two", "r0|PASS\nr1|PASS"])
    def test_all_mode_batches_every_row(self, generate) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "in.json", root / "out.json"
            source.write_text(json.dumps({"results": [
                {"failure_probability": 0.1, "failure_explanation": {}},
                {"failure_probability": 0.2, "failure_explanation": {}},
            ]}))
            summary = explain_selected(str(source), str(target), all_rows=True, batch_size=32)
            self.assertEqual(summary["selected"], 2)
            self.assertEqual(summary["batches"], 1)
            self.assertEqual(generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
