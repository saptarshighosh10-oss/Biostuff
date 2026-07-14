from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model.local_explainer import explain_selected


class TestLocalExplainer(unittest.TestCase):
    @patch("model.local_explainer._generate", side_effect=["Likely hydrophobic.", "PASS"])
    def test_only_selected_rows_call_local_models(self, _generate) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "in.json"
            target = root / "out.json"
            source.write_text(json.dumps({"results": [
                {"failure_probability": 0.9, "failure_explanation": {"evidence": ["x"]}},
                {"failure_probability": 0.2, "failure_explanation": {"evidence": ["y"]}},
            ]}))
            summary = explain_selected(str(source), str(target), top_n=1)
            result = json.loads(target.read_text())["results"]
            self.assertEqual(summary["selected"], 1)
            self.assertIn("local_description", result[0])
            self.assertNotIn("local_description", result[1])
            self.assertEqual(_generate.call_count, 2)

    @patch("model.local_explainer._generate", side_effect=["one", "PASS", "two", "PASS"])
    def test_all_mode_processes_every_row(self, _generate) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "in.json"
            target = root / "out.json"
            source.write_text(json.dumps({"results": [
                {"failure_probability": 0.1, "failure_explanation": {}},
                {"failure_probability": 0.2, "failure_explanation": {}},
            ]}))
            summary = explain_selected(str(source), str(target), all_rows=True, checkpoint_every=1)
            self.assertEqual(summary["selected"], 2)
            self.assertEqual(_generate.call_count, 4)


if __name__ == "__main__":
    unittest.main()
