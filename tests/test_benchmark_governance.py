from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model.benchmark_governance import (
    PolicyError,
    build_status,
    load_policy,
    read_json_pointer,
    validate_policy,
)


def minimal_policy() -> dict:
    return {
        "schema_version": "1",
        "metric_definitions": {
            "primary": "macro_source_roc_auc",
            "required": ["source_roc_auc", "source_pr_auc"],
        },
        "properties": {
            name: {
                "source_roles": {
                    "development": [{"name": f"{name}_dev", "prior_use": True}],
                    "locked_validation": [],
                    "final": [{"name": "gdpa3", "prior_use": False}],
                },
                "baselines": [],
                "count_evidence": [],
            }
            for name in ("polyreactivity", "aggregation", "hydrophobicity")
        },
        "frozen_artifacts": [],
    }


class TestPolicyValidation(unittest.TestCase):
    def test_project_policy_is_valid_and_gdpa3_is_unspent_final_only(self) -> None:
        policy = load_policy(Path("tools/benchmark_policy.json"))
        validate_policy(policy)
        for property_policy in policy["properties"].values():
            final = property_policy["source_roles"]["final"]
            self.assertEqual(final, [{"name": "gdpa3", "prior_use": False}])
            self.assertNotIn(
                "gdpa3",
                {
                    source["name"]
                    for role in ("development", "locked_validation")
                    for source in property_policy["source_roles"][role]
                },
            )

    def test_rejects_a_source_in_multiple_roles(self) -> None:
        policy = minimal_policy()
        policy["properties"]["polyreactivity"]["source_roles"]["locked_validation"] = [
            {"name": "polyreactivity_dev", "prior_use": False}
        ]
        with self.assertRaisesRegex(PolicyError, "multiple roles"):
            validate_policy(policy)

    def test_rejects_a_final_source_marked_as_previously_used(self) -> None:
        policy = minimal_policy()
        policy["properties"]["polyreactivity"]["source_roles"]["final"][0]["prior_use"] = True
        with self.assertRaisesRegex(PolicyError, "final.*unspent"):
            validate_policy(policy)

    def test_load_policy_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(PolicyError, "valid JSON"):
                load_policy(path)


class TestEvidenceStatus(unittest.TestCase):
    def test_json_pointer_reads_nested_values_and_rejects_missing_fields(self) -> None:
        payload = {"a": {"b": [10, {"c": 0.625}]}}
        self.assertEqual(read_json_pointer(payload, "/a/b/1/c"), 0.625)
        with self.assertRaisesRegex(PolicyError, "JSON pointer"):
            read_json_pointer(payload, "/a/missing")

    def test_status_resolves_values_hashes_and_streamed_csv_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.json"
            rows = root / "rows.csv"
            artifact = root / "artifact.bin"
            evidence.write_text(
                '{"score": 0.625, "counts": {"heldout": {"test_n": 97}}}\n',
                encoding="utf-8",
            )
            rows.write_text("source,label\na,0\na,1\nb,1\n", encoding="utf-8")
            artifact.write_bytes(b"frozen")
            policy = minimal_policy()
            prop = policy["properties"]["polyreactivity"]
            prop["baselines"] = [
                {
                    "name": "hard_transfer",
                    "metric": "macro_source_roc_auc",
                    "path": "evidence.json",
                    "json_pointer": "/score",
                    "scope": "test",
                }
            ]
            prop["count_evidence"] = [
                {
                    "name": "rows",
                    "kind": "csv_group_count",
                    "path": "rows.csv",
                    "column": "source",
                }
            ]
            policy["frozen_artifacts"] = [{"kind": "test", "path": "artifact.bin"}]

            report = build_status(policy, repo_root=root, ledger_path=root / "uses.jsonl")

            baseline = report["properties"]["polyreactivity"]["baselines"][0]
            self.assertEqual(baseline["value"], 0.625)
            self.assertEqual(baseline["status"], "ok")
            self.assertEqual(
                report["properties"]["polyreactivity"]["counts"][0]["value"],
                {"a": 2, "b": 1},
            )
            self.assertEqual(report["frozen_artifacts"][0]["size_bytes"], 6)
            self.assertEqual(len(report["frozen_artifacts"][0]["sha256"]), 64)

    def test_missing_evidence_is_reported_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = minimal_policy()
            policy["properties"]["hydrophobicity"]["baselines"] = [
                {
                    "name": "missing",
                    "metric": "roc_auc",
                    "path": "absent.json",
                    "json_pointer": "/auc",
                    "scope": "test",
                }
            ]
            report = build_status(policy, repo_root=root, ledger_path=root / "uses.jsonl")
            item = report["properties"]["hydrophobicity"]["baselines"][0]
            self.assertEqual(item["status"], "missing")
            self.assertIsNone(item["value"])
            self.assertIn("absent.json", item["error"])


if __name__ == "__main__":
    unittest.main()
