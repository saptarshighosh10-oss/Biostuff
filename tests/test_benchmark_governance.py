from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model.benchmark_governance import (
    PolicyError,
    build_status,
    load_policy,
    read_json_pointer,
    read_ledger,
    record_holdout_use,
    validate_policy,
)
from tools.benchmark_status import render_status


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


class TestHoldoutLedger(unittest.TestCase):
    def test_development_use_appends_a_traceable_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "uses.jsonl"
            evidence = root / "trial.json"
            evidence.write_text('{"auc": 0.63}\n', encoding="utf-8")
            record = record_holdout_use(
                minimal_policy(),
                ledger_path=ledger,
                property_name="polyreactivity",
                source="polyreactivity_dev",
                experiment_id="exp-001",
                purpose="fixed development comparison",
                evidence_path=evidence,
            )
            self.assertEqual(record["role"], "development")
            self.assertEqual(len(record["evidence_sha256"]), 64)
            self.assertEqual(read_ledger(ledger), [record])

    def test_locked_use_requires_confirmation_and_is_single_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "uses.jsonl"
            policy = minimal_policy()
            policy["properties"]["polyreactivity"]["source_roles"]["locked_validation"] = [
                {"name": "new_locked", "prior_use": False}
            ]
            evidence = root / "trial.json"
            evidence.write_text("{}\n", encoding="utf-8")
            arguments = dict(
                policy=policy,
                ledger_path=ledger,
                property_name="polyreactivity",
                source="new_locked",
                experiment_id="exp-locked",
                purpose="one locked evaluation",
                evidence_path=evidence,
            )
            with self.assertRaisesRegex(PolicyError, "confirm_locked_use"):
                record_holdout_use(**arguments)
            record_holdout_use(**arguments, confirm_locked_use=True)
            with self.assertRaisesRegex(PolicyError, "already recorded"):
                record_holdout_use(
                    **(arguments | {"experiment_id": "exp-locked-repeat"}),
                    confirm_locked_use=True,
                )

    def test_final_use_is_confirmed_and_single_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "uses.jsonl"
            evidence = root / "final.json"
            evidence.write_text("{}\n", encoding="utf-8")
            arguments = dict(
                policy=minimal_policy(),
                ledger_path=ledger,
                property_name="polyreactivity",
                source="gdpa3",
                experiment_id="final-001",
                purpose="single-shot final evaluation",
                evidence_path=evidence,
            )
            with self.assertRaisesRegex(PolicyError, "confirm_final_use"):
                record_holdout_use(**arguments)
            record_holdout_use(**arguments, confirm_final_use=True)
            report = build_status(arguments["policy"], repo_root=root, ledger_path=ledger)
            final = report["properties"]["polyreactivity"]["source_roles"]["final"][0]
            self.assertTrue(final["used"])
            self.assertFalse(final["prior_use"])
            with self.assertRaisesRegex(PolicyError, "already recorded"):
                record_holdout_use(**arguments, confirm_final_use=True)

    def test_unknown_source_is_rejected_before_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(PolicyError, "not assigned"):
                record_holdout_use(
                    minimal_policy(),
                    ledger_path=root / "uses.jsonl",
                    property_name="polyreactivity",
                    source="unknown",
                    experiment_id="exp",
                    purpose="test",
                    evidence_path=root / "missing.json",
                )


class TestStatusRendering(unittest.TestCase):
    def test_render_keeps_historical_and_hard_transfer_separate(self) -> None:
        report = {
            "generated_at": "2026-07-17T00:00:00+00:00",
            "metric_definitions": {"primary": "macro_source_roc_auc", "required": []},
            "properties": {
                "polyreactivity": {
                    "source_roles": {
                        "development": [{"name": "aims_flu", "prior_use": True}],
                        "locked_validation": [],
                        "final": [{"name": "gdpa3", "prior_use": False}],
                    },
                    "baselines": [
                        {"name": "historical_platform_scorecard", "status": "ok", "value": 0.7},
                        {"name": "hard_library_to_diverse_transfer", "status": "ok", "value": 0.6269},
                    ],
                    "counts": [],
                }
            },
            "frozen_artifacts": [],
            "holdout_uses": [],
        }
        text = render_status(report)
        self.assertIn("historical_platform_scorecard: 0.700", text)
        self.assertIn("hard_library_to_diverse_transfer: 0.627", text)
        self.assertIn("final (UNSPENT): gdpa3", text)
        self.assertIn("locked validation: NONE ASSIGNED", text)


if __name__ == "__main__":
    unittest.main()
