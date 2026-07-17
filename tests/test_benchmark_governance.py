from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model.benchmark_governance import PolicyError, load_policy, validate_policy


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


if __name__ == "__main__":
    unittest.main()
