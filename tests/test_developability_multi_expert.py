from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model.developability_matrix import IdentityCalibrator
from model.developability_multi_expert import (
    CandidateSpec,
    evaluate_specialists,
    family_references,
    nested_generalist,
    predict_artifact,
    prepare_outer_split,
    summarize_generalists,
)


def item(record_id: str, source: str, target: int, vh: str, **extra) -> dict:
    default_light = {"a": "A", "b": "C", "c": "D"}.get(source, "E") * 100
    return {
        "record_id": record_id,
        "source": source,
        "source_class": extra.pop("source_class", "DIVERSE"),
        "property": "polyreactivity",
        "assay_metric": "psr",
        "target": target,
        "endpoint_direction": "higher_bad",
        "supervision_tier": "paired_fv_supervised",
        "training_eligible": extra.pop("training_eligible", True),
        "holdout_eligible": extra.pop("holdout_eligible", True),
        "vh": vh,
        "vl": extra.pop("vl", default_light),
        "cdrh1": "",
        "cdrh2": "",
        "cdrh3": "",
        **extra,
    }


class TestNestedGeneralist(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict]:
        rows = []
        parents = {"a": "A", "b": "C", "c": "D"}
        for source, residue in parents.items():
            for index in range(24):
                tail = "F" if index % 2 else "G"
                rows.append(item(
                    f"{source}-{index}", source, index % 2,
                    residue * 117 + f"{index % 3}{tail}".replace("0", "H").replace("1", "I").replace("2", "K"),
                ))
        return rows

    def test_outer_split_removes_overlap_and_ineligible_holdouts(self) -> None:
        held = item("held", "a", 1, "A" * 120)
        rows = [
            held,
            item("reconstructed", "a", 0, "C" * 120, holdout_eligible=False),
            item("exact", "b", 0, "A" * 120, vl="F" * 100),
            item("near", "b", 0, "A" * 119 + "C", vl="G" * 100),
            item("far", "b", 0, "D" * 120, vl="H" * 100),
        ]
        train, test, audit = prepare_outer_split(rows, "a", 0.90)
        self.assertEqual([row["record_id"] for row in test], ["held"])
        self.assertEqual([row["record_id"] for row in train], ["far"])
        self.assertEqual(audit["exact_vh_removed"], 1)
        self.assertEqual(audit["near_vh_removed"], 1)

    @patch("model.developability_multi_expert._predict_spec")
    def test_outer_labels_do_not_change_inner_selection(self, predict) -> None:
        def fake(_spec, _train, test, _seed):
            return {
                "records": {
                    row["record_id"]: (int(row["target"]), float(row["target"]))
                    for row in test
                },
                "coverage": {"train": 1.0, "test": 1.0},
                "calibration_method": "identity_test",
            }

        predict.side_effect = fake
        candidate = CandidateSpec("one", "biophysical_charge", "mixed", "lgbm")
        first = nested_generalist(
            self.rows(), ["a", "b", "c"], "polyreactivity", 42,
            candidates=(candidate,),
        )
        changed = self.rows()
        for row in changed:
            if row["source"] == "a":
                row["target"] = 1 - row["target"]
        second = nested_generalist(
            changed, ["a", "b", "c"], "polyreactivity", 42,
            candidates=(candidate,),
        )
        self.assertEqual(
            first["per_source"]["a"]["trial_members"],
            second["per_source"]["a"]["trial_members"],
        )
        confirmation = nested_generalist(
            self.rows(), ["a", "b", "c"], "polyreactivity", 42,
            candidates=(candidate,), outer_sources=("c",),
        )
        self.assertEqual(set(confirmation["per_source"]), {"c"})

    def test_seed_summary_keeps_baseline_and_trial_separate(self) -> None:
        runs = []
        for seed, baseline, trial in ((42, 0.60, 0.62), (43, 0.61, 0.63), (44, 0.62, 0.64)):
            runs.append({
                "property": "polyreactivity",
                "seed": seed,
                "per_source": {
                    "a": {
                        "baseline": {"roc_auc": baseline, "enrichment_at_15pct": 1.5},
                        "trial": {"roc_auc": trial, "enrichment_at_15pct": 1.7},
                    },
                    "b": {
                        "baseline": {"roc_auc": baseline, "enrichment_at_15pct": 1.5},
                        "trial": {"roc_auc": trial, "enrichment_at_15pct": 1.7},
                    },
                },
            })
        summary = summarize_generalists(runs)["polyreactivity"]
        self.assertAlmostEqual(summary["baseline"]["mean_macro_source_auc"], 0.61)
        self.assertAlmostEqual(summary["trial"]["mean_macro_source_auc"], 0.63)
        self.assertEqual(summary["trial"]["source_mean_auc"], {"a": 0.63, "b": 0.63})
        self.assertEqual(summary["trial"]["seed_macro_source_auc"], [0.62, 0.63, 0.64])


class ConstantModel:
    def __init__(self, risk: float):
        self.risk = risk

    def predict_proba(self, rows):
        import numpy as np

        return np.asarray([[1.0 - self.risk, self.risk] for _ in rows])


class TestExperimentalArtifact(unittest.TestCase):
    @staticmethod
    def _bundle(risk: float, seed: int) -> dict:
        return {
            "component": "constant",
            "seed": seed,
            "learner": "lgbm",
            "feature_mode": "biophysical_charge",
            "model": ConstantModel(risk),
            "calibrator": IdentityCalibrator("identity_test"),
        }

    def test_family_references_cover_separated_clusters_deterministically(self) -> None:
        sequences = ["A" * 120, "A" * 119 + "C", "C" * 120]
        first = family_references(sequences, threshold=0.90, max_references=4)
        second = family_references(reversed(sequences), threshold=0.90, max_references=4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    @patch("model.developability_multi_expert.score_applicability")
    def test_sequence_identity_routes_without_source_name(self, applicability) -> None:
        applicability.return_value = [{
            "ood_distance": 0.1,
            "applicability": "in_domain",
            "nearest_training_source": "train",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.pkl"
            artifact = {
                "schema_version": "1",
                "router_threshold": 0.90,
                "training_manifest": {"registry_hash": "abc"},
                "properties": {
                    "polyreactivity": {
                        "generalist": [self._bundle(0.2, seed) for seed in (42, 43, 44)],
                        "applicability_index": {},
                        "specialists": {
                            "tessier_lib1": {
                                "references": ["A" * 120],
                                "models": [self._bundle(0.9, seed) for seed in (42, 43, 44)],
                            }
                        },
                    }
                },
            }
            path.write_bytes(pickle.dumps(artifact))
            near = predict_artifact(path, "A" * 120)
            far = predict_artifact(path, "C" * 120)
            self.assertEqual(near["polyreactivity"]["expert_name"], "tessier_lib1")
            self.assertAlmostEqual(near["polyreactivity"]["risk_score"], 0.9)
            self.assertEqual(far["polyreactivity"]["expert_type"], "generalist")
            self.assertAlmostEqual(far["polyreactivity"]["risk_score"], 0.2)
            self.assertFalse(far["polyreactivity"]["abstain"])

    def test_invalid_vh_is_rejected_before_artifact_read(self) -> None:
        with self.assertRaisesRegex(ValueError, "standard 20-amino-acid"):
            predict_artifact(Path("unused.pkl"), "ABZ")

    @patch("model.developability_multi_expert.make_featurizer")
    def test_selected_plm_cache_miss_fails_with_exact_command(self, make) -> None:
        make.return_value = lambda _row: None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.pkl"
            bundle = self._bundle(0.2, 42)
            bundle["feature_mode"] = "frozen_plm"
            path.write_bytes(pickle.dumps({
                "schema_version": "1",
                "router_threshold": 0.90,
                "training_manifest": {"registry_hash": "abc"},
                "properties": {
                    "polyreactivity": {
                        "generalist": [bundle],
                        "applicability_index": {},
                        "specialists": {},
                    }
                },
            }))
            with self.assertRaisesRegex(RuntimeError, "features.plm --precompute"):
                predict_artifact(path, "A" * 120)

    def test_specialist_evaluation_reports_router_coverage(self) -> None:
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        rows = []
        for index in range(120):
            suffix = "".join(
                alphabet[(index // divisor) % len(alphabet)]
                for divisor in (1, 20, 400)
            )
            rows.append(item(
                f"lib-{index}", "library", index % 2,
                "A" * 117 + suffix,
                source_class="LIBRARY",
            ))
        report = evaluate_specialists(
            rows, "polyreactivity", 42, n_splits=3, max_per_source_class=60
        )
        self.assertIn("library", report["per_source"])
        result = report["per_source"]["library"]
        self.assertEqual(result["n"], 120)
        self.assertTrue(0.0 <= result["router_coverage"] <= 1.0)
        self.assertIn("roc_auc", result)


if __name__ == "__main__":
    unittest.main()
