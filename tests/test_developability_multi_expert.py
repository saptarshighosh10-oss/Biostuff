from __future__ import annotations

import unittest
from unittest.mock import patch

from model.developability_multi_expert import (
    CandidateSpec,
    nested_generalist,
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


if __name__ == "__main__":
    unittest.main()
