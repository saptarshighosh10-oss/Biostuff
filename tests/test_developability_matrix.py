from __future__ import annotations

import pickle
import unittest
import warnings

from model.developability_matrix import (
    assign_group_labels,
    classification_metrics,
    eligible_split,
    fit_applicability_index,
    fit_candidate,
    make_featurizer,
    score_applicability,
    source_balanced_sample,
    strip_holdout_overlap,
)


VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGGYFDYWGQGTLVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPYTFGQGTKLEIK"


def row(record_id: str, source: str, value: float, **extra) -> dict:
    return {
        "record_id": record_id,
        "source": source,
        "assay_metric": extra.pop("assay_metric", "assay"),
        "label_type": extra.pop("label_type", "continuous"),
        "endpoint_value": value,
        "endpoint_direction": extra.pop("endpoint_direction", "higher_bad"),
        "vh": extra.pop("vh", VH),
        "vl": extra.pop("vl", VL),
        **extra,
    }


class TestDevelopabilityMatrix(unittest.TestCase):
    def test_group_labels_keep_binary_and_use_exact_top_tail(self) -> None:
        binary = [row("b0", "s", 0.0), row("b1", "s", 1.0)]
        assign_group_labels(binary)
        self.assertEqual([item["target"] for item in binary], [0, 1])

        continuous = [row(f"c{i}", "s", float(i)) for i in range(20)]
        assign_group_labels(continuous, bad_fraction=0.15)
        self.assertEqual(sum(item["target"] for item in continuous), 3)
        self.assertEqual(
            {item["record_id"] for item in continuous if item["target"]},
            {"c17", "c18", "c19"},
        )
        self.assertEqual(continuous[0]["rank_target"], 0.0)
        self.assertEqual(continuous[-1]["rank_target"], 1.0)

        lower_bad = [
            row("low0", "s", 1.0, endpoint_direction="lower_bad"),
            row("low1", "s", 2.0, endpoint_direction="lower_bad"),
        ]
        assign_group_labels(lower_bad)
        self.assertEqual(
            [item["rank_target"] for item in lower_bad], [1.0, 0.0]
        )

        react = [row("r0", "s", 1, label_type="num_react"), row("r1", "s", 2, label_type="num_react")]
        assign_group_labels(react)
        self.assertEqual([item["target"] for item in react], [0, 1])

    def test_overlap_stripping_removes_exact_chains_and_near_vh(self) -> None:
        near = VH[:20] + ("A" if VH[20] != "A" else "C") + VH[21:]
        unrelated = "A" * len(VH)
        train = [
            row("exact_vh", "train", 0, vh=VH, vl="C" * len(VL)),
            row("exact_vl", "train", 0, vh=unrelated, vl=VL),
            row("near", "train", 0, vh=near, vl="D" * len(VL)),
            row("keep", "train", 0, vh=unrelated, vl="E" * len(VL)),
        ]
        test = [row("test", "held", 1)]
        kept, audit = strip_holdout_overlap(train, test, near_identity=0.90)
        self.assertEqual([item["record_id"] for item in kept], ["keep"])
        self.assertEqual(audit["exact_vh_removed"], 1)
        self.assertEqual(audit["exact_vl_removed"], 1)
        self.assertEqual(audit["near_vh_removed"], 1)

    def test_source_balancing_caps_each_source_and_class_deterministically(self) -> None:
        rows = [row(f"{source}-{label}-{i}", source, label, target=label)
                for source in ("a", "b") for label in (0, 1) for i in range(8)]
        first = source_balanced_sample(rows, max_per_source_class=3, seed=42)
        second = source_balanced_sample(rows, max_per_source_class=3, seed=42)
        self.assertEqual([item["record_id"] for item in first], [item["record_id"] for item in second])
        self.assertEqual(len(first), 12)

    def test_eligibility_keeps_reconstructions_out_of_holdouts(self) -> None:
        rows = [
            {"record_id": "direct-held", "source": "held",
             "training_eligible": 1, "holdout_eligible": 1},
            {"record_id": "reconstructed-held", "source": "held",
             "training_eligible": 1, "holdout_eligible": 0},
            {"record_id": "unlabeled-train", "source": "train",
             "training_eligible": 0, "holdout_eligible": 0},
            {"record_id": "reconstructed-train", "source": "train",
             "training_eligible": 1, "holdout_eligible": 0},
        ]
        train, test = eligible_split(rows, "held")
        self.assertEqual([item["record_id"] for item in train], ["reconstructed-train"])
        self.assertEqual([item["record_id"] for item in test], ["direct-held"])

    def test_metrics_report_auc_pr_screening_and_calibration(self) -> None:
        metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(metrics["catch_rate_at_15pct"], 0.5)
        self.assertGreater(metrics["enrichment_at_15pct"], 1.0)
        self.assertLess(metrics["brier"], 0.1)


class TestCandidateFitting(unittest.TestCase):
    @staticmethod
    def _featured(count: int = 40) -> list[tuple[dict, list[float]]]:
        return [
            (
                row(f"r{i}", "a" if i % 4 < 2 else "b", i % 2, target=i % 2),
                [float(i % 2), float(i)],
            )
            for i in range(count)
        ]

    def test_supported_learners_return_pickleable_calibrated_probabilities(self) -> None:
        featured = self._featured()
        for learner in (
            "lgbm", "histgb", "ridge", "rank_lgbm", "rank_lgbm_source",
            "rank_lgbm_hybrid",
            "rank_histgb", "rank_ridge"
        ):
            if learner.startswith("rank_"):
                for item, _ in featured:
                    item["rank_target"] = float(item["target"])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                result = fit_candidate(
                    learner, featured[:30], featured[30:35], featured[35:], 42
                )
            self.assertEqual(len(result["scores"]), 5)
            self.assertTrue(all(0.0 <= score <= 1.0 for score in result["scores"]))
            restored = pickle.loads(pickle.dumps(result["calibrator"]))
            self.assertEqual(len(restored(result["raw_scores"])), 5)

    def test_rank_learner_uses_train_only_platt_calibration(self) -> None:
        featured = self._featured(100)
        for item, _ in featured:
            item["rank_target"] = float(item["target"])
        result = fit_candidate(
            "rank_lgbm_hybrid", featured[:60], featured[60:90], featured[90:], 42
        )
        self.assertEqual(result["calibration_method"], "platt_logistic_train_only")
        self.assertEqual(len(result["scores"]), 10)

    def test_unknown_candidate_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported learner"):
            fit_candidate("forest", self._featured(), [], self._featured(4), 42)

    def test_applicability_index_round_trips_to_scoring(self) -> None:
        featured = self._featured()
        index = fit_applicability_index(featured)
        restored = pickle.loads(pickle.dumps(index))
        records = score_applicability(restored, featured[:3])
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["nearest_training_source"] in {"a", "b"}
                            for record in records))
        self.assertTrue(all(record["applicability"] in {
            "in_domain", "borderline", "out_of_domain"
        } for record in records))

    def test_paired_features_add_light_chain_signal_and_presence_flag(self) -> None:
        base = row("paired", "a", 1, target=1)
        base.update({"vh": VH, "vl": VL})
        paired = make_featurizer("paired_biophysical_charge")(base)
        heavy_only = make_featurizer("biophysical_charge")(base)
        missing = make_featurizer("paired_biophysical_charge")({**base, "vl": ""})
        self.assertGreater(len(paired), len(heavy_only))
        self.assertEqual(paired[-1], 1.0)
        self.assertEqual(missing[-1], 0.0)
        self.assertNotEqual(paired, missing)

    def test_paired_interactions_add_fixed_whole_fv_descriptors(self) -> None:
        base = row("interaction", "a", 1, target=1)
        paired = make_featurizer("paired_biophysical_charge")(base)
        interaction = make_featurizer(
            "paired_interaction_biophysical_charge"
        )(base)
        missing = make_featurizer(
            "paired_interaction_biophysical_charge"
        )({**base, "vl": ""})
        self.assertEqual(len(interaction), len(paired) + 21)
        self.assertEqual(interaction[-1], 1.0)
        self.assertEqual(missing[-1], 0.0)
        self.assertTrue(all(value == 0.0 for value in missing[-22:-1]))


if __name__ == "__main__":
    unittest.main()
