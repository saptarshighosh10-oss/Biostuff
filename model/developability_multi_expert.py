"""Nested evaluation and opt-in artifact for the routed developability platform."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from model.developability_experts import select_blend
from model.developability_matrix import (
    MIN_CLASS_ROWS,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    _bootstrap_source_ci,
    _fit_calibration_split,
    _training_arm_rows,
    classification_metrics,
    fit_applicability_index,
    fit_candidate,
    make_featurizer,
    score_applicability,
    source_balanced_sample,
    strip_holdout_overlap,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    feature_mode: str
    training_arm: str
    learner: str
    training_tiers: tuple[str, ...] = ("paired_fv_supervised",)


CANDIDATES = {
    "polyreactivity": (
        CandidateSpec("cdr_library_lgbm", "cdr_aware", "library_only", "lgbm"),
        CandidateSpec("bio_mixed_lgbm", "biophysical_charge", "mixed", "lgbm"),
        CandidateSpec("plm_mixed_lgbm", "frozen_plm", "mixed", "lgbm"),
    ),
    "aggregation": (
        CandidateSpec("bio_diverse_lgbm", "biophysical_charge", "diverse_only", "lgbm"),
        CandidateSpec("plm_diverse_lgbm", "frozen_plm", "diverse_only", "lgbm"),
        CandidateSpec("plm_diverse_histgb", "frozen_plm", "diverse_only", "histgb"),
        CandidateSpec("plm_diverse_ridge", "frozen_plm", "diverse_only", "ridge"),
    ),
    "hydrophobicity": (
        CandidateSpec(
            "bio_mixed_lgbm", "biophysical_charge", "mixed", "lgbm",
            ("paired_fv_supervised", "single_chain_supervised"),
        ),
        CandidateSpec("plm_mixed_lgbm", "frozen_plm", "mixed", "lgbm"),
        CandidateSpec(
            "bio_mixed_histgb", "biophysical_charge", "mixed", "histgb",
            ("paired_fv_supervised", "single_chain_supervised"),
        ),
        CandidateSpec(
            "bio_mixed_ridge", "biophysical_charge", "mixed", "ridge",
            ("paired_fv_supervised", "single_chain_supervised"),
        ),
    ),
}

_METRICS = (
    "roc_auc", "pr_auc", "catch_rate_at_10pct", "catch_rate_at_15pct",
    "enrichment_at_10pct", "enrichment_at_15pct", "brier", "ece_10bin",
)


def prepare_outer_split(
    rows: list[dict],
    held_source: str,
    near_identity: float,
    test_tier: str = "paired_fv_supervised",
) -> tuple[list[dict], list[dict], dict]:
    """Return eligible train/test rows after universal held-source overlap removal."""
    test = [
        row for row in rows
        if row["source"] == held_source
        and row.get("holdout_eligible")
        and row.get("supervision_tier") == test_tier
    ]
    train = [
        row for row in rows
        if row["source"] != held_source and row.get("training_eligible")
    ]
    held_rows = [row for row in rows if row["source"] == held_source]
    train, audit = strip_holdout_overlap(train, held_rows, near_identity=near_identity)
    audit["eligible_test_rows"] = len(test)
    audit["test_tier"] = test_tier
    return train, test, audit


def _predict_spec(
    spec: CandidateSpec,
    train: list[dict],
    test: list[dict],
    seed: int,
) -> dict:
    """Fit one fixed candidate and return calibrated record-level predictions."""
    eligible = [
        row for row in train
        if row.get("supervision_tier") in spec.training_tiers
    ]
    eligible = _training_arm_rows(eligible, spec.training_arm)
    featurize = make_featurizer(spec.feature_mode)
    train_featured = [(row, featurize(row)) for row in eligible]
    test_featured = [(row, featurize(row)) for row in test]
    train_featured = [(row, vector) for row, vector in train_featured if vector is not None]
    test_featured = [(row, vector) for row, vector in test_featured if vector is not None]
    coverage = {
        "train": len(train_featured) / len(eligible) if eligible else 0.0,
        "test": len(test_featured) / len(test) if test else 0.0,
    }
    if min(coverage.values()) < 0.80 or len(train_featured) < MIN_TRAIN_ROWS:
        return {
            "records": {},
            "coverage": coverage,
            "skip": "feature coverage below 80% or fewer than 40 training rows",
        }
    labels = [int(row["target"]) for row, _ in train_featured]
    if min(sum(labels), len(labels) - sum(labels)) < MIN_CLASS_ROWS:
        return {
            "records": {},
            "coverage": coverage,
            "skip": "training rows have too few rows in one class",
        }
    fit_featured, calibration_featured = _fit_calibration_split(train_featured, seed)
    if (
        len(fit_featured) < MIN_TRAIN_ROWS
        or len({int(row["target"]) for row, _ in fit_featured}) != 2
    ):
        fit_featured, calibration_featured = train_featured, []
    fitted = fit_candidate(
        spec.learner, fit_featured, calibration_featured, test_featured, seed
    )
    return {
        "records": {
            row["record_id"]: (int(row["target"]), float(score))
            for (row, _), score in zip(test_featured, fitted["scores"])
        },
        "coverage": coverage,
        "calibration_method": fitted["calibration_method"],
        "model": fitted["model"],
        "calibrator": fitted["calibrator"],
        "fit_n": len(fit_featured),
        "calibration_n": len(calibration_featured),
    }


def _single_selection(predictions: dict, eligible_names: list[str]) -> dict:
    singles = []
    for name in eligible_names:
        result = select_blend({
            source: {name: candidates[name]}
            for source, candidates in predictions.items()
        })
        singles.append(result)
    return min(
        singles,
        key=lambda result: (
            -result["macro_source_auc"], result["members"]
        ),
    )


def _source_metrics(rows: list[dict], scores: dict[str, float]) -> dict | None:
    by_assay: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["record_id"] in scores:
            by_assay[str(row["assay_metric"])].append(row)
    per_assay = {}
    for assay, assay_rows in sorted(by_assay.items()):
        labels = [int(row["target"]) for row in assay_rows]
        if len(assay_rows) >= MIN_TEST_ROWS and len(set(labels)) == 2:
            per_assay[assay] = classification_metrics(
                labels, [scores[row["record_id"]] for row in assay_rows]
            )
    if not per_assay:
        return None
    return {
        **{
            metric: statistics.fmean(result[metric] for result in per_assay.values())
            for metric in _METRICS
        },
        "per_assay": per_assay,
        "test_n": sum(result["n"] for result in per_assay.values()),
        "test_positives": sum(result["positives"] for result in per_assay.values()),
    }


def _subgroup_metrics(rows: list[dict], scores: dict[str, float]) -> dict:
    aligned = [row for row in rows if row["record_id"] in scores]
    if not aligned:
        return {}
    median_length = statistics.median(len(str(row.get("vh") or "")) for row in aligned)
    groups = {
        "vh_shorter": [row for row in aligned if len(str(row.get("vh") or "")) < median_length],
        "vh_longer_or_equal": [
            row for row in aligned if len(str(row.get("vh") or "")) >= median_length
        ],
        "vl_present": [row for row in aligned if row.get("vl")],
        "vl_missing": [row for row in aligned if not row.get("vl")],
    }
    output = {}
    for name, group in groups.items():
        labels = [int(row["target"]) for row in group]
        if len(group) >= MIN_TEST_ROWS and len(set(labels)) == 2:
            output[name] = classification_metrics(
                labels, [scores[row["record_id"]] for row in group]
            )
    return output


def nested_generalist(
    rows: list[dict],
    held_sources: list[str],
    property_name: str,
    seed: int,
    max_per_source_class: int = 5000,
    near_identity: float = 0.90,
    candidates: tuple[CandidateSpec, ...] | list[CandidateSpec] | None = None,
) -> dict:
    """Nested candidate selection with untouched outer-source labels."""
    specs = tuple(candidates or CANDIDATES[property_name])
    by_name = {spec.name: spec for spec in specs}
    per_source = {}
    skipped = {}

    for outer in held_sources:
        outer_pool, outer_test, outer_audit = prepare_outer_split(
            rows, outer, near_identity
        )
        outer_labels = [int(row["target"]) for row in outer_test]
        if len(outer_test) < MIN_TEST_ROWS or (
            outer_labels
            and min(sum(outer_labels), len(outer_labels) - sum(outer_labels)) < MIN_CLASS_ROWS
        ):
            skipped[outer] = f"only {len(outer_test)} compatible test rows or too few classes"
            continue

        inner_predictions = {}
        inner_audits = {}
        for inner in sorted(set(held_sources) - {outer}):
            inner_test = [
                row for row in outer_pool
                if row["source"] == inner
                and row.get("holdout_eligible")
                and row.get("supervision_tier") == "paired_fv_supervised"
            ]
            inner_labels = [int(row["target"]) for row in inner_test]
            if len(inner_test) < MIN_TEST_ROWS or len(set(inner_labels)) != 2:
                continue
            inner_train = [row for row in outer_pool if row["source"] != inner]
            held_inner_rows = [row for row in outer_pool if row["source"] == inner]
            inner_train = source_balanced_sample(
                inner_train, max_per_source_class=max_per_source_class, seed=seed
            )
            inner_train, inner_audit = strip_holdout_overlap(
                inner_train, held_inner_rows, near_identity=near_identity
            )
            predicted = {}
            candidate_audit = {}
            for spec in specs:
                result = _predict_spec(spec, inner_train, inner_test, seed)
                candidate_audit[spec.name] = {
                    key: value for key, value in result.items()
                    if key not in {"records", "model", "calibrator"}
                }
                if result["records"]:
                    predicted[spec.name] = result["records"]
            if predicted:
                inner_predictions[inner] = predicted
                inner_audits[inner] = {
                    "overlap": inner_audit,
                    "candidates": candidate_audit,
                }
        if not inner_predictions:
            skipped[outer] = "no scorable inner held sources"
            continue

        trial_selection = select_blend(inner_predictions)
        baseline_selection = _single_selection(
            inner_predictions, trial_selection["eligible_candidates"]
        )
        baseline_members = baseline_selection["members"]
        trial_members = trial_selection["members"]
        needed = sorted(set(baseline_members + trial_members))
        outer_train = source_balanced_sample(
            outer_pool, max_per_source_class=max_per_source_class, seed=seed
        )
        outer_predictions = {
            name: _predict_spec(by_name[name], outer_train, outer_test, seed)
            for name in needed
        }
        if any(not outer_predictions[name]["records"] for name in needed):
            missing = [name for name in needed if not outer_predictions[name]["records"]]
            skipped[outer] = f"selected candidates unavailable on outer source: {', '.join(missing)}"
            continue
        common_ids = sorted(set.intersection(*(
            set(outer_predictions[name]["records"]) for name in needed
        )))
        if len(common_ids) < MIN_TEST_ROWS:
            skipped[outer] = f"only {len(common_ids)} common outer predictions"
            continue
        baseline_scores = {
            record_id: statistics.fmean(
                outer_predictions[name]["records"][record_id][1]
                for name in baseline_members
            )
            for record_id in common_ids
        }
        trial_scores = {
            record_id: statistics.fmean(
                outer_predictions[name]["records"][record_id][1]
                for name in trial_members
            )
            for record_id in common_ids
        }
        baseline_metrics = _source_metrics(outer_test, baseline_scores)
        trial_metrics = _source_metrics(outer_test, trial_scores)
        if baseline_metrics is None or trial_metrics is None:
            skipped[outer] = "no common assay has enough rows and both classes"
            continue

        baseline_featurize = make_featurizer("biophysical_charge")
        app_train = [(row, baseline_featurize(row)) for row in outer_train]
        test_by_id = {row["record_id"]: row for row in outer_test}
        app_test = [
            (test_by_id[record_id], baseline_featurize(test_by_id[record_id]))
            for record_id in common_ids
        ]
        applicability_index = fit_applicability_index(app_train)
        applicability = score_applicability(applicability_index, app_test)
        record_predictions = []
        for record_id, domain in zip(common_ids, applicability):
            row = test_by_id[record_id]
            record_predictions.append({
                "record_id": record_id,
                "label": int(row["target"]),
                "assay": row["assay_metric"],
                "baseline_score": baseline_scores[record_id],
                "trial_score": trial_scores[record_id],
                **domain,
            })
        per_source[outer] = {
            "baseline": {
                **baseline_metrics,
                "members": baseline_members,
                "subgroups": _subgroup_metrics(outer_test, baseline_scores),
            },
            "trial": {
                **trial_metrics,
                "members": trial_members,
                "subgroups": _subgroup_metrics(outer_test, trial_scores),
            },
            "baseline_members": baseline_members,
            "trial_members": trial_members,
            "outer_overlap_audit": outer_audit,
            "inner_selection": {
                "baseline_macro_source_auc": baseline_selection["macro_source_auc"],
                "trial_macro_source_auc": trial_selection["macro_source_auc"],
                "eligible_candidates": trial_selection["eligible_candidates"],
                "inner_sources": sorted(inner_predictions),
                "audits": inner_audits,
            },
            "outer_candidate_audit": {
                name: {
                    key: value for key, value in outer_predictions[name].items()
                    if key not in {"records", "model", "calibrator"}
                }
                for name in needed
            },
            "applicability_fit": {
                "distance": applicability_index["distance"],
                "train_p95": applicability_index["train_p95"],
                "train_p99": applicability_index["train_p99"],
                "source_centroids": applicability_index["source_names"],
            },
            "record_predictions": record_predictions,
        }

    return {
        "property": property_name,
        "seed": seed,
        "per_source": per_source,
        "skipped_sources": skipped,
    }


def _mode_summary(runs: list[dict], mode: str) -> dict:
    seed_macros = []
    source_values: dict[str, list[float]] = defaultdict(list)
    enrichments = []
    for run in sorted(runs, key=lambda item: item["seed"]):
        values = []
        for source, result in run["per_source"].items():
            metrics = result[mode]
            value = float(metrics["roc_auc"])
            values.append(value)
            source_values[source].append(value)
            enrichments.append(float(metrics["enrichment_at_15pct"]))
        if values:
            seed_macros.append(statistics.fmean(values))
    source_means = {
        source: statistics.fmean(values) for source, values in sorted(source_values.items())
    }
    return {
        "seed_macro_source_auc": seed_macros,
        "mean_macro_source_auc": statistics.fmean(seed_macros) if seed_macros else None,
        "seed_std_auc": statistics.pstdev(seed_macros) if len(seed_macros) > 1 else 0.0,
        "source_mean_auc": source_means,
        "source_bootstrap_ci95": _bootstrap_source_ci(source_means),
        "mean_enrichment_at_15pct": (
            statistics.fmean(enrichments) if enrichments else None
        ),
        "audits_present": bool(runs) and all(
            "outer_overlap_audit" in result and "inner_selection" in result
            for run in runs for result in run["per_source"].values()
        ),
    }


def summarize_generalists(runs: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["property"]].append(run)

    def selection_counts(property_runs: list[dict], field: str) -> dict[str, int]:
        counts = Counter(
            "+".join(result[field])
            for run in property_runs for result in run["per_source"].values()
            if result.get(field)
        )
        return dict(sorted(counts.items()))

    return {
        property_name: {
            "baseline": _mode_summary(property_runs, "baseline"),
            "trial": _mode_summary(property_runs, "trial"),
            "per_seed": sorted(property_runs, key=lambda item: item["seed"]),
            "selection_counts": {
                "baseline": selection_counts(property_runs, "baseline_members"),
                "trial": selection_counts(property_runs, "trial_members"),
            },
        }
        for property_name, property_runs in sorted(grouped.items())
    }
