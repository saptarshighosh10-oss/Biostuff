"""Nested evaluation and opt-in artifact for the routed developability platform."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from model.benchmark_governance import load_policy, validate_policy
from model.developability_experts import (
    choose_specialist,
    positional_identity,
    select_blend,
)
from model.developability_matrix import (
    MIN_CLASS_ROWS,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    _bootstrap_source_ci,
    _fit_calibration_split,
    _near_vh_matches,
    _training_arm_rows,
    classification_metrics,
    load_property_rows,
    fit_applicability_index,
    fit_candidate,
    make_featurizer,
    predict_positive_probabilities,
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
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def family_references(
    sequences,
    threshold: float = 0.90,
    max_references: int = 8,
) -> list[str]:
    """Return a small deterministic set of conservative family references."""
    unique = sorted(set(str(sequence or "").strip().upper() for sequence in sequences))
    unique = [sequence for sequence in unique if sequence]
    if not unique or max_references < 1:
        return []
    # ponytail: 64 deterministic probes keep library routing cheap; increase
    # only if measured router coverage is poor on a future locked family.
    if len(unique) > 64:
        indexes = [round(index * (len(unique) - 1) / 63) for index in range(64)]
        sample = [unique[index] for index in indexes]
    else:
        sample = unique
    medoid = min(
        sample,
        key=lambda candidate: (
            -statistics.fmean(positional_identity(candidate, other) for other in sample),
            candidate,
        ),
    )
    references = [medoid]
    while len(references) < max_references:
        uncovered = [
            sequence for sequence in sample
            if max(positional_identity(sequence, ref) for ref in references) < threshold
        ]
        if not uncovered:
            break
        references.append(min(
            uncovered,
            key=lambda sequence: (
                max(positional_identity(sequence, ref) for ref in references),
                sequence,
            ),
        ))
    return references


def _validated_sequence(sequence: str, name: str, required: bool) -> str:
    value = str(sequence or "").strip().upper()
    if not value and not required:
        return ""
    if len(value) < 20 or set(value) - STANDARD_AA:
        raise ValueError(
            f"{name} must be a standard 20-amino-acid sequence of length >= 20"
        )
    return value


def _query_vector(feature_mode: str, vh: str, vl: str) -> list[float]:
    row = {
        "record_id": "query",
        "vh": vh,
        "vl": vl,
        "cdrh1": "",
        "cdrh2": "",
        "cdrh3": "",
    }
    vector = make_featurizer(feature_mode)(row)
    if vector is None:
        raise RuntimeError(
            "frozen PLM cache miss; run: ~/Biostuff/.venv/bin/python -m "
            "features.plm --precompute results/query_sequences.txt --model both"
        )
    return vector


def _score_models(models: list[dict], vh: str, vl: str) -> tuple[float, float]:
    scores = []
    for item in models:
        vector = _query_vector(item["feature_mode"], vh, vl)
        raw = predict_positive_probabilities(item["model"], [vector])[0]
        scores.append(float(item["calibrator"]([raw])[0]))
    if not scores:
        raise ValueError("selected expert has no fitted models")
    return (
        statistics.fmean(scores),
        statistics.pstdev(scores) if len(scores) > 1 else 0.0,
    )


def predict_artifact(path: Path, vh: str, vl: str = "") -> dict:
    """Score one antibody with sequence-only routing and explicit abstention."""
    vh = _validated_sequence(vh, "VH", True)
    vl = _validated_sequence(vl, "VL", False)
    with Path(path).open("rb") as stream:
        artifact = pickle.load(stream)
    if artifact.get("schema_version") != "1":
        raise ValueError("unsupported experimental platform artifact schema")

    output = {}
    for property_name, head in artifact["properties"].items():
        references = {
            name: specialist["references"]
            for name, specialist in head["specialists"].items()
        }
        route = choose_specialist(vh, references, artifact["router_threshold"])
        if route:
            expert_type = "specialist"
            expert_name = route["name"]
            models = head["specialists"][expert_name]["models"]
        else:
            expert_type = "generalist"
            expert_name = "diverse_generalist"
            models = head["generalist"]
        risk_score, disagreement = _score_models(models, vh, vl)
        baseline = _query_vector("biophysical_charge", vh, vl)
        domain = score_applicability(
            head["applicability_index"],
            [({"record_id": "query", "source": "query"}, baseline)],
        )[0]
        reasons = []
        if domain["applicability"] == "out_of_domain":
            reasons.append("out_of_domain")
        if disagreement > 0.15:
            reasons.append("high_seed_disagreement")
        output[property_name] = {
            "risk_score": risk_score,
            "expert_type": expert_type,
            "expert_name": expert_name,
            "selected_components": sorted({item["component"] for item in models}),
            "nearest_family_identity": route["identity"] if route else None,
            **domain,
            "ensemble_disagreement": disagreement,
            "abstain": bool(reasons),
            "abstention_reason": ";".join(reasons),
            "artifact_schema": artifact["schema_version"],
            "training_manifest_id": artifact["training_manifest"].get("registry_hash"),
        }
    return output


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

        def combined_calibration(members: list[str]) -> str:
            methods = [outer_predictions[name]["calibration_method"] for name in members]
            return (
                "platt_component_train_only"
                if all(method.startswith("platt_") for method in methods)
                else "identity_component_calibration"
            )

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
                "calibration_method": combined_calibration(baseline_members),
                "subgroups": _subgroup_metrics(outer_test, baseline_scores),
            },
            "trial": {
                **trial_metrics,
                "members": trial_members,
                "calibration_method": combined_calibration(trial_members),
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


def evaluate_specialists(
    rows: list[dict],
    property_name: str,
    seed: int,
    n_splits: int = 5,
    max_per_source_class: int = 40000,
    router_threshold: float = 0.90,
) -> dict:
    """Evaluate library specialists only in their explicit known-family regime."""
    from sklearn.model_selection import StratifiedKFold

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source_class") == "LIBRARY" and row.get("training_eligible"):
            by_source[str(row["source"])].append(row)
    per_source = {}
    skipped = {}
    for source, source_rows in sorted(by_source.items()):
        by_vh = {}
        conflicting = set()
        for row in sorted(source_rows, key=lambda item: str(item["record_id"])):
            vh = str(row.get("vh") or "")
            if vh in by_vh and int(by_vh[vh]["target"]) != int(row["target"]):
                conflicting.add(vh)
            else:
                by_vh.setdefault(vh, row)
        selected = [row for vh, row in by_vh.items() if vh not in conflicting]
        selected = source_balanced_sample(
            selected, max_per_source_class=max_per_source_class, seed=seed
        )
        labels = [int(row["target"]) for row in selected]
        if (
            len(selected) < MIN_TRAIN_ROWS
            or len(set(labels)) != 2
            or min(sum(labels), len(labels) - sum(labels)) < 10
        ):
            skipped[source] = "too few deduplicated rows or classes"
            continue
        featurize = make_featurizer("biophysical_charge")
        featured = [(row, featurize(row)) for row in selected]
        folds = StratifiedKFold(
            n_splits=min(n_splits, sum(labels), len(labels) - sum(labels)),
            shuffle=True,
            random_state=seed,
        )
        oof = [0.0] * len(selected)
        routed = 0
        calibration_methods = []
        for train_indexes, test_indexes in folds.split(featured, labels):
            train_featured = [featured[index] for index in train_indexes]
            test_featured = [featured[index] for index in test_indexes]
            fit_featured, calibration_featured = _fit_calibration_split(
                train_featured, seed
            )
            if (
                len(fit_featured) < MIN_TRAIN_ROWS
                or len({int(row["target"]) for row, _ in fit_featured}) != 2
            ):
                fit_featured, calibration_featured = train_featured, []
            fitted = fit_candidate(
                "lgbm", fit_featured, calibration_featured, test_featured, seed
            )
            calibration_methods.append(fitted["calibration_method"])
            for index, score in zip(test_indexes, fitted["scores"]):
                oof[index] = float(score)
            references = family_references(
                (selected[index]["vh"] for index in train_indexes),
                threshold=router_threshold,
            )
            matched = _near_vh_matches(
                {selected[index]["vh"] for index in test_indexes},
                set(references),
                router_threshold,
            )
            routed += sum(selected[index]["vh"] in matched for index in test_indexes)
        metrics = classification_metrics(labels, oof)
        per_source[source] = {
            **metrics,
            "n": len(selected),
            "positives": sum(labels),
            "router_coverage": routed / len(selected),
            "conflicting_vh_removed": len(conflicting),
            "calibration_methods": sorted(set(calibration_methods)),
        }
    return {
        "property": property_name,
        "seed": seed,
        "per_source": per_source,
        "skipped_sources": skipped,
    }


def summarize_specialists(runs: list[dict]) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for source, result in run["per_source"].items():
            grouped[run["property"]][source].append(result)
    return {
        property_name: {
            source: {
                "n": results[0]["n"],
                "positives": results[0]["positives"],
                "seed_auc": [result["roc_auc"] for result in results],
                "mean_auc": statistics.fmean(result["roc_auc"] for result in results),
                "seed_std_auc": statistics.pstdev(
                    [result["roc_auc"] for result in results]
                ) if len(results) > 1 else 0.0,
                "mean_pr_auc": statistics.fmean(result["pr_auc"] for result in results),
                "mean_enrichment_at_15pct": statistics.fmean(
                    result["enrichment_at_15pct"] for result in results
                ),
                "mean_router_coverage": statistics.fmean(
                    result["router_coverage"] for result in results
                ),
            }
            for source, results in sorted(sources.items())
        }
        for property_name, sources in sorted(grouped.items())
    }


def _fit_spec_bundle(
    spec: CandidateSpec,
    rows: list[dict],
    seed: int,
    max_per_source_class: int,
) -> tuple[dict, list[dict]]:
    eligible = [
        row for row in rows
        if row.get("training_eligible")
        and row.get("supervision_tier") in spec.training_tiers
    ]
    eligible = _training_arm_rows(eligible, spec.training_arm)
    eligible = source_balanced_sample(
        eligible, max_per_source_class=max_per_source_class, seed=seed
    )
    featurize = make_featurizer(spec.feature_mode)
    featured = [(row, featurize(row)) for row in eligible]
    featured = [(row, vector) for row, vector in featured if vector is not None]
    coverage = len(featured) / len(eligible) if eligible else 0.0
    if coverage < 0.80 or len(featured) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"cannot fit {spec.name}: coverage={coverage:.3f}, rows={len(featured)}"
        )
    fit_featured, calibration_featured = _fit_calibration_split(featured, seed)
    if (
        len(fit_featured) < MIN_TRAIN_ROWS
        or len({int(row["target"]) for row, _ in fit_featured}) != 2
    ):
        fit_featured, calibration_featured = featured, []
    fitted = fit_candidate(
        spec.learner, fit_featured, calibration_featured, [], seed
    )
    return ({
        "component": spec.name,
        "seed": seed,
        "learner": spec.learner,
        "feature_mode": spec.feature_mode,
        "model": fitted["model"],
        "calibrator": fitted["calibrator"],
        "calibration_method": fitted["calibration_method"],
        "n_train": len(featured),
        "feature_coverage": coverage,
        "train_sources": sorted({str(row["source"]) for row, _ in featured}),
    }, [row for row, _ in featured])


def _selected_trial_members(property_summary: dict) -> list[str]:
    counts = property_summary.get("selection_counts", {}).get("trial", {})
    if not counts:
        raise ValueError("no nested trial selection is available")
    selected = min(
        counts,
        key=lambda name: (-int(counts[name]), len(name.split("+")), name),
    )
    return selected.split("+")


def train_experimental_artifact(
    rows_by_property: dict[str, list[dict]],
    report: dict,
    path: Path,
    max_per_source_class: int = 5000,
    max_per_specialist_class: int = 40000,
) -> dict:
    """Fit the opt-in artifact after nested evaluation, never the serving artifact."""
    artifact = {
        "schema_version": "1",
        "head": "developability_multi_expert",
        "router_threshold": 0.90,
        "seeds": [42, 43, 44],
        "feature_schema": "candidate-specific; no zero fill",
        "training_manifest": report["training_manifest"],
        "properties": {},
    }
    for property_name, rows in sorted(rows_by_property.items()):
        property_summary = report["generalists"].get(property_name)
        if not property_summary:
            continue
        members = _selected_trial_members(property_summary)
        specs = {spec.name: spec for spec in CANDIDATES[property_name]}
        generalists = []
        for seed in artifact["seeds"]:
            for name in members:
                bundle, _ = _fit_spec_bundle(
                    specs[name], rows, seed, max_per_source_class
                )
                generalists.append(bundle)

        app_rows = source_balanced_sample(
            [row for row in rows if row.get("training_eligible")],
            max_per_source_class=max_per_source_class,
            seed=42,
        )
        app_featurize = make_featurizer("biophysical_charge")
        app_featured = [(row, app_featurize(row)) for row in app_rows]
        applicability_index = fit_applicability_index(app_featured)

        specialists = {}
        specialist_summary = report.get("specialists", {}).get(property_name, {})
        for source in sorted(specialist_summary):
            source_rows = [
                row for row in rows
                if row.get("source") == source
                and row.get("source_class") == "LIBRARY"
                and row.get("training_eligible")
            ]
            tiers = tuple(sorted({str(row["supervision_tier"]) for row in source_rows}))
            spec = CandidateSpec(
                f"specialist_{source}", "biophysical_charge", "mixed", "lgbm", tiers
            )
            models = []
            fitted_rows = []
            for seed in artifact["seeds"]:
                bundle, selected_rows = _fit_spec_bundle(
                    spec, source_rows, seed, max_per_specialist_class
                )
                models.append(bundle)
                if seed == 42:
                    fitted_rows = selected_rows
            specialists[source] = {
                "references": family_references(row["vh"] for row in fitted_rows),
                "models": models,
            }
        artifact["properties"][property_name] = {
            "generalist": generalists,
            "generalist_members": members,
            "applicability_index": applicability_index,
            "specialists": specialists,
        }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        pickle.dump(artifact, stream)
    return artifact


def _uncertainty_summary(runs: list[dict]) -> dict:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for run in runs:
        for source, result in run["per_source"].items():
            for record in result["record_predictions"]:
                groups[(run["property"], source, record["record_id"])].append(record)
    domain_order = {"in_domain": 0, "borderline": 1, "out_of_domain": 2}
    domain_counts = Counter()
    abstention_counts = Counter()
    disagreements = []
    for records in groups.values():
        domain = max(
            (record["applicability"] for record in records),
            key=lambda value: domain_order[value],
        )
        disagreement = statistics.pstdev(
            [float(record["trial_score"]) for record in records]
        ) if len(records) > 1 else 0.0
        domain_counts[domain] += 1
        disagreements.append(disagreement)
        if len(records) < 3:
            abstention_counts["fewer_than_three_seeds"] += 1
        if domain == "out_of_domain":
            abstention_counts["out_of_domain"] += 1
        if disagreement > 0.15:
            abstention_counts["high_seed_disagreement"] += 1
    ordered = sorted(disagreements)
    return {
        "unique_prediction_records": len(groups),
        "domain_counts": dict(sorted(domain_counts.items())),
        "abstention_counts": dict(sorted(abstention_counts.items())),
        "mean_seed_disagreement": (
            statistics.fmean(disagreements) if disagreements else None
        ),
        "p95_seed_disagreement": (
            ordered[round(0.95 * (len(ordered) - 1))] if ordered else None
        ),
    }


def _promotion_matrix(generalists: dict) -> dict:
    summaries = []
    experiments = []
    for property_name, summary in sorted(generalists.items()):
        experiment = f"{property_name}:multi_expert_trial"
        trial = summary["trial"]
        summaries.append({
            "experiment": experiment,
            "property": property_name,
            "test_tier": "paired_fv_supervised",
            "mean_macro_source_auc": trial["mean_macro_source_auc"],
            "source_mean_auc": trial["source_mean_auc"],
            "seed_macro_source_auc": trial["seed_macro_source_auc"],
            "seed_std_auc": trial["seed_std_auc"],
            "source_bootstrap_ci95": trial["source_bootstrap_ci95"],
        })
        for run in summary["per_seed"]:
            experiments.append({
                "experiment": experiment,
                "property": property_name,
                "seed": run["seed"],
                "per_source": [
                    {"source": source, **result["trial"]}
                    for source, result in sorted(run["per_source"].items())
                ],
            })
    return {"summaries": summaries, "experiments": experiments}


def _write_predictions(runs: list[dict], path: Path) -> int:
    fields = (
        "property", "seed", "held_source", "record_id", "assay", "label",
        "baseline_score", "trial_score", "ood_distance", "applicability",
        "nearest_training_source",
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda item: (item["property"], item["seed"])):
            for source, result in sorted(run["per_source"].items()):
                for record in result["record_predictions"]:
                    writer.writerow({
                        "property": run["property"],
                        "seed": run["seed"],
                        "held_source": source,
                        **record,
                    })
                    count += 1
    return count


def run_experiment(
    database: Path,
    policy_path: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    max_per_source_class: int = 5000,
    near_identity: float = 0.90,
) -> tuple[dict, dict[str, list[dict]]]:
    policy = load_policy(policy_path)
    validate_policy(policy)
    if tuple(seeds) != (42, 43, 44):
        raise ValueError("the governed multi-expert run requires seeds 42,43,44")

    rows_by_property = {}
    manifests = {}
    generalist_runs = []
    specialist_runs = []
    prefetch = max(40000, max_per_source_class * 3)
    for property_name in ("polyreactivity", "aggregation", "hydrophobicity"):
        print(f"loading {property_name} registry rows", flush=True)
        rows, manifest = load_property_rows(
            database, property_name, prefetch_per_binary_class=prefetch
        )
        if any(str(row["source"]).lower() == "gdpa3" for row in rows):
            raise RuntimeError("GDPa3 appeared in development rows; refusing to continue")
        rows_by_property[property_name] = rows
        manifests[property_name] = manifest
        held_sources = [
            item["name"]
            for item in policy["properties"][property_name]["source_roles"]["development"]
        ]
        for seed in seeds:
            print(f"  {property_name}: seed {seed} generalist", flush=True)
            generalist_runs.append(nested_generalist(
                rows,
                held_sources,
                property_name,
                seed,
                max_per_source_class=max_per_source_class,
                near_identity=near_identity,
            ))
            print(f"  {property_name}: seed {seed} specialists", flush=True)
            specialist_runs.append(evaluate_specialists(
                rows, property_name, seed, max_per_source_class=40000
            ))
    generalists = summarize_generalists(generalist_runs)
    specialists = summarize_specialists(specialist_runs)
    registry_hashes = {
        manifest.get("registry_hash") for manifest in manifests.values()
        if manifest.get("registry_hash")
    }
    training_manifest = {
        "registry_hash": next(iter(registry_hashes)) if len(registry_hashes) == 1 else None,
        "properties": manifests,
    }
    report = {
        "schema_version": "1",
        "benchmark_role": "development_only",
        "seeds": list(seeds),
        "near_vh_identity_filter": near_identity,
        "max_per_source_class": max_per_source_class,
        "generalists": generalists,
        "specialists": specialists,
        "uncertainty": _uncertainty_summary(generalist_runs),
        "training_manifest": training_manifest,
        "promotion_matrix": _promotion_matrix(generalists),
        "important_limitations": [
            "Known-family specialist and unseen-source generalist metrics are separate.",
            "No genuinely locked validation source is assigned.",
            "GDPa3 was not loaded or inspected.",
            "Development-source results cannot support a lab-ready claim.",
        ],
    }
    return report, rows_by_property


def _print_report(report: dict) -> None:
    print("\nMulti-expert developability experiment — development evidence only")
    print("\nUnseen-source generalists")
    for property_name, summary in report["generalists"].items():
        baseline = summary["baseline"]["mean_macro_source_auc"]
        trial = summary["trial"]["mean_macro_source_auc"]
        print(f"  {property_name:<18} baseline={baseline:.3f} trial={trial:.3f} delta={trial-baseline:+.3f}")
        for source, value in summary["trial"]["source_mean_auc"].items():
            base = summary["baseline"]["source_mean_auc"][source]
            print(f"    {source:<25} {base:.3f} -> {value:.3f} ({value-base:+.3f})")
    print("\nKnown-family specialists (not pooled with generalists)")
    for property_name, sources in report["specialists"].items():
        for source, result in sources.items():
            print(
                f"  {property_name:<18} {source:<18} n={result['n']:<6} "
                f"AUC={result['mean_auc']:.3f} route={result['mean_router_coverage']:.3f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("results/developability_registry.sqlite"))
    parser.add_argument("--policy", type=Path, default=Path("tools/benchmark_policy.json"))
    parser.add_argument("--out", type=Path, default=Path("results/developability_multi_expert.json"))
    parser.add_argument("--predictions", type=Path, default=Path("results/developability_multi_expert_predictions.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("model/saved/platform_experimental.pkl"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-per-source-class", type=int, default=5000)
    parser.add_argument("--near-identity", type=float, default=0.90)
    args = parser.parse_args(argv)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    report, rows_by_property = run_experiment(
        args.database,
        args.policy,
        seeds,
        args.max_per_source_class,
        args.near_identity,
    )
    prediction_count = _write_predictions(
        [run for summary in report["generalists"].values() for run in summary["per_seed"]],
        args.predictions,
    )
    artifact = train_experimental_artifact(
        rows_by_property, report, args.artifact, args.max_per_source_class
    )
    report["predictions"] = str(args.predictions)
    report["prediction_rows"] = prediction_count
    report["artifact"] = str(args.artifact)
    report["artifact_properties"] = sorted(artifact["properties"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_report(report)
    print(f"\nwrote {args.out.resolve()}")
    print(f"wrote {args.predictions.resolve()} ({prediction_count} rows)")
    print(f"wrote {args.artifact.resolve()} (experimental only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
