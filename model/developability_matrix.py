"""Reproducible source-held-out baseline matrix for developability properties.

The primary evidence is always macro AUC across whole held-out development
sources. Narrow libraries are source/class capped, held-out VH/VL overlaps are
removed, and near-identical heavy chains are removed before training. Frozen
PLM rows are evaluated only when both cached embeddings exist; cache misses are
reported and never disguised as zero-filled PLM features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from data.contract import hash_payload
from data.run_manifest import file_sha256
from model.benchmark_governance import load_policy, validate_policy
from model.cdr_features import cdr_feature_vector, parse_rate
from model.head_b_acsins_charge import charge_features
from model.head_b_gbm import build_feature_vector


SEEDS = (42, 43, 44)
BAD_FRACTION = 0.15
MIN_TRAIN_ROWS = 40
MIN_TEST_ROWS = 20
MIN_CLASS_ROWS = 3
FEATURE_MODES = ("biophysical_charge", "cdr_aware", "frozen_plm")
TRAINING_ARMS = ("library_only", "diverse_only", "mixed")


def _is_binary(values: Iterable[float]) -> bool:
    observed = set(values)
    return bool(observed) and observed <= {0.0, 1.0}


def assign_group_labels(rows: list[dict], bad_fraction: float = BAD_FRACTION) -> None:
    """Assign deterministic binary badness labels within one source/assay."""
    if not rows:
        return
    values = [float(row["endpoint_value"]) for row in rows]
    label_types = {str(row.get("label_type") or "") for row in rows}
    if label_types == {"num_react"}:
        for row in rows:
            row["target"] = int(float(row["endpoint_value"]) >= 2.0)
        return
    if _is_binary(values):
        for row in rows:
            row["target"] = int(float(row["endpoint_value"]))
        return

    directions = {str(row.get("endpoint_direction") or "higher_bad") for row in rows}
    if len(directions) != 1:
        raise ValueError("one source/assay group has mixed endpoint directions")
    reverse = directions.pop() == "higher_bad"
    bad_count = max(1, math.ceil(len(rows) * bad_fraction))
    ordered = sorted(
        rows,
        key=lambda row: (float(row["endpoint_value"]), str(row["record_id"])),
        reverse=reverse,
    )
    bad_ids = {row["record_id"] for row in ordered[:bad_count]}
    for row in rows:
        row["target"] = int(row["record_id"] in bad_ids)


def _near_vh_matches(train_vh: set[str], test_vh: set[str], threshold: float) -> set[str]:
    """Find aligned near VHs with exact vectorized equal-length identity."""
    import numpy as np

    matched: set[str] = set()
    train_by_length: dict[int, list[str]] = defaultdict(list)
    test_by_length: dict[int, list[str]] = defaultdict(list)
    for sequence in train_vh:
        train_by_length[len(sequence)].append(sequence)
    for sequence in test_vh:
        test_by_length[len(sequence)].append(sequence)
    for length, train_sequences in train_by_length.items():
        test_sequences = test_by_length.get(length)
        if not test_sequences:
            continue
        test_matrix = np.frombuffer("".join(test_sequences).encode("ascii"), dtype=np.uint8).reshape(-1, length)
        limit = math.floor((1.0 - threshold) * length + 1e-9)
        for start in range(0, len(train_sequences), 256):
            batch = train_sequences[start:start + 256]
            train_matrix = np.frombuffer("".join(batch).encode("ascii"), dtype=np.uint8).reshape(-1, length)
            minimum = (train_matrix[:, None, :] != test_matrix[None, :, :]).sum(axis=2).min(axis=1)
            matched.update(sequence for sequence, distance in zip(batch, minimum) if int(distance) <= limit)

    return matched


def strip_holdout_overlap(
    train: list[dict],
    test: list[dict],
    *,
    near_identity: float = 0.90,
) -> tuple[list[dict], dict[str, int | float]]:
    """Remove exact VH, exact VL, then near-VH matches to a held-out source."""
    test_vh = {str(row.get("vh") or "") for row in test if row.get("vh")}
    test_vl = {str(row.get("vl") or "") for row in test if row.get("vl")}
    candidate_train_vh = {
        str(row.get("vh") or "") for row in train
        if row.get("vh") and str(row.get("vh")) not in test_vh
    }
    near_vh = (
        _near_vh_matches(candidate_train_vh, test_vh, near_identity)
        if near_identity < 1.0 and test_vh else set()
    )
    kept: list[dict] = []
    audit: dict[str, int | float] = {
        "input_train_rows": len(train),
        "exact_vh_removed": 0,
        "exact_vl_removed": 0,
        "near_vh_removed": 0,
        "near_identity_threshold": near_identity,
        "near_identity_method": "exact positional identity for equal-length VH sequences",
    }
    for row in train:
        vh = str(row.get("vh") or "")
        vl = str(row.get("vl") or "")
        if vh and vh in test_vh:
            audit["exact_vh_removed"] += 1
        elif vl and vl in test_vl:
            audit["exact_vl_removed"] += 1
        elif vh and vh in near_vh:
            audit["near_vh_removed"] += 1
        else:
            kept.append(row)
    audit["retained_train_rows"] = len(kept)
    return kept, audit


def source_balanced_sample(
    rows: list[dict], *, max_per_source_class: int, seed: int
) -> list[dict]:
    """Cap every source/class independently with a reproducible sample."""
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source"]), int(row["target"]))].append(row)
    selected: list[dict] = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: str(row["record_id"]))
        if len(group) > max_per_source_class:
            group_seed = int(hash_payload({"seed": seed, "source_class": key})[:16], 16)
            group = random.Random(group_seed).sample(group, max_per_source_class)
        selected.extend(group)
    return sorted(selected, key=lambda row: str(row["record_id"]))


def _ece(labels: list[int], scores: list[float], bins: int = 10) -> float:
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [i for i, score in enumerate(scores)
                   if lower <= score < upper or (index == bins - 1 and score == 1.0)]
        if members:
            confidence = statistics.fmean(scores[i] for i in members)
            frequency = statistics.fmean(labels[i] for i in members)
            error += len(members) / total * abs(confidence - frequency)
    return error


def _screening_metrics(labels: list[int], scores: list[float], fraction: float) -> tuple[float, float]:
    positives = sum(labels)
    if positives == 0:
        return 0.0, 0.0
    count = max(1, math.ceil(len(labels) * fraction))
    top = sorted(range(len(scores)), key=lambda i: (scores[i], -i), reverse=True)[:count]
    caught = sum(labels[i] for i in top)
    catch = caught / positives
    enrichment = (caught / count) / (positives / len(labels))
    return catch, enrichment


def classification_metrics(labels: list[int], scores: list[float]) -> dict[str, float | int]:
    """Required ranking, screening, and uncalibrated probability metrics."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    if len(labels) != len(scores) or len(set(labels)) != 2:
        raise ValueError("classification metrics require equal lengths and both classes")
    catch10, enrichment10 = _screening_metrics(labels, scores, 0.10)
    catch15, enrichment15 = _screening_metrics(labels, scores, 0.15)
    return {
        "n": len(labels),
        "positives": sum(labels),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "catch_rate_at_10pct": catch10,
        "catch_rate_at_15pct": catch15,
        "enrichment_at_10pct": enrichment10,
        "enrichment_at_15pct": enrichment15,
        "brier": float(brier_score_loss(labels, scores)),
        "ece_10bin": _ece(labels, scores),
    }


def _row_from_sql(row: sqlite3.Row) -> dict:
    return {
        "record_id": row["record_id"],
        "source": row["source"],
        "source_class": row["source_class"],
        "property": row["property"],
        "assay_metric": row["assay_metric"],
        "label_type": row["label_type"],
        "endpoint_value": float(row["endpoint_value"]),
        "endpoint_direction": row["endpoint_direction"],
        "supervision_tier": row["supervision_tier"],
        "evidence_tier": row["evidence_tier"],
        "training_eligible": bool(row["training_eligible"]),
        "holdout_eligible": bool(row["holdout_eligible"]),
        "vh": row["vh"] or "",
        "vl": row["vl"] or "",
        "molecule_key": row["molecule_key"],
        "split_group": row["split_group"],
        "cdrh1": row["cdr_h1"] or "",
        "cdrh2": row["cdr_h2"] or "",
        "cdrh3": row["cdr_h3"] or "",
    }


_SELECT_COLUMNS = (
    "record_id,source,source_class,property,assay_metric,label_type,endpoint_value,"
    "endpoint_direction,supervision_tier,evidence_tier,training_eligible,holdout_eligible,"
    "vh,vl,molecule_key,split_group,cdr_h1,cdr_h2,cdr_h3"
)


def load_property_rows(
    database: str | Path,
    property_name: str,
    *,
    prefetch_per_binary_class: int,
) -> tuple[list[dict], dict]:
    """Load all small groups and a large balanced prefetch from binary libraries."""
    rows: list[dict] = []
    manifest_groups: list[dict] = []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        metadata = connection.execute(
            "SELECT source,supervision_tier,evidence_tier,training_eligible,holdout_eligible,"
            "assay_metric,label_type,COUNT(*) n,"
            "MIN(endpoint_value) lo,MAX(endpoint_value) hi,COUNT(DISTINCT endpoint_value) distinct_n "
            "FROM records WHERE status='accepted' AND property=? "
            "AND (training_eligible=1 OR holdout_eligible=1) "
            "GROUP BY source,supervision_tier,evidence_tier,training_eligible,holdout_eligible,"
            "assay_metric,label_type "
            "ORDER BY source,supervision_tier,evidence_tier,assay_metric,label_type",
            (property_name,),
        ).fetchall()
        for group in metadata:
            params = (
                property_name, group["source"], group["supervision_tier"],
                group["evidence_tier"], group["training_eligible"], group["holdout_eligible"],
                group["assay_metric"], group["label_type"],
            )
            where = (
                "status='accepted' AND property=? AND source=? AND supervision_tier=? "
                "AND evidence_tier=? AND training_eligible=? AND holdout_eligible=? "
                "AND assay_metric=? AND label_type=?"
            )
            is_large_binary = (
                int(group["n"]) > 2 * prefetch_per_binary_class
                and int(group["distinct_n"]) <= 2
                and float(group["lo"]) >= 0.0
                and float(group["hi"]) <= 1.0
            )
            fetched: list[sqlite3.Row] = []
            if is_large_binary:
                for target in (0.0, 1.0):
                    fetched.extend(connection.execute(
                        f"SELECT {_SELECT_COLUMNS} FROM records WHERE {where} "
                        "AND endpoint_value=? ORDER BY record_id LIMIT ?",
                        (*params, target, prefetch_per_binary_class),
                    ).fetchall())
            else:
                fetched = connection.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM records WHERE {where} ORDER BY record_id",
                    params,
                ).fetchall()
            converted = [_row_from_sql(row) for row in fetched]
            assign_group_labels(converted)
            rows.extend(converted)
            manifest_groups.append({
                "source": group["source"],
                "tier": group["supervision_tier"],
                "evidence_tier": group["evidence_tier"],
                "training_eligible": bool(group["training_eligible"]),
                "holdout_eligible": bool(group["holdout_eligible"]),
                "assay": group["assay_metric"],
                "available_n": int(group["n"]),
                "loaded_n": len(converted),
                "large_binary_prefetch": is_large_binary,
            })
        registry_hash_row = connection.execute(
            "SELECT value FROM registry_meta WHERE key='registry_hash'"
        ).fetchone()
    return rows, {
        "property": property_name,
        "available_rows": sum(group["available_n"] for group in manifest_groups),
        "loaded_rows": len(rows),
        "groups": manifest_groups,
        "registry_hash": registry_hash_row[0] if registry_hash_row else None,
    }


def _model(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.03,
        min_child_samples=20,
        reg_lambda=5.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )


def _candidate_model(learner: str, seed: int):
    if learner == "lgbm":
        return _model(seed)
    if learner == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=seed,
        )
    if learner == "ridge":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, random_state=seed),
        )
    raise ValueError(f"unsupported learner: {learner}")


def _weights(rows: list[dict]) -> list[float]:
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        counts[(str(row["source"]), int(row["target"]))] += 1
    raw = [1.0 / counts[(str(row["source"]), int(row["target"]))] for row in rows]
    scale = len(raw) / sum(raw)
    return [weight * scale for weight in raw]


def _fit_calibration_split(
    featured: list[tuple[dict, list[float]]], seed: int
) -> tuple[list[tuple[dict, list[float]]], list[tuple[dict, list[float]]]]:
    """Deterministic source/class-stratified 80/20 train-only split."""
    groups: dict[tuple[str, int], list[tuple[dict, list[float]]]] = defaultdict(list)
    for item in featured:
        groups[(str(item[0]["source"]), int(item[0]["target"]))].append(item)
    fit, calibration = [], []
    for key, group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda item: hash_payload({"seed": seed, "key": key, "record": item[0]["record_id"]}),
        )
        calibration_count = max(1, round(len(ordered) * 0.20)) if len(ordered) >= 5 else 0
        calibration.extend(ordered[:calibration_count])
        fit.extend(ordered[calibration_count:])
    return fit, calibration


@dataclass(frozen=True)
class IdentityCalibrator:
    reason: str

    def __call__(self, scores) -> list[float]:
        return [float(score) for score in scores]


@dataclass(frozen=True)
class PlattCalibrator:
    coefficient: float
    intercept: float

    def __call__(self, scores) -> list[float]:
        import numpy as np

        values = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(values / (1.0 - values))
        transformed = self.coefficient * logits + self.intercept
        return [float(value) for value in 1.0 / (1.0 + np.exp(-transformed))]


def _platt_calibrator(labels: list[int], raw_scores: list[float]):
    """Return a pickleable train-only monotone Platt transform."""
    if len(labels) < 20 or len(set(labels)) != 2:
        reason = "identity_insufficient_calibration_rows"
        return IdentityCalibrator(reason), reason
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(np.asarray(raw_scores, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    fitted = LogisticRegression(C=1.0, random_state=0).fit(logits, labels)
    coefficient = float(fitted.coef_[0, 0])
    if coefficient <= 0:
        reason = "identity_nonpositive_platt_slope"
        return IdentityCalibrator(reason), reason
    return (
        PlattCalibrator(coefficient, float(fitted.intercept_[0])),
        "platt_logistic_train_only",
    )


def fit_candidate(
    learner: str,
    fit_featured: list[tuple[dict, list[float]]],
    calibration_featured: list[tuple[dict, list[float]]],
    test_featured: list[tuple[dict, list[float]]],
    seed: int,
) -> dict:
    """Fit one comparable classifier and its train-only calibration."""
    model = _candidate_model(learner, seed)
    X = [vector for _, vector in fit_featured]
    y = [int(row["target"]) for row, _ in fit_featured]
    weights = _weights([row for row, _ in fit_featured])
    if learner == "ridge":
        model.fit(X, y, logisticregression__sample_weight=weights)
    else:
        model.fit(X, y, sample_weight=weights)

    calibration_raw = (
        [float(value) for value in model.predict_proba(
            [vector for _, vector in calibration_featured]
        )[:, 1]]
        if calibration_featured else []
    )
    calibrator, calibration_method = _platt_calibrator(
        [int(row["target"]) for row, _ in calibration_featured], calibration_raw
    )
    raw_scores = [float(value) for value in model.predict_proba(
        [vector for _, vector in test_featured]
    )[:, 1]]
    return {
        "model": model,
        "calibrator": calibrator,
        "raw_scores": raw_scores,
        "scores": calibrator(raw_scores),
        "calibration_method": calibration_method,
    }


def fit_applicability_index(
    train_featured: list[tuple[dict, list[float]]],
) -> dict:
    """Fit a serializable normalized source-centroid applicability index."""
    import numpy as np

    if not train_featured:
        raise ValueError("applicability index requires training rows")
    train_matrix = np.asarray([vector for _, vector in train_featured], dtype=float)
    center = train_matrix.mean(axis=0)
    scale = train_matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_scaled = (train_matrix - center) / scale
    source_indexes: dict[str, list[int]] = defaultdict(list)
    for index, (row, _) in enumerate(train_featured):
        source_indexes[str(row["source"])].append(index)
    source_names = sorted(source_indexes)
    centroids = np.asarray([
        train_scaled[source_indexes[source]].mean(axis=0) for source in source_names
    ])
    own_distances = []
    for source, indexes in source_indexes.items():
        centroid = centroids[source_names.index(source)]
        own_distances.extend(
            np.sqrt(((train_scaled[indexes] - centroid) ** 2).mean(axis=1))
        )
    return {
        "distance": "RMS normalized feature distance to nearest source centroid",
        "center": center.tolist(),
        "scale": scale.tolist(),
        "source_names": source_names,
        "centroids": centroids.tolist(),
        "train_p95": float(np.quantile(own_distances, 0.95)),
        "train_p99": float(np.quantile(own_distances, 0.99)),
    }


def score_applicability(
    index: dict,
    test_featured: list[tuple[dict, list[float]]],
) -> list[dict]:
    """Score rows with a fitted applicability index."""
    import numpy as np

    if not test_featured:
        return []
    matrix = np.asarray([vector for _, vector in test_featured], dtype=float)
    center = np.asarray(index["center"], dtype=float)
    scale = np.asarray(index["scale"], dtype=float)
    centroids = np.asarray(index["centroids"], dtype=float)
    scaled = (matrix - center) / scale
    distances = np.sqrt(((scaled[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2))
    nearest = distances.argmin(axis=1)
    records = []
    for row_index, source_index in enumerate(nearest):
        distance = float(distances[row_index, source_index])
        label = (
            "in_domain" if distance <= index["train_p95"]
            else "borderline" if distance <= index["train_p99"]
            else "out_of_domain"
        )
        records.append({
            "ood_distance": distance,
            "applicability": label,
            "nearest_training_source": index["source_names"][int(source_index)],
        })
    return records


def _applicability(
    train_featured: list[tuple[dict, list[float]]],
    test_featured: list[tuple[dict, list[float]]],
) -> tuple[list[dict], dict]:
    """Compatibility wrapper preserving the existing compact report schema."""
    index = fit_applicability_index(train_featured)
    records = score_applicability(index, test_featured)
    summary = {
        "distance": index["distance"],
        "train_p95": index["train_p95"],
        "train_p99": index["train_p99"],
        "source_centroids": index["source_names"],
    }
    return records, summary


def _feature_row(row: dict) -> dict:
    return {
        "variant_sequence": row["vh"],
        "vh_sequence": row["vh"],
        "vl_sequence": row["vl"] or None,
        "cdrh1": row.get("cdrh1", ""),
        "cdrh2": row.get("cdrh2", ""),
        "cdrh3": row.get("cdrh3", ""),
    }


def make_featurizer(
    mode: str,
    baseline_cache: dict[str, list[float]] | None = None,
) -> Callable[[dict], list[float] | None]:
    if mode not in FEATURE_MODES:
        raise ValueError(f"unsupported feature mode: {mode}")
    cache: dict[str, list[float] | None] = {}
    shared_baseline = baseline_cache if baseline_cache is not None else {}

    def feature(row: dict) -> list[float] | None:
        record_id = str(row["record_id"])
        if record_id in cache:
            return cache[record_id]
        model_row = _feature_row(row)
        baseline = shared_baseline.get(record_id)
        if baseline is None:
            baseline = build_feature_vector(model_row, use_plm=False) + charge_features(model_row)
            shared_baseline[record_id] = baseline
        if mode == "biophysical_charge":
            vector: list[float] | None = baseline
        elif mode == "cdr_aware":
            vector = baseline + cdr_feature_vector(model_row, "heavy_all")
        else:
            from features.plm import embed_combined
            try:
                vector = list(embed_combined(row["vh"])) + baseline
            except RuntimeError:
                vector = None
        cache[record_id] = vector
        return vector

    return feature


def _source_result(
    *,
    property_name: str,
    held_source: str,
    train: list[dict],
    test: list[dict],
    seed: int,
    feature_mode: str,
    overlap_audit: dict[str, int | float],
    featurize: Callable[[dict], list[float] | None],
    prediction_writer: csv.DictWriter | None,
    experiment_key: str,
) -> dict:
    if len(test) < MIN_TEST_ROWS:
        return {"source": held_source, "skip": f"only {len(test)} compatible test rows"}
    labels = [int(row["target"]) for row in test]
    if min(sum(labels), len(labels) - sum(labels)) < MIN_CLASS_ROWS:
        return {"source": held_source, "skip": "test source has too few rows in one class"}

    train_featured = [(row, featurize(row)) for row in train]
    test_featured = [(row, featurize(row)) for row in test]
    train_featured = [(row, vector) for row, vector in train_featured if vector is not None]
    test_featured = [(row, vector) for row, vector in test_featured if vector is not None]
    coverage = {
        "train": len(train_featured) / len(train) if train else 0.0,
        "test": len(test_featured) / len(test) if test else 0.0,
    }
    if feature_mode == "frozen_plm" and (coverage["train"] < 0.80 or coverage["test"] < 0.80):
        return {
            "source": held_source,
            "skip": "frozen PLM cache coverage below 80%; zero-fill is forbidden",
            "feature_coverage": coverage,
            "overlap_audit": overlap_audit,
        }
    if len(train_featured) < MIN_TRAIN_ROWS:
        return {"source": held_source, "skip": f"only {len(train_featured)} training rows after filtering",
                "feature_coverage": coverage, "overlap_audit": overlap_audit}
    train_labels = [int(row["target"]) for row, _ in train_featured]
    if min(sum(train_labels), len(train_labels) - sum(train_labels)) < MIN_CLASS_ROWS:
        return {"source": held_source, "skip": "training rows have too few rows in one class",
                "feature_coverage": coverage, "overlap_audit": overlap_audit}

    fit_featured, calibration_featured = _fit_calibration_split(train_featured, seed)
    if len(fit_featured) < MIN_TRAIN_ROWS or len({int(row["target"]) for row, _ in fit_featured}) != 2:
        fit_featured, calibration_featured = train_featured, []
    fitted = fit_candidate(
        "lgbm", fit_featured, calibration_featured, test_featured, seed
    )
    calibration_method = fitted["calibration_method"]
    test_rows = [row for row, _ in test_featured]
    raw_scores = fitted["raw_scores"]
    scores = fitted["scores"]
    applicability, applicability_fit = _applicability(fit_featured, test_featured)

    per_assay: dict[str, dict] = {}
    assay_indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test_rows):
        assay_indexes[str(row["assay_metric"])].append(index)
    for assay, indexes in sorted(assay_indexes.items()):
        assay_labels = [int(test_rows[index]["target"]) for index in indexes]
        if len(indexes) >= MIN_TEST_ROWS and len(set(assay_labels)) == 2:
            per_assay[assay] = classification_metrics(
                assay_labels, [scores[index] for index in indexes]
            )
    if not per_assay:
        return {"source": held_source, "skip": "no assay has enough rows and both classes",
                "feature_coverage": coverage, "overlap_audit": overlap_audit}

    metric_names = (
        "roc_auc", "pr_auc", "catch_rate_at_10pct", "catch_rate_at_15pct",
        "enrichment_at_10pct", "enrichment_at_15pct", "brier", "ece_10bin",
    )
    result = {
        "source": held_source,
        "test_n": len(test_rows),
        "test_positives": sum(int(row["target"]) for row in test_rows),
        "train_n": len(train_featured),
        "model_fit_n": len(fit_featured),
        "calibration_n": len(calibration_featured),
        "calibration_method": calibration_method,
        "applicability_fit": applicability_fit,
        "train_sources": sorted({str(row["source"]) for row, _ in train_featured}),
        "feature_coverage": coverage,
        "overlap_audit": overlap_audit,
        "per_assay": per_assay,
        **{
            name: statistics.fmean(float(metrics[name]) for metrics in per_assay.values())
            for name in metric_names
        },
    }
    if feature_mode == "cdr_aware":
        result["cdr_parse_rate"] = parse_rate([_feature_row(row) for row in test_rows], "heavy_all")
    if prediction_writer is not None:
        for row, raw_score, score, domain in zip(test_rows, raw_scores, scores, applicability):
            prediction_writer.writerow({
                "experiment": experiment_key,
                "seed": seed,
                "property": property_name,
                "feature_mode": feature_mode,
                "held_source": held_source,
                "assay": row["assay_metric"],
                "record_id": row["record_id"],
                "label": row["target"],
                "raw_score": raw_score,
                "score": score,
                **domain,
            })
    return result


def _training_arm_rows(rows: list[dict], arm: str) -> list[dict]:
    if arm == "library_only":
        return [row for row in rows if row["source_class"] == "LIBRARY"]
    if arm == "diverse_only":
        return [row for row in rows if row["source_class"] == "DIVERSE"]
    if arm == "mixed":
        return rows
    raise ValueError(f"unsupported training arm: {arm}")


def eligible_split(rows: list[dict], held_source: str) -> tuple[list[dict], list[dict]]:
    """Apply evidence eligibility before supervision-tier and arm filters."""
    train = [
        row for row in rows
        if row["source"] != held_source and row["training_eligible"]
    ]
    test = [
        row for row in rows
        if row["source"] == held_source and row["holdout_eligible"]
    ]
    return train, test


def _run_configuration(
    *,
    property_name: str,
    rows: list[dict],
    held_sources: list[str],
    seed: int,
    feature_mode: str,
    training_arm: str,
    training_tiers: tuple[str, ...],
    test_tier: str,
    max_per_source_class: int,
    near_identity: float,
    prediction_writer: csv.DictWriter | None,
    overlap_cache: dict[tuple, tuple[list[dict], dict[str, int | float]]],
    featurizers: dict[str, Callable[[dict], list[float] | None]],
) -> dict:
    key = f"{property_name}:{feature_mode}:{training_arm}:{'+'.join(training_tiers)}:{test_tier}"
    per_source = []
    for held_source in held_sources:
        eligible_train, eligible_test = eligible_split(rows, held_source)
        test = [row for row in eligible_test if row["supervision_tier"] == test_tier]
        cache_key = (held_source, training_arm, training_tiers, test_tier, seed)
        cached_overlap = overlap_cache.get(cache_key)
        if cached_overlap is None:
            eligible = _training_arm_rows(
                [row for row in eligible_train if row["supervision_tier"] in training_tiers],
                training_arm,
            )
            raw_train = source_balanced_sample(
                eligible,
                max_per_source_class=max_per_source_class,
                seed=seed,
            )
            held_rows = [row for row in rows if row["source"] == held_source]
            cached_overlap = strip_holdout_overlap(
                raw_train,
                held_rows,
                near_identity=near_identity,
            )
            overlap_cache[cache_key] = cached_overlap
        train, universal_audit = cached_overlap
        overlap_audit = {
            **universal_audit,
            "eligible_train_rows_after_overlap": len(train),
            "audit_scope": "sampled eligible training rows versus all held-source formats",
        }
        per_source.append(_source_result(
            property_name=property_name,
            held_source=held_source,
            train=train,
            test=test,
            seed=seed,
            feature_mode=feature_mode,
            overlap_audit=overlap_audit,
            featurize=featurizers[feature_mode],
            prediction_writer=prediction_writer,
            experiment_key=key,
        ))
    scored = [result for result in per_source if "roc_auc" in result]
    return {
        "experiment": key,
        "property": property_name,
        "seed": seed,
        "feature_mode": feature_mode,
        "training_arm": training_arm,
        "training_tiers": list(training_tiers),
        "test_tier": test_tier,
        "macro_source_roc_auc": statistics.fmean(result["roc_auc"] for result in scored) if scored else None,
        "macro_source_pr_auc": statistics.fmean(result["pr_auc"] for result in scored) if scored else None,
        "scored_sources": len(scored),
        "per_source": per_source,
    }


def _bootstrap_source_ci(values: dict[str, float], seed: int = 20260717, draws: int = 2000) -> list[float] | None:
    if not values:
        return None
    sources = sorted(values)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(statistics.fmean(values[rng.choice(sources)] for _ in sources))
    samples.sort()
    return [samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]]


def summarize_experiments(experiments: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for experiment in experiments:
        grouped[experiment["experiment"]].append(experiment)
    summaries = []
    for key, runs in sorted(grouped.items()):
        seed_scores = [run["macro_source_roc_auc"] for run in runs
                       if run["macro_source_roc_auc"] is not None]
        source_values: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            for result in run["per_source"]:
                if "roc_auc" in result:
                    source_values[result["source"]].append(float(result["roc_auc"]))
        source_means = {source: statistics.fmean(values) for source, values in source_values.items()}
        first = runs[0]
        summaries.append({
            "experiment": key,
            "property": first["property"],
            "feature_mode": first["feature_mode"],
            "training_arm": first["training_arm"],
            "training_tiers": first["training_tiers"],
            "test_tier": first["test_tier"],
            "seeds_requested": [run["seed"] for run in runs],
            "seed_macro_source_auc": seed_scores,
            "mean_macro_source_auc": statistics.fmean(seed_scores) if seed_scores else None,
            "seed_std_auc": statistics.pstdev(seed_scores) if len(seed_scores) > 1 else 0.0 if seed_scores else None,
            "source_mean_auc": source_means,
            "source_bootstrap_ci95": _bootstrap_source_ci(source_means),
        })
    return summaries


def _development_sources(policy: dict, property_name: str) -> list[str]:
    return [entry["name"] for entry in policy["properties"][property_name]["source_roles"]["development"]]


def run_matrix(
    *,
    database: str | Path,
    policy_path: str | Path,
    seeds: tuple[int, ...] = SEEDS,
    max_per_source_class: int = 5000,
    near_identity: float = 0.90,
    prediction_writer: csv.DictWriter | None = None,
) -> dict:
    policy = load_policy(policy_path)
    validate_policy(policy)
    if any(entry["name"] == "gdpa3" and entry.get("prior_use")
           for item in policy["properties"].values()
           for entries in item["source_roles"].values() for entry in entries):
        raise ValueError("GDPa3 must remain unspent")

    experiments: list[dict] = []
    manifests: dict[str, dict] = {}
    for property_name in ("polyreactivity", "aggregation", "hydrophobicity"):
        print(f"loading {property_name} registry rows", flush=True)
        rows, manifest = load_property_rows(
            database,
            property_name,
            prefetch_per_binary_class=max_per_source_class * 3,
        )
        manifests[property_name] = manifest
        held_sources = _development_sources(policy, property_name)
        overlap_cache: dict[tuple, tuple[list[dict], dict[str, int | float]]] = {}
        baseline_cache: dict[str, list[float]] = {}
        featurizers = {mode: make_featurizer(mode, baseline_cache) for mode in FEATURE_MODES}
        configurations = [
            (feature_mode, arm, ("paired_fv_supervised",), "paired_fv_supervised")
            for feature_mode in FEATURE_MODES
            for arm in TRAINING_ARMS
        ]
        configurations.append((
            "biophysical_charge", "mixed",
            ("paired_fv_supervised", "single_chain_supervised"),
            "paired_fv_supervised",
        ))
        configurations.append((
            "biophysical_charge", "diverse_only",
            ("single_chain_supervised",), "single_chain_supervised",
        ))
        for seed in seeds:
            print(f"  seed {seed}", flush=True)
            for feature_mode, arm, train_tiers, test_tier in configurations:
                experiments.append(_run_configuration(
                    property_name=property_name,
                    rows=rows,
                    held_sources=held_sources,
                    seed=seed,
                    feature_mode=feature_mode,
                    training_arm=arm,
                    training_tiers=train_tiers,
                    test_tier=test_tier,
                    max_per_source_class=max_per_source_class,
                    near_identity=near_identity,
                    prediction_writer=prediction_writer,
                    overlap_cache=overlap_cache,
                    featurizers=featurizers,
                ))
            if property_name == "polyreactivity":
                cdr_sources = sorted({
                    str(row["source"])
                    for row in rows
                    if row["supervision_tier"] == "cdr_supervised" and row["holdout_eligible"]
                })
                experiments.append(_run_configuration(
                    property_name=property_name,
                    rows=rows,
                    held_sources=cdr_sources,
                    seed=seed,
                    feature_mode="cdr_aware",
                    training_arm="diverse_only",
                    training_tiers=("cdr_supervised",),
                    test_tier="cdr_supervised",
                    max_per_source_class=max_per_source_class,
                    near_identity=near_identity,
                    prediction_writer=prediction_writer,
                    overlap_cache=overlap_cache,
                    featurizers=featurizers,
                ))
    summaries = summarize_experiments(experiments)
    return {
        "schema_version": "1",
        "benchmark_role": "development_only",
        "primary_metric": "macro_source_roc_auc",
        "seeds": list(seeds),
        "max_per_source_class": max_per_source_class,
        "near_vh_identity_filter": near_identity,
        "near_vh_identity_method": "exact positional identity for equal-length VH sequences",
        "bad_fraction_for_continuous_endpoints": BAD_FRACTION,
        "policy_hash": hash_payload(policy),
        "registry_manifests": manifests,
        "experiments": experiments,
        "summaries": summaries,
        "important_limitations": [
            "No locked-validation source is assigned; all scores are development evidence.",
            "GDPa3 was not loaded or inspected.",
            "Frozen PLM comparisons require at least 80% real cache coverage on train and test.",
            "Single-chain and CDR-only evidence is auxiliary and cannot promote the paired-Fv head.",
            "Verified reconstructed evidence may train but is never eligible for held-out evaluation.",
        ],
    }


def _print_report(report: dict) -> None:
    print("\nDevelopability baseline matrix — development sources only")
    for property_name in ("polyreactivity", "aggregation", "hydrophobicity"):
        candidates = [item for item in report["summaries"]
                      if item["property"] == property_name and item["mean_macro_source_auc"] is not None]
        candidates.sort(key=lambda item: item["mean_macro_source_auc"], reverse=True)
        print(f"\n  {property_name}")
        for item in candidates[:8]:
            ci = item["source_bootstrap_ci95"]
            print(
                f"    {item['feature_mode']:<20} {item['training_arm']:<13} "
                f"{'+'.join(item['training_tiers']):<48} "
                f"AUC={item['mean_macro_source_auc']:.3f} "
                f"seed_sd={item['seed_std_auc']:.3f} sources={len(item['source_mean_auc'])} "
                f"CI95={ci[0]:.3f}-{ci[1]:.3f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("results/developability_registry.sqlite"))
    parser.add_argument("--policy", type=Path, default=Path("tools/benchmark_policy.json"))
    parser.add_argument("--out", type=Path, default=Path("results/developability_baseline_matrix.json"))
    parser.add_argument("--predictions", type=Path, default=Path("results/developability_baseline_predictions.csv"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-per-source-class", type=int, default=5000)
    parser.add_argument("--near-identity", type=float, default=0.90)
    args = parser.parse_args(argv)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    fields = ("experiment", "seed", "property", "feature_mode", "held_source",
              "assay", "record_id", "label", "raw_score", "score", "ood_distance",
              "applicability", "nearest_training_source")
    with args.predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        report = run_matrix(
            database=args.database,
            policy_path=args.policy,
            seeds=seeds,
            max_per_source_class=args.max_per_source_class,
            near_identity=args.near_identity,
            prediction_writer=writer,
        )
    report["predictions"] = str(args.predictions.resolve())
    report["predictions_sha256"] = file_sha256(args.predictions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_report(report)
    print(f"\nwrote {args.out.resolve()}")
    print(f"wrote {args.predictions.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
