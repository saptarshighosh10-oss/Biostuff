"""Head B — antibody-aggregation risk driver (PLM + biophysical features).

Ties the Phase 1/2/3 pieces together for the ANTIBODY model, kept strictly
separate from Head A (general ProteinGym fitness) — separate metrics, separate
report, separate artifact:

  normalized antibody rows
    → feature vector [ PLM(esm2|ablang2, cached) | biophysical(28) ]   (model.features + features.plm)
    → leave-cluster-out grouped folds                                  (model.plan_c_cv)
    → per-assay regressor (LightGBM by default; pluggable)
    → per-assay Spearman / MAE / top-K enrichment + group-bootstrap CI (model.metrics)

Heavy libs (lightgbm/sklearn/torch) are imported lazily so the orchestration is
testable with a stub regressor and no PLM cache. Everything degrades to
biophysical-only features when the PLM cache is absent (``use_plm=False``), so a
first pass runs before any weights are downloaded.
"""

from __future__ import annotations

from collections import defaultdict
import math
import json
import pickle
from pathlib import Path
from typing import Callable

from model.features import extract_from_sequence, features_to_vector
from model.metrics import spearman_rho, mae, topk_enrichment, group_bootstrap_ci
from model.plan_c_cv import composite_group_key, grouped_folds, assert_no_group_leakage
from data.cohort import is_head_b_aggregation_row
from features import plm

# GDPa ships its own homology-cluster fold column; honor it over a recomputed
# split when present (design: leave-homology-cluster-out).
_BENCHMARK_FOLD_FIELDS = (
    "hierarchical_cluster_IgG_isotype_stratified_fold",
    "cluster_fold",
    "fold",
)
MIN_ROWS_PER_ASSAY = 20
_PLM_FEATURE_WIDTH: int | None = None
def _plm_feature_width() -> int:
    """Return the cached combined PLM width, or zero when no complete cache exists."""
    global _PLM_FEATURE_WIDTH
    if _PLM_FEATURE_WIDTH is not None:
        return _PLM_FEATURE_WIDTH
    try:
        from features import plm
        widths = []
        for tag in (plm.MODEL_TAG_ESM2, plm.MODEL_TAG_ABLANG2):
            files = sorted((plm.CACHE_ROOT / tag).glob("*.json"))
            if not files:
                _PLM_FEATURE_WIDTH = 0
                return 0
            widths.append(len(json.loads(files[0].read_text(encoding="utf-8"))))
        _PLM_FEATURE_WIDTH = sum(widths)
    except (OSError, TypeError, ValueError):
        _PLM_FEATURE_WIDTH = 0
    return _PLM_FEATURE_WIDTH


def build_feature_vector(
    row: dict,
    use_plm: bool = True,
    add_paired_ablang: bool = False,
) -> list[float]:
    """Build ``[base PLM | paired AbLang2 | flags | biophysical]`` features."""
    seq = row.get("variant_sequence") or row.get("vh_sequence") or row.get("sequence", "")
    biophysical = features_to_vector(extract_from_sequence(seq, row.get("mutations", [])))
    features: list[float] = []
    if use_plm:
        try:
            features.extend(plm.embed_combined(seq))
        except RuntimeError:
            features.extend([0.0] * _plm_feature_width())
    if add_paired_ablang:
        vh = str(row.get("vh_sequence") or "").strip()
        vl = str(row.get("vl_sequence") or "").strip()
        if vh and vl:
            features.extend(plm.embed_ablang2_pairs([(vh, vl)])[0])
            features.append(1.0)
        else:
            features.extend([0.0] * plm.ABLANG2_SEQUENCE_CODING_WIDTH)
            features.append(0.0)
    return features + biophysical


def _group_key(row: dict) -> str:
    """Prefer the benchmark's provided homology-cluster fold; else the composite
    (study, campaign, identity_component) key."""
    molecule_group = row.get("molecule_group_id")
    if molecule_group:
        return f"molecule:{molecule_group}"
    for field in _BENCHMARK_FOLD_FIELDS:
        val = row.get(field)
        if val is not None and str(val).strip() != "":
            return f"benchfold:{val}"
    return composite_group_key(row)


def _assay_key(row: dict) -> tuple[str, str, str, str]:
    """Keep identical metric names separate when they come from different assays."""
    return (
        str(row.get("source") or "unknown"),
        str(row.get("assay_id") or row.get("assay_metric") or "unknown"),
        str(row.get("assay_metric") or "unknown"),
        "",
    )


def _failure_labels(values: list[float], direction: str, percentile: float) -> list[int]:
    """1 = developability failure. `higher_bad` → top tail fails; else bottom tail."""
    if not values:
        return []
    ordered = sorted(values)
    n = len(ordered)
    if direction == "higher_bad":
        thresh = ordered[min(n - 1, int(n * percentile))]
        return [1 if v >= thresh else 0 for v in values]
    thresh = ordered[max(0, int(n * (1 - percentile)))]
    return [1 if v <= thresh else 0 for v in values]


def _risk_scores(preds: list[float], direction: str) -> list[float]:
    """Orient predictions so higher = more likely to fail (for enrichment/ranking)."""
    return preds if direction == "higher_bad" else [-p for p in preds]


def _default_regressor_factory() -> Callable[[], object]:
    def factory():
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=300, num_leaves=31, random_state=42, verbose=-1)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=42)
    return factory


def evaluate_assay(
    rows: list[dict],
    use_plm: bool = False,
    regressor_factory: Callable[[], object] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    failure_percentile: float = 0.90,
    progress_every: int = 0,
    progress_callback: Callable[[dict], None] | None = None,
    row_filter: Callable[[dict], bool] | None = None,
) -> dict:
    """Grouped-CV evaluate one assay's rows. Returns a report dict or an
    ``{"abstain": True, "reason": ...}`` when the data can't support honest CV."""
    predicate = row_filter or is_head_b_aggregation_row
    rows = [r for r in rows if predicate(r) and _is_number(r.get("endpoint_value"))]
    if len(rows) < MIN_ROWS_PER_ASSAY:
        return {"abstain": True, "reason": f"only {len(rows)} rows (<{MIN_ROWS_PER_ASSAY})"}

    factory = regressor_factory or _default_regressor_factory()
    directions = {str(r.get("endpoint_direction") or "higher_bad") for r in rows}
    if len(directions) != 1:
        return {"abstain": True, "reason": "mixed endpoint directions in one assay"}
    direction = directions.pop()

    X = []
    for index, row in enumerate(rows, start=1):
        X.append(build_feature_vector(row, use_plm))
        if progress_every > 0 and index % progress_every == 0:
            print(f"  progress {index}/{len(rows)} rows ({row.get('assay_metric', 'assay')})", flush=True)
        if progress_callback is not None:
            progress_callback({"completed": index, "total": len(rows), "assay": row.get("assay_metric", "assay")})
    y = [float(r["endpoint_value"]) for r in rows]
    group_keys = [_group_key(r) for r in rows]
    labels = _failure_labels(y, direction, failure_percentile)

    try:
        folds = grouped_folds(group_keys, labels, n_splits=n_splits, seed=seed)
    except ValueError as e:
        return {"abstain": True, "reason": f"insufficient groups for CV: {e}"}
    assert_no_group_leakage(folds, group_keys)

    oof_true: list[float] = []
    oof_pred: list[float] = []
    oof_group: list[str] = []
    oof_label: list[int] = []
    oof_records: list[dict] = []
    for train_idx, test_idx in folds:
        model = factory()
        model.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        preds = list(model.predict([X[i] for i in test_idx]))
        for pos, i in enumerate(test_idx):
            oof_true.append(y[i]); oof_pred.append(float(preds[pos]))
            oof_group.append(group_keys[i]); oof_label.append(labels[i])
            oof_records.append({
                "row_index": i,
                "endpoint_value": y[i],
                "prediction": float(preds[pos]),
                "group": group_keys[i],
                "failure_label": labels[i],
            })

    risk = _risk_scores(oof_pred, direction)
    for record, risk_score in zip(oof_records, risk):
        record["risk_score"] = float(risk_score)
    by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for g, yt, yp in zip(oof_group, oof_true, oof_pred):
        by_group[g].append((yt, yp))
    point, lo, hi = group_bootstrap_ci(dict(by_group), spearman_rho, seed=seed)

    return {
        "n": len(rows),
        "n_groups": len(set(group_keys)),
        "n_folds": len(folds),
        "endpoint_direction": direction,
        "spearman": round(spearman_rho(oof_true, oof_pred), 4),
        "spearman_ci95": [round(lo, 4), round(hi, 4)],
        "mae": round(mae(oof_true, oof_pred), 4),
        "enrichment_top10pct": round(topk_enrichment(oof_label, risk, 0.10), 4),
        "features": "plm+biophysical" if use_plm else "biophysical_only",
        "oof_predictions": oof_records,
    }


def train_head_b(
    rows: list[dict],
    use_plm: bool = False,
    regressor_factory: Callable[[], object] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    progress_every: int = 0,
    progress_file: str | None = None,
    artifact_file: str | Path = "model/saved/head_b_aggregation.pkl",
) -> dict:
    """Per-assay grouped-CV report for the antibody aggregation model. Rows are
    grouped by ``assay_metric`` so each endpoint is evaluated on its own scale —
    no pooling of incommensurate assays."""
    target_rows = [r for r in rows if is_head_b_aggregation_row(r)]
    progress_path = Path(progress_file) if progress_file else None
    completed_rows = 0
    events: list[str] = []

    def write_progress(payload: dict) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps({
            "completed": completed_rows,
            "total": len(target_rows),
            "fraction": completed_rows / len(target_rows) if target_rows else 1.0,
            "stage": payload.get("stage", "starting"),
            "operation": payload.get("operation", ""),
            "current_assay": payload.get("current_assay", ""),
            "current_assay_completed": payload.get("current_assay_completed", 0),
            "current_assay_total": payload.get("current_assay_total", 0),
            "events": events[-12:],
        }, sort_keys=True) + "\n", encoding="utf-8")

    write_progress({"stage": "starting", "operation": "building assay buckets"})
    by_assay: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in target_rows:
        by_assay[_assay_key(r)].append(r)

    metric_counts: dict[str, int] = defaultdict(int)
    for _, _, metric, _ in by_assay:
        metric_counts[metric] += 1

    import datetime
    from model.features import feature_schema_hash, FEATURE_NAMES
    provenance = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "feature_schema_hash": feature_schema_hash(FEATURE_NAMES),
        "features": "plm+biophysical" if use_plm else "biophysical_only",
        "n_splits": n_splits,
        "seed": seed,
        "sources": sorted({r.get("source", "unknown") for r in rows}),
    }
    report = {"head": "antibody_aggregation", "target_scope": "aggregation_endpoints_only",
              "n_rows": len(target_rows), "excluded_non_target_rows": len(rows) - len(target_rows),
              "provenance": provenance, "assays": {}, "quarantined_assays": []}
    fitted_models: dict[str, dict] = {}
    for source, assay_id, metric, _ in sorted(by_assay):
        bucket = by_assay[(source, assay_id, metric, _)]
        study = str(bucket[0].get("study_id") or bucket[0].get("dataset") or "unknown")
        events.append(f"Starting {study}/{metric} ({len(bucket)} rows)")
        write_progress({"stage": "featurizing", "operation": "building PLM and biophysical features",
                        "current_assay": f"{study}/{metric}", "current_assay_total": len(bucket)})

        def on_row(info: dict) -> None:
            nonlocal completed_rows
            completed_rows += 1
            if progress_every <= 1 or completed_rows % max(progress_every, 1) == 0:
                write_progress({"stage": "featurizing", "operation": "building PLM and biophysical features",
                                "current_assay": f"{study}/{metric}",
                                "current_assay_completed": info["completed"],
                                "current_assay_total": info["total"]})
        report_key = metric if metric_counts[metric] == 1 else f"{study}/{assay_id}/{metric}"
        result = evaluate_assay(
            bucket, use_plm=use_plm, regressor_factory=regressor_factory,
            n_splits=n_splits, seed=seed, progress_every=progress_every,
            progress_callback=on_row,
        )
        events.append(f"Completed {report_key}")
        write_progress({"stage": "completed_assay", "operation": "assay CV complete",
                        "current_assay": report_key, "current_assay_completed": len(bucket),
                        "current_assay_total": len(bucket)})
        result.update({
            "source": source,
            "study_id": study,
            "assay_id": assay_id,
            "assay_metric": metric,
            "assay_family": str(bucket[0].get("assay_family") or "unknown"),
            "endpoint_unit": str(bucket[0].get("endpoint_unit") or "native"),
            "paired_rows": sum(1 for row in bucket if row.get("pair_id")),
            "molecule_groups": len({str(row.get("molecule_group_id") or _group_key(row)) for row in bucket}),
            "specificity_context_rows": sum(
                1 for row in bucket
                if any(row.get(key) for key in ("antigen_id", "target_id", "specificity_assay", "binding_context"))
            ),
        })
        report["assays"][report_key] = result
        if not result.get("abstain"):
            if result.get("spearman") is not None and result["spearman"] < 0:
                result["serving_status"] = "quarantined_negative_grouped_cv"
                report["quarantined_assays"].append(report_key)
                events.append(f"Quarantined weak assay {report_key}")
                continue
            result["serving_status"] = "active"
            final_model = (regressor_factory or _default_regressor_factory())()
            final_features = [build_feature_vector(row, use_plm) for row in bucket]
            final_targets = [float(row["endpoint_value"]) for row in bucket]
            final_model.fit(final_features, final_targets)
            fitted_models[report_key] = {
                "model": final_model,
                "source": source,
                "assay_id": assay_id,
                "assay_metric": metric,
                "endpoint_direction": result.get("endpoint_direction"),
            }
            events.append(f"Fitted final model {report_key}")
    report["specificity_context"] = {
        "rows_with_target_context": sum(item["specificity_context_rows"] for item in report["assays"].values()),
        "note": "Target/specificity metadata is reported only; it is not used as an aggregation label.",
    }
    artifact = Path(artifact_file)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("wb") as handle:
        pickle.dump({
            "head": "antibody_aggregation",
            "feature_schema_hash": provenance["feature_schema_hash"],
            "features": provenance["features"],
            "assays": fitted_models,
        }, handle)
    report["artifact"] = str(artifact)
    report["fitted_assays"] = len(fitted_models)
    events.append("Head B evaluation complete")
    write_progress({"stage": "complete", "operation": "report and model artifact written"})
    return report


def _is_number(v: object) -> bool:
    try:
        float(v)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def train_developability_heads(
    rows: list[dict],
    use_plm: bool = False,
    n_splits: int = 5,
    seed: int = 42,
    artifact_file: str | Path = "model/saved/head_d_developability.pkl",
) -> dict:
    """Grouped-CV baselines and final models for GDPa measured properties."""
    families = {"thermostability", "hydrophobicity", "aggregation", "expression", "polyreactivity"}
    target_rows = [r for r in rows if r.get("assay_family") in families]
    by_assay: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in target_rows:
        by_assay[(str(row.get("study_id")), str(row.get("assay_metric")), str(row.get("assay_family")))].append(row)
    report = {"head": "antibody_developability_panel", "features": "biophysical_only",
              "n_rows": len(target_rows), "assays": {}, "quarantined_assays": []}
    fitted_models: dict[str, dict] = {}
    for (study, metric, family), bucket in sorted(by_assay.items()):
        key = f"{study}/{metric}"
        result = evaluate_assay(bucket, use_plm=use_plm, n_splits=n_splits, seed=seed, row_filter=lambda _: True)
        result.update({"study_id": study, "assay_metric": metric, "assay_family": family,
                       "endpoint_direction": bucket[0].get("endpoint_direction")})
        report["assays"][key] = result
        if result.get("abstain") or result.get("spearman", 0) < 0.20:
            result["serving_status"] = "abstain" if result.get("abstain") else "quarantined_weak_grouped_cv"
            if not result.get("abstain"):
                report["quarantined_assays"].append(key)
            continue
        model = _default_regressor_factory()()
        model.fit([build_feature_vector(row, use_plm) for row in bucket], [float(row["endpoint_value"]) for row in bucket])
        fitted_models[key] = {"model": model, "assay_family": family, "assay_metric": metric,
                              "endpoint_direction": bucket[0].get("endpoint_direction")}
        result["serving_status"] = "active"
    artifact = Path(artifact_file)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("wb") as handle:
        pickle.dump({"head": report["head"], "features": report["features"], "assays": fitted_models}, handle)
    report["artifact"] = str(artifact)
    report["fitted_assays"] = len(fitted_models)
    return report


if __name__ == "__main__":
    import argparse, json
    from data.cohort import build_antibody_cohort, emit_ledgers

    parser = argparse.ArgumentParser(description="Head B (antibody aggregation) grouped-CV eval")
    parser.add_argument("--use-plm", action="store_true", help="concat cached ESM-2+AbLang2 embeddings")
    parser.add_argument("--no-gdpa", action="store_true", help="FLAb only; skip GDPa CSVs")
    parser.add_argument("--developability", action="store_true", help="run GDPa Tm, HIC, SEC, expression, and PR heads")
    parser.add_argument("--ledgers", action="store_true", help="also emit conflicts.tsv/exclusions.tsv")
    parser.add_argument("--out", default="results/head_b_report.json")
    parser.add_argument("--progress-every", type=int, default=0,
                        help="print feature progress every N rows; use 1 or 2 for live updates")
    parser.add_argument("--progress-file", default="results/head_b_progress.json")
    parser.add_argument("--artifact", default="model/saved/head_b_aggregation.pkl")
    args = parser.parse_args()

    if args.developability:
        from data.gdpa import load_gdpa
        from data.cohort import _attach_pairing
        rows = []
        for dataset in ("gdpa1", "gdpa2", "gdpa4.1"):
            rows.extend(load_gdpa(dataset))
        _attach_pairing(rows)
        artifact = args.artifact if args.artifact != "model/saved/head_b_aggregation.pkl" else "model/saved/head_d_developability.pkl"
        report = train_developability_heads(rows, use_plm=args.use_plm, artifact_file=artifact)
    else:
        rows = build_antibody_cohort(include_gdpa=not args.no_gdpa)
        if args.ledgers:
            emit_ledgers(rows)
        report = train_head_b(rows, use_plm=args.use_plm, progress_every=args.progress_every,
                              progress_file=args.progress_file, artifact_file=args.artifact)
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
