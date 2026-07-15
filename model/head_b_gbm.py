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
from typing import Callable

from model.features import extract_from_sequence, features_to_vector
from model.metrics import spearman_rho, mae, topk_enrichment, group_bootstrap_ci
from model.plan_c_cv import composite_group_key, grouped_folds, assert_no_group_leakage
from data.cohort import is_head_b_aggregation_row

# GDPa ships its own homology-cluster fold column; honor it over a recomputed
# split when present (design: leave-homology-cluster-out).
_BENCHMARK_FOLD_FIELDS = (
    "hierarchical_cluster_IgG_isotype_stratified_fold",
    "cluster_fold",
    "fold",
)
MIN_ROWS_PER_ASSAY = 20


def build_feature_vector(row: dict, use_plm: bool = True) -> list[float]:
    """[ PLM | biophysical ] for one row. Falls back to biophysical-only when
    use_plm is False or the PLM cache has no entry for this sequence."""
    seq = row.get("variant_sequence") or row.get("vh_sequence") or row.get("sequence", "")
    biophysical = features_to_vector(extract_from_sequence(seq, row.get("mutations", [])))
    if not use_plm:
        return biophysical
    from features.plm import embed_combined  # cache read is stdlib; guarded for clarity
    try:
        return list(embed_combined(seq)) + biophysical
    except RuntimeError:
        # no cached embedding for this sequence — degrade rather than crash.
        return biophysical


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
) -> dict:
    """Grouped-CV evaluate one assay's rows. Returns a report dict or an
    ``{"abstain": True, "reason": ...}`` when the data can't support honest CV."""
    rows = [r for r in rows if is_head_b_aggregation_row(r) and _is_number(r.get("endpoint_value"))]
    if len(rows) < MIN_ROWS_PER_ASSAY:
        return {"abstain": True, "reason": f"only {len(rows)} rows (<{MIN_ROWS_PER_ASSAY})"}

    factory = regressor_factory or _default_regressor_factory()
    directions = {str(r.get("endpoint_direction") or "higher_bad") for r in rows}
    if len(directions) != 1:
        return {"abstain": True, "reason": "mixed endpoint directions in one assay"}
    direction = directions.pop()

    X = [build_feature_vector(r, use_plm) for r in rows]
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
    for train_idx, test_idx in folds:
        model = factory()
        model.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        preds = list(model.predict([X[i] for i in test_idx]))
        for pos, i in enumerate(test_idx):
            oof_true.append(y[i]); oof_pred.append(float(preds[pos]))
            oof_group.append(group_keys[i]); oof_label.append(labels[i])

    risk = _risk_scores(oof_pred, direction)
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
    }


def train_head_b(
    rows: list[dict],
    use_plm: bool = False,
    regressor_factory: Callable[[], object] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Per-assay grouped-CV report for the antibody aggregation model. Rows are
    grouped by ``assay_metric`` so each endpoint is evaluated on its own scale —
    no pooling of incommensurate assays."""
    target_rows = [r for r in rows if is_head_b_aggregation_row(r)]
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
              "provenance": provenance, "assays": {}}
    for source, assay_id, metric, _ in sorted(by_assay):
        bucket = by_assay[(source, assay_id, metric, _)]
        study = str(bucket[0].get("study_id") or bucket[0].get("dataset") or "unknown")
        report_key = metric if metric_counts[metric] == 1 else f"{study}/{assay_id}/{metric}"
        result = evaluate_assay(
            bucket, use_plm=use_plm, regressor_factory=regressor_factory,
            n_splits=n_splits, seed=seed,
        )
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
    report["specificity_context"] = {
        "rows_with_target_context": sum(item["specificity_context_rows"] for item in report["assays"].values()),
        "note": "Target/specificity metadata is reported only; it is not used as an aggregation label.",
    }
    return report


def _is_number(v: object) -> bool:
    try:
        float(v)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    import argparse, json
    from data.cohort import build_antibody_cohort, emit_ledgers

    parser = argparse.ArgumentParser(description="Head B (antibody aggregation) grouped-CV eval")
    parser.add_argument("--use-plm", action="store_true", help="concat cached ESM-2+AbLang2 embeddings")
    parser.add_argument("--no-gdpa", action="store_true", help="FLAb only; skip GDPa CSVs")
    parser.add_argument("--ledgers", action="store_true", help="also emit conflicts.tsv/exclusions.tsv")
    parser.add_argument("--out", default="results/head_b_report.json")
    args = parser.parse_args()

    rows = build_antibody_cohort(include_gdpa=not args.no_gdpa)
    if args.ledgers:
        emit_ledgers(rows)
    report = train_head_b(rows, use_plm=args.use_plm)
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
