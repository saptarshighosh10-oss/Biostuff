"""Pooled multi-task Head B (antibody aggregation).

The baseline (model/head_b_gbm.py) trains an INDEPENDENT regressor per assay.
For small assays (n=22-43) that means fitting a high-capacity model on ~17 rows
per CV fold — it memorises noise, and the out-of-fold Spearman lands anywhere,
often negative. This module instead trains ONE model across every aggregation
assay at once (hard parameter sharing / multi-task), so a 22-row assay learns
from the full pool of thousands of rows.

How the pooling stays honest:
  - Targets are made commensurate across assays by per-assay percentile-rank
    normalisation (each endpoint -> its rank in [0,1] within its own assay).
    Spearman is rank-based, so this monotone per-assay transform does not alter
    any per-assay evaluation; it only lets one regressor see all assays on a
    common scale.
  - Assay identity is a one-hot feature block, so the shared model can still
    specialise per assay (the multi-task signal).
  - Cross-validation holds out WHOLE molecules across the entire pool (a molecule
    measured in several assays is never split train/test). Same grouped-CV idea
    as the baseline, applied globally -> no cross-assay molecule leakage.
  - Non-aggregation rows (FcRn / pharmacokinetics) and replicate-scatter (*_std)
    endpoints are dropped from the pool; they are assay noise, not aggregation.

Run:
  python -m model.head_b_pooled --use-plm --out results/head_b_pooled_report.json
  # add --baseline results/... to print a per-assay pooled-vs-baseline delta table
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from data.cohort import build_antibody_cohort, is_head_b_aggregation_row
from model.head_b_gbm import _assay_key, _group_key, build_feature_vector, MIN_ROWS_PER_ASSAY
from model.metrics import spearman_rho
from model.plan_c_cv import grouped_folds, assert_no_group_leakage
from model.serving_policy import is_scatter_endpoint, NON_AGGREGATION_FAMILIES


def _regressor():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                             subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=42)


def _percentile_targets(values: list[float]) -> list[float]:
    """Map each value to its rank percentile in [0,1] within the assay (ties averaged)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank / max(len(values) - 1, 1)
        i = j + 1
    return ranks


def _pool_rows(include_gdpa: bool = True) -> list[dict]:
    rows = build_antibody_cohort(include_gdpa=include_gdpa)
    pool = []
    for r in rows:
        if not is_head_b_aggregation_row(r):
            continue
        try:
            float(r.get("endpoint_value"))
        except (TypeError, ValueError):
            continue
        family = str(r.get("assay_family") or "").lower()
        metric = str(r.get("assay_metric") or "")
        if family in NON_AGGREGATION_FAMILIES or is_scatter_endpoint(metric):
            continue
        pool.append(r)
    return pool


def run_pooled(use_plm: bool = True, n_splits: int = 5, seed: int = 42,
               include_gdpa: bool = True) -> dict:
    rows = _pool_rows(include_gdpa)

    # Bucket by assay; build per-assay percentile targets and an assay index.
    by_assay: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_assay[_assay_key(r)].append(i)
    assay_ids = sorted(by_assay)
    assay_index = {key: k for k, key in enumerate(assay_ids)}

    y_norm = [0.0] * len(rows)
    for key, idxs in by_assay.items():
        vals = [float(rows[i]["endpoint_value"]) for i in idxs]
        for i, p in zip(idxs, _percentile_targets(vals)):
            y_norm[i] = p

    # Features: [ base feature vector | one-hot(assay) ]. Memoise base vectors per sequence.
    n_assays = len(assay_ids)
    base_cache: dict[str, list[float]] = {}

    def feat(i: int) -> list[float]:
        r = rows[i]
        seq = r.get("variant_sequence") or r.get("vh_sequence") or r.get("sequence", "")
        ck = f"{seq}|{use_plm}"
        base = base_cache.get(ck)
        if base is None:
            base = build_feature_vector(r, use_plm)
            base_cache[ck] = base
        onehot = [0.0] * n_assays
        onehot[assay_index[_assay_key(r)]] = 1.0
        return base + onehot

    X = [feat(i) for i in range(len(rows))]
    group_keys = [_group_key(r) for r in rows]
    labels = [0] * len(rows)  # grouped_folds only needs group_keys; labels unused for splitting
    folds = grouped_folds(group_keys, labels, n_splits=n_splits, seed=seed)
    assert_no_group_leakage(folds, group_keys)

    oof_pred = [float("nan")] * len(rows)
    for train_idx, test_idx in folds:
        model = _regressor()
        model.fit([X[i] for i in train_idx], [y_norm[i] for i in train_idx])
        preds = model.predict([X[i] for i in test_idx])
        for pos, i in enumerate(test_idx):
            oof_pred[i] = float(preds[pos])

    # Per-assay Spearman on out-of-fold predictions (normalised true vs pred).
    assays_report: dict[str, dict] = {}
    for key, idxs in by_assay.items():
        yt = [y_norm[i] for i in idxs]
        yp = [oof_pred[i] for i in idxs]
        r0 = rows[idxs[0]]
        study = str(r0.get("study_id") or "?")
        metric = key[2]
        report_key = f"{study}/{metric}"
        rho = spearman_rho(yt, yp) if len(idxs) >= 3 else None
        assays_report[report_key] = {
            "assay_metric": metric,
            "assay_family": r0.get("assay_family"),
            "n": len(idxs),
            "n_groups": len({group_keys[i] for i in idxs}),
            "spearman": round(rho, 4) if rho is not None else None,
        }

    scored = [a["spearman"] for a in assays_report.values() if a["spearman"] is not None]
    servable = [a["spearman"] for a in assays_report.values()
                if a["spearman"] is not None and a["n"] >= MIN_ROWS_PER_ASSAY]
    return {
        "head": "antibody_aggregation_pooled_multitask",
        "features": "plm+biophysical" if use_plm else "biophysical_only",
        "pool_rows": len(rows),
        "n_assays": len(assay_ids),
        "mean_spearman_all": round(mean(scored), 4) if scored else None,
        "mean_spearman_n>=20": round(mean(servable), 4) if servable else None,
        "assays": assays_report,
    }


def _print_compare(report: dict, baseline_path: str | None) -> None:
    base = {}
    if baseline_path and Path(baseline_path).exists():
        bd = json.loads(Path(baseline_path).read_text())
        for k, a in bd.get("assays", {}).items():
            if isinstance(a, dict) and a.get("spearman") is not None:
                base[str(a.get("assay_metric"))] = a["spearman"]
    print(f"Pooled multi-task Head B  ({report['features']}, pool={report['pool_rows']} rows, "
          f"{report['n_assays']} assays)")
    print(f"  mean Spearman (all scored)   : {report['mean_spearman_all']}")
    print(f"  mean Spearman (n>=20)        : {report['mean_spearman_n>=20']}")
    print()
    header = f"  {'assay':<40}{'n':>5}{'pooled':>9}"
    if base:
        header += f"{'baseline':>10}{'delta':>8}"
    print(header)
    for k, a in sorted(report["assays"].items(), key=lambda kv: -(kv[1]["spearman"] or -9)):
        line = f"  {k[:39]:<40}{a['n']:>5}{(a['spearman'] if a['spearman'] is not None else float('nan')):>9.3f}"
        if base:
            b = base.get(a["assay_metric"])
            if b is not None and a["spearman"] is not None:
                line += f"{b:>10.3f}{a['spearman']-b:>+8.3f}"
            else:
                line += f"{'—':>10}{'—':>8}"
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pooled multi-task Head B aggregation eval")
    p.add_argument("--use-plm", action="store_true")
    p.add_argument("--no-gdpa", action="store_true")
    p.add_argument("--baseline", default=None, help="baseline report JSON for a delta table")
    p.add_argument("--out", default="results/head_b_pooled_report.json")
    args = p.parse_args()

    report = run_pooled(use_plm=args.use_plm, include_gdpa=not args.no_gdpa)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_compare(report, args.baseline)
    print(f"\nwrote {args.out}")
