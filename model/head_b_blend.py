"""Blended Head B — combine the pooled and independent models honestly.

The hybrid (model/head_b_hybrid.py) makes a hard per-assay choice: pooled OR
independent. Blending keeps a weighted mix of BOTH predictions per row, which
usually beats either alone because their errors are partly uncorrelated.

To blend fairly, both models must predict on the SAME held-out rows under ONE
shared cross-validation split (molecule-grouped over the whole pool). This module
builds that aligned out-of-fold (OOF) prediction harness, then reports, under
identical folds:
  - independent  (per-assay model, trained only on that assay's rows in each fold)
  - pooled       (one multi-task model over the whole pool)
  - blend@w      (w*pooled + (1-w)*independent, both on the 0..1 percentile scale)
  - size-routed  (pooled for n<threshold, independent otherwise)

Everything is scored per assay with Spearman on the assay's OOF rows.

Note on honesty: the blend-weight curve is REPORTED, not fit-to-test. Pick w by
the principled default (0.5) or by assay size; do not cherry-pick the w that
maximises the reported mean, that would leak the test set.

Run:
  python -m model.head_b_blend --use-plm --out results/head_b_blend_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from model.head_b_gbm import _assay_key, _group_key, build_feature_vector, MIN_ROWS_PER_ASSAY
from model.head_b_pooled import _pool_rows, _percentile_targets
from model.metrics import spearman_rho
from model.plan_c_cv import grouped_folds, assert_no_group_leakage

ROUTE_THRESHOLD = 50
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # w on the pooled model


def _pooled_regressor():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                         subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)


def _independent_regressor():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, num_leaves=31, random_state=42, verbose=-1)


def _aligned_oof(use_plm: bool, n_splits: int, seed: int, include_gdpa: bool):
    """Return (rows, by_assay, group_keys, y, pooled_oof, indep_oof) — pooled and
    independent OOF predictions on the SAME molecule-grouped folds."""
    rows = _pool_rows(include_gdpa)
    by_assay: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_assay[_assay_key(r)].append(i)
    assay_ids = sorted(by_assay)
    assay_index = {k: j for j, k in enumerate(assay_ids)}
    n_assays = len(assay_ids)

    # per-assay percentile targets (common 0..1 scale)
    y = [0.0] * len(rows)
    for key, idxs in by_assay.items():
        for i, p in zip(idxs, _percentile_targets([float(rows[i]["endpoint_value"]) for i in idxs])):
            y[i] = p

    # base feature vectors, memoised per (sequence, plm)
    base_cache: dict[str, list[float]] = {}

    def base_feat(i: int) -> list[float]:
        r = rows[i]
        seq = r.get("variant_sequence") or r.get("vh_sequence") or r.get("sequence", "")
        ck = f"{seq}|{use_plm}"
        v = base_cache.get(ck)
        if v is None:
            v = build_feature_vector(r, use_plm)
            base_cache[ck] = v
        return v

    base = [base_feat(i) for i in range(len(rows))]

    def pooled_feat(i: int) -> list[float]:
        onehot = [0.0] * n_assays
        onehot[assay_index[_assay_key(rows[i])]] = 1.0
        return base[i] + onehot

    group_keys = [_group_key(r) for r in rows]
    folds = grouped_folds(group_keys, [0] * len(rows), n_splits=n_splits, seed=seed)
    assert_no_group_leakage(folds, group_keys)

    pooled_oof = [float("nan")] * len(rows)
    indep_oof = [float("nan")] * len(rows)
    for train_idx, test_idx in folds:
        train_set = set(train_idx)
        # pooled: one model over all training rows
        pm = _pooled_regressor()
        pm.fit([pooled_feat(i) for i in train_idx], [y[i] for i in train_idx])
        for pos, i in enumerate(pm.predict([pooled_feat(i) for i in test_idx])):
            pooled_oof[test_idx[pos]] = float(i)
        # independent: one model per assay, trained on that assay's TRAIN rows only
        test_by_assay: dict[tuple, list[int]] = defaultdict(list)
        for i in test_idx:
            test_by_assay[_assay_key(rows[i])].append(i)
        for key, test_rows in test_by_assay.items():
            tr = [i for i in by_assay[key] if i in train_set]
            if len(tr) < 5:  # too few to fit anything honest -> predict the neutral 0.5
                for i in test_rows:
                    indep_oof[i] = 0.5
                continue
            im = _independent_regressor()
            im.fit([base[i] for i in tr], [y[i] for i in tr])
            for pos, i in enumerate(im.predict([base[i] for i in test_rows])):
                indep_oof[test_rows[pos]] = float(i)
    return rows, by_assay, y, pooled_oof, indep_oof


def _per_assay_spearman(by_assay, y, pred) -> dict[str, dict]:
    out = {}
    for key, idxs in by_assay.items():
        r0_metric = key[2]
        if len(idxs) < 3:
            continue
        rho = spearman_rho([y[i] for i in idxs], [pred[i] for i in idxs])
        out[key] = {"metric": r0_metric, "n": len(idxs), "spearman": round(rho, 4)}
    return out


def run_blend(use_plm: bool = True, n_splits: int = 5, seed: int = 42,
              include_gdpa: bool = True, route_threshold: int = ROUTE_THRESHOLD) -> dict:
    rows, by_assay, y, pooled_oof, indep_oof = _aligned_oof(use_plm, n_splits, seed, include_gdpa)

    def served_mean(pred) -> float:
        vals = []
        for key, idxs in by_assay.items():
            if len(idxs) < MIN_ROWS_PER_ASSAY:
                continue
            rho = spearman_rho([y[i] for i in idxs], [pred[i] for i in idxs])
            if rho >= 0:
                vals.append(rho)
        return round(mean(vals), 4) if vals else float("nan")

    # blend curve
    blends = {}
    blend_preds = {}
    for w in BLEND_WEIGHTS:
        pred = [w * pooled_oof[i] + (1 - w) * indep_oof[i] for i in range(len(rows))]
        blend_preds[w] = pred
        blends[f"blend@{w:.2f}"] = served_mean(pred)

    # size-routed prediction
    routed = [
        (pooled_oof[i] if len(by_assay[_assay_key(rows[i])]) < route_threshold else indep_oof[i])
        for i in range(len(rows))
    ]

    summary = {
        "independent": served_mean(indep_oof),
        "pooled": served_mean(pooled_oof),
        **blends,
        f"size_routed(<{route_threshold})": served_mean(routed),
    }

    # per-assay detail for the principled default: 0.5 blend
    default_pred = blend_preds[0.5]
    detail = {}
    for key, idxs in by_assay.items():
        metric = key[2]
        study = str(rows[idxs[0]].get("study_id") or "?")
        n = len(idxs)
        rk = f"{study}/{metric}"
        detail[rk] = {
            "n": n,
            "independent": round(spearman_rho([y[i] for i in idxs], [indep_oof[i] for i in idxs]), 4) if n >= 3 else None,
            "pooled": round(spearman_rho([y[i] for i in idxs], [pooled_oof[i] for i in idxs]), 4) if n >= 3 else None,
            "blend@0.5": round(spearman_rho([y[i] for i in idxs], [default_pred[i] for i in idxs]), 4) if n >= 3 else None,
        }

    return {
        "head": "antibody_aggregation_blend",
        "features": "plm+biophysical" if use_plm else "biophysical_only",
        "pool_rows": len(rows),
        "n_assays": len(by_assay),
        "served_mean_by_strategy": summary,
        "assays": detail,
        "note": "Blend weights are reported, not fit-to-test. Default = blend@0.50.",
    }


def _print(report: dict) -> None:
    print(f"Blended Head B  ({report['features']}, pool={report['pool_rows']} rows, "
          f"{report['n_assays']} assays)")
    print("  served mean Spearman by strategy:")
    for k, v in report["served_mean_by_strategy"].items():
        print(f"    {k:<22}: {v}")
    print()
    print(f"  {'assay':<40}{'n':>5}{'indep':>8}{'pooled':>8}{'blend.5':>9}")
    for k, a in sorted(report["assays"].items(), key=lambda kv: -((kv[1]['blend@0.5'] or -9))):
        def f(x): return f"{x:+.3f}" if x is not None else "  n/a"
        print(f"  {k[:39]:<40}{a['n']:>5}{f(a['independent']):>8}{f(a['pooled']):>8}{f(a['blend@0.5']):>9}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Blend pooled + independent Head B models under shared folds")
    p.add_argument("--use-plm", action="store_true")
    p.add_argument("--no-gdpa", action="store_true")
    p.add_argument("--out", default="results/head_b_blend_report.json")
    args = p.parse_args()

    report = run_blend(use_plm=args.use_plm, include_gdpa=not args.no_gdpa)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(report)
    print(f"\nwrote {args.out}")
