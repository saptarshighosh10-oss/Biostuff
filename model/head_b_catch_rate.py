"""Smoke-detector scorecard for Head B — how well does it CATCH the bad antibodies?

Spearman grades "rank everything perfectly". The useful question for triage is
easier and more relevant: "of the genuinely sticky antibodies, how many land in
the slice you'd actually test?" That is a detection problem, scored by:
  - ROC AUC          : chance a random bad ranks above a random good (0.5 = coin flip)
  - recall@top-15%   : fraction of the truly-bad caught in the predicted-worst 15%
  - enrichment@15%   : how many times better than random that slice is

All measured on the HONEST setting: leave a whole study out, train on the rest,
predict the held-out study cold (unseen antibodies, unseen lab). Anchored on the
self-association (AC-SINS) family — the model's strongest, most-generalising signal.

Run:
  python -m model.head_b_catch_rate --use-plm --out results/head_b_catch_rate.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from model.head_b_gbm import _assay_key, build_feature_vector
from model.head_b_pooled import _pool_rows

BAD_FRACTION = 0.15  # the worst 15% by the assay's true value are the "bad" antibodies


def _study(r: dict) -> str:
    return str(r.get("study_id") or r.get("source") or "unknown")


def _is_acsins(key: tuple) -> bool:
    m = str(key[2] or "").lower()
    return ("acsins" in m or "ac-sins" in m) and "sg" not in m  # core AC-SINS, not SGAC-SINS


def _regressor(use_plm: bool):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                         subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)


def _bad_labels(values: list[float], direction: str, frac: float) -> list[int]:
    """1 = bad (in the failing tail). higher_bad -> top frac; lower_bad -> bottom frac."""
    n = len(values)
    k = max(1, round(n * frac))
    order = sorted(range(n), key=lambda i: values[i], reverse=(direction == "higher_bad"))
    bad = set(order[:k])
    return [1 if i in bad else 0 for i in range(n)]


def _metrics(y_true_bad: list[int], scores: list[float], frac: float) -> dict | None:
    from sklearn.metrics import roc_auc_score
    n = len(y_true_bad)
    n_bad = sum(y_true_bad)
    if n_bad == 0 or n_bad == n:
        return None
    auc = roc_auc_score(y_true_bad, scores)
    k = max(1, round(n * frac))
    top = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    caught = sum(y_true_bad[i] for i in top)
    recall = caught / n_bad
    enrichment = (caught / k) / (n_bad / n)
    return {"n": n, "n_bad": n_bad, "auc": round(auc, 3),
            "recall_at_top15pct": round(recall, 3), "enrichment_at_15pct": round(enrichment, 2)}


def run_catch_rate(use_plm: bool = True, include_gdpa: bool = True, frac: float = BAD_FRACTION) -> dict:
    rows = _pool_rows(include_gdpa)
    studies = sorted({_study(r) for r in rows})
    base_cache: dict[str, list[float]] = {}

    def feat(i: int) -> list[float]:
        r = rows[i]
        seq = r.get("variant_sequence") or r.get("vh_sequence") or r.get("sequence", "")
        ck = f"{seq}|{use_plm}"
        v = base_cache.get(ck)
        if v is None:
            v = build_feature_vector(r, use_plm)
            base_cache[ck] = v
        return v

    per_assay: list[dict] = []
    for held in studies:
        train_idx = [i for i in range(len(rows)) if _study(rows[i]) != held]
        test_idx = [i for i in range(len(rows)) if _study(rows[i]) == held]
        # only bother if the held-out study has AC-SINS and there is AC-SINS training signal
        held_acsins = [i for i in test_idx if _is_acsins(_assay_key(rows[i]))]
        train_acsins = any(_is_acsins(_assay_key(rows[i])) for i in train_idx)
        if not held_acsins or not train_acsins or len(train_idx) < 50:
            continue
        model = _regressor(use_plm)
        model.fit([feat(i) for i in train_idx], [float(rows[i]["endpoint_value"]) for i in train_idx])
        pred = {i: float(p) for i, p in zip(test_idx, model.predict([feat(i) for i in test_idx]))}

        by_assay: dict[tuple, list[int]] = defaultdict(list)
        for i in held_acsins:
            by_assay[_assay_key(rows[i])].append(i)
        for key, idxs in by_assay.items():
            if len(idxs) < 20:
                continue
            direction = str(rows[idxs[0]].get("endpoint_direction") or "higher_bad")
            vals = [float(rows[i]["endpoint_value"]) for i in idxs]
            bad = _bad_labels(vals, direction, frac)
            # score: model prediction oriented so higher = more bad
            scores = [pred[i] if direction == "higher_bad" else -pred[i] for i in idxs]
            m = _metrics(bad, scores, frac)
            if m:
                m.update({"study": held, "assay": key[2]})
                per_assay.append(m)

    def avg(field): return round(mean([a[field] for a in per_assay]), 3) if per_assay else None
    return {
        "head": "antibody_aggregation_catch_rate",
        "task": "flag the worst 15% (sticky) AC-SINS antibodies on UNSEEN studies",
        "features": "plm+biophysical" if use_plm else "biophysical_only",
        "n_evals": len(per_assay),
        "mean_auc": avg("auc"),
        "mean_recall_at_top15pct": avg("recall_at_top15pct"),
        "mean_enrichment_at_15pct": avg("enrichment_at_15pct"),
        "per_assay": sorted(per_assay, key=lambda a: -a["auc"]),
    }


def _print(r: dict) -> None:
    print(f"Smoke-detector scorecard — catch the sticky ones on UNSEEN antibodies ({r['features']})")
    print(f"  mean AUC (0.5=coin flip, 1.0=perfect): {r['mean_auc']}")
    print(f"  mean catch-rate of bad ones in top 15%: {r['mean_recall_at_top15pct']}")
    print(f"  mean enrichment vs random            : {r['mean_enrichment_at_15pct']}x")
    print(f"  ({r['n_evals']} held-out AC-SINS evaluations)\n")
    print(f"  {'study':<26}{'assay':<24}{'n':>5}{'AUC':>7}{'catch%':>8}{'enrich':>8}")
    for a in r["per_assay"]:
        print(f"  {a['study'][:25]:<26}{str(a['assay'])[:23]:<24}{a['n']:>5}{a['auc']:>7}"
              f"{a['recall_at_top15pct']*100:>7.0f}%{a['enrichment_at_15pct']:>7}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Catch-the-bad-ones scorecard for Head B (AC-SINS, unseen studies)")
    p.add_argument("--use-plm", action="store_true")
    p.add_argument("--no-gdpa", action="store_true")
    p.add_argument("--out", default="results/head_b_catch_rate.json")
    args = p.parse_args()

    report = run_catch_rate(use_plm=args.use_plm, include_gdpa=not args.no_gdpa)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(report)
    print(f"\nwrote {args.out}")
