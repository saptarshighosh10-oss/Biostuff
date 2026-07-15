"""Head C — antibody polyreactivity / non-specificity (developability property #2).

Separate from Head B (aggregation/AC-SINS). Built from public polyreactivity datasets
consolidated into ~/Downloads/antibody_sources/polyreactivity_dataset.csv
(columns: source, vh, vl, label, label_type; higher label = more polyreactive).

Honest evaluation: within each source, sequence-deduplicated k-fold CV, scored by ROC AUC
for "flag the polyreactive ones" — the same catch-the-bad-ones framing used for Head B.
Cross-source transfer is NOT pooled here (sources measure different reactivity assays on
different antibody sets).

Features: biophysical (28) + charge-at-pH. NOTE: PLM embeddings are zero-filled for these
new sequences (not in the AC-SINS cache), so this is effectively a biophysical+charge model
— a first honest baseline. For the Emi mutant library (near-identical sequences) PLM/one-hot
of the mutated positions is what the original paper needed, so treat Emi's number as a
lower bound until PLM embeddings are computed.

Run:
  python -m model.head_c_polyreactivity --out results/head_c_polyreactivity.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from model.head_b_gbm import build_feature_vector
from model.head_b_acsins_charge import charge_features

DATA = os.path.expanduser("~/Downloads/antibody_sources/polyreactivity_dataset.csv")


def _load():
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in csv.DictReader(open(DATA)):
        vh = (r["vh"] or "").strip().upper()
        if len(vh) < 20:
            continue
        try:
            label = float(r["label"])
        except ValueError:
            continue
        by_source[r["source"]].append({
            "variant_sequence": vh, "vh_sequence": vh,
            "vl_sequence": (r["vl"] or "").strip().upper() or None,
            "label": label, "label_type": r["label_type"],
        })
    return by_source


def _feat(row):
    return build_feature_vector(row, use_plm=False) + charge_features(row)


def _model():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.03,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=42, verbose=-1)


def _binary_labels(rows: list[dict]) -> list[int]:
    labs = [r["label"] for r in rows]
    if set(labs) <= {0.0, 1.0}:
        return [int(l) for l in labs]
    thr = median(labs)  # continuous -> above-median = polyreactive
    return [1 if l > thr else 0 for l in labs]


def evaluate_source(rows: list[dict], n_splits: int = 5, seed: int = 42) -> dict | None:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    # dedup identical sequences
    seen = {}
    for r in rows:
        seen.setdefault(r["variant_sequence"], r)
    rows = list(seen.values())
    y = _binary_labels(rows)
    if sum(y) < 10 or sum(y) > len(y) - 10:
        return {"n": len(rows), "positives": sum(y), "skip": "too few of one class"}
    X = [_feat(r) for r in rows]
    oof = [0.0] * len(rows)
    skf = StratifiedKFold(n_splits=min(n_splits, sum(y), len(y) - sum(y)), shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = _model()
        m.fit([X[i] for i in tr], [y[i] for i in tr])
        proba = m.predict_proba([X[i] for i in te])[:, 1]
        for pos, i in enumerate(te):
            oof[i] = float(proba[pos])
    auc = roc_auc_score(y, oof)
    # catch-rate: of the polyreactive ones, how many in the top-scored 15%?
    k = max(1, round(len(rows) * 0.15))
    top = sorted(range(len(rows)), key=lambda i: oof[i], reverse=True)[:k]
    recall = sum(y[i] for i in top) / sum(y)
    return {"n": len(rows), "positives": sum(y), "auc": round(auc, 3),
            "recall_at_top15pct": round(recall, 3)}


def run() -> dict:
    by_source = _load()
    out = {}
    for src, rows in sorted(by_source.items()):
        res = evaluate_source(rows)
        if res:
            out[src] = res
    scored = [v["auc"] for v in out.values() if "auc" in v]
    return {"head": "antibody_polyreactivity", "features": "biophysical+charge (PLM zero-filled)",
            "mean_auc": round(mean(scored), 3) if scored else None, "per_source": out}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Head C polyreactivity within-source CV")
    p.add_argument("--out", default="results/head_c_polyreactivity.json")
    args = p.parse_args()
    report = run()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Head C — polyreactivity ({report['features']})")
    print(f"  mean AUC across scorable sources: {report['mean_auc']}\n")
    print(f"  {'source':<14}{'n':>6}{'pos':>6}{'AUC':>7}{'catch%':>9}")
    for s, v in report["per_source"].items():
        if "auc" in v:
            print(f"  {s:<14}{v['n']:>6}{v['positives']:>6}{v['auc']:>7}{v['recall_at_top15pct']*100:>8.0f}%")
        else:
            print(f"  {s:<14}{v['n']:>6}{v['positives']:>6}   skip ({v.get('skip')})")
    print(f"\nwrote {args.out}")