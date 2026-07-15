"""Unified AC-SINS (self-association) anchor model + smoke-detector re-measure.

Motivation. The generic pooled model trains on the raw endpoint values of EVERY
aggregation assay at once — AC-SINS (nm shift), HIC (retention minutes), SEC (%),
etc. — which are incommensurate, so the target is muddy. This module anchors on
the model's strongest, most-generalising signal — self-association (AC-SINS) — and
UNIFIES it:

  1. Keep only AC-SINS-family rows across ALL studies (one coherent phenomenon).
  2. Put every study's AC-SINS on a common 0..1 badness scale (per-study percentile;
     AC-SINS is higher=stickier=worse, so percentile == badness rank). Now hundreds
     of scattered, differently-named AC-SINS measurements become ONE training target.
  3. Train features -> unified badness, and score the honest smoke-detector task
     (catch the worst 15% on a held-out, unseen study).

Compared head-to-head with the mixed-assay baseline (model/head_b_catch_rate.py).

Run:
  python -m model.head_b_acsins_unified --use-plm --out results/head_b_acsins_unified.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from model.head_b_gbm import _assay_key, build_feature_vector
from model.head_b_pooled import _pool_rows, _percentile_targets
from model.head_b_catch_rate import _study, _is_acsins, _bad_labels, _metrics, _regressor, BAD_FRACTION


def _acsins_rows(include_gdpa: bool) -> list[dict]:
    return [r for r in _pool_rows(include_gdpa) if _is_acsins(_assay_key(r))]


def _seqkey(r: dict) -> str:
    return (r.get("variant_sequence") or r.get("sequence") or "").strip().upper()


def run_unified(use_plm: bool = True, include_gdpa: bool = True, frac: float = BAD_FRACTION,
                strict_dedup: bool = False) -> dict:
    rows = _acsins_rows(include_gdpa)

    # Unified target: per-(study, assay) percentile of the raw value (all AC-SINS is
    # higher=worse, so percentile == badness). One coherent 0..1 target across studies.
    by_assay: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_assay[(_study(r), _assay_key(r))].append(i)
    y = [0.0] * len(rows)
    for key, idxs in by_assay.items():
        vals = [float(rows[i]["endpoint_value"]) for i in idxs]
        for i, p in zip(idxs, _percentile_targets(vals)):
            y[i] = p  # already oriented: higher raw -> higher percentile -> worse

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

    studies = sorted({_study(r) for r in rows})
    per_assay: list[dict] = []
    for held in studies:
        test_idx = [i for i in range(len(rows)) if _study(rows[i]) == held]
        held_seqs = {_seqkey(rows[i]) for i in test_idx} if strict_dedup else set()
        # strict: drop any training row whose antibody also appears in the held-out study
        train_idx = [i for i in range(len(rows))
                     if _study(rows[i]) != held and _seqkey(rows[i]) not in held_seqs]
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue
        model = _regressor(use_plm)
        model.fit([feat(i) for i in train_idx], [y[i] for i in train_idx])
        pred = {i: float(p) for i, p in zip(test_idx, model.predict([feat(i) for i in test_idx]))}

        held_assays: dict[tuple, list[int]] = defaultdict(list)
        for i in test_idx:
            held_assays[_assay_key(rows[i])].append(i)
        for key, idxs in held_assays.items():
            if len(idxs) < 20:
                continue
            vals = [float(rows[i]["endpoint_value"]) for i in idxs]
            bad = _bad_labels(vals, "higher_bad", frac)  # AC-SINS: higher = bad
            scores = [pred[i] for i in idxs]              # higher pred = more bad
            m = _metrics(bad, scores, frac)
            if m:
                m.update({"study": held, "assay": key[2]})
                per_assay.append(m)

    def avg(f): return round(mean([a[f] for a in per_assay]), 3) if per_assay else None
    # solid = big held-out studies (n>=100) — the trustworthy transfer evidence
    solid = [a for a in per_assay if a["n"] >= 100]
    return {
        "head": "acsins_unified_self_association",
        "task": "flag worst 15% sticky AC-SINS on UNSEEN studies (unified anchor model)",
        "features": "plm+biophysical" if use_plm else "biophysical_only",
        "n_acsins_rows": len(rows),
        "n_evals": len(per_assay),
        "mean_auc": avg("auc"),
        "mean_recall_at_top15pct": avg("recall_at_top15pct"),
        "mean_enrichment_at_15pct": avg("enrichment_at_15pct"),
        "mean_auc_solid_studies": round(mean([a["auc"] for a in solid]), 3) if solid else None,
        "per_assay": sorted(per_assay, key=lambda a: -a["auc"]),
    }


def _print(r: dict) -> None:
    print(f"UNIFIED AC-SINS smoke-detector — catch sticky ones on UNSEEN antibodies ({r['features']})")
    print(f"  AC-SINS rows unified          : {r['n_acsins_rows']}")
    print(f"  mean AUC (all {r['n_evals']} evals)       : {r['mean_auc']}   [baseline mixed-assay was 0.539]")
    print(f"  mean AUC (solid studies n>=100): {r['mean_auc_solid_studies']}   [baseline solid was ~0.64]")
    print(f"  mean catch-rate in top 15%    : {r['mean_recall_at_top15pct']}")
    print(f"  mean enrichment vs random     : {r['mean_enrichment_at_15pct']}x\n")
    print(f"  {'study':<26}{'assay':<24}{'n':>5}{'AUC':>7}{'catch%':>8}{'enrich':>8}")
    for a in r["per_assay"]:
        print(f"  {a['study'][:25]:<26}{str(a['assay'])[:23]:<24}{a['n']:>5}{a['auc']:>7}"
              f"{a['recall_at_top15pct']*100:>7.0f}%{a['enrichment_at_15pct']:>7}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified AC-SINS anchor model + smoke-detector re-measure")
    p.add_argument("--use-plm", action="store_true")
    p.add_argument("--no-gdpa", action="store_true")
    p.add_argument("--strict-dedup", action="store_true",
                   help="remove shared antibodies from training (leakage-free number)")
    p.add_argument("--out", default="results/head_b_acsins_unified.json")
    args = p.parse_args()

    report = run_unified(use_plm=args.use_plm, include_gdpa=not args.no_gdpa,
                         strict_dedup=args.strict_dedup)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(report)
    print(f"\nwrote {args.out}")
