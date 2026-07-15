"""Honest, leakage-free re-grade of the whole GDPa developability panel.

The panel (head_b_gbm --developability) reports grouped-CV Spearman — the in-distribution,
optimistic number. This re-grades every property the SAME honest way AC-SINS got:
  - leave one whole GDPa dataset out (gdpa1 / gdpa2 / gdpa4.1 are molecule-disjoint),
  - remove any antibody shared with the held-out set from training (strict dedup),
  - score by ROC AUC for "flag the bad ones" (respecting each assay's good/bad direction).

Same data source as the panel (load_gdpa), same families. Features: biophysical + charge
(PLM zero-filled), matching the panel's biophysical-only setup.

Run:
  python -m model.head_developability_honest --out results/head_developability_honest.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from model.head_b_gbm import build_feature_vector
from model.head_b_acsins_charge import charge_features
from model.head_b_catch_rate import _bad_labels, _metrics, BAD_FRACTION

FAMILIES = {"thermostability", "hydrophobicity", "aggregation", "self_association",
            "colloidal_stability", "expression", "polyreactivity"}


def _seqkey(r: dict) -> str:
    return (r.get("variant_sequence") or r.get("vh_sequence") or "").strip().upper()


def _tuned_lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.02,
                         min_child_samples=30, reg_lambda=5.0, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)


def _load_rows():
    from data.gdpa import load_gdpa
    from data.cohort import _attach_pairing
    rows = []
    for ds in ("gdpa1", "gdpa2", "gdpa4.1"):
        rows.extend(load_gdpa(ds))
    _attach_pairing(rows)
    out = []
    for r in rows:
        if str(r.get("assay_family")) not in FAMILIES:
            continue
        try:
            float(r.get("endpoint_value"))
        except (TypeError, ValueError):
            continue
        out.append(r)
    return out


def run(frac: float = BAD_FRACTION, use_plm: bool = False, add_struct: bool = False,
        struct_path: str = "data/external/structures/structure_features_v2.csv") -> dict:
    rows = _load_rows()
    struct = struct_cols = None
    if add_struct:
        import hashlib
        from model.head_b_acsins_struct import _load_structure_features
        struct, struct_cols = _load_structure_features(struct_path)
        _sid = lambda s: hashlib.sha256((s or "").strip().upper().encode()).hexdigest()[:16]
    n_struct = len(struct_cols) if struct_cols else 0
    feat_cache: dict[str, list[float]] = {}

    def feat(r):
        s = _seqkey(r)
        v = feat_cache.get(s)
        if v is None:
            v = build_feature_vector(r, use_plm=use_plm) + charge_features(r)
            if add_struct:
                v = v + struct.get(_sid(r.get("variant_sequence") or ""), [0.0] * n_struct)
            feat_cache[s] = v
        return v

    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[str(r.get("assay_family"))].append(r)

    report: dict[str, dict] = {}
    for family, frows in sorted(by_family.items()):
        studies = sorted({str(r.get("study_id")) for r in frows})
        evals = []
        for held in studies:
            test = [r for r in frows if str(r.get("study_id")) == held]
            held_seqs = {_seqkey(r) for r in test}
            train = [r for r in frows if str(r.get("study_id")) != held and _seqkey(r) not in held_seqs]
            if len(train) < 40 or len(test) < 20:
                continue
            model = _tuned_lgbm()
            model.fit([feat(r) for r in train], [float(r["endpoint_value"]) for r in train])
            preds = {id(r): float(p) for r, p in zip(test, model.predict([feat(r) for r in test]))}
            by_assay = defaultdict(list)
            for r in test:
                by_assay[str(r.get("assay_metric"))].append(r)
            for metric, arows in by_assay.items():
                if len(arows) < 20:
                    continue
                direction = str(arows[0].get("endpoint_direction") or "higher_bad")
                vals = [float(r["endpoint_value"]) for r in arows]
                # orient score so higher = more likely bad
                scores = [preds[id(r)] if direction == "higher_bad" else -preds[id(r)] for r in arows]
                m = _metrics(_bad_labels(vals, direction, frac), scores, frac)
                if m:
                    m.update({"held_out_study": held, "assay": metric})
                    evals.append(m)
        aucs = [e["auc"] for e in evals]
        report[family] = {
            "honest_mean_auc": round(mean(aucs), 3) if aucs else None,
            "n_cross_study_evals": len(evals),
            "evals": sorted(evals, key=lambda e: -e["auc"]),
        }
    scored = [v["honest_mean_auc"] for v in report.values() if v["honest_mean_auc"] is not None]
    return {"head": "developability_panel_honest_leave_one_gdpa_out",
            "features": "biophysical+charge (PLM zero-filled)",
            "panel_mean_honest_auc": round(mean(scored), 3) if scored else None,
            "per_property": report}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Honest leakage-free re-grade of the developability panel")
    p.add_argument("--use-plm", action="store_true", help="add cached PLM embeddings (fold context)")
    p.add_argument("--struct", action="store_true", help="add IgFold structure features (v2)")
    p.add_argument("--out", default="results/head_developability_honest.json")
    args = p.parse_args()
    rep = run(use_plm=args.use_plm, add_struct=args.struct)
    rep["config"] = {"use_plm": args.use_plm, "add_struct": args.struct}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Honest developability panel (leave-one-GDPa-out, leakage-free, AUC)")
    print(f"  features: {rep['features']}\n")
    print(f"  {'property':<22}{'honest AUC':>12}{'#tests':>9}")
    for fam, v in rep["per_property"].items():
        a = v["honest_mean_auc"]
        print(f"  {fam:<22}{(str(a) if a is not None else 'no cross-study data'):>12}{v['n_cross_study_evals']:>9}")
    print(f"\n  panel mean honest AUC: {rep['panel_mean_honest_auc']}")
    print(f"\nwrote {args.out}")