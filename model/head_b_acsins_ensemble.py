"""Ensemble + tuned AC-SINS smoke-detector — the push from ~0.68 toward 0.75.

Same honest test as model/head_b_acsins_unified.py --strict-dedup (unified AC-SINS
target, leave-one-study-out, shared antibodies removed from training), but the
single tuned LightGBM is replaced by a rank-averaged ensemble of three different
learners that make different mistakes:
  - LightGBM (tuned: more trees, stronger regularisation)
  - HistGradientBoosting (sklearn, different boosting implementation)
  - Ridge on standardised features (captures the linear PLM signal the trees miss)

Predictions are converted to within-study ranks and averaged — a robust blend that
needs no per-model scaling and matches the rank-based scoring.

Run:
  python -m model.head_b_acsins_ensemble --use-plm --out results/head_b_acsins_ensemble.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
import re
from pathlib import Path
from statistics import mean

from model.head_b_gbm import _assay_key, build_feature_vector
from model.head_b_pooled import _percentile_targets
from model.head_b_catch_rate import _study, _is_acsins, _bad_labels, _metrics, BAD_FRACTION
from model.head_b_acsins_unified import _acsins_rows, _seqkey
from model.cdr_features import cdr_feature_vector


def _rank01(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    for r, i in enumerate(order):
        out[i] = r / max(len(vals) - 1, 1)
    return out


def _ensemble_predict(Xtr, ytr, Xte) -> list[float]:
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    preds = []
    lgb = LGBMRegressor(n_estimators=600, num_leaves=63, learning_rate=0.03,
                        min_child_samples=30, reg_lambda=1.0, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
                        n_jobs=1)
    lgb.fit(Xtr, ytr)
    preds.append(_rank01(list(lgb.predict(Xte))))

    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                        l2_regularization=1.0, random_state=42)
    hgb.fit(Xtr, ytr)
    preds.append(_rank01(list(hgb.predict(Xte))))

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(Xtr, ytr)
    preds.append(_rank01(list(ridge.predict(Xte))))

    n = len(Xte)
    return [mean(p[i] for p in preds) for i in range(n)]


def assay_context_features(row: dict) -> list[float]:
    """Return 6 assay-context flags from assay_metric/assay_id text."""
    raw = f"{row.get('assay_metric', '')} {row.get('assay_id', '')}".lower().replace("Δ", "delta")
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9./+\- ]", " ", raw))
    text = text.strip()

    ph6 = bool(re.search(r"\bph\s*6(?:\.0+)?\b", text))
    ph74 = bool(re.search(r"\bph\s*7(?:\s*\.?\s*4)\b", text))
    pbs = bool(re.search(r"\bpbs\b", text))
    his_arg = bool(re.search(r"\bhis\s*[/\-]?\s*arg(?:inine)?\b", text))
    his_nacl = bool(re.search(r"\bhis\s*[/\-]?\s*na\s*cl\b", text))
    delta_lmax = bool(re.search(r"\bdelta\s*[-_ ]?\s*lmax\b", text) or re.search(r"\bdeltalmax\b", text))
    return [float(ph6), float(ph74), float(pbs), float(his_arg), float(his_nacl), float(delta_lmax)]


def run_ensemble(use_plm: bool = True, include_gdpa: bool = True, frac: float = BAD_FRACTION,
                 add_charge: bool = False, add_struct: bool = False,
                 struct_path: str = "data/external/structures/structure_features_v2.csv",
                 add_assay_context: bool = False, add_cdr_h3: bool = False) -> dict:
    rows = _acsins_rows(include_gdpa)
    if add_charge:
        from model.head_b_acsins_charge import charge_features
    struct = struct_cols = None
    if add_struct:
        import hashlib
        from model.head_b_acsins_struct import _load_structure_features
        struct, struct_cols = _load_structure_features(struct_path)
        _sid = lambda s: hashlib.sha256((s or "").strip().upper().encode()).hexdigest()[:16]

    by_assay: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_assay[(_study(r), _assay_key(r))].append(i)
    y = [0.0] * len(rows)
    for key, idxs in by_assay.items():
        for i, p in zip(idxs, _percentile_targets([float(rows[i]["endpoint_value"]) for i in idxs])):
            y[i] = p

    base_cache: dict[str, list[float]] = {}
    charge_cache: dict[int, list[float]] = {}
    assay_context_cache: dict[int, list[float]] = {}
    n_struct = len(struct_cols) if struct_cols else 0

    def feat(i: int) -> list[float]:
        r = rows[i]
        seq = r.get("variant_sequence") or r.get("vh_sequence") or r.get("sequence", "")
        cache_key = {
            "seq": seq,
            "use_plm": use_plm,
            "vh": r.get("vh_sequence") or "",
            "vl": r.get("vl_sequence") or "",
        }
        ck = json.dumps(cache_key, sort_keys=True)
        v = base_cache.get(ck)
        if v is None:
            v = build_feature_vector(r, use_plm)
            base_cache[ck] = v
        v = list(v)
        if add_charge:
            if i not in charge_cache:
                charge_cache[i] = charge_features(r)
            v = v + charge_cache[i]
        if add_assay_context:
            if i not in assay_context_cache:
                assay_context_cache[i] = assay_context_features(r)
            v = v + assay_context_cache[i]
        if add_struct:
            v = v + struct.get(_sid(r.get("variant_sequence") or r.get("sequence") or ""), [0.0] * n_struct)
        if add_cdr_h3:
            v = v + cdr_feature_vector(r, "h3")
        return v

    studies = sorted({_study(r) for r in rows})
    per_assay: list[dict] = []
    for held in studies:
        test_idx = [i for i in range(len(rows)) if _study(rows[i]) == held]
        held_seqs = {_seqkey(rows[i]) for i in test_idx}
        train_idx = [i for i in range(len(rows))
                     if _study(rows[i]) != held and _seqkey(rows[i]) not in held_seqs]
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue
        Xtr = [feat(i) for i in train_idx]
        ytr = [y[i] for i in train_idx]
        pred = dict(zip(test_idx, _ensemble_predict(Xtr, ytr, [feat(i) for i in test_idx])))

        held_assays: dict[tuple, list[int]] = defaultdict(list)
        for i in test_idx:
            held_assays[_assay_key(rows[i])].append(i)
        for key, idxs in held_assays.items():
            if len(idxs) < 20:
                continue
            vals = [float(rows[i]["endpoint_value"]) for i in idxs]
            bad = _bad_labels(vals, "higher_bad", frac)
            m = _metrics(bad, [pred[i] for i in idxs], frac)
            if m:
                m.update({"study": held, "assay": key[2]})
                per_assay.append(m)

    def avg(f): return round(mean([a[f] for a in per_assay]), 3) if per_assay else None
    solid = [a for a in per_assay if a["n"] >= 100]
    features = "plm+biophysical" if use_plm else "biophysical_only"
    if add_assay_context:
        features += "+assay_context"
    if add_cdr_h3:
        features += "+cdr_h3"
    return {
        "head": "acsins_ensemble_self_association",
        "features": features,
        "model": "rank-avg ensemble (tuned LGBM + HistGBR + Ridge), strict leakage-free",
        "n_evals": len(per_assay),
        "mean_auc": avg("auc"),
        "mean_auc_solid_studies": round(mean([a["auc"] for a in solid]), 3) if solid else None,
        "mean_recall_at_top15pct": avg("recall_at_top15pct"),
        "mean_enrichment_at_15pct": avg("enrichment_at_15pct"),
        "per_assay": sorted(per_assay, key=lambda a: -a["auc"]),
    }


def _print(r: dict) -> None:
    print(f"ENSEMBLE AC-SINS smoke-detector (leakage-free) — {r['model']}")
    print(f"  mean AUC (all {r['n_evals']})            : {r['mean_auc']}   [strict single-model was 0.633]")
    print(f"  mean AUC (solid studies n>=100): {r['mean_auc_solid_studies']}   [strict single-model was 0.681]")
    print(f"  mean catch-rate in top 15%    : {r['mean_recall_at_top15pct']}")
    print(f"  mean enrichment vs random     : {r['mean_enrichment_at_15pct']}x\n")
    print(f"  {'study':<26}{'assay':<24}{'n':>5}{'AUC':>7}{'catch%':>8}{'enrich':>8}")
    for a in r["per_assay"]:
        print(f"  {a['study'][:25]:<26}{str(a['assay'])[:23]:<24}{a['n']:>5}{a['auc']:>7}"
              f"{a['recall_at_top15pct']*100:>7.0f}%{a['enrichment_at_15pct']:>7}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ensemble + tuned AC-SINS smoke-detector (leakage-free)")
    p.add_argument("--use-plm", action="store_true")
    p.add_argument("--no-gdpa", action="store_true")
    p.add_argument("--ab-charge", action="store_true",
                   help="run BOTH without and with charge features in one process (clean A/B)")
    p.add_argument("--ab-full", action="store_true",
                   help="A/B baseline vs charge + structure(v2) features together")
    p.add_argument("--ab-context", action="store_true",
                  help="A/B baseline vs assay-context metadata flags in one process")
    p.add_argument("--ab-cdr-h3", action="store_true",
                   help="A/B baseline vs dependency-free CDR-H3 descriptors in one process")
    p.add_argument("--assay-context", action="store_true",
                  help="run with assay-context metadata flags")
    p.add_argument("--out", default="results/head_b_acsins_ensemble.json")
    args = p.parse_args()

    if args.ab_cdr_h3:
        base = run_ensemble(use_plm=args.use_plm, include_gdpa=not args.no_gdpa)
        combo = run_ensemble(use_plm=args.use_plm, include_gdpa=not args.no_gdpa, add_cdr_h3=True)
        print("ENSEMBLE + CDR-H3 A/B (leakage-free, same process)")
        print(f"  ensemble BASELINE         solid AUC : {base['mean_auc_solid_studies']}")
        print(f"  ensemble + CDR-H3         solid AUC : {combo['mean_auc_solid_studies']}")
        d = (combo['mean_auc_solid_studies'] or 0) - (base['mean_auc_solid_studies'] or 0)
        print(f"  >>> delta on ensemble               : {d:+.4f}")
        print(f"  (all-eval: {base['mean_auc']} -> {combo['mean_auc']})")
        report = {"baseline": base, "with_features": combo, "features": "cdr_h3",
                  "solid_delta": round(d, 4)}
    elif args.ab_charge or args.ab_full:
        add_struct = bool(args.ab_full)
        base = run_ensemble(use_plm=args.use_plm, include_gdpa=not args.no_gdpa, add_charge=False)
        combo = run_ensemble(use_plm=args.use_plm, include_gdpa=not args.no_gdpa,
                             add_charge=True, add_struct=add_struct)
        label = "charge+structure" if add_struct else "charge"
        print(f"ENSEMBLE + {label} A/B (leakage-free, same process)")
        print(f"  ensemble BASELINE         solid AUC : {base['mean_auc_solid_studies']}")
        print(f"  ensemble + {label:<16} solid AUC : {combo['mean_auc_solid_studies']}")
        d = (combo['mean_auc_solid_studies'] or 0) - (base['mean_auc_solid_studies'] or 0)
        print(f"  >>> delta on ensemble               : {d:+.4f}")
        print(f"  (all-eval: {base['mean_auc']} -> {combo['mean_auc']})")
        report = {"baseline": base, "with_features": combo, "features": label,
                  "solid_delta": round(d, 4)}
    elif args.ab_context:
        base = run_ensemble(use_plm=args.use_plm, include_gdpa=not args.no_gdpa, add_assay_context=False)
        combo = run_ensemble(use_plm=args.use_plm, include_gdpa=not args.no_gdpa, add_assay_context=True)
        print("ENSEMBLE + assay_context A/B (leakage-free, same process)")
        print(f"  ensemble BASELINE                 solid AUC : {base['mean_auc_solid_studies']}")
        print(f"  ensemble + assay_context         solid AUC : {combo['mean_auc_solid_studies']}")
        d = (combo['mean_auc_solid_studies'] or 0) - (base['mean_auc_solid_studies'] or 0)
        print(f"  >>> delta on ensemble                  : {d:+.4f}")
        print(f"  (all-eval: {base['mean_auc']} -> {combo['mean_auc']})")
        report = {"baseline": base, "with_features": combo, "features": "assay_context",
                  "solid_delta": round(d, 4)}
    else:
        report = run_ensemble(use_plm=args.use_plm, include_gdpa=not args.no_gdpa,
                              add_assay_context=args.assay_context)
        _print(report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
