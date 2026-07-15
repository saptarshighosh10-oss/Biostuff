"""Does adding charge-at-assay-pH features raise the honest AC-SINS AUC?

AC-SINS self-association is driven largely by electrostatics, and the assay is run at
pH 6.0 and pH 7.4. Net charge at those exact pH values is a classic, physics-grounded
predictor — and it's free to compute from sequence. This runs the same strict,
leakage-free leave-one-study-out A/B as the structure test: identical model and folds,
once WITHOUT charge features and once WITH, so the delta is purely the charge signal.

Features added per antibody (computed on VH+VL when available):
  net_charge_pH6, net_charge_pH7.4, charge_delta_6_to_74, isoelectric_point,
  frac_positive, frac_negative, chain_charge_asymmetry (|VH - VL| at pH 7.4)

Run:
  python -m model.head_b_acsins_charge --use-plm --out results/head_b_acsins_charge.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from model.head_b_gbm import _assay_key, build_feature_vector
from model.head_b_catch_rate import _study, _bad_labels, _metrics, BAD_FRACTION
from model.head_b_acsins_unified import _acsins_rows, _seqkey

# pKa values (Bjellqvist/EMBOSS-style)
_POS = {"K": 10.5, "R": 12.5, "H": 6.0}   # plus N-terminus 9.0
_NEG = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}  # plus C-terminus 2.0
_NTERM, _CTERM = 9.0, 2.0


def net_charge(seq: str, pH: float) -> float:
    if not seq:
        return 0.0
    q = 1.0 / (1.0 + 10 ** (pH - _NTERM))           # N-terminus (positive)
    q -= 1.0 / (1.0 + 10 ** (_CTERM - pH))          # C-terminus (negative)
    for aa, pka in _POS.items():
        q += seq.count(aa) / (1.0 + 10 ** (pH - pka))
    for aa, pka in _NEG.items():
        q -= seq.count(aa) / (1.0 + 10 ** (pka - pH))
    return q


def isoelectric_point(seq: str) -> float:
    lo, hi = 2.0, 12.0
    for _ in range(40):  # bisection
        mid = (lo + hi) / 2
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _tuned_lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.02,
                         min_child_samples=30, reg_lambda=5.0, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)


def charge_features(row: dict) -> list[float]:
    vh = (row.get("vh_sequence") or "").strip().upper()
    vl = (row.get("vl_sequence") or "").strip().upper()
    full = (vh + vl) if (vh or vl) else (row.get("variant_sequence") or "").strip().upper()
    n = max(len(full), 1)
    asym = abs(net_charge(vh, 7.4) - net_charge(vl, 7.4)) if (vh and vl) else 0.0
    return [
        net_charge(full, 6.0),
        net_charge(full, 7.4),
        net_charge(full, 6.0) - net_charge(full, 7.4),
        isoelectric_point(full),
        sum(full.count(a) for a in "KRH") / n,
        sum(full.count(a) for a in "DE") / n,
        asym,
    ]


def run(use_plm: bool = True, include_gdpa: bool = True, frac: float = BAD_FRACTION) -> dict:
    rows = _acsins_rows(include_gdpa)
    base_cache: dict[str, list[float]] = {}
    charge_cache: dict[int, list[float]] = {}

    def base_feat(i: int) -> list[float]:
        r = rows[i]
        seq = r.get("variant_sequence") or r.get("vh_sequence") or r.get("sequence", "")
        ck = f"{seq}|{use_plm}"
        v = base_cache.get(ck)
        if v is None:
            v = build_feature_vector(r, use_plm)
            base_cache[ck] = v
        return v

    def chg_feat(i: int) -> list[float]:
        if i not in charge_cache:
            charge_cache[i] = charge_features(rows[i])
        return charge_cache[i]

    studies = sorted({_study(r) for r in rows})

    def evaluate(with_charge: bool) -> list[dict]:
        per_assay = []
        for held in studies:
            test_idx = [i for i in range(len(rows)) if _study(rows[i]) == held]
            held_seqs = {_seqkey(rows[i]) for i in test_idx}
            train_idx = [i for i in range(len(rows))
                         if _study(rows[i]) != held and _seqkey(rows[i]) not in held_seqs]
            if len(train_idx) < 50 or len(test_idx) < 20:
                continue

            def feat(i):
                return base_feat(i) + chg_feat(i) if with_charge else base_feat(i)

            model = _tuned_lgbm()
            model.fit([feat(i) for i in train_idx],
                      [float(rows[i]["endpoint_value"]) for i in train_idx])
            pred = dict(zip(test_idx, model.predict([feat(i) for i in test_idx])))
            by_assay = defaultdict(list)
            for i in test_idx:
                by_assay[_assay_key(rows[i])].append(i)
            for key, idxs in by_assay.items():
                if len(idxs) < 20:
                    continue
                vals = [float(rows[i]["endpoint_value"]) for i in idxs]
                m = _metrics(_bad_labels(vals, "higher_bad", frac), [pred[i] for i in idxs], frac)
                if m:
                    m.update({"study": held, "assay": key[2]})
                    per_assay.append(m)
        return per_assay

    without, withc = evaluate(False), evaluate(True)

    def solid(rep):
        s = [a["auc"] for a in rep if a["n"] >= 100]
        return round(mean(s), 4) if s else None

    key = lambda a: (a["study"], a["assay"])
    wmap = {key(a): a["auc"] for a in withc}
    deltas = [{"study": a["study"], "assay": a["assay"], "n": a["n"], "without": a["auc"],
               "with": wmap[key(a)], "delta": round(wmap[key(a)] - a["auc"], 3)}
              for a in without if key(a) in wmap]

    return {
        "head": "acsins_charge_ablation",
        "solid_auc_without_charge": solid(without),
        "solid_auc_with_charge": solid(withc),
        "per_study": sorted(deltas, key=lambda d: -d["delta"]),
    }


def _print(r: dict) -> None:
    a, b = r["solid_auc_without_charge"], r["solid_auc_with_charge"]
    print("Charge-at-pH ablation (leakage-free, tuned LGBM)")
    print(f"  solid AUC  WITHOUT charge : {a}")
    print(f"  solid AUC  WITH charge    : {b}")
    print(f"  >>> charge delta          : {(b or 0)-(a or 0):+.4f}")
    print()
    print(f"  {'study':<22}{'assay':<22}{'n':>5}{'without':>9}{'with':>8}{'delta':>8}")
    for d in r["per_study"]:
        print(f"  {d['study'][:21]:<22}{str(d['assay'])[:21]:<22}{d['n']:>5}{d['without']:>9}{d['with']:>8}{d['delta']:>+8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Charge-at-pH feature A/B for AC-SINS (leakage-free)")
    p.add_argument("--use-plm", action="store_true")
    p.add_argument("--no-gdpa", action="store_true")
    p.add_argument("--out", default="results/head_b_acsins_charge.json")
    args = p.parse_args()

    report = run(use_plm=args.use_plm, include_gdpa=not args.no_gdpa)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(report)
    print(f"\nwrote {args.out}")
