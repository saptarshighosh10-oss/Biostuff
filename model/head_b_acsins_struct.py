"""Does adding 3D-structure features raise the honest AC-SINS AUC?

Same strict, leakage-free leave-one-study-out test as head_b_acsins_unified
(--strict-dedup), but runs an A/B: identical model and folds, once WITHOUT the
structure features and once WITH them, so the delta is purely the structure signal.

Structure features come from data/external/structures/structure_features.csv
(produced by folding on Colab; keyed by the same seq_id hash as the export).
The key ones are sap_score / max_hydrophobic_patch — surface hydrophobic patches,
which you can only measure in 3D, and which drive self-association.

Run:
  python -m model.head_b_acsins_struct --use-plm --out results/head_b_acsins_struct.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from model.head_b_gbm import _assay_key, build_feature_vector
from model.head_b_catch_rate import _study, _bad_labels, _metrics, BAD_FRACTION
from model.head_b_acsins_unified import _acsins_rows, _seqkey

STRUCT_PATH = "data/external/structures/structure_features.csv"


def _load_structure_features(path: str = STRUCT_PATH):
    """Load any structure-feature CSV. Auto-detects numeric columns (all but seq_id) and
    drops any column that is constant across rows (e.g. v1's low_confidence_fraction)."""
    reader = list(csv.DictReader(open(path)))
    cols = [c for c in reader[0].keys() if c != "seq_id"]
    # drop constant columns (no signal)
    keep = [c for c in cols if len({r.get(c) for r in reader}) > 1]
    out: dict[str, list[float]] = {}
    for r in reader:
        out[r["seq_id"]] = [float(r.get(c, 0.0) or 0.0) for c in keep]
    return out, keep


def _seq_id(seq: str) -> str:
    return hashlib.sha256(seq.strip().upper().encode()).hexdigest()[:16]


def _tuned_lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.02,
                         min_child_samples=30, reg_lambda=5.0, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)


def run(use_plm: bool = True, include_gdpa: bool = True, frac: float = BAD_FRACTION,
        struct_path: str = STRUCT_PATH) -> dict:
    rows = _acsins_rows(include_gdpa)
    struct, struct_cols = _load_structure_features(struct_path)
    n_struct = len(struct_cols)
    matched = sum(1 for r in rows if _seq_id(r.get("variant_sequence") or r.get("sequence") or "") in struct)

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

    def struct_feat(i: int) -> list[float]:
        seq = rows[i].get("variant_sequence") or rows[i].get("sequence") or ""
        return struct.get(_seq_id(seq), [0.0] * n_struct)

    studies = sorted({_study(r) for r in rows})

    def evaluate(with_struct: bool) -> list[dict]:
        per_assay = []
        for held in studies:
            test_idx = [i for i in range(len(rows)) if _study(rows[i]) == held]
            held_seqs = {_seqkey(rows[i]) for i in test_idx}
            train_idx = [i for i in range(len(rows))
                         if _study(rows[i]) != held and _seqkey(rows[i]) not in held_seqs]
            if len(train_idx) < 50 or len(test_idx) < 20:
                continue

            def feat(i):
                return base_feat(i) + struct_feat(i) if with_struct else base_feat(i)

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
                bad = _bad_labels(vals, "higher_bad", frac)
                m = _metrics(bad, [pred[i] for i in idxs], frac)
                if m:
                    m.update({"study": held, "assay": key[2]})
                    per_assay.append(m)
        return per_assay

    without = evaluate(False)
    withs = evaluate(True)

    def solid_mean(rep):
        s = [a["auc"] for a in rep if a["n"] >= 100]
        return round(mean(s), 4) if s else None

    def all_mean(rep):
        return round(mean([a["auc"] for a in rep]), 4) if rep else None

    # per-study delta table (match on study+assay)
    key = lambda a: (a["study"], a["assay"])
    wmap = {key(a): a["auc"] for a in withs}
    deltas = []
    for a in without:
        k = key(a)
        if k in wmap:
            deltas.append({"study": a["study"], "assay": a["assay"], "n": a["n"],
                           "without": a["auc"], "with": wmap[k], "delta": round(wmap[k] - a["auc"], 3)})

    return {
        "head": "acsins_structure_ablation",
        "structure_file": struct_path,
        "structure_sequences_matched": f"{matched}/{len(rows)} rows",
        "structure_features": struct_cols,
        "solid_auc_without_structure": solid_mean(without),
        "solid_auc_with_structure": solid_mean(withs),
        "all_auc_without_structure": all_mean(without),
        "all_auc_with_structure": all_mean(withs),
        "per_study": sorted(deltas, key=lambda d: -d["delta"]),
    }


def _print(r: dict) -> None:
    print(f"Structure ablation (leakage-free, tuned LGBM)  [{r['structure_sequences_matched']} matched]")
    print(f"  solid AUC  WITHOUT structure : {r['solid_auc_without_structure']}")
    print(f"  solid AUC  WITH structure    : {r['solid_auc_with_structure']}")
    d = (r['solid_auc_with_structure'] or 0) - (r['solid_auc_without_structure'] or 0)
    print(f"  >>> structure delta          : {d:+.4f}")
    print()
    print(f"  {'study':<22}{'assay':<22}{'n':>5}{'without':>9}{'with':>8}{'delta':>8}")
    for a in r["per_study"]:
        print(f"  {a['study'][:21]:<22}{str(a['assay'])[:21]:<22}{a['n']:>5}{a['without']:>9}{a['with']:>8}{a['delta']:>+8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Structure-feature A/B for AC-SINS (leakage-free)")
    p.add_argument("--use-plm", action="store_true")
    p.add_argument("--no-gdpa", action="store_true")
    p.add_argument("--struct-path", default=STRUCT_PATH,
                   help="structure-feature CSV (v1 default, or the v2 IgFold file)")
    p.add_argument("--out", default="results/head_b_acsins_struct.json")
    args = p.parse_args()

    report = run(use_plm=args.use_plm, include_gdpa=not args.no_gdpa, struct_path=args.struct_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(report)
    print(f"\nwrote {args.out}")
