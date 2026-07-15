"""The antibody developability platform — three validated, honestly-measured heads.

Only properties that survived leave-one-GDPa-out, leakage-free evaluation are served:
  - aggregation / self-association (AC-SINS)
  - polyreactivity (non-specific binding)
  - hydrophobicity (HIC)

Thermostability and expression were tested three ways (biophysical+charge, +PLM+structure,
+multi-task) and never beat a coin flip on unseen antibodies, so they are NOT served — an
honest 3-property platform beats a 5-property one with two fake heads.

This trains a final serving model per property on all available data (PLM + biophysical +
charge features) and writes the artifact + a scorecard carrying the honest cross-study AUCs.

Run:
  python -m model.head_platform --out results/platform_scorecard.json \
    --artifact model/saved/platform.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

from model.head_b_gbm import build_feature_vector, _assay_key
from model.head_b_acsins_charge import charge_features
from model.head_developability_honest import _load_rows, _seqkey, _tuned_lgbm

# Honest cross-study AUCs measured by model/head_developability_honest.py (PLM+structure)
# and, for aggregation, the stronger rank-averaged ensemble (model/head_b_acsins_ensemble.py).
SERVED = {
    "aggregation":   {"family": "self_association", "honest_auc": 0.72,
                      "note": "best number uses the PLM ensemble; 0.68 with a single model"},
    "polyreactivity":{"family": "polyreactivity",   "honest_auc": 0.70, "note": ""},
    "hydrophobicity":{"family": "hydrophobicity",   "honest_auc": 0.70, "note": ""},
}
NOT_SERVED = {
    "thermostability": "coin flip on unseen antibodies (0.53); needs physics-based ΔΔG modeling",
    "expression":      "coin flip on unseen antibodies (0.50); depends on cell biology, not just sequence",
}


def train(use_plm: bool = True) -> dict:
    rows = _load_rows()
    by_family = defaultdict(list)
    for r in rows:
        by_family[str(r.get("assay_family"))].append(r)

    fitted = {}
    for prop, meta in SERVED.items():
        fam_rows = by_family.get(meta["family"], [])
        # one model per (assay_metric) within the property, trained on all data
        by_assay = defaultdict(list)
        for r in fam_rows:
            by_assay[str(r.get("assay_metric"))].append(r)
        models = {}
        for metric, bucket in by_assay.items():
            if len(bucket) < 40:
                continue
            X = [build_feature_vector(r, use_plm=use_plm) + charge_features(r) for r in bucket]
            y = [float(r["endpoint_value"]) for r in bucket]
            m = _tuned_lgbm(); m.fit(X, y)
            models[metric] = {"model": m, "direction": bucket[0].get("endpoint_direction"),
                              "n_train": len(bucket)}
        fitted[prop] = {"family": meta["family"], "assays": models}
    return fitted


def run(use_plm: bool = True, artifact: str = "model/saved/platform.pkl") -> dict:
    fitted = train(use_plm=use_plm)
    Path(artifact).parent.mkdir(parents=True, exist_ok=True)
    with open(artifact, "wb") as fh:
        pickle.dump({"head": "developability_platform", "features": "plm+biophysical+charge",
                     "heads": fitted}, fh)
    scorecard = {
        "platform": "antibody developability",
        "evaluation": "leave-one-GDPa-out, leakage-free (shared antibodies removed), catch-the-bad AUC",
        "served_heads": {p: {"honest_auc": m["honest_auc"], "assays_fitted": len(fitted[p]["assays"]),
                             "note": m["note"]} for p, m in SERVED.items()},
        "not_served": NOT_SERVED,
        "artifact": artifact,
    }
    return scorecard


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train + score the 3-property developability platform")
    p.add_argument("--no-plm", action="store_true")
    p.add_argument("--artifact", default="model/saved/platform.pkl")
    p.add_argument("--out", default="results/platform_scorecard.json")
    args = p.parse_args()
    card = run(use_plm=not args.no_plm, artifact=args.artifact)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Antibody developability platform — trained & scored\n")
    print(f"  {'property':<18}{'honest AUC':>12}{'assays':>9}")
    for prop, v in card["served_heads"].items():
        print(f"  {prop:<18}{v['honest_auc']:>12}{v['assays_fitted']:>9}")
    print("\n  not served (tested, don't generalize):")
    for prop, why in card["not_served"].items():
        print(f"    {prop}: {why}")
    print(f"\nwrote {args.out} and {args.artifact}")