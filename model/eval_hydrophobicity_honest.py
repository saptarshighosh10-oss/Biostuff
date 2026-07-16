"""Honest hydrophobicity (HIC) evaluation with Shehata's diverse data wired in.

Pools GDPa HIC + Shehata 2019 HIC and runs leave-one-SOURCE-out (strict molecule dedup,
catch-the-bad AUC) — the same honest test the rest of the platform uses. Answers: does the
diverse Shehata set help hydrophobicity generalize, and does GDPa-trained transfer to Shehata?

Run:
  python -m model.eval_hydrophobicity_honest
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from statistics import mean

from model.head_b_gbm import build_feature_vector
from model.head_b_acsins_charge import charge_features
from model.head_b_catch_rate import _bad_labels, _metrics, BAD_FRACTION
from model.head_developability_honest import _tuned_lgbm

HIC_CSV = os.path.expanduser("~/Downloads/antibody_sources/hydrophobicity_dataset.csv")


def _seqkey(r):
    return (r.get("vh_sequence") or r.get("variant_sequence") or "").strip().upper()


def _rows():
    from data.gdpa import load_gdpa
    from data.cohort import _attach_pairing
    rows = []
    # GDPa hydrophobicity
    g = []
    for ds in ("gdpa1", "gdpa2", "gdpa4.1"):
        g.extend(load_gdpa(ds))
    _attach_pairing(g)
    for r in g:
        if str(r.get("assay_family")) != "hydrophobicity":
            continue
        try:
            float(r.get("endpoint_value"))
        except (TypeError, ValueError):
            continue
        rows.append({"source": str(r.get("study_id")), "vh_sequence": r.get("vh_sequence"),
                     "vl_sequence": r.get("vl_sequence"), "variant_sequence": r.get("variant_sequence"),
                     "value": float(r["endpoint_value"])})
    # Shehata HIC
    if os.path.exists(HIC_CSV):
        for r in csv.DictReader(open(HIC_CSV)):
            try:
                v = float(r["label"])
            except (TypeError, ValueError):
                continue
            vh = (r.get("vh") or "").strip().upper()
            if len(vh) < 20:
                continue
            rows.append({"source": r.get("source", "shehata2019"), "vh_sequence": vh,
                         "vl_sequence": (r.get("vl") or "").strip().upper(),
                         "variant_sequence": vh, "value": v})
    return rows


def run(frac=BAD_FRACTION):
    rows = _rows()
    feat_cache = {}

    def feat(r):
        s = _seqkey(r)
        v = feat_cache.get(s)
        if v is None:
            v = build_feature_vector(r, use_plm=False) + charge_features(r)
            feat_cache[s] = v
        return v

    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    sources = sorted(by_source)
    results = {}
    for held in sources:
        test = by_source[held]
        if len(test) < 20:
            continue
        held_seqs = {_seqkey(r) for r in test}
        train = [r for r in rows if r["source"] != held and _seqkey(r) not in held_seqs]
        if len(train) < 40:
            continue
        m = _tuned_lgbm()
        m.fit([feat(r) for r in train], [r["value"] for r in train])
        preds = list(m.predict([feat(r) for r in test]))
        vals = [r["value"] for r in test]
        bad = _bad_labels(vals, "higher_bad", frac)  # higher HIC = more hydrophobic = bad
        res = _metrics(bad, preds, frac)
        if res:
            results[held] = {"n": len(test), "auc": res["auc"], "catch": res["recall_at_top15pct"]}
    return results


if __name__ == "__main__":
    r = run()
    print("Honest hydrophobicity (leave-one-source-out, GDPa + Shehata):\n")
    print(f"  {'source':<20}{'n':>6}{'AUC':>8}{'catch%':>9}")
    aucs = []
    for s, v in sorted(r.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"  {s:<20}{v['n']:>6}{v['auc']:>8}{v['catch']*100:>8.0f}%")
        aucs.append(v["auc"])
    print(f"\n  mean AUC across sources: {round(mean(aucs), 3) if aucs else None}")
    print("  (platform hydrophobicity baseline was ~0.70; Shehata standalone within-source was 0.712)")
