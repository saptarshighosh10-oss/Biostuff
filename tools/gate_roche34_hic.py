"""Holdout-safe HIC gate for the Roche-34 candidate.

GDPa HIC measurements form the accepted training baseline.  Shehata HIC is a
diverse held-out test set and is never used to fit either model.  Because HIC
scales differ by study, each study is converted to its own top-15% "higher is
worse" label before the cross-study comparison.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from model.head_b_acsins_charge import charge_features
from model.head_b_gbm import build_feature_vector
from model.head_c_polyreactivity import _model
from model.head_developability_honest import _load_rows


DATA = Path.home() / "Downloads/antibody_sources/hydrophobicity_dataset.csv"
FRACTION = 0.15


def source_rows(source: str) -> list[dict]:
    seen = {}
    with DATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] != source:
                continue
            vh = row["vh"].strip().upper()
            if len(vh) < 20:
                continue
            try:
                value = float(row["label"])
            except ValueError:
                continue
            seen.setdefault(vh, {"variant_sequence": vh, "vh_sequence": vh,
                                 "vl_sequence": row["vl"].strip().upper() or None,
                                 "endpoint_value": value, "study_id": source})
    return list(seen.values())


def mark_top_fraction(rows: list[dict]) -> list[dict]:
    """Attach an exact top-15% high-HIC label independently per study."""
    by_study = defaultdict(list)
    for row in rows:
        by_study[row["study_id"]].append(row)
    labelled = []
    for bucket in by_study.values():
        n_bad = max(1, round(len(bucket) * FRACTION))
        bad = {id(row) for row in sorted(bucket, key=lambda row: row["endpoint_value"], reverse=True)[:n_bad]}
        for row in bucket:
            labelled.append({**row, "bad": int(id(row) in bad)})
    return labelled


def baseline_rows() -> list[dict]:
    rows = []
    for row in _load_rows():
        if row["assay_family"] != "hydrophobicity":
            continue
        rows.append({"variant_sequence": row["variant_sequence"], "vh_sequence": row.get("vh_sequence"),
                     "vl_sequence": row.get("vl_sequence"), "endpoint_value": float(row["endpoint_value"]),
                     "study_id": row["study_id"]})
    seen = {}
    for row in rows:
        seen.setdefault((row["study_id"], row["variant_sequence"]), row)
    return list(seen.values())


def features(row: dict) -> list[float]:
    return build_feature_vector(row, use_plm=False) + charge_features(row)


def score(train: list[dict], test: list[dict]) -> dict:
    from sklearn.metrics import roc_auc_score

    test_vh = {row["variant_sequence"] for row in test}
    train = [row for row in train if row["variant_sequence"] not in test_vh]
    model = _model()
    model.fit([features(row) for row in train], [row["bad"] for row in train])
    predictions = model.predict_proba([features(row) for row in test])[:, 1]
    return {"train_n": len(train), "test_n": len(test), "test_bad": sum(row["bad"] for row in test),
            "auc": float(roc_auc_score([row["bad"] for row in test], predictions))}


def run() -> dict:
    baseline = mark_top_fraction(baseline_rows())
    shehata = mark_top_fraction(source_rows("shehata2019"))
    roche = mark_top_fraction(source_rows("roche34_2022"))
    without = score(baseline, shehata)
    with_roche = score(baseline + roche, shehata)
    return {"baseline": without, "trial": with_roche,
            "accepted": with_roche["auc"] >= without["auc"] - 0.005,
            "roche_n": len(roche), "holdout": "shehata2019"}


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
