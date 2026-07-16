"""Add a source-normalized PSR risk tower after the Jain-2024 gate passes.

This adds a new classifier to the existing platform rather than replacing the
legacy assay-specific polyreactivity regressors.  For every assay/study, the
highest 15% measured values are labelled as higher-risk, avoiding a false
assumption that absolute PSR scales are interchangeable.
"""

from __future__ import annotations

import csv
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

from model.head_b_acsins_charge import charge_features
from model.head_b_gbm import build_feature_vector
from model.head_c_polyreactivity import _model
from model.head_developability_honest import _load_rows


DATA = Path.home() / "Downloads" / "antibody_sources" / "polyreactivity_dataset.csv"
ARTIFACT = Path("model/saved/platform.pkl")
BACKUP = Path("model/saved/platform.before_jain2024_psr.pkl")
REPORT = Path("results/psr_platform_refresh.json")
FRACTION = 0.15


def rows() -> list[dict]:
    """Return per-assay, VH-deduplicated GDPa PR plus accepted Jain PSR rows."""
    output, seen = [], set()
    for record in _load_rows():
        if record.get("assay_family") != "polyreactivity":
            continue
        vh = (record.get("variant_sequence") or "").strip().upper()
        study = f"{record.get('study_id')}:{record.get('assay_metric')}"
        if not vh or (study, vh) in seen:
            continue
        seen.add((study, vh))
        output.append({"variant_sequence": vh, "vh_sequence": record.get("vh_sequence"),
                       "vl_sequence": record.get("vl_sequence"),
                       "value": float(record["endpoint_value"]), "study": study})
    with DATA.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record.get("source") != "jain2024_psr":
                continue
            vh = (record.get("vh") or "").strip().upper()
            try:
                value = float(record.get("label") or "")
            except ValueError:
                continue
            if len(vh) < 20 or ("jain2024_psr", vh) in seen:
                continue
            seen.add(("jain2024_psr", vh))
            output.append({"variant_sequence": vh, "vh_sequence": vh,
                           "vl_sequence": (record.get("vl") or "").strip().upper() or None,
                           "value": value, "study": "jain2024_psr"})
    return output


def label(rows_: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows_:
        grouped[row["study"]].append(row)
    labelled = []
    for bucket in grouped.values():
        k = max(1, round(len(bucket) * FRACTION))
        bad = {id(row) for row in sorted(bucket, key=lambda row: row["value"], reverse=True)[:k]}
        labelled.extend({**row, "bad": int(id(row) in bad)} for row in bucket)
    return labelled


def run() -> dict:
    if not ARTIFACT.exists():
        raise FileNotFoundError(f"Missing current platform artifact: {ARTIFACT}")
    training = label(rows())
    counts = defaultdict(int)
    for row in training:
        counts[row["study"]] += 1
    if counts.get("jain2024_psr") != 43:
        raise RuntimeError("Expected exactly 43 accepted Jain-2024 PSR rows before refresh")
    X = [build_feature_vector(row, use_plm=True) + charge_features(row) for row in training]
    y = [row["bad"] for row in training]
    model = _model()
    model.fit(X, y)
    with ARTIFACT.open("rb") as handle:
        platform = pickle.load(handle)
    shutil.copy2(ARTIFACT, BACKUP)
    platform["heads"]["polyreactivity"]["assays"]["PSR_high_risk_top15pct"] = {
        "model": model, "direction": "higher_bad", "n_train": len(training),
        "training_note": "Source-normalized PSR risk: highest 15% per assay/study is higher_bad; includes gate-accepted Jain-2024 PSR.",
    }
    with ARTIFACT.open("wb") as handle:
        pickle.dump(platform, handle)
    report = {"artifact": str(ARTIFACT), "backup": str(BACKUP), "head": "PSR_high_risk_top15pct",
              "n_train": len(training), "n_bad": sum(y), "feature_width": len(X[0]),
              "study_counts": dict(sorted(counts.items()))}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
