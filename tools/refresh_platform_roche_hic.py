"""Refresh the served HIC head after external HIC sources pass their gates.

HIC retention values from different laboratories are not on a common numerical
scale.  This script therefore trains one shared, source-normalized classifier:
within each study/assay, the highest 15% retention values are ``higher_bad``.
It replaces only the hydrophobicity head in the existing platform artifact;
the aggregation and polyreactivity heads are preserved byte-for-byte.

External rows are used only after their holdout-safe gates pass.
``shehata2019`` is also included here because final retraining may use all
accepted data; it was excluded from the gate's training folds.
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


DATA = Path.home() / "Downloads/antibody_sources/hydrophobicity_dataset.csv"
ARTIFACT = Path("model/saved/platform.pkl")
BACKUP = Path("model/saved/platform.before_jain2017_hic.pkl")
REPORT = Path("results/hic_platform_refresh.json")
FRACTION = 0.15


def _external_rows() -> list[dict]:
    """Read strict, VH-deduplicated external HIC records."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    with DATA.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            source = record.get("source", "")
            if source not in {"shehata2019", "roche34_2022", "jain2017_hic"}:
                continue
            vh = (record.get("vh") or "").strip().upper()
            vl = (record.get("vl") or "").strip().upper() or None
            try:
                value = float(record.get("label") or "")
            except ValueError:
                continue
            if len(vh) < 20 or (source, vh) in seen:
                continue
            seen.add((source, vh))
            rows.append({"variant_sequence": vh, "vh_sequence": vh, "vl_sequence": vl,
                         "endpoint_value": value, "study_id": source})
    return rows


def _gdpa_rows() -> list[dict]:
    """Read strict, per-study VH-deduplicated GDPa HIC records."""
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for record in _load_rows():
        if record.get("assay_family") != "hydrophobicity":
            continue
        vh = (record.get("variant_sequence") or record.get("vh_sequence") or "").strip().upper()
        key = (str(record.get("study_id")), vh)
        if not vh or key in seen:
            continue
        seen.add(key)
        rows.append({"variant_sequence": vh, "vh_sequence": record.get("vh_sequence"),
                     "vl_sequence": record.get("vl_sequence"),
                     "endpoint_value": float(record["endpoint_value"]),
                     "study_id": str(record.get("study_id"))})
    return rows


def _label_top_fraction(rows: list[dict]) -> list[dict]:
    """Label the highest HIC values in each assay/study as hydrophobic risk."""
    by_study: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_study[row["study_id"]].append(row)
    labelled: list[dict] = []
    for bucket in by_study.values():
        n_bad = max(1, round(len(bucket) * FRACTION))
        bad_ids = {id(row) for row in sorted(
            bucket, key=lambda row: row["endpoint_value"], reverse=True
        )[:n_bad]}
        labelled.extend({**row, "bad": int(id(row) in bad_ids)} for row in bucket)
    return labelled


def _feature(row: dict) -> list[float]:
    return build_feature_vector(row, use_plm=True) + charge_features(row)


def run() -> dict:
    if not ARTIFACT.exists():
        raise FileNotFoundError(f"Missing current platform artifact: {ARTIFACT}")
    rows = _label_top_fraction(_gdpa_rows() + _external_rows())
    source_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        source_counts[row["study_id"]] += 1
    if source_counts.get("roche34_2022") != 17 or source_counts.get("jain2017_hic") != 134:
        raise RuntimeError("Expected exactly 17 Roche-34 and 134 Jain-2017 HIC records before refresh")

    X = [_feature(row) for row in rows]
    y = [row["bad"] for row in rows]
    model = _model()
    model.fit(X, y)

    with ARTIFACT.open("rb") as handle:
        platform = pickle.load(handle)
    shutil.copy2(ARTIFACT, BACKUP)
    platform["heads"]["hydrophobicity"] = {
        "family": "hydrophobicity",
        "assays": {
            "HIC_high_risk_top15pct": {
                "model": model,
                "direction": "higher_bad",
                "n_train": len(rows),
                "training_note": (
                    "Binary source-normalized HIC-risk model: each assay/study's "
                    "highest 15% retention values are labelled higher_bad. Includes "
                    "accepted Roche-34 and Jain-2017 HIC data after holdout gates."
                ),
            }
        },
    }
    with ARTIFACT.open("wb") as handle:
        pickle.dump(platform, handle)

    report = {
        "artifact": str(ARTIFACT), "backup": str(BACKUP),
        "head": "HIC_high_risk_top15pct", "n_train": len(rows),
        "n_bad": sum(y), "feature_width": len(X[0]),
        "source_counts": dict(sorted(source_counts.items())),
        "note": "Final training only; the separate holdout gate supplies the generalization estimate.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
