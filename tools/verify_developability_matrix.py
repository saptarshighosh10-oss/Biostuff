"""Independently recompute the saved baseline-matrix metrics from predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from data.run_manifest import file_sha256


def verify(report_path: str | Path, predictions_path: str | Path) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    report_file = Path(report_path)
    predictions_file = Path(predictions_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    actual_hash = file_sha256(predictions_file)
    if report.get("predictions_sha256") != actual_hash:
        raise ValueError("prediction file hash does not match the matrix report")

    grouped: dict[tuple[str, int, str, str], list[tuple[int, float]]] = defaultdict(list)
    with predictions_file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["experiment"], int(row["seed"]), row["held_source"], row["assay"])
            grouped[key].append((int(row["label"]), float(row["score"])))

    checked_assays = 0
    checked_sources = 0
    for experiment in report["experiments"]:
        source_aucs = []
        source_prs = []
        for source in experiment["per_source"]:
            if "roc_auc" not in source:
                continue
            assay_aucs = []
            assay_prs = []
            for assay, expected in source["per_assay"].items():
                key = (experiment["experiment"], int(experiment["seed"]), source["source"], assay)
                observations = grouped.get(key)
                if not observations:
                    raise ValueError(f"missing predictions for {key}")
                labels = [label for label, _ in observations]
                scores = [score for _, score in observations]
                auc = float(roc_auc_score(labels, scores))
                pr = float(average_precision_score(labels, scores))
                if not math.isclose(auc, float(expected["roc_auc"]), abs_tol=1e-12):
                    raise ValueError(f"ROC AUC mismatch for {key}: {auc} != {expected['roc_auc']}")
                if not math.isclose(pr, float(expected["pr_auc"]), abs_tol=1e-12):
                    raise ValueError(f"PR AUC mismatch for {key}: {pr} != {expected['pr_auc']}")
                assay_aucs.append(auc)
                assay_prs.append(pr)
                checked_assays += 1
            source_auc = fmean(assay_aucs)
            source_pr = fmean(assay_prs)
            if not math.isclose(source_auc, float(source["roc_auc"]), abs_tol=1e-12):
                raise ValueError(f"source AUC mismatch for {source['source']}")
            source_aucs.append(source_auc)
            source_prs.append(source_pr)
            checked_sources += 1
        if source_aucs:
            if not math.isclose(fmean(source_aucs), float(experiment["macro_source_roc_auc"]), abs_tol=1e-12):
                raise ValueError(f"macro AUC mismatch for {experiment['experiment']} seed {experiment['seed']}")
            if not math.isclose(fmean(source_prs), float(experiment["macro_source_pr_auc"]), abs_tol=1e-12):
                raise ValueError(f"macro PR mismatch for {experiment['experiment']} seed {experiment['seed']}")

    return {
        "verified": True,
        "report": str(report_file.resolve()),
        "predictions": str(predictions_file.resolve()),
        "predictions_sha256": actual_hash,
        "checked_experiments": len(report["experiments"]),
        "checked_sources": checked_sources,
        "checked_assays": checked_assays,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/developability_baseline_matrix.json"))
    parser.add_argument("--predictions", type=Path, default=Path("results/developability_baseline_predictions.csv"))
    args = parser.parse_args(argv)
    result = verify(args.report, args.predictions)
    print("Independent matrix verification: PASS")
    print(f"  experiments: {result['checked_experiments']}")
    print(f"  sources: {result['checked_sources']}")
    print(f"  assays: {result['checked_assays']}")
    print(f"  predictions sha256: {result['predictions_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
