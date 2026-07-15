"""Fit diagnostic per-assay calibrators from grouped out-of-fold predictions.

The labels are the existing empirical failure-tail labels, not physical assay
thresholds. The output is therefore a calibration aid until real thresholds are
registered; it must not be described as a universal failure probability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fit_predict(train_x, train_y, test_x):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    model.fit([[x] for x in train_x], train_y)
    return model, [float(p) for p in model.predict_proba([[x] for x in test_x])[:, 1]]


def _ece(probs: list[float], labels: list[int], bins: int = 10) -> float:
    errors = []
    for bucket in range(bins):
        picked = [i for i, p in enumerate(probs) if bucket / bins <= p < (bucket + 1) / bins]
        if not picked:
            continue
        observed = sum(labels[i] for i in picked) / len(picked)
        predicted = sum(probs[i] for i in picked) / len(picked)
        errors.append(abs(observed - predicted) * len(picked))
    return sum(errors) / len(probs) if probs else 0.0


def calibrate_report(report_path: str | Path) -> dict:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    output = {"source_report": str(report_path), "threshold_basis": "empirical_failure_tail", "assays": {}}
    for name, assay in report.get("assays", {}).items():
        if assay.get("serving_status") != "active":
            continue
        records = assay.get("oof_predictions", [])
        if len(records) < 20 or len({r.get("failure_label") for r in records}) < 2:
            continue
        x = [float(r["risk_score"]) for r in records]
        y = [int(r["failure_label"]) for r in records]
        folds = [([i for i in range(len(x)) if i % 3 != fold],
                  [i for i in range(len(x)) if i % 3 == fold]) for fold in range(3)]
        oof = [None] * len(x)
        for train_idx, test_idx in folds:
            _, predictions = _fit_predict([x[i] for i in train_idx], [y[i] for i in train_idx], [x[i] for i in test_idx])
            for i, prediction in zip(test_idx, predictions):
                oof[i] = prediction
        final_model, _ = _fit_predict(x, y, [x[0]])
        coefficient = float(final_model.coef_[0][0])
        intercept = float(final_model.intercept_[0])
        probs = [float(p) for p in oof]
        brier = sum((p - label) ** 2 for p, label in zip(probs, y)) / len(y)
        output["assays"][name] = {
            "n": len(y),
            "coefficient": round(coefficient, 8),
            "intercept": round(intercept, 8),
            "crossfit_brier": round(brier, 8),
            "crossfit_ece": round(_ece(probs, y), 8),
            "calibrator_status": "diagnostic_empirical_tail_only",
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Head B assay risk scores")
    parser.add_argument("--reports", nargs="+", default=["results/autonomous_lab/igg_plm.json"])
    parser.add_argument("--out", default="results/head_b_calibration.json")
    args = parser.parse_args()
    result = {"calibrators": [calibrate_report(path) for path in args.reports]}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote calibrators for {sum(len(r['assays']) for r in result['calibrators'])} assays to {args.out}")


if __name__ == "__main__":
    main()
