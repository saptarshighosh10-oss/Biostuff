"""Honest within-source HIC hydrophobicity evaluation for Shehata 2019."""

from __future__ import annotations

import csv
from pathlib import Path

from model.head_b_acsins_charge import charge_features
from model.head_b_gbm import build_feature_vector
from model.head_c_polyreactivity import _model


DATA = Path.home() / "Downloads/antibody_sources/hydrophobicity_dataset.csv"
SOURCE = "shehata2019"
BAD_FRACTION = 0.15


def load_source() -> list[dict]:
    """Load one HIC source and retain one observation per VH sequence."""
    unique: dict[str, dict] = {}
    with DATA.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record["source"] != SOURCE:
                continue
            vh = record["vh"].strip().upper()
            if len(vh) < 20:
                continue
            try:
                label = float(record["label"])
            except ValueError:
                continue
            unique.setdefault(vh, {
                "variant_sequence": vh,
                "vh_sequence": vh,
                "vl_sequence": record["vl"].strip().upper() or None,
                "label": label,
            })
    return list(unique.values())


def bad_labels(rows: list[dict]) -> list[int]:
    """Flag exactly the top 15% of HIC values as hydrophobicity failures."""
    count = max(1, round(len(rows) * BAD_FRACTION))
    worst = sorted(range(len(rows)), key=lambda index: rows[index]["label"], reverse=True)[:count]
    flagged = set(worst)
    return [int(index in flagged) for index in range(len(rows))]


def features(row: dict) -> list[float]:
    return build_feature_vector(row, use_plm=False) + charge_features(row)


def evaluate(rows: list[dict]) -> dict:
    """Run Head C-style stratified five-fold CV on the source-held HIC labels."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    labels = bad_labels(rows)
    positives = sum(labels)
    if positives < 2 or positives == len(rows):
        raise ValueError("not enough HIC high/low examples for stratified CV")
    matrix = [features(row) for row in rows]
    predictions = [0.0] * len(rows)
    folds = StratifiedKFold(
        n_splits=min(5, positives, len(rows) - positives), shuffle=True, random_state=42
    )
    for train_indices, test_indices in folds.split(matrix, labels):
        model = _model()
        model.fit([matrix[index] for index in train_indices], [labels[index] for index in train_indices])
        probabilities = model.predict_proba([matrix[index] for index in test_indices])[:, 1]
        for position, index in enumerate(test_indices):
            predictions[index] = float(probabilities[position])
    top_count = max(1, round(len(rows) * BAD_FRACTION))
    top_predictions = sorted(range(len(rows)), key=lambda index: predictions[index], reverse=True)[:top_count]
    catch_rate = sum(labels[index] for index in top_predictions) / positives
    return {
        "n": len(rows),
        "positives": positives,
        "auc": float(roc_auc_score(labels, predictions)),
        "catch_rate_at_15pct": catch_rate,
    }


if __name__ == "__main__":
    report = evaluate(load_source())
    print(
        f"shehata2019 hydrophobicity ({report['n']} diverse antibodies): "
        f"AUC={report['auc']:.3f}"
    )
    print(
        f"  most-hydrophobic HIC rows: {report['positives']}/{report['n']}; "
        f"catch-rate@15%={report['catch_rate_at_15pct']:.3f}"
    )
