"""Source-held-out evaluation for CDR-only AIMS polyreactivity cohorts.

This deliberately does not feed CDR-only records into the paired-Fv model.
Numerical reactivity labels use the AIMS convention: 0-1 is non-polyreactive,
and 2 or more bound panel reagents is polyreactive.  The PLOS cohort supplies
the same endpoint directly as Y/N.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from model.cdr_features import cdr_feature_vector
from model.head_c_polyreactivity import _model


DATA = Path.home() / "Downloads" / "antibody_sources" / "polyreactivity_cdr_dataset.csv"
CDR_FIELDS = ("cdr_h1", "cdr_h2", "cdr_h3", "cdr_l1", "cdr_l2", "cdr_l3")

# LightGBM's sklearn wrapper emits this cosmetic warning for plain list features.
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")


def load() -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = {}
    with DATA.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            label = int(record["label"])
            row = {field: record[field] for field in CDR_FIELDS}
            row.update({"cdrh1": record["cdr_h1"], "cdrh2": record["cdr_h2"],
                        "cdrh3": record["cdr_h3"], "vh_sequence": "", "variant_sequence": "",
                        "label": int(label >= 2)
                        if record["label_type"] == "num_react" else label})
            by_source.setdefault(record["source"], []).append(row)
    return by_source


def key(row: dict) -> tuple[str, ...]:
    return tuple(row[field] for field in CDR_FIELDS)


def features(rows: list[dict]) -> list[list[float]]:
    # cdr_feature_vector uses supplied cdrh1/cdrh2/cdrh3 before sequence parsing.
    return [cdr_feature_vector(row, "heavy_all") for row in rows]


def score(train: list[dict], test: list[dict]) -> float:
    held = {key(row) for row in test}
    unique = {key(row): row for row in train if key(row) not in held}
    train = list(unique.values())
    y_train = [row["label"] for row in train]
    y_test = [row["label"] for row in test]
    if len(set(y_train)) != 2 or len(set(y_test)) != 2:
        raise ValueError("each held-out evaluation requires both label classes")
    model = _model()
    model.set_params(n_jobs=1)
    model.fit(features(train), y_train)
    prediction = model.predict_proba(features(test), validate_features=False)[:, 1]
    return float(roc_auc_score(y_test, prediction)), len(train)


def main() -> None:
    by_source = load()
    aucs = []
    print("AIMS CDR-only transfer — leave one source out; H1+H2+H3 descriptors")
    for source, test in sorted(by_source.items()):
        train = [row for name, rows in by_source.items() if name != source for row in rows]
        auc, train_n = score(train, test)
        positives = sum(row["label"] for row in test)
        aucs.append(auc)
        print(f"  {source}: train={train_n}, test={len(test)}, positives={positives}, AUC={auc:.3f}")
    print(f"  mean leave-one-source-out AUC={np.mean(aucs):.3f}")


if __name__ == "__main__":
    main()
