"""Train and serve the separate general ProteinGym mutation-fitness head.

This head is deliberately not the antibody aggregation model. Its metrics are
reported only for the protein-disjoint ProteinGym cohort and its output enters
the antibody model only as ``proteingym_fitness_score``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from model.features import (
    extract_features,
    features_to_vector,
    feature_schema_hash,
    HEAD_A_FEATURE_NAMES,
)


class ProteinGymFitnessModel:
    """A small, independently persisted binary mutation-fitness scorer.

    Trained and served over HEAD_A_FEATURE_NAMES — the subset of features that
    is computable identically from any bare sequence (no mutation deltas, no
    structure/risk fields, not its own output). This keeps the score valid when
    applied to raw antibody sequences that have none of those fields.
    """

    def __init__(self) -> None:
        self.scaler = None
        self.classifier = None
        self.feature_names = list(HEAD_A_FEATURE_NAMES)
        self.feature_schema_hash = feature_schema_hash(HEAD_A_FEATURE_NAMES)
        self.trained = False

    def train(self, X: list[list[float]], y: list[int]) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
        scaled = self.scaler.fit_transform(np.asarray(X, dtype=float))
        self.classifier.fit(scaled, np.asarray(y, dtype=int))
        self.trained = True

    def score_features(self, X: list[list[float]]) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("ProteinGym fitness head is not trained")
        return self.classifier.predict_proba(self.scaler.transform(np.asarray(X, dtype=float)))[:, 1]

    def score_general_fitness(self, sequence: str) -> float:
        # mutations are intentionally empty: Head A's vector excludes all
        # mutation-derived features, so a raw sequence and its mutant form
        # produce the same Head A inputs — the score is sequence-intrinsic.
        features = extract_features({"variant_sequence": sequence, "mutations": []})
        return float(self.score_features([features_to_vector(features, HEAD_A_FEATURE_NAMES)])[0])


MODEL_PATH = Path("model/saved/proteingym_fitness.pkl")


def _load_rows(path: str | Path) -> tuple[list[list[float]], list[int], list[str]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    X, y, groups = [], [], []
    for row in rows:
        X.append(features_to_vector(extract_features(row), HEAD_A_FEATURE_NAMES))
        y.append(1 if row.get("label") == "confirmed_failure" else 0)
        groups.append(str(row.get("protein_group_id") or row.get("group_id", "")))
    return X, y, groups


def train_head(train_file: str | Path, test_file: str | Path, model_file: str | Path = MODEL_PATH) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    X_train, y_train, train_groups = _load_rows(train_file)
    X_test, y_test, test_groups = _load_rows(test_file)
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise ValueError(f"ProteinGym split is not protein-disjoint: {len(overlap)} overlapping groups")

    model = ProteinGymFitnessModel()
    model.train(X_train, y_train)
    probabilities = model.score_features(X_test)
    report = {
        "head": "general_protein_mutation_fitness",
        "split_unit": "protein_group_id",
        "train_count": len(y_train),
        "test_count": len(y_test),
        "train_protein_groups": len(set(train_groups)),
        "test_protein_groups": len(set(test_groups)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "feature_names": list(HEAD_A_FEATURE_NAMES),
        "feature_schema_hash": feature_schema_hash(HEAD_A_FEATURE_NAMES),
    }
    destination = Path(model_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(model, handle)
    report_path = destination.with_name("proteingym_fitness_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def load_general_fitness_model(model_file: str | Path = MODEL_PATH) -> ProteinGymFitnessModel:
    with Path(model_file).open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, ProteinGymFitnessModel):
        raise TypeError(f"Unexpected ProteinGym head artifact: {model_file}")
    expected = feature_schema_hash(HEAD_A_FEATURE_NAMES)
    actual = getattr(model, "feature_schema_hash", None)
    if actual != expected:
        raise ValueError(
            f"ProteinGym head feature schema mismatch (artifact={actual}, code={expected}); "
            f"retrain via `python -m model.pretrain_proteingym_fitness`."
        )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the separate ProteinGym mutation-fitness head")
    parser.add_argument("--train", default="data/external/proteingym/train.json")
    parser.add_argument("--test", default="data/external/proteingym/test.json")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()
    train_head(args.train, args.test, args.model)
