"""Evaluate the frozen Head A artifact on the protein-disjoint test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.pretrain_proteingym_fitness import _load_rows, load_general_fitness_model


def evaluate(test_file: str, report_file: str, progress_file: str, progress_every: int) -> dict:
    model = load_general_fitness_model()
    X, y, groups = _load_rows(
        test_file, progress_file=progress_file, stage="head_a_test_evaluation",
        progress_every=progress_every,
    )
    probabilities = model.score_features(X)
    from sklearn.metrics import average_precision_score, roc_auc_score
    report = {
        "head": "general_protein_mutation_fitness",
        "evaluation": "frozen_test_split",
        "test_count": len(y),
        "test_protein_groups": len(set(groups)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
    }
    target = Path(report_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate frozen Head A")
    parser.add_argument("--test", default="data/external/proteingym/test.json")
    parser.add_argument("--report", default="results/head_a_test_report.json")
    parser.add_argument("--progress-file", default="results/head_a_progress.json")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.test, args.report, args.progress_file, args.progress_every), indent=2))
