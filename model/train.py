"""
Training orchestrator.

Input:  results/labeled_results.json
        List of candidate dicts (Phase 1/2 format) each with a "label" field:
          "confirmed_failure" → positive class
          "working"           → negative class

Output: model/saved/model.pkl   (trained AggregationFailureModel)

Usage:
    python model/train.py
    python model/train.py --input results/labeled_results.json --min-failures 30
"""

import json
import argparse
from pathlib import Path

from model.failure_model import AggregationFailureModel, MIN_TRAINING_FAILURES
from model.features import extract_features, features_to_vector, FEATURE_NAMES

SAVE_DIR = Path("model/saved")


def load_labeled_data(input_file: str) -> tuple[list, list]:
    """Load labeled candidates. Returns (failures, working)."""
    with open(input_file) as f:
        data = json.load(f)

    failures, working = [], []
    skipped = 0
    for entry in data:
        label = entry.get("label", "").strip().lower()
        if label == "confirmed_failure":
            failures.append(entry)
        elif label == "working":
            working.append(entry)
        else:
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} entries with missing/unknown label")
    return failures, working


def train(
    input_file: str = "results/labeled_results.json",
    min_failures: int = MIN_TRAINING_FAILURES,
):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING: Aggregation Failure Model")
    print("=" * 60)

    print(f"\nLoading labeled data from {input_file}...")
    failures, working = load_labeled_data(input_file)
    print(f"  confirmed_failure: {len(failures)}")
    print(f"  working:           {len(working)}")

    if len(failures) < min_failures:
        print(
            f"\nNot enough failures to train ({len(failures)} < {min_failures}). "
            f"Keep collecting wet-lab results."
        )
        return None

    if len(working) == 0:
        print("\nNo working sequences labeled — need both classes to train.")
        return None

    # extract features
    print("\nExtracting features...")
    X, y = [], []
    for entry in failures:
        feats = extract_features(entry)
        X.append(features_to_vector(feats))
        y.append(1)
    for entry in working:
        feats = extract_features(entry)
        X.append(features_to_vector(feats))
        y.append(0)

    print(f"  Feature vector size: {len(FEATURE_NAMES)}")
    print(f"  Total samples: {len(X)} ({sum(y)} failures, {len(y) - sum(y)} working)")

    # train
    print("\nTraining (logistic regression + random forest with 5-fold CV)...")
    model = AggregationFailureModel()
    model.train(X, y, FEATURE_NAMES)

    # report CV scores
    cv = model.cv_scores
    lr_auc = cv["logistic_regression_auc"]
    rf_auc = cv["random_forest_auc"]
    print(f"\nCross-validation ROC-AUC:")
    print(f"  Logistic Regression: {lr_auc.mean():.3f} ± {lr_auc.std():.3f}")
    print(f"  Random Forest:       {rf_auc.mean():.3f} ± {rf_auc.std():.3f}")

    if lr_auc.mean() < 0.6 and rf_auc.mean() < 0.6:
        print("\n  WARNING: AUC near 0.5 — model may not be learning meaningful signal.")
        print("  Consider collecting more diverse failures before deploying.")

    # feature importance
    print("\nTop 10 most predictive features:")
    for name, score in model.feature_importance()[:10]:
        bar = "█" * int(score * 100)
        print(f"  {name:35s} {bar} ({score:.4f})")

    # save
    save_path = str(SAVE_DIR / "model.pkl")
    model.save(save_path)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"  Failures used:   {model.n_failures}")
    print(f"  Working used:    {model.n_working}")
    print(f"  Best AUC:        {max(lr_auc.mean(), rf_auc.mean()):.3f}")
    print(f"  Model saved to:  {save_path}")
    print(f"{'=' * 60}")
    print(f"\nTo score new sequences: python model/predict.py --sequence EVQLVES...")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the aggregation failure model")
    parser.add_argument("--input", default="results/labeled_results.json")
    parser.add_argument("--min-failures", type=int, default=MIN_TRAINING_FAILURES)
    args = parser.parse_args()
    train(input_file=args.input, min_failures=args.min_failures)
