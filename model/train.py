"""
Training orchestrator.

Three data sources (can be combined):
  --input  results/labeled_results.json   your own wet-lab labeled data
  --flab                                  FLAb 31-dataset experimental aggregation data
  --antiref                               AntiRef human germline negatives

Output: model/saved/model.pkl

Usage:
    # train from FLAb + AntiRef right now (no wet-lab needed)
    python model/train.py --flab --antiref

    # add your own wet-lab data on top
    python model/train.py --flab --antiref --input results/labeled_results.json
"""

import json
import argparse
from pathlib import Path

from model.failure_model import AggregationFailureModel, MIN_TRAINING_FAILURES
from model.features import extract_features, features_to_vector, FEATURE_NAMES

SAVE_DIR = Path("model/saved")


def load_labeled_data(input_file: str) -> tuple[list, list]:
    if not Path(input_file).exists():
        return [], []
    with open(input_file) as f:
        data = json.load(f)
    failures, working, skipped = [], [], 0
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
    input_file: str | None = None,
    use_flab: bool = False,
    use_antiref: bool = False,
    min_failures: int = MIN_TRAINING_FAILURES,
    flab_percentile: float = 0.25,
    antiref_max: int = 500,
):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING: Aggregation Failure Model")
    print("=" * 60)

    all_failures: list = []
    all_working:  list = []

    # ── Source 1: user-provided labeled data ────────────────────────
    if input_file and Path(input_file).exists():
        print(f"\n[1/3] Loading labeled data from {input_file}...")
        f, w = load_labeled_data(input_file)
        print(f"  confirmed_failure: {len(f)}  |  working: {len(w)}")
        all_failures.extend(f)
        all_working.extend(w)
    else:
        print("\n[1/3] No --input file provided (or file not found) — skipping")

    # ── Source 2: FLAb experimental datasets ─────────────────────────
    if use_flab:
        print("\n[2/3] Loading FLAb aggregation datasets...")
        from data.flab import load_all_flab_data
        f, w = load_all_flab_data(percentile=flab_percentile)
        all_failures.extend(f)
        all_working.extend(w)
    else:
        print("\n[2/3] FLAb skipped (pass --flab to include)")

    # ── Source 3: AntiRef negatives ───────────────────────────────────
    if use_antiref:
        print("\n[3/3] Loading AntiRef negative sequences...")
        from data.antiref import load_antiref_negatives
        negatives = load_antiref_negatives(max_sequences=antiref_max)
        all_working.extend(negatives)
        print(f"  Added {len(negatives)} AntiRef working sequences")
    else:
        print("\n[3/3] AntiRef skipped (pass --antiref to include)")

    # ── Deduplicate across sources ───────────────────────────────────
    seen: set[str] = set()
    failures_dedup, working_dedup = [], []
    for entry in all_failures:
        seq = entry.get("variant_sequence", "")
        if seq and seq not in seen:
            seen.add(seq)
            failures_dedup.append(entry)
    for entry in all_working:
        seq = entry.get("variant_sequence", "")
        if seq and seq not in seen:
            seen.add(seq)
            working_dedup.append(entry)

    print(f"\nCombined (after dedup):")
    print(f"  confirmed_failure: {len(failures_dedup)}")
    print(f"  working:           {len(working_dedup)}")

    if len(failures_dedup) < min_failures:
        print(
            f"\nNot enough failures to train ({len(failures_dedup)} < {min_failures}).\n"
            f"Try: python model/train.py --flab --antiref"
        )
        return None

    if not working_dedup:
        print("\nNo working sequences — need both classes to train.")
        return None

    # ── Extract features ─────────────────────────────────────────────
    print("\nExtracting features...")
    X, y = [], []
    for entry in failures_dedup:
        feats = extract_features(entry)
        X.append(features_to_vector(feats))
        y.append(1)
        if (len(X)) % 200 == 0:
            print(f"  {len(X)} features extracted...")
    for entry in working_dedup:
        feats = extract_features(entry)
        X.append(features_to_vector(feats))
        y.append(0)

    print(f"  Feature vector size: {len(FEATURE_NAMES)}")
    print(f"  Total samples: {len(X)} ({sum(y)} failures, {len(y)-sum(y)} working)")

    # ── Train ────────────────────────────────────────────────────────
    print("\nTraining (logistic regression + random forest, 5-fold CV)...")
    model = AggregationFailureModel()
    model.train(X, y, FEATURE_NAMES)

    cv = model.cv_scores
    lr_auc = cv["logistic_regression_auc"]
    rf_auc = cv["random_forest_auc"]
    print(f"\nCross-validation ROC-AUC:")
    print(f"  Logistic Regression: {lr_auc.mean():.3f} ± {lr_auc.std():.3f}")
    print(f"  Random Forest:       {rf_auc.mean():.3f} ± {rf_auc.std():.3f}")

    if lr_auc.mean() < 0.6 and rf_auc.mean() < 0.6:
        print("\n  WARNING: AUC near 0.5 — model not learning well.")
        print("  If using only FLAb: the sequence features alone may not separate")
        print("  HIC/SEC scores cleanly. Add Phase 2 multi-predictor features via")
        print("  --input with Phase 2 annotated candidates for better signal.")

    print("\nTop 10 most predictive features:")
    for name, score in model.feature_importance()[:10]:
        bar = "█" * int(score * 100)
        print(f"  {name:35s} {bar} ({score:.4f})")

    save_path = str(SAVE_DIR / "model.pkl")
    model.save(save_path)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"  Failures:   {model.n_failures}")
    print(f"  Working:    {model.n_working}")
    print(f"  Best AUC:   {max(lr_auc.mean(), rf_auc.mean()):.3f}")
    print(f"  Saved to:   {save_path}")
    print(f"{'=' * 60}")
    print(f"\nRe-score your candidates:")
    print(f"  python model/predict.py --file results/phase3_candidates.json")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the aggregation failure model")
    parser.add_argument("--input",    default=None,  help="Path to labeled_results.json")
    parser.add_argument("--flab",     action="store_true", help="Include FLAb experimental data")
    parser.add_argument("--antiref",  action="store_true", help="Include AntiRef negatives")
    parser.add_argument("--min-failures", type=int, default=MIN_TRAINING_FAILURES)
    parser.add_argument("--flab-percentile", type=float, default=0.25,
                        help="Score percentile cutoff for FLAb labeling (default 0.25)")
    parser.add_argument("--antiref-max", type=int, default=500,
                        help="Max AntiRef negatives to load (default 500)")
    args = parser.parse_args()

    train(
        input_file=args.input,
        use_flab=args.flab,
        use_antiref=args.antiref,
        min_failures=args.min_failures,
        flab_percentile=args.flab_percentile,
        antiref_max=args.antiref_max,
    )
