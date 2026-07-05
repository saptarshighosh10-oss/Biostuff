"""
Active learning feedback loop.

Takes wet-lab results, merges them with existing training data,
and retrains the model. Run this after each round of wet-lab testing.

Usage:
    python -m model.feedback --results results/wetlab_results.json

The wetlab_results.json file should look like:
    [
      {
        "variant_sequence": "EVQLVESGG...",
        "label": "confirmed_failure",
        "anchor_pdb": "1NFD",
        "mutations": [[201, "V", "E"], [92, "N", "D"]],
        "notes": "aggregated at 37C within 24h"
      },
      ...
    ]

Labels: "confirmed_failure" or "working"
"""

import json
import argparse
from pathlib import Path

TRAINING_DATA_PATH = Path("results/training_data.json")
MODEL_PATH         = Path("model/saved/model.pkl")


def load_existing(path: Path) -> list[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def merge_results(existing: list[dict], new_results: list[dict]) -> tuple[list[dict], int]:
    """
    Merge new wet-lab results into existing training data.
    Deduplicates by variant_sequence.
    Returns (merged_list, n_added).
    """
    seen = {e["variant_sequence"] for e in existing}
    added = 0
    merged = list(existing)
    for entry in new_results:
        seq = entry.get("variant_sequence", "").strip().upper()
        label = entry.get("label", "").strip().lower()
        if not seq or label not in ("confirmed_failure", "working"):
            print(f"  Skipping entry — missing sequence or invalid label: {entry.get('label')}")
            continue
        if seq in seen:
            print(f"  Duplicate skipped: {seq[:20]}...")
            continue
        seen.add(seq)
        merged.append({
            "variant_sequence": seq,
            "label": label,
            "source": "wetlab",
            "anchor_pdb":  entry.get("anchor_pdb", ""),
            "mutations":   entry.get("mutations", []),
            "notes":       entry.get("notes", ""),
        })
        added += 1
    return merged, added


def print_priority_list(candidates_file: str, top_n: int = 10):
    """
    Print the next candidates to test based on high risk + high uncertainty gap.
    Call this to decide what to test in the next wet-lab round.
    """
    if not Path(candidates_file).exists():
        print(f"  {candidates_file} not found — run the pipeline first.")
        return

    if not MODEL_PATH.exists():
        print("  No trained model yet — run model/train.py first.")
        return

    from model.failure_model import AggregationFailureModel
    from model.features import extract_from_sequence, features_to_vector

    model = AggregationFailureModel.load(str(MODEL_PATH))

    with open(candidates_file) as f:
        candidates = json.load(f)

    # score each candidate with uncertainty
    scored = []
    for c in candidates:
        seq = c.get("variant_sequence", "")
        if not seq:
            continue
        muts = c.get("mutations", [])
        feat = extract_from_sequence(seq, muts)
        vec  = features_to_vector(feat)
        probs, gaps = model.predict_with_uncertainty([vec])
        prob = float(probs[0])
        gap  = float(gaps[0])
        # wet-lab value: high risk and high uncertainty → most informative to test
        wetlab_value = prob * (1 + gap) / 2
        scored.append({
            "anchor_pdb":  c.get("anchor_pdb", "?"),
            "mutations":   c.get("mutations", []),
            "sequence":    seq,
            "prob":        round(prob, 3),
            "gap":         round(gap, 3),
            "confidence":  "high" if gap < 0.15 else "medium" if gap < 0.30 else "low",
            "wetlab_value": round(wetlab_value, 3),
        })

    scored.sort(key=lambda x: x["wetlab_value"], reverse=True)

    print(f"\nNext {top_n} candidates to test (ranked by wet-lab value):")
    print(f"{'#':>3}  {'PDB':6}  {'prob':>5}  {'gap':>5}  {'conf':8}  {'value':>6}  mutations")
    print("-" * 75)
    for i, s in enumerate(scored[:top_n]):
        muts = " ".join(f"{o}{p}{m}" for p, o, m in s["mutations"])
        print(f"{i+1:>3}  {s['anchor_pdb']:6}  {s['prob']:>5.3f}  "
              f"{s['gap']:>5.3f}  {s['confidence']:8}  {s['wetlab_value']:>6.3f}  {muts}")


def run_feedback(results_file: str, retrain: bool = True):
    print("=" * 60)
    print("FEEDBACK: Merging wet-lab results")
    print("=" * 60)

    # load new results
    with open(results_file) as f:
        new_results = json.load(f)
    print(f"\nNew wet-lab results loaded: {len(new_results)} entries")

    # merge with existing training data
    existing = load_existing(TRAINING_DATA_PATH)
    print(f"Existing training data: {len(existing)} entries")

    merged, n_added = merge_results(existing, new_results)
    print(f"Added {n_added} new entries ({len(merged)} total)")

    if n_added == 0:
        print("Nothing new to add — all sequences already in training data.")
        return

    # save merged training data
    TRAINING_DATA_PATH.parent.mkdir(exist_ok=True)
    with open(TRAINING_DATA_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved to {TRAINING_DATA_PATH}")

    # count labels
    failures = sum(1 for e in merged if e["label"] == "confirmed_failure")
    working  = sum(1 for e in merged if e["label"] == "working")
    print(f"\nTraining data now: {failures} failures | {working} working")

    if retrain:
        print("\nRetraining model with updated data...")
        from model.train import train
        train(
            input_file=str(TRAINING_DATA_PATH),
            use_flab=True,
            use_abdev=True,
            use_anchors=True,
            flab_percentile=0.4,
        )
        print("\nModel updated. Run predict.py to re-score your candidates.")
    else:
        print("\nSkipped retrain (--no-retrain). Run model/train.py manually when ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge wet-lab results and retrain")
    parser.add_argument("--results",    required=False,
                        help="Path to wetlab_results.json")
    parser.add_argument("--priority",   default="results/phase3_candidates.json",
                        help="Show next candidates to test (default: phase3_candidates.json)")
    parser.add_argument("--top",        type=int, default=10,
                        help="How many candidates to show (default: 10)")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Merge data but skip retraining")
    args = parser.parse_args()

    if args.results:
        run_feedback(args.results, retrain=not args.no_retrain)
    else:
        # no results file → just show what to test next
        print_priority_list(args.priority, top_n=args.top)
