"""
Fast inference on new sequences using the trained failure model.
No ESMFold API calls — runs entirely on CPU from sequence alone.

Usage:
    python -m model.predict --sequence EVQLVESGGGLVQPGG...
    python -m model.predict --file results/phase2_candidates.json --top 10
"""

import json
import argparse
from pathlib import Path

MODEL_PATH = "model/saved/model.pkl"


def score_sequence(sequence: str, mutations: list | None = None) -> dict:
    """
    Score a single sequence. Returns risk probability + top contributing features.
    No API calls — uses only sequence-based and multi-predictor features.
    """
    from model.failure_model import AggregationFailureModel
    from model.features import extract_from_sequence, features_to_vector, FEATURE_NAMES

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python -m model.train` first."
        )

    model = AggregationFailureModel.load(MODEL_PATH)
    feat_dict = extract_from_sequence(sequence, mutations)
    vec = features_to_vector(feat_dict)
    probs, gaps = model.predict_with_uncertainty([vec])
    prob = float(probs[0])
    gap  = float(gaps[0])

    # top 5 contributing features (value × importance)
    importance = dict(model.feature_importance())
    contributions = sorted(
        [(name, feat_dict.get(name, 0.0), importance.get(name, 0.0))
         for name in FEATURE_NAMES],
        key=lambda x: abs(x[1]) * x[2],
        reverse=True,
    )

    confidence = "high" if gap < 0.15 else "medium" if gap < 0.30 else "low"

    # closest known reference of each class — "this looks like X"
    working_neighbors = model.nearest_neighbors(vec, label=0, top_k=1)
    failure_neighbors = model.nearest_neighbors(vec, label=1, top_k=1)

    return {
        "failure_probability": round(prob, 4),
        "confidence_gap":      round(gap, 4),
        "confidence":          confidence,
        "risk_level": "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low",
        "top_features": [
            {"feature": name, "value": round(val, 4), "importance": round(imp, 4)}
            for name, val, imp in contributions[:5]
        ],
        "closest_working": working_neighbors[0] if working_neighbors else None,
        "closest_failure": failure_neighbors[0] if failure_neighbors else None,
    }


def score_file(input_file: str, top_n: int = 10):
    """Re-score candidates from a JSON file using the trained model."""
    with open(input_file) as f:
        candidates = json.load(f)

    results = []
    for c in candidates[:top_n]:
        seq = c.get("variant_sequence", "")
        mutations = c.get("mutations", [])
        if not seq:
            continue
        score = score_sequence(seq, mutations)
        results.append({
            "anchor_pdb": c.get("anchor_pdb", "?"),
            "mutations": mutations,
            "phase1_risk": c.get("risk", {}).get("combined_risk", 0.0),
            **score,
        })

    results.sort(key=lambda x: x["failure_probability"], reverse=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score sequences with the trained failure model")
    parser.add_argument("--sequence", help="Single sequence to score")
    parser.add_argument("--file", help="Score candidates from a JSON file")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    if args.sequence:
        result = score_sequence(args.sequence)
        print(f"\nFailure probability: {result['failure_probability']:.4f} ({result['risk_level']})")
        print(f"Confidence:          {result['confidence']} (gap={result['confidence_gap']:.4f})")
        print("Top contributing features:")
        for f in result["top_features"]:
            print(f"  {f['feature']:35s} value={f['value']:.3f}  importance={f['importance']:.4f}")
        cw, cf = result.get("closest_working"), result.get("closest_failure")
        if cw:
            print(f"\nClosest known WORKING reference:  {cw['name']} (source={cw['source']}, distance={cw['distance']:.3f})")
        if cf:
            print(f"Closest known FAILURE reference:  {cf['name']} (source={cf['source']}, distance={cf['distance']:.3f})")

    elif args.file:
        results = score_file(args.file, top_n=args.top)
        print(f"\nTop {len(results)} candidates re-scored by trained model:\n")
        for i, r in enumerate(results):
            conf_str = f"confidence={r['confidence']} gap={r['confidence_gap']:.3f}"
            print(f"  {i+1}. {r['anchor_pdb']} | prob={r['failure_probability']:.3f} "
                  f"({r['risk_level']}) | {conf_str} | phase1_risk={r['phase1_risk']:.3f}")
            print(f"       mutations: {r['mutations']}")
            cw, cf = r.get("closest_working"), r.get("closest_failure")
            if cw:
                print(f"       closest working: {cw['name']} (distance={cw['distance']:.3f})")
            top_feat = r['top_features'][0] if r['top_features'] else {}
            if top_feat:
                print(f"       top driver: {top_feat['feature']} = {top_feat['value']:.3f}")

    else:
        parser.print_help()
