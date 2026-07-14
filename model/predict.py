"""
Fast inference on new sequences using the trained failure model.
No ESMFold API calls — runs entirely on CPU from sequence alone.

Usage:
    python -m model.predict --sequence EVQLVESGGGLVQPGG...
    python -m model.predict --file results/phase2_candidates.json --top 10
"""

import json
import argparse
import hashlib
from pathlib import Path

MODEL_PATH = "model/saved/model.pkl"


def score_sequence(sequence: str, mutations: list | None = None, model=None, n_neighbors: int = 3) -> dict:
    """
    Score a single sequence. Returns risk probability + top contributing features.
    No API calls — uses only sequence-based and multi-predictor features.
    Pass a pre-loaded `model` to avoid re-reading model.pkl from disk when
    scoring many sequences in a loop (e.g. from model/report.py).

    n_neighbors: how many closest known references to return per class
    (working/failure), ranked nearest first — not just the single closest.
    """
    from model.failure_model import AggregationFailureModel
    from model.features import extract_from_sequence, features_to_vector, FEATURE_NAMES
    from model.explanations import explain_failure

    if model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python -m model.train` first."
            )
        model = AggregationFailureModel.load(MODEL_PATH)

    from data.contract import canonical_sequence

    sequence = canonical_sequence(sequence)
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
    top_features = [
        {"feature": name, "value": round(val, 4), "importance": round(imp, 4)}
        for name, val, imp in contributions[:5]
    ]

    confidence = "high" if gap < 0.15 else "medium" if gap < 0.30 else "low"
    grouped_cv = bool(getattr(model, "used_grouped_cv", False))
    if not grouped_cv:
        decision = "abstain"
        decision_reason = "model was not evaluated with grouped cross-validation"
    elif confidence == "low":
        decision = "abstain"
        decision_reason = "ensemble disagreement exceeds the uncertainty ceiling"
    elif prob >= 0.7:
        decision = "prioritize_wet_lab"
        decision_reason = "high modeled failure risk with acceptable model agreement"
    else:
        decision = "review"
        decision_reason = "candidate is below the high-risk prioritization threshold"

    failure_explanation = explain_failure(
        mutations=mutations or [],
        features=feat_dict,
        probability=prob,
        confidence=confidence,
        top_features=top_features,
    )

    # closest known references of each class, ranked nearest first —
    # "this looks like X, and if not, here's the next-closest match"
    working_neighbors = model.nearest_neighbors(vec, label=0, top_k=n_neighbors)
    failure_neighbors = model.nearest_neighbors(vec, label=1, top_k=n_neighbors)

    return {
        "schema_version": "1",
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "failure_probability": round(prob, 4),
        "confidence_gap":      round(gap, 4),
        "confidence":          confidence,
        "risk_level": "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low",
        "decision": decision,
        "decision_reason": decision_reason,
        "cv_strategy": getattr(model, "cv_strategy", None),
        "top_features": top_features,
        "failure_explanation": failure_explanation,
        "closest_working": working_neighbors[0] if working_neighbors else None,
        "closest_failure": failure_neighbors[0] if failure_neighbors else None,
        "closest_working_list": working_neighbors,
        "closest_failure_list": failure_neighbors,
    }


def score_file(input_file: str, top_n: int = 10, n_neighbors: int = 3, output_file: str | None = None):
    """Re-score candidates from a JSON file using the trained model."""
    from model.failure_model import AggregationFailureModel

    with open(input_file) as f:
        candidates = json.load(f)

    model = AggregationFailureModel.load(MODEL_PATH)
    results = []
    for c in candidates[:top_n]:
        seq = c.get("variant_sequence", "")
        mutations = c.get("mutations", [])
        if not seq:
            continue
        score = score_sequence(seq, mutations, model=model, n_neighbors=n_neighbors)
        results.append({
            "anchor_pdb": c.get("anchor_pdb", "?"),
            "mutations": mutations,
            "phase1_risk": c.get("risk", {}).get("combined_risk", 0.0),
            **score,
        })

    results.sort(key=lambda x: x["failure_probability"], reverse=True)
    if output_file:
        payload = {
            "schema_version": "1",
            "input_file": str(input_file),
            "model_file": MODEL_PATH,
            "results": results,
        }
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score sequences with the trained failure model")
    parser.add_argument("--sequence", help="Single sequence to score")
    parser.add_argument("--file", help="Score candidates from a JSON file")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=3,
                        help="How many closest known references to show per class (default 3)")
    parser.add_argument("--out", help="Write machine-readable prediction artifact to this JSON path")
    args = parser.parse_args()

    if args.sequence:
        result = score_sequence(args.sequence, n_neighbors=args.neighbors)
        print(f"\nFailure probability: {result['failure_probability']:.4f} ({result['risk_level']})")
        print(f"Confidence:          {result['confidence']} (gap={result['confidence_gap']:.4f})")
        print("Top contributing features:")
        for f in result["top_features"]:
            print(f"  {f['feature']:35s} value={f['value']:.3f}  importance={f['importance']:.4f}")

        cw_list = result.get("closest_working_list") or []
        cf_list = result.get("closest_failure_list") or []
        if cw_list:
            print(f"\nClosest known WORKING references (nearest first):")
            for n in cw_list:
                print(f"  #{n['rank']}  {n['name']:35s} (source={n['source']}, distance={n['distance']:.3f})")
        if cf_list:
            print(f"\nClosest known FAILURE references (nearest first):")
            for n in cf_list:
                print(f"  #{n['rank']}  {n['name']:35s} (source={n['source']}, distance={n['distance']:.3f})")

    elif args.file:
        results = score_file(args.file, top_n=args.top, n_neighbors=args.neighbors, output_file=args.out)
        print(f"\nTop {len(results)} candidates re-scored by trained model:\n")
        for i, r in enumerate(results):
            conf_str = f"confidence={r['confidence']} gap={r['confidence_gap']:.3f}"
            print(f"  {i+1}. {r['anchor_pdb']} | prob={r['failure_probability']:.3f} "
                  f"({r['risk_level']}) | {conf_str} | phase1_risk={r['phase1_risk']:.3f}")
            print(f"       mutations: {r['mutations']}")
            cw_list = r.get("closest_working_list") or []
            if cw_list:
                names = ", ".join(f"{n['name']} ({n['distance']:.2f})" for n in cw_list)
                print(f"       closest working (nearest first): {names}")
            top_feat = r['top_features'][0] if r['top_features'] else {}
            if top_feat:
                print(f"       top driver: {top_feat['feature']} = {top_feat['value']:.3f}")

    else:
        parser.print_help()
