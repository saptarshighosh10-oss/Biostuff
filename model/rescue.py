"""
Rescue-mutation suggester.

For a risky candidate (anchor/wild-type sequence + a list of mutations
applied to it), searches for the smallest change that most reduces the
model's predicted failure risk:

  1. Revert search — try reverting each mutation back to the anchor's
     original residue, one at a time (holding the others fixed).
  2. Substitution search — try replacing the mutated residue at each
     position with every other amino acid, holding everything else fixed.
     This can find a better-than-wild-type fix, not just a revert.

This is a computational proposal from the same model that scored the
candidate — it is not a synthesis guarantee. Treat suggestions as
hypotheses to prioritize for the wet-lab queue, not as verified fixes.

Usage:
    python -m model.rescue --file results/phase1_candidates.json --index 0
"""

import json
import argparse

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _apply_mutations(
    anchor_seq: str,
    mutations: list,
    skip_index: int | None = None,
    override: tuple[int, str] | None = None,
) -> str:
    """
    Apply a mutation list (pos, orig, mut) to the anchor sequence.
    skip_index: leave that mutation's position at the anchor's original
                residue (a "revert").
    override:   (index_in_list, new_residue) — replace that mutation's
                target residue with a different one.
    """
    seq = list(anchor_seq)
    for i, (pos, orig, mut) in enumerate(mutations):
        if skip_index is not None and i == skip_index:
            continue
        target = mut
        if override is not None and override[0] == i:
            target = override[1]
        if 0 <= pos < len(seq):
            seq[pos] = target
    return "".join(seq)


def _score(model, sequence: str) -> float:
    from model.features import extract_from_sequence, features_to_vector
    feat = extract_from_sequence(sequence)
    vec = features_to_vector(feat)
    return float(model.predict_proba([vec])[0])


def suggest_rescue_mutations(
    anchor_sequence: str,
    mutations: list,
    model=None,
    model_path: str = "model/saved/model.pkl",
    substitution_search: bool = True,
    top_n: int = 5,
) -> dict:
    """
    Returns {"baseline_risk": float, "suggestions": [ranked fix dicts]}.
    Each suggestion: {"type", "position", "from", "to", "predicted_risk", "risk_reduction"}.
    "type" is "revert" (back to wild-type) or "substitute" (a different residue
    entirely, found by exhaustive single-position search).
    """
    if model is None:
        from model.failure_model import AggregationFailureModel
        model = AggregationFailureModel.load(model_path)

    if not mutations:
        return {"baseline_risk": None, "suggestions": []}

    baseline_seq = _apply_mutations(anchor_sequence, mutations)
    baseline_risk = _score(model, baseline_seq)

    suggestions = []

    # 1. Revert-one-mutation search
    for i, (pos, orig, mut) in enumerate(mutations):
        reverted_seq = _apply_mutations(anchor_sequence, mutations, skip_index=i)
        risk = _score(model, reverted_seq)
        suggestions.append({
            "type": "revert",
            "position": pos,
            "from": mut,
            "to": orig,
            "predicted_risk": round(risk, 4),
            "risk_reduction": round(baseline_risk - risk, 4),
        })

    # 2. Full substitution search at each mutated position
    if substitution_search:
        for i, (pos, orig, mut) in enumerate(mutations):
            best_aa, best_risk = None, baseline_risk
            for aa in AMINO_ACIDS:
                if aa == mut:
                    continue
                test_seq = _apply_mutations(anchor_sequence, mutations, override=(i, aa))
                risk = _score(model, test_seq)
                if risk < best_risk:
                    best_risk = risk
                    best_aa = aa
            if best_aa is not None:
                suggestions.append({
                    "type": "substitute",
                    "position": pos,
                    "from": mut,
                    "to": best_aa,
                    "predicted_risk": round(best_risk, 4),
                    "risk_reduction": round(baseline_risk - best_risk, 4),
                })

    suggestions.sort(key=lambda s: s["risk_reduction"], reverse=True)
    return {
        "baseline_risk": round(baseline_risk, 4),
        "suggestions": suggestions[:top_n],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Suggest mutations to reduce a candidate's predicted failure risk"
    )
    parser.add_argument("--file", required=True, help="Candidates JSON (phase1/2/3 format)")
    parser.add_argument("--index", type=int, default=0, help="Which candidate to analyze (default 0)")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--no-substitution-search", action="store_true",
                        help="Skip the exhaustive 19-residue-per-position search (revert-only, faster)")
    args = parser.parse_args()

    with open(args.file) as f:
        candidates = json.load(f)
    if args.index >= len(candidates):
        raise SystemExit(f"--index {args.index} out of range (file has {len(candidates)} candidates)")

    c = candidates[args.index]
    anchor_seq = c.get("anchor_sequence", "")
    mutations = c.get("mutations", [])
    if not anchor_seq:
        raise SystemExit(
            "Candidate has no 'anchor_sequence' field — the wild-type sequence "
            "is required to compute reverts. Use a phase1_candidates.json-format file."
        )
    if not mutations:
        raise SystemExit("Candidate has no mutations to rescue.")

    result = suggest_rescue_mutations(
        anchor_seq, mutations,
        substitution_search=not args.no_substitution_search,
        top_n=args.top,
    )

    print(f"Candidate: {c.get('anchor_pdb', '?')} ({c.get('chain_type', '?')})")
    print(f"Mutations: {mutations}")
    print(f"Baseline predicted risk: {result['baseline_risk']}")
    print(f"\nTop {len(result['suggestions'])} rescue suggestions (ranked by risk reduction):")
    print(f"{'type':11s} {'pos':>5s}  {'change':10s} {'new risk':>9s}  {'reduction':>9s}")
    print("-" * 55)
    for s in result["suggestions"]:
        change = f"{s['from']}->{s['to']}"
        print(f"{s['type']:11s} {s['position']:>5d}  {change:10s} "
              f"{s['predicted_risk']:>9.4f}  {s['risk_reduction']:>+9.4f}")
