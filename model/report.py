"""
Unified candidate report.

Stitches everything the pipeline knows about one candidate into a single
readable narrative: predicted risk + confidence, the biological drivers
behind that score, the closest known working/failure reference proteins,
a rescue-mutation suggestion, and (when present in the input file) the
Phase 2 multi-predictor hotspots and Phase 3 database evidence.

This doesn't compute anything new — it's a presentation layer over
model.predict, model.rescue, and whatever phase2/phase3 already annotated
onto the candidate. Use it to read one candidate closely instead of
scanning separate command outputs.

Usage:
    python -m model.report --file results/phase3_candidates.json --top 5
    python -m model.report --file results/phase3_candidates.json --index 2
"""

import json
import argparse

FEATURE_PHRASES = {
    "hydrophobicity_mean":       "mean hydrophobicity",
    "net_charge":                "net charge",
    "charge_density":            "charge density",
    "hydrophobic_patch_score":   "largest hydrophobic patch",
    "camsol_score":               "CamSol solubility score",
    "aromatic_fraction":         "aromatic residue content (pi-stacking risk)",
    "beta_sheet_propensity":     "beta-sheet-forming propensity",
    "longest_hydrophobic_run":   "longest contiguous hydrophobic stretch",
    "gatekeeper_density":        "protective gatekeeper-residue density",
    "max_beta_aromatic_patch":   "aromatic cross-beta 'zipper' motif strength",
    "n_mutations":                "number of mutations",
    "mean_hydrophobicity_delta": "mean hydrophobicity shift from mutations",
    "total_charge_delta":        "total charge shift from mutations",
    "max_hydrophobicity_delta":  "largest single hydrophobicity shift",
    "tango_mean":                 "TANGO beta-aggregation score",
    "aggrescan_mean":            "AGGRESCAN aggregation-propensity score",
    "zyggregator_mean":          "Zyggregator aggregation score",
    "n_consensus_hotspots":      "predictor-consensus aggregation hotspots",
    "mutation_hits_hotspot":     "a mutation lands inside a consensus hotspot",
    "plddt_mean":                 "mean ESMFold structural confidence",
    "plddt_variance":            "ESMFold confidence variance",
    "low_confidence_fraction":   "fraction of low-confidence structure",
    "disagreement_score":        "sequence-vs-structure predictor disagreement",
}


def _phrase(name: str) -> str:
    return FEATURE_PHRASES.get(name, name.replace("_", " "))


def build_report(candidate: dict, model=None, model_path: str = "model/saved/model.pkl") -> dict:
    """
    Assemble a unified report dict for one candidate. Reuses model.predict's
    scorer and model.rescue's suggester; pulls phase2/phase3 fields directly
    from the candidate dict when present.
    """
    from model.predict import score_sequence
    from model.rescue import suggest_rescue_mutations

    seq = candidate.get("variant_sequence", "")
    mutations = candidate.get("mutations", [])
    anchor_seq = candidate.get("anchor_sequence", "")

    score = score_sequence(seq, mutations, model=model) if seq else {}

    rescue = None
    if anchor_seq and mutations:
        try:
            rescue = suggest_rescue_mutations(anchor_seq, mutations, model=model,
                                              model_path=model_path, top_n=1)
        except Exception:
            rescue = None

    return {
        "anchor_pdb": candidate.get("anchor_pdb", "?"),
        "chain_type": candidate.get("chain_type", "?"),
        "mutations": mutations,
        "score": score,
        "rescue": rescue,
        "phase1_risk": candidate.get("risk", {}).get("combined_risk"),
        "phase2_hotspots": candidate.get("multi_predictor", {}).get("consensus_hotspots"),
        "phase2_explanation": candidate.get("explanation"),
        "phase3_db_confidence": candidate.get("database_evidence", {}).get("database_confidence"),
        "phase3_explanation": candidate.get("db_explanation"),
    }


def render_report(report: dict) -> str:
    """Render a build_report() dict as readable multi-line text."""
    lines = []
    s = report["score"]
    lines.append(f"{'=' * 64}")
    lines.append(f"{report['anchor_pdb']} ({report['chain_type']}) — mutations: {report['mutations']}")
    lines.append(f"{'=' * 64}")

    if s:
        lines.append(f"\nPredicted failure risk: {s['failure_probability']:.3f} ({s['risk_level']})")
        lines.append(f"Confidence:             {s['confidence']} (LR/RF gap={s['confidence_gap']:.3f})")

        lines.append("\nWhy — top contributing signals:")
        for f in s["top_features"][:3]:
            lines.append(f"  - {_phrase(f['feature'])}: value={f['value']:.3f} (importance={f['importance']:.3f})")

        cw, cf = s.get("closest_working"), s.get("closest_failure")
        if cw or cf:
            lines.append("\nClosest known references:")
            if cw:
                lines.append(f"  - working:  {cw['name']} (source={cw['source']}, distance={cw['distance']:.2f})")
            if cf:
                lines.append(f"  - failure:  {cf['name']} (source={cf['source']}, distance={cf['distance']:.2f})")
    else:
        lines.append("\n(no sequence available to score)")

    if report.get("phase2_hotspots"):
        lines.append(f"\nPhase 2 consensus hotspots: {report['phase2_hotspots']}")
    if report.get("phase2_explanation"):
        lines.append(f"Phase 2 explanation: {report['phase2_explanation']}")

    if report.get("phase3_db_confidence"):
        lines.append(f"\nPhase 3 database confidence: {report['phase3_db_confidence']}")
    if report.get("phase3_explanation"):
        lines.append(f"Phase 3 evidence: {report['phase3_explanation'][:200]}")

    rescue = report.get("rescue")
    if rescue and rescue.get("suggestions"):
        top = rescue["suggestions"][0]
        lines.append(f"\nTop rescue suggestion ({top['type']}): "
                     f"position {top['position']} {top['from']}->{top['to']}  "
                     f"predicted risk {top['predicted_risk']:.3f} "
                     f"(reduction {top['risk_reduction']:+.3f})")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified per-candidate report")
    parser.add_argument("--file", required=True, help="Candidates JSON (phase1/2/3 format)")
    parser.add_argument("--index", type=int, default=None, help="Report on a single candidate by index")
    parser.add_argument("--top", type=int, default=3, help="Report on the top N candidates (default 3)")
    parser.add_argument("--save", default=None, help="Also write the report(s) to a text file")
    args = parser.parse_args()

    with open(args.file) as f:
        candidates = json.load(f)

    indices = [args.index] if args.index is not None else list(range(min(args.top, len(candidates))))

    from model.failure_model import AggregationFailureModel
    model = None
    try:
        model = AggregationFailureModel.load("model/saved/model.pkl")
    except FileNotFoundError:
        pass

    output_chunks = []
    for i in indices:
        if i >= len(candidates):
            continue
        report = build_report(candidates[i], model=model)
        text = render_report(report)
        print(f"\n{text}")
        output_chunks.append(text)

    if args.save:
        with open(args.save, "w") as f:
            f.write("\n\n".join(output_chunks))
        print(f"\n\nSaved report(s) to {args.save}")
