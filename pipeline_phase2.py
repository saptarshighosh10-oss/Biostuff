"""
Phase 2 pipeline — mechanistic explanation layer.

Reads results/phase1_candidates.json, runs multi-predictor consensus
and optional MD on the top N candidates, writes results/phase2_candidates.json.
"""

import json
import argparse
from pathlib import Path

from predictors.multi_predictor import run_all_predictors, mutation_hits_hotspot
from features.structure_dynamics import run_short_md, rmsf_at_mutations

RESULTS_DIR = Path("results")


def _build_explanation(candidate: dict, predictor_results: dict, md_results: dict | None) -> str:
    mutations = candidate["mutations"]
    risk = candidate["risk"]
    consensus = predictor_results.get("consensus_hotspots", [])
    hits = mutation_hits_hotspot(mutations, consensus)

    parts = []

    # Describe each mutation's physicochemical effect
    HYDROPHOBICITY = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }
    CHARGE = {"R": 1.0, "K": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}

    for pos, orig, mut in mutations:
        dH = HYDROPHOBICITY.get(mut, 0.0) - HYDROPHOBICITY.get(orig, 0.0)
        dC = CHARGE.get(mut, 0.0) - CHARGE.get(orig, 0.0)
        desc = f"{orig}{pos}{mut}"
        effects = []
        if dH > 1.0:
            effects.append(f"adds hydrophobicity (+{dH:.1f})")
        elif dH < -1.0:
            effects.append(f"reduces hydrophobicity ({dH:.1f})")
        if dC < -0.5:
            effects.append(f"removes charge ({dC:+.1f})")
        elif dC > 0.5:
            effects.append(f"adds charge (+{dC:.1f})")
        if effects:
            parts.append(f"{desc} {' and '.join(effects)}")
        else:
            parts.append(f"{desc} (conservative substitution)")

    # Hotspot consensus
    n_predictors_agreeing = max((h[2] for h in consensus), default=0)
    if hits and consensus:
        hotspot_str = ", ".join(f"{h[0]}-{h[1]}" for h in consensus[:3])
        parts.append(
            f"{n_predictors_agreeing}/3 predictors flag aggregation hotspot(s) at positions {hotspot_str}"
        )
    elif consensus:
        parts.append(
            f"aggregation hotspot(s) detected but mutations do not directly overlap"
        )
    else:
        parts.append("no consensus aggregation hotspots detected")

    # MD flexibility
    if md_results:
        rmsf_vals = rmsf_at_mutations(md_results["per_residue_rmsf"], mutations)
        mean_rmsf = md_results["mean_rmsf"]
        high_flex = [
            f"{orig}{pos}{mut} (RMSF={r:.1f}Å vs mean {mean_rmsf:.1f}Å)"
            for (pos, orig, mut), r in zip(mutations, rmsf_vals)
            if r > mean_rmsf * 1.5
        ]
        if high_flex:
            parts.append(f"MD shows elevated flexibility at: {', '.join(high_flex)}")
        else:
            parts.append(f"MD shows normal flexibility at mutation sites (mean RMSF={mean_rmsf:.1f}Å)")

    return ". ".join(parts) + "."


def run_phase2(
    input_file: str = "results/phase1_candidates.json",
    output_file: str = "results/phase2_candidates.json",
    top_n: int = 20,
    run_md: bool = False,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    Path("results/structures").mkdir(exist_ok=True)

    print("=" * 60)
    print("PHASE 2: Multi-predictor consensus + mechanistic explanation")
    print("=" * 60)

    with open(input_file) as f:
        candidates = json.load(f)

    candidates.sort(key=lambda v: v["risk"].get("combined_risk", 0), reverse=True)
    top = candidates[:top_n]
    print(f"\nLoaded {len(candidates)} candidates — processing top {len(top)}")

    output = []
    for i, candidate in enumerate(top):
        pdb = candidate["anchor_pdb"]
        seq = candidate["variant_sequence"]
        print(f"\n[{i+1}/{len(top)}] {pdb} | risk={candidate['risk']['combined_risk']:.3f}")

        # Multi-predictor
        print("  Running TANGO / AGGRESCAN / Zyggregator...")
        predictor_results = run_all_predictors(seq)
        n_hotspots = len(predictor_results.get("consensus_hotspots", []))
        print(f"  Consensus hotspots: {n_hotspots}")

        # MD (optional, only if ESMFold PDB string is available)
        md_results = None
        if run_md:
            pdb_string = candidate["risk"].get("pdb_string")
            if pdb_string:
                print("  Running short MD simulation...")
                md_results = run_short_md(pdb_string)
                if md_results:
                    print(f"  MD done — mean RMSF={md_results['mean_rmsf']:.2f}Å")
                else:
                    print("  MD skipped (failed or OpenMM not installed)")
            else:
                print("  MD skipped — no ESMFold PDB string in Phase 1 output")

        # Explanation
        explanation = _build_explanation(candidate, predictor_results, md_results)
        print(f"  Explanation: {explanation}")

        # Strip per_residue_scores from output to keep JSON small
        compact_predictors = {
            k: {sk: sv for sk, sv in v.items() if sk != "per_residue_scores"}
            for k, v in predictor_results.items()
            if isinstance(v, dict)
        }
        compact_predictors["consensus_hotspots"] = predictor_results["consensus_hotspots"]

        enriched = dict(candidate)
        enriched["multi_predictor"] = compact_predictors
        if md_results:
            enriched["md"] = {
                "mean_rmsf": md_results["mean_rmsf"],
                "high_flexibility_segments": md_results["high_flexibility_segments"],
                "rmsf_at_mutations": rmsf_at_mutations(
                    md_results["per_residue_rmsf"], candidate["mutations"]
                ),
            }
        enriched["explanation"] = explanation
        output.append(enriched)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"PHASE 2 COMPLETE")
    print(f"  Candidates processed: {len(output)}")
    print(f"  MD used: {'yes' if run_md else 'no'}")
    print(f"  Saved to: {output_file}")
    print(f"{'=' * 60}")

    print("\nTop 5 with explanations:")
    for i, c in enumerate(output[:5]):
        print(f"\n  {i+1}. {c['anchor_pdb']} | risk={c['risk']['combined_risk']:.3f}")
        print(f"     {c['explanation']}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 mechanistic explanation pipeline")
    parser.add_argument("--input", default="results/phase1_candidates.json")
    parser.add_argument("--output", default="results/phase2_candidates.json")
    parser.add_argument("--top", type=int, default=20, help="How many candidates to process")
    parser.add_argument("--md", action="store_true", help="Run short MD simulation (requires OpenMM)")
    args = parser.parse_args()

    run_phase2(
        input_file=args.input,
        output_file=args.output,
        top_n=args.top,
        run_md=args.md,
    )
