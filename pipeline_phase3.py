"""
Phase 3 pipeline — database cross-referencing.

For each candidate:
  1. Maps anchor PDB → UniProt via EBI PDBe API
  2. Checks UniProt for aggregation/disease annotations
  3. Checks EBI Proteins API for known pathogenic variants at mutation sites
  4. Boosts combined_risk score where database confirms aggregation
  5. Adds plain-English database explanation

Reads:  results/phase2_candidates.json  (falls back to phase1 if missing)
Writes: results/phase3_candidates.json
"""

import json
import time
import argparse
from pathlib import Path

from predictors.database_lookup import lookup_candidate

RESULTS_DIR = Path("results")


def _build_db_explanation(evidence: dict) -> str:
    parts = []

    name = evidence.get("protein_name", "")
    if name:
        parts.append(f"Anchor maps to {name}")

    if evidence["has_aggregation_annotation"]:
        parts.append("UniProt annotates this protein as aggregation-prone")
        if evidence["relevant_comments"]:
            snippet = evidence["relevant_comments"][0][:120]
            parts.append(f'("{snippet}")')

    if evidence["disease_associations"]:
        parts.append(
            f'associated with: {", ".join(evidence["disease_associations"][:2])}'
        )

    for v in evidence["variants_at_mutation_sites"][:2]:
        pos = v["position_0idx"]
        desc = v.get("description", "")
        sig = v.get("clinical_significance", "")
        if v["is_pathogenic"]:
            parts.append(
                f"position {pos} is a known pathogenic variant in UniProt"
                + (f" ({desc})" if desc else "")
            )
        elif desc:
            parts.append(f"position {pos}: known variant — {desc}")

    if not parts:
        return "No matching aggregation evidence found in UniProt or EBI databases."

    return ". ".join(parts) + "."


def run_phase3(
    input_file: str = "results/phase2_candidates.json",
    output_file: str = "results/phase3_candidates.json",
    top_n: int = 50,
    delay: float = 0.5,
):
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("PHASE 3: Database cross-referencing")
    print("  Sources: EBI PDBe · UniProt REST · EBI Proteins API")
    print("=" * 60)

    # graceful fallback if phase2 not run yet
    if not Path(input_file).exists():
        fallback = str(Path(input_file).parent / "phase1_candidates.json")
        if Path(fallback).exists():
            print(f"\n  {input_file} not found — falling back to {fallback}")
            input_file = fallback
        else:
            print(f"\nError: {input_file} not found. Run Phase 1 first.")
            return []

    with open(input_file) as f:
        candidates = json.load(f)

    candidates.sort(key=lambda v: v["risk"].get("combined_risk", 0), reverse=True)
    top = candidates[:top_n]
    print(f"\nLoaded {len(candidates)} candidates — querying top {len(top)}")
    print(f"(~{len(top) * delay:.0f}s minimum for rate-limited API calls)\n")

    output = []
    confidence_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}

    for i, candidate in enumerate(top):
        pdb = candidate["anchor_pdb"]
        print(f"[{i+1:3d}/{len(top)}] {pdb}...", end=" ", flush=True)

        evidence = lookup_candidate(candidate)
        db_explanation = _build_db_explanation(evidence)
        conf = evidence["database_confidence"]
        confidence_counts[conf] += 1

        # print one-line status
        protein = evidence.get("protein_name", "")[:30]
        agg_flag = " AGG" if evidence["has_aggregation_annotation"] else ""
        path_flag = " PATH" if any(v["is_pathogenic"] for v in evidence["variants_at_mutation_sites"]) else ""
        print(f"{conf:6s} | {protein}{agg_flag}{path_flag}")

        # risk boost based on database confidence
        boost = {"high": 0.15, "medium": 0.07, "low": 0.02, "none": 0.0}[conf]
        original_risk = candidate["risk"].get("combined_risk", 0.0)
        db_boosted_risk = min(1.0, original_risk + boost)

        enriched = dict(candidate)
        enriched["risk"] = dict(candidate["risk"])
        enriched["risk"]["db_boosted_risk"] = round(db_boosted_risk, 4)
        enriched["database_evidence"] = evidence
        enriched["db_explanation"] = db_explanation

        output.append(enriched)
        time.sleep(delay)

    # re-sort by boosted risk
    output.sort(key=lambda v: v["risk"].get("db_boosted_risk", 0), reverse=True)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"PHASE 3 COMPLETE")
    print(f"  Candidates queried:  {len(output)}")
    print(f"  High confidence:     {confidence_counts['high']}  (UniProt aggregation + pathogenic variant)")
    print(f"  Medium confidence:   {confidence_counts['medium']} (UniProt aggregation OR known variant)")
    print(f"  Low confidence:      {confidence_counts['low']}  (disease association only)")
    print(f"  No DB evidence:      {confidence_counts['none']}")
    print(f"  Saved to:            {output_file}")
    print(f"{'=' * 60}")

    print("\nTop 5 after database re-ranking:")
    for i, c in enumerate(output[:5]):
        ev = c.get("database_evidence", {})
        print(f"\n  {i+1}. {c['anchor_pdb']} ({c.get('chain_type','?')}) "
              f"| boosted={c['risk']['db_boosted_risk']:.3f} "
              f"| db={ev.get('database_confidence','?')}")
        print(f"     {c['db_explanation'][:120]}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: database cross-referencing")
    parser.add_argument("--input",  default="results/phase2_candidates.json")
    parser.add_argument("--output", default="results/phase3_candidates.json")
    parser.add_argument("--top",    type=int,   default=50)
    parser.add_argument("--delay",  type=float, default=0.5,
                        help="Seconds between API calls (default 0.5)")
    args = parser.parse_args()

    run_phase3(
        input_file=args.input,
        output_file=args.output,
        top_n=args.top,
        delay=args.delay,
    )
