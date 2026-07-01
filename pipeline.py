"""
Phase 1 pipeline — fully wired end-to-end.

Steps:
  1. Fetch anchor sequences from PDB
  2. Extract CDR regions
  3. Generate BLOSUM62-guided variants
  4. Score each variant (fast sequence-based, then ESMFold for high-risk)
  5. Flag high-disagreement candidates for wet-lab testing
  6. Save results to JSON
"""

import json
import time
from pathlib import Path

from data.fetch_pdb import search_antibody_entries
from data.fetch_sequences import fetch_antibody_dataset, AntibodyChain
from data.perturb import generate_variants, load_esm2
from features.cdr import extract_cdrs
from predictors.aggregation_risk import fast_risk_score, full_risk_score

RESULTS_DIR = Path("results")


def run_phase1(
    max_pdb_entries: int = 300,
    variants_per_sequence: int = 5,
    mutations_per_variant: int = 2,
    use_esmfold: bool = False,
    use_esm2: bool = False,
    top_n: int = 20,
    output_file: str = "results/phase1_candidates.json",
):
    """
    Run the full Phase 1 pipeline.

    Args:
        max_pdb_entries: how many PDB entries to pull anchors from
        variants_per_sequence: how many mutated variants to generate per anchor
        mutations_per_variant: mutations per variant (keep <=3 for attribution)
        use_esmfold: whether to call ESMFold API for structure-based disagreement
        output_file: where to save flagged candidates
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("PHASE 1: Anchor → Perturb → Predict → Flag")
    print("=" * 60)

    # ── Load ESM-2 if requested ──────────────────────────────────
    esm2_model, esm2_alphabet = None, None
    if use_esm2:
        print("\nLoading ESM-2 model for guided mutations...")
        esm2_model, esm2_alphabet = load_esm2()
        if esm2_model is None:
            print("  fair-esm not installed — falling back to BLOSUM62.")
            print("  Install with: pip install fair-esm")
        else:
            print("  ESM-2 loaded.")

    # ── Step 1: Fetch anchor sequences ──────────────────────────
    print(f"\n[1/5] Fetching up to {max_pdb_entries} antibody PDB entries...")
    pdb_ids = search_antibody_entries(max_results=max_pdb_entries)
    print(f"      Found {len(pdb_ids)} PDB IDs")

    print("\n[2/5] Fetching sequences for each entry...")
    anchors: list[AntibodyChain] = fetch_antibody_dataset(pdb_ids, max_per_entry=2)
    print(f"      Retrieved {len(anchors)} antibody chains")

    if not anchors:
        print("\nNo antibody chains found. This usually means the PDB FASTA")
        print("headers didn't match the keyword filter. Try --entries 30.")
        return []

    # ── Step 2: Extract CDRs (informational) ────────────────────
    print("\n[3/5] Extracting CDR regions...")
    for chain in anchors:
        cdrs = extract_cdrs(chain.sequence, chain.chain_type)
        if cdrs:
            print(f"      {chain.pdb_id} chain {chain.chain_id}: "
                  f"CDR1={cdrs.cdr1[:6]}... CDR3={cdrs.cdr3[:6]}...")

    # ── Step 3: Generate variants ────────────────────────────────
    print(f"\n[4/5] Generating {variants_per_sequence} variants per anchor "
          f"({mutations_per_variant} mutations each)...")
    all_variants = []
    for chain in anchors:
        variants = generate_variants(
            chain.sequence,
            n_variants=variants_per_sequence,
            mutations_per_variant=mutations_per_variant,
            esm2_model=esm2_model,
            esm2_alphabet=esm2_alphabet,
        )
        for variant_seq, mutations in variants:
            if not mutations:
                continue
            all_variants.append({
                "anchor_pdb": chain.pdb_id,
                "anchor_chain": chain.chain_id,
                "chain_type": chain.chain_type,
                "anchor_sequence": chain.sequence,
                "variant_sequence": variant_seq,
                "mutations": [(pos, orig, mut) for pos, orig, mut in mutations],
                "n_mutations": len(mutations),
            })

    print(f"      Generated {len(all_variants)} variants total")

    # ── Step 4a: Fast sequence scoring on all variants ──────────
    print(f"\n[5/5] Scoring variants...")
    print(f"      Step 1: fast sequence scoring on all {len(all_variants)}...")

    scored = []
    for i, variant in enumerate(all_variants):
        risk = fast_risk_score(variant["variant_sequence"])
        variant["risk"] = {
            k: v for k, v in risk.items()
            if k not in ("sequence_features", "pdb_string")
        }
        scored.append(variant)
        if (i + 1) % 50 == 0:
            print(f"      Scored {i + 1}/{len(all_variants)}...")

    # ── Step 4b: ESMFold on top N only ──────────────────────────
    scored.sort(key=lambda v: v["risk"].get("combined_risk", 0), reverse=True)

    if use_esmfold:
        esmfold_n = min(top_n, len(scored))
        print(f"      Step 2: ESMFold on top {esmfold_n} candidates...")
        from predictors.esmfold import fold_sequence
        from features import structure as struct_features
        from predictors.ensemble import _compute_disagreement

        for i, variant in enumerate(scored[:esmfold_n]):
            fold = fold_sequence(variant["variant_sequence"])
            if fold:
                struct = struct_features.compute_all(fold["plddt_scores"])
                disagreement = _compute_disagreement({
                    "camsol_score": variant["risk"]["camsol_score"],
                    "structure_features": struct,
                })
                variant["risk"]["structure_features"] = struct
                variant["risk"]["disagreement_score"] = disagreement
                variant["risk"]["combined_risk"] = max(
                    variant["risk"]["combined_risk"],
                    disagreement,
                )
            time.sleep(1)
            print(f"      ESMFold {i + 1}/{esmfold_n} done")

        # re-sort after ESMFold updates
        scored.sort(key=lambda v: v["risk"].get("combined_risk", 0), reverse=True)

    # ── Output ───────────────────────────────────────────────────
    top_candidates = scored[:top_n]

    with open(output_file, "w") as f:
        json.dump(top_candidates, f, indent=2, default=str)

    scores = [v["risk"]["combined_risk"] for v in scored]
    print(f"\n{'=' * 60}")
    print(f"PHASE 1 COMPLETE")
    print(f"  Anchors:          {len(anchors)} chains")
    print(f"  Variants tested:  {len(all_variants)}")
    print(f"  Risk score range: {min(scores):.3f} – {max(scores):.3f}")
    print(f"  ESMFold used:     {'yes, top ' + str(min(top_n, len(scored))) if use_esmfold else 'no'}")
    print(f"  Top {top_n} saved to: {output_file}")
    print(f"{'=' * 60}")
    mutation_mode = "ESM-2" if (use_esm2 and esm2_model is not None) else "BLOSUM62"
    print(f"\nTop 5 candidates:")
    for i, v in enumerate(top_candidates[:5]):
        dscore = v["risk"].get("disagreement_score")
        dscore_str = f" | disagreement={dscore:.3f}" if dscore is not None else ""
        print(f"  {i+1}. {v['anchor_pdb']} {v['chain_type']} | "
              f"mutations={v['mutations']} | risk={v['risk']['combined_risk']:.3f}{dscore_str}")
    print(f"\n  Mutation mode: {mutation_mode}")
    print(f"Once you have >=30 confirmed failures, run model/train.py.")

    return top_candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 1 antibody aggregation pipeline")
    parser.add_argument("--entries", type=int, default=100, help="PDB entries to fetch")
    parser.add_argument("--variants", type=int, default=5, help="Variants per sequence")
    parser.add_argument("--mutations", type=int, default=2, help="Mutations per variant")
    parser.add_argument("--esmfold", action="store_true", help="Enable ESMFold API calls on top candidates")
    parser.add_argument("--esm2", action="store_true", help="Use ESM-2 for guided mutations (requires: pip install fair-esm)")
    parser.add_argument("--top", type=int, default=20, help="How many top candidates to save")
    args = parser.parse_args()

    run_phase1(
        max_pdb_entries=args.entries,
        variants_per_sequence=args.variants,
        mutations_per_variant=args.mutations,
        use_esmfold=args.esmfold,
        use_esm2=args.esm2,
        top_n=args.top,
    )
