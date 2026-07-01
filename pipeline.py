"""
End-to-end pipeline orchestration.
Phase 1: fetch → compute features → predict → flag disagreements.
Training happens only after sufficient wet-lab data is collected.
"""

from data.fetch_pdb import search_antibody_entries
from features import sequence as seq_features
from predictors.ensemble import run_all
from labeling.auto_label import auto_label


def run_phase1(max_entries: int = 20):
    """
    Phase 1: Pull anchors, compute features, flag high-disagreement variants.
    No training yet — just building the signal layer.
    """
    print("=== Phase 1: Anchor + Predict + Flag ===\n")

    print(f"Fetching up to {max_entries} antibody PDB entries...")
    pdb_ids = search_antibody_entries(max_results=max_entries)
    print(f"Found {len(pdb_ids)} entries: {pdb_ids[:5]}...\n")

    # placeholder: in practice, fetch actual sequences from PDB entries
    print("Next step: fetch sequences for each PDB ID and run predictors.")
    print("Run predictors/ensemble.py on each sequence.")
    print("Flag any with disagreement_score > 0.5 for wet-lab testing.\n")

    print("=== Phase 1 scaffold ready. Implement sequence fetching next. ===")


if __name__ == "__main__":
    run_phase1()
