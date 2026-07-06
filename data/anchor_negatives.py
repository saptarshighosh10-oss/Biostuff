"""
Extract anchor sequences from Phase 1 output as clean negatives.

Anchor sequences are real PDB antibody chains that were:
  1. Experimentally determined (X-ray / cryo-EM)
  2. Successfully crystallized → confirmed soluble
  3. Already on disk — no downloads needed

This is the cheapest possible source of working examples.
"""

import json
import re
from pathlib import Path

AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{20,}$", re.IGNORECASE)

PHASE1_FILE = "results/phase1_candidates.json"


def load_anchor_negatives(
    candidates_file: str = PHASE1_FILE,
    max_sequences: int = 500,
) -> list[dict]:
    """
    Extract unique anchor sequences from phase1_candidates.json as working examples.
    Returns list of dicts with variant_sequence + label="working" + source="pdb_anchor".
    """
    path = Path(candidates_file)
    if not path.exists():
        print(f"  {candidates_file} not found — run the pipeline first.")
        return []

    with open(path) as f:
        candidates = json.load(f)

    seen: set[str] = set()
    result = []

    for c in candidates:
        seq = c.get("anchor_sequence", "").strip().upper()
        if not seq or not AA_PATTERN.match(seq):
            continue
        if seq in seen:
            continue
        seen.add(seq)
        result.append({
            "variant_sequence": seq,
            "label": "working",
            "source": "pdb_anchor",
            "anchor_pdb": c.get("anchor_pdb", ""),
            "chain_type": c.get("chain_type", ""),
            "group_id": c.get("anchor_pdb", ""),
        })
        if len(result) >= max_sequences:
            break

    print(f"  Loaded {len(result)} PDB anchor sequences as negatives")
    return result
