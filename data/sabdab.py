"""
SAbDab — Structural Antibody Database (Oxford Protein Informatics Group).
Fetches therapeutic/clinical antibody sequences as clean negatives.

API: http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/
All deposited antibodies are experimentally determined structures →
confirmed soluble → label="working".
"""

import csv
import io
import re
import requests

SABDAB_SUMMARY_URL = (
    "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/"
    "?format=csv&CDRdef=chothia&rfactor=&resolution=3.0&otype=All&otype=&method=X-RAY+DIFFRACTION"
)
SABDAB_SEQ_URL = (
    "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/sequence/{pdb_id}/"
    "?chain_type={chain_type}&fasta=1"
)

AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{20,}$", re.IGNORECASE)


def _fetch_sabdab_summary(max_entries: int = 500) -> list[dict]:
    """Fetch the SAbDab summary CSV — one row per PDB entry."""
    try:
        r = requests.get(SABDAB_SUMMARY_URL, timeout=30)
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        rows = list(reader)
        print(f"  SAbDab summary: {len(rows)} entries")
        return rows[:max_entries]
    except requests.RequestException as e:
        print(f"  SAbDab summary fetch failed: {e}")
        return []


def _extract_sequences_from_summary(rows: list[dict]) -> list[str]:
    """
    Extract VH/VL sequences directly from the summary CSV.
    SAbDab summary includes Hseq and Lseq columns.
    """
    sequences = []
    seen: set[str] = set()

    for row in rows:
        for col in ("Hseq", "Lseq", "hseq", "lseq", "heavy_seq", "light_seq"):
            seq = row.get(col, "").strip().upper()
            if seq and seq not in seen and AA_PATTERN.match(seq):
                seen.add(seq)
                sequences.append(seq)

    return sequences


def load_sabdab_negatives(
    max_sequences: int = 500,
    max_entries: int = 600,
) -> list[dict]:
    """
    Fetch therapeutic antibody sequences from SAbDab as clean negatives.
    Returns list of dicts with variant_sequence + label="working" + source="sabdab".
    """
    print("Fetching SAbDab antibody sequences...")
    rows = _fetch_sabdab_summary(max_entries=max_entries)

    if not rows:
        print("  SAbDab unavailable. Skipping.")
        return []

    sequences = _extract_sequences_from_summary(rows)
    print(f"  Extracted {len(sequences)} sequences from summary")

    if not sequences:
        print("  No sequences found in SAbDab summary columns.")
        return []

    sequences = sequences[:max_sequences]

    result = []
    seen: set[str] = set()
    for seq in sequences:
        if seq not in seen:
            seen.add(seq)
            result.append({
                "variant_sequence": seq,
                "label": "working",
                "source": "sabdab",
            })

    print(f"  Loaded {len(result)} SAbDab working sequences")
    return result
