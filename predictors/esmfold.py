"""
ESMFold via the free ESM Metagenomic Atlas API.
No local GPU needed — sends sequence, gets pLDDT scores back.
"""

import requests
import time

ESM_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def fold_sequence(sequence: str, retries: int = 3) -> dict | None:
    """
    Submit a sequence to ESMFold API.
    Returns dict with 'pdb_string' and parsed pLDDT scores, or None on failure.
    """
    for attempt in range(retries):
        try:
            response = requests.post(
                ESM_API_URL,
                data=sequence,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
            )
            response.raise_for_status()
            pdb_string = response.text
            plddt_scores = _parse_plddt_from_pdb(pdb_string)
            return {"pdb_string": pdb_string, "plddt_scores": plddt_scores}
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"ESMFold API failed after {retries} attempts: {e}")
                return None


def _parse_plddt_from_pdb(pdb_string: str) -> list[float]:
    """
    Extract per-residue pLDDT from ESMFold PDB output.
    ESMFold stores pLDDT in the B-factor column of ATOM records.
    """
    scores = []
    seen_residues = set()

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        residue_id = (line[21], line[22:26].strip())
        if residue_id in seen_residues:
            continue
        seen_residues.add(residue_id)
        try:
            scores.append(float(line[60:66].strip()))
        except ValueError:
            continue

    return scores
