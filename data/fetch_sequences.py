"""
Fetch actual antibody sequences (FASTA) from PDB entries.
Filters chains by keywords that indicate VH/VL antibody chains.
"""

import re
import requests
from dataclasses import dataclass

FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdb_id}"

ANTIBODY_KEYWORDS = {
    "heavy chain", "light chain", "vh", "vl", "fab",
    "immunoglobulin", "antibody", "variable domain",
}


@dataclass
class AntibodyChain:
    pdb_id: str
    chain_id: str
    chain_type: str   # "heavy" | "light" | "unknown"
    sequence: str
    description: str


def fetch_fasta(pdb_id: str) -> str | None:
    """Fetch raw FASTA text for a PDB entry."""
    url = FASTA_URL.format(pdb_id=pdb_id.upper())
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch {pdb_id}: {e}")
        return None


def parse_fasta(fasta_text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into (header, sequence) pairs."""
    records = []
    header, seq_parts = None, []
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            if header:
                records.append((header, "".join(seq_parts)))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    if header:
        records.append((header, "".join(seq_parts)))
    return records


def _classify_chain(description: str) -> str:
    desc_lower = description.lower()
    if any(k in desc_lower for k in ("heavy", "vh", "heavy chain")):
        return "heavy"
    if any(k in desc_lower for k in ("light", "vl", "light chain", "kappa", "lambda")):
        return "light"
    return "unknown"


def _is_antibody_chain(description: str) -> bool:
    desc_lower = description.lower()
    return any(k in desc_lower for k in ANTIBODY_KEYWORDS)


def _extract_chain_id(header: str) -> str:
    """Extract chain ID from RCSB FASTA header like '4D5_1|Chains A,B|...'"""
    match = re.search(r"Chain[s]?\s+([A-Z])", header)
    return match.group(1) if match else "?"


def fetch_antibody_chains(pdb_id: str) -> list[AntibodyChain]:
    """Fetch and return antibody chains for a PDB entry."""
    fasta_text = fetch_fasta(pdb_id)
    if not fasta_text:
        return []

    chains = []
    for header, sequence in parse_fasta(fasta_text):
        if not sequence or len(sequence) < 50:
            continue
        if not _is_antibody_chain(header):
            continue
        chains.append(AntibodyChain(
            pdb_id=pdb_id,
            chain_id=_extract_chain_id(header),
            chain_type=_classify_chain(header),
            sequence=sequence,
            description=header,
        ))
    return chains


def fetch_antibody_dataset(
    pdb_ids: list[str], max_per_entry: int = 2, workers: int = 20
) -> list[AntibodyChain]:
    """Fetch antibody chains from a list of PDB IDs in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, list[AntibodyChain]] = {}

    def _fetch(pdb_id: str) -> tuple[str, list[AntibodyChain]]:
        return pdb_id, fetch_antibody_chains(pdb_id)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, pid): pid for pid in pdb_ids}
        done = 0
        for future in as_completed(futures):
            pdb_id, chains = future.result()
            results[pdb_id] = chains[:max_per_entry]
            done += 1
            if done % 50 == 0:
                print(f"  fetched {done}/{len(pdb_ids)}...")

    # preserve input order
    all_chains = []
    for pdb_id in pdb_ids:
        all_chains.extend(results.get(pdb_id, []))
    print(f"  fetched {len(pdb_ids)} entries → {len(all_chains)} chains")
    return all_chains


def deduplicate_chains(
    chains: list[AntibodyChain], identity_threshold: float = 0.9
) -> list[AntibodyChain]:
    """
    Remove chains whose sequence is >=identity_threshold similar to an already-kept chain.
    Prevents near-identical antibody families (e.g. MC-series Bence-Jones proteins)
    from flooding variant generation with redundant mutations.
    """
    from difflib import SequenceMatcher

    unique: list[AntibodyChain] = []
    for chain in chains:
        for kept in unique:
            if SequenceMatcher(None, chain.sequence, kept.sequence).ratio() >= identity_threshold:
                break
        else:
            unique.append(chain)
    return unique
