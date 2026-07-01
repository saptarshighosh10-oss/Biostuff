"""
AntiRef — clustered human antibody sequence collections from OAS.
Used as clean negative training examples (germline-like, non-aggregating).

We sample a small representative set rather than downloading the full dataset
(which is hundreds of MB). Uses the GitHub API to find available files,
then fetches a partial FASTA stream.
"""

import re
import requests

ANTIREF_API_URL  = "https://api.github.com/repos/brineylab/antiref/contents"
ANTIREF_RAW_BASE = "https://raw.githubusercontent.com/brineylab/antiref/main"

AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{20,}$", re.IGNORECASE)


def _list_fasta_files() -> list[str]:
    try:
        r = requests.get(ANTIREF_API_URL, timeout=15,
                         headers={"Accept": "application/vnd.github.v3+json"})
        r.raise_for_status()
        return [f["name"] for f in r.json()
                if f["name"].endswith(".fasta") or f["name"].endswith(".fa")]
    except requests.RequestException:
        return []


def _parse_fasta_stream(text: str, max_seqs: int) -> list[str]:
    sequences = []
    current = []
    for line in text.splitlines():
        if line.startswith(">"):
            if current:
                seq = "".join(current).upper()
                if AA_PATTERN.match(seq):
                    sequences.append(seq)
                    if len(sequences) >= max_seqs:
                        return sequences
                current = []
        else:
            current.append(line.strip())
    if current:
        seq = "".join(current).upper()
        if AA_PATTERN.match(seq) and len(sequences) < max_seqs:
            sequences.append(seq)
    return sequences


def load_antiref_negatives(
    max_sequences: int = 500,
    prefer_clustered: bool = True,
) -> list[dict]:
    """
    Download a sample of AntiRef sequences as negative (working) training examples.
    Returns list of dicts with variant_sequence + label="working".
    """
    print("Fetching AntiRef file list from GitHub...")
    files = _list_fasta_files()

    if not files:
        print("  No FASTA files found in AntiRef repo. Check network access.")
        return []

    # prefer antiref90 (less redundant than antiref100, more diverse than antiref70)
    if prefer_clustered:
        files.sort(key=lambda f: (
            0 if "90" in f else
            1 if "85" in f else
            2 if "95" in f else
            3
        ))

    sequences: list[str] = []
    for fname in files:
        if len(sequences) >= max_sequences:
            break
        url = f"{ANTIREF_RAW_BASE}/{fname}"
        print(f"  Downloading {fname}...")
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            # only read enough text to get max_sequences
            chunk = ""
            for part in r.iter_content(chunk_size=65536, decode_unicode=True):
                chunk += part
                seqs = _parse_fasta_stream(chunk, max_sequences - len(sequences))
                sequences.extend(seqs)
                if len(sequences) >= max_sequences:
                    break
        except requests.RequestException as e:
            print(f"  Failed to download {fname}: {e}")
            continue

    print(f"  Loaded {len(sequences)} AntiRef negative sequences")

    # deduplicate
    seen: set[str] = set()
    result = []
    for seq in sequences:
        if seq not in seen:
            seen.add(seq)
            result.append({
                "variant_sequence": seq,
                "label": "working",
                "source": "antiref",
            })

    return result
