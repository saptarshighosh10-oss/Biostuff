"""
AbDev — 246 clinical-stage antibody sequences compiled from public sources.
Used as high-quality negatives: these are all approved/in-trial antibodies
that cleared formulation, meaning they're real-world "working" sequences.

GitHub: https://github.com/Lailabcode/AbDev

The repo contains FASTA or CSV files with VH/VL sequences.
We treat all AbDev sequences as label="working" (confirmed non-aggregators
by virtue of having passed clinical development).
"""

import csv
import io
import re
import requests

ABDEV_API_URL  = "https://api.github.com/repos/Lailabcode/AbDev/contents"
ABDEV_RAW_BASE = "https://raw.githubusercontent.com/Lailabcode/AbDev/main"

AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{10,}$", re.IGNORECASE)

# Known sequence files in Lailabcode/AbDev — fallback when the contents API is
# unavailable. raw.githubusercontent is reachable where api.github.com is not.
KNOWN_ABDEV_FILES = [
    "seq_H.fasta",
    "seq_L.fasta",
    "Sequence_Info.csv",
]


def _known_files() -> list[dict]:
    """Fallback file list built from known raw URLs (no API needed)."""
    return [{"name": n, "download_url": f"{ABDEV_RAW_BASE}/{n}", "path": n}
            for n in KNOWN_ABDEV_FILES]


def _list_repo_files(subdir: str = "") -> list[dict]:
    """List files in the AbDev repo (root or subdirectory)."""
    url = f"{ABDEV_API_URL}/{subdir}" if subdir else ABDEV_API_URL
    try:
        r = requests.get(url, timeout=15,
                         headers={"Accept": "application/vnd.github.v3+json"})
        r.raise_for_status()
        items = r.json()
        files = []
        for item in items:
            if item["type"] == "file":
                name = item["name"].lower()
                if name.endswith((".fasta", ".fa", ".csv", ".tsv", ".txt")):
                    files.append({
                        "name": item["name"],
                        "download_url": item.get("download_url", ""),
                        "path": item.get("path", ""),
                    })
            elif item["type"] == "dir":
                subname = item["name"].lower()
                if subname in ("data", "sequences", "antibodies", "ab", "fasta", "csv"):
                    files.extend(_list_repo_files(item["name"]))
        # if the API returned nothing usable, fall back to known files
        return files if files else (_known_files() if not subdir else [])
    except requests.RequestException:
        return _known_files() if not subdir else []


def _parse_fasta(text: str, max_seqs: int) -> list[str]:
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


def _parse_csv_sequences(text: str, max_seqs: int) -> list[str]:
    """Extract sequences from a CSV/TSV file."""
    # auto-detect delimiter
    delim = "\t" if "\t" in text[:500] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    rows = list(reader)
    if not rows:
        return []

    header = list(rows[0].keys())
    # find sequence column(s) — VH, VL, sequence, heavy, light, CDR, etc.
    seq_cols = [
        c for c in header
        if any(k in c.lower() for k in ("seq", "vh", "vl", "heavy", "light",
                                         "sequence", "peptide", "cdr", "variable"))
    ]
    if not seq_cols:
        # fallback: detect by content
        for c in header:
            vals = [rows[i].get(c, "") for i in range(min(5, len(rows)))]
            if all(AA_PATTERN.match(v.strip()) for v in vals if v.strip()):
                seq_cols.append(c)

    sequences = []
    for row in rows:
        for col in seq_cols:
            seq = row.get(col, "").strip().upper()
            if seq and AA_PATTERN.match(seq):
                sequences.append(seq)
                if len(sequences) >= max_seqs:
                    return sequences
    return sequences


def load_abdev_negatives(max_sequences: int = 300) -> list[dict]:
    """
    Download AbDev clinical antibody sequences as clean negatives.
    Returns list of dicts with variant_sequence + label="working" + source="abdev".
    """
    print("Fetching AbDev file list from GitHub...")
    files = _list_repo_files()

    if not files:
        print("  No sequence files found in AbDev repo. Skipping.")
        return []

    # prefer FASTA files, then CSV/TSV
    files.sort(key=lambda f: (
        0 if f["name"].lower().endswith((".fasta", ".fa")) else 1
    ))

    sequences: list[str] = []
    for fmeta in files:
        if len(sequences) >= max_sequences:
            break
        url = fmeta.get("download_url", "")
        if not url:
            continue
        name = fmeta["name"]
        print(f"  Downloading {name}...")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"    Failed: {e}")
            continue

        if name.lower().endswith((".fasta", ".fa")):
            seqs = _parse_fasta(r.text, max_sequences - len(sequences))
        elif name.lower().endswith((".csv", ".tsv", ".txt")):
            seqs = _parse_csv_sequences(r.text, max_sequences - len(sequences))
        else:
            continue

        sequences.extend(seqs)
        print(f"    {name}: {len(seqs)} sequences")

    if not sequences:
        print("  AbDev data unavailable or no parseable sequences found. Skipping.")
        return []

    # deduplicate
    seen: set[str] = set()
    result = []
    for seq in sequences:
        if seq not in seen:
            seen.add(seq)
            result.append({
                "variant_sequence": seq,
                "label": "working",
                "source": "abdev",
            })

    print(f"  Loaded {len(result)} AbDev negative sequences")
    return result
