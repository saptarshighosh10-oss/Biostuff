"""
FLAb (Fitness Landscapes for Antibodies) data loader.
Downloads aggregation CSVs from Graylab/FLAb on GitHub and converts them
to binary-labeled training examples without any wet-lab work.

FLAb contains 31 experimental datasets (HIC, SEC, PSR, etc.) from published
developability studies. We threshold score distributions to get:
  top quartile    → "confirmed_failure"
  bottom quartile → "working"

No API key needed — uses GitHub raw content and the GitHub contents API.
"""

import csv
import io
import re
import requests

GITHUB_API_URL  = "https://api.github.com/repos/Graylab/FLAb/contents/data/aggregation"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Graylab/FLAb/main/data/aggregation"

AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{20,}$", re.IGNORECASE)

# Known aggregation CSVs in Graylab/FLAb — used as a fallback when the GitHub
# contents API is unavailable (rate-limited or proxy-blocked). raw.githubusercontent
# is far more reliable than api.github.com, so we probe these directly first.
KNOWN_FLAB_FILES = [
    "jain2017biophyscial_HICRT.csv",
    "jain2017biophysical_ACSINS.csv",
    "jain2017biophysical_CSIBLI.csv",
    "jain2017biophysical_SAS.csv",
    "jain2017biophysical_SGACSINS.csv",
    "jain2024assessment_ACSINS.csv",
    "jain2024assessment_CIC.csv",
    "jain2024assessment_CSSINS.csv",
    "jain2024assessment_Fab_pI.csv",
    "jain2024assessment_HIC.csv",
    "jain2024assessment_SEC.csv",
    "jain2024assessment_cIEF.csv",
    "jetha2019homology_RT.csv",
    "kraft2019herapin_relrt.csv",
    "shanehsazzadeh2023unlocking_ACSINS.csv",
    "shanehsazzadeh2023unlocking_CGE.csv",
    "shanehsazzadeh2023unlocking_HICRRT.csv",
    "shanehsazzadeh2023unlocking_NRCGE.csv",
    "shanehsazzadeh2023unlocking_SEC.csv",
]


# ── GitHub file discovery ────────────────────────────────────────────────────

def list_flab_datasets() -> list[str]:
    """
    Return list of CSV filenames in FLAb's aggregation directory.
    Tries the GitHub contents API first; on any failure, falls back to the
    known filename list (verified reachable via raw.githubusercontent.com).
    """
    try:
        r = requests.get(GITHUB_API_URL, timeout=20,
                         headers={"Accept": "application/vnd.github.v3+json"})
        r.raise_for_status()
        names = [f["name"] for f in r.json() if f["name"].endswith(".csv")]
        if names:
            return names
    except requests.RequestException as e:
        print(f"  GitHub API unavailable ({e}); using known FLAb file list")
    return list(KNOWN_FLAB_FILES)


def download_csv(filename: str) -> str | None:
    """Download a single FLAb CSV and return its text content."""
    url = f"{GITHUB_RAW_BASE}/{filename}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"  Failed to download {filename}: {e}")
        return None


# ── Column auto-detection ────────────────────────────────────────────────────

def _detect_sequence_col(header: list[str], rows: list[dict]) -> str | None:
    """Find the column that contains amino acid sequences."""
    candidates = []
    for col in header:
        col_lower = col.lower()
        # prefer columns with obvious names
        if any(k in col_lower for k in ("vh", "vl", "heavy", "light", "sequence", "seq", "cdr")):
            candidates.insert(0, col)
        else:
            candidates.append(col)

    for col in candidates:
        sample = [row.get(col, "") for row in rows[:10] if row.get(col)]
        if sample and all(AA_PATTERN.match(v.strip()) for v in sample[:5]):
            return col
    return None


def _detect_score_col(header: list[str], rows: list[dict], seq_col: str) -> str | None:
    """
    Find the numeric score column.
    Prefers columns named after known assays; excludes the sequence column.
    """
    assay_keywords = ["hic", "sec", "psr", "ac", "score", "agg", "fitness",
                      "monomer", "retention", "percent", "solubility"]
    candidates = [c for c in header if c != seq_col]

    # score by keyword match
    ranked = []
    for col in candidates:
        col_lower = col.lower()
        score = sum(kw in col_lower for kw in assay_keywords)
        ranked.append((score, col))
    ranked.sort(reverse=True)

    for _, col in ranked:
        vals = []
        for row in rows[:20]:
            try:
                vals.append(float(row[col]))
            except (ValueError, TypeError, KeyError):
                pass
        if len(vals) >= 5:
            return col
    return None


# ── Labeling ─────────────────────────────────────────────────────────────────

def _scores_to_labels(
    sequences: list[str],
    scores: list[float],
    percentile: float = 0.25,
    score_col_name: str = "",
) -> tuple[list[dict], list[dict]]:
    """
    Threshold scores to produce binary labels.
    For HIC/aggregation scores: high = bad.
    For SEC monomer %: low = bad (we detect by column name).
    """
    if not scores:
        return [], []

    col_lower = score_col_name.lower()
    low_is_bad = any(k in col_lower for k in ("sec", "monomer", "solubil", "percent"))

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    low_threshold  = sorted_scores[int(n * percentile)]
    high_threshold = sorted_scores[int(n * (1 - percentile))]

    failures, working = [], []
    for seq, score in zip(sequences, scores):
        if not seq:
            continue
        if low_is_bad:
            label = "confirmed_failure" if score <= low_threshold else (
                    "working"           if score >= high_threshold else None)
        else:
            label = "confirmed_failure" if score >= high_threshold else (
                    "working"           if score <= low_threshold else None)

        if label is not None:
            entry = {"variant_sequence": seq.upper(), "label": label,
                     "score": score, "score_col": score_col_name, "source": "flab"}
            (failures if label == "confirmed_failure" else working).append(entry)

    return failures, working


# ── Public API ───────────────────────────────────────────────────────────────

def load_dataset(
    filename: str,
    percentile: float = 0.25,
    max_rows: int = 500,
) -> tuple[list[dict], list[dict]]:
    """
    Download and parse one FLAb CSV.
    Returns (failures, working) as lists of labeled dicts.
    """
    text = download_csv(filename)
    if not text:
        return [], []

    reader = csv.DictReader(io.StringIO(text))
    rows = [row for row in reader]
    if not rows:
        return [], []

    header = list(rows[0].keys())
    seq_col   = _detect_sequence_col(header, rows)
    score_col = _detect_score_col(header, rows, seq_col or "") if seq_col else None

    if not seq_col or not score_col:
        print(f"    {filename}: could not detect seq/score columns (headers: {header[:6]})")
        return [], []

    sequences, scores = [], []
    for row in rows[:max_rows]:
        seq = row.get(seq_col, "").strip().upper()
        try:
            score = float(row[score_col])
        except (ValueError, TypeError):
            continue
        if AA_PATTERN.match(seq):
            sequences.append(seq)
            scores.append(score)

    failures, working = _scores_to_labels(sequences, scores, percentile, score_col)
    # tag with the source dataset so grouped CV never splits variants from the
    # same study across train/test folds (they often share a parent antibody)
    for entry in failures + working:
        entry["group_id"] = filename
    print(f"    {filename}: {len(failures)} failures, {len(working)} working "
          f"(seq={seq_col}, score={score_col})")
    return failures, working


def load_all_flab_data(
    percentile: float = 0.25,
    max_per_dataset: int = 500,
) -> tuple[list[dict], list[dict]]:
    """
    Load all FLAb aggregation datasets.
    Returns (all_failures, all_working) combined across all datasets.
    """
    print("Fetching FLAb dataset list from GitHub...")
    filenames = list_flab_datasets()
    if not filenames:
        print("  No datasets found. Check network access.")
        return [], []

    print(f"Found {len(filenames)} CSV files. Downloading...")

    all_failures, all_working = [], []
    seen_sequences: set[str] = set()

    for fname in filenames:
        failures, working = load_dataset(fname, percentile, max_per_dataset)
        # deduplicate across datasets
        for entry in failures + working:
            seq = entry["variant_sequence"]
            if seq not in seen_sequences:
                seen_sequences.add(seq)
                if entry["label"] == "confirmed_failure":
                    all_failures.append(entry)
                else:
                    all_working.append(entry)

    print(f"\nFLAb total: {len(all_failures)} unique failures, "
          f"{len(all_working)} unique working sequences")
    return all_failures, all_working
