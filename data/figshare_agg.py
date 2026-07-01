"""
Figshare A3D database — Aggrescan3D scores for 144,612 PDB structures.
DOI: 10.6084/m9.figshare.22492606.v1

We query the Figshare API to find downloadable files, then fetch a
manageable CSV subset (not the full ~500 MB dataset).  If a per-chain
aggregation summary CSV is available we use that directly; otherwise
we fall back to the first parseable CSV in the archive.

Label convention:
  avg_a3d_score > threshold  →  confirmed_failure  (aggregation-prone)
  avg_a3d_score ≤ threshold  →  working
Typical Aggrescan3D safe threshold: 0.0 (negative = stable, positive = prone)
"""

import csv
import io
import re
import zipfile
import requests

FIGSHARE_FILES_URL = "https://api.figshare.com/v2/articles/22492606/files"
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{4,}$", re.IGNORECASE)

# A3D score above this → aggregation-prone
A3D_FAILURE_THRESHOLD = 0.0


def _list_figshare_files() -> list[dict]:
    """Return file metadata list from the Figshare article."""
    try:
        r = requests.get(FIGSHARE_FILES_URL, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return []


def _detect_columns(header: list[str]) -> tuple[str | None, str | None]:
    """Return (seq_col, score_col) from a CSV header."""
    seq_col = next(
        (c for c in header if any(k in c.lower() for k in ("seq", "peptide", "sequence", "prot"))),
        None,
    )
    score_col = next(
        (c for c in header if any(k in c.lower() for k in
         ("a3d", "aggrescan", "score", "avg_score", "mean_score", "aggreg"))),
        None,
    )
    return seq_col, score_col


def _parse_a3d_csv(text: str, max_sequences: int) -> tuple[list[dict], list[dict]]:
    """
    Parse an Aggrescan3D CSV.
    Returns (failures, working) dicts with variant_sequence + label + source.
    """
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []

    header = list(rows[0].keys())
    seq_col, score_col = _detect_columns(header)

    # If no sequence column, try to find one by content inspection
    if not seq_col:
        for c in header:
            vals = [rows[i].get(c, "") for i in range(min(5, len(rows)))]
            if all(AA_PATTERN.match(v.strip()) for v in vals if v.strip()):
                seq_col = c
                break

    # If no score column, use last numeric-looking column
    if not score_col:
        for c in reversed(header):
            sample = [rows[i].get(c, "") for i in range(min(5, len(rows)))]
            try:
                [float(v) for v in sample if v.strip()]
                score_col = c
                break
            except ValueError:
                continue

    if not score_col:
        return [], []

    failures, working = [], []
    for row in rows:
        raw_score = row.get(score_col, "").strip()
        try:
            score = float(raw_score)
        except ValueError:
            continue

        # If we have a sequence column, use it; otherwise use PDB ID as placeholder
        if seq_col:
            seq = row.get(seq_col, "").strip().upper()
            if not seq or not AA_PATTERN.match(seq):
                continue
        else:
            # Use PDB ID + chain as a synthetic key; skip — we need sequences
            continue

        label = "confirmed_failure" if score > A3D_FAILURE_THRESHOLD else "working"
        entry = {
            "variant_sequence": seq,
            "label": label,
            "source": "figshare_a3d",
            "a3d_score": score,
        }

        if label == "confirmed_failure":
            if len(failures) < max_sequences // 2:
                failures.append(entry)
        else:
            if len(working) < max_sequences // 2:
                working.append(entry)

        if len(failures) + len(working) >= max_sequences:
            break

    return failures, working


def _try_zip_csv(content: bytes, max_sequences: int) -> tuple[list[dict], list[dict]]:
    """Extract the first useful CSV from a zip archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            # prefer summary/chain files over per-residue detail files
            csv_names.sort(key=lambda n: (
                0 if any(k in n.lower() for k in ("summary", "chain", "protein")) else 1
            ))
            for name in csv_names[:5]:
                text = zf.read(name).decode("utf-8", errors="replace")
                failures, working = _parse_a3d_csv(text, max_sequences)
                if failures or working:
                    print(f"    Parsed {name}: {len(failures)} failures, {len(working)} working")
                    return failures, working
    except (zipfile.BadZipFile, Exception):
        pass
    return [], []


def load_figshare_agg_data(
    max_sequences: int = 2000,
) -> tuple[list[dict], list[dict]]:
    """
    Load Aggrescan3D data from Figshare article 22492606.
    Returns (failures, working).

    Note: the full dataset is ~500 MB; we download only the smallest
    useful file (typically a per-chain summary CSV, ~5–20 MB).
    """
    print("Fetching Figshare A3D file list...")
    files = _list_figshare_files()

    if not files:
        print("  Could not reach Figshare API. Skipping A3D data.")
        return [], []

    # Sort: prefer CSV over zip, prefer smaller files
    files.sort(key=lambda f: (
        0 if f.get("name", "").endswith(".csv") else 1,
        f.get("size", 10**9),
    ))

    all_failures, all_working = [], []

    for fmeta in files:
        name = fmeta.get("name", "")
        size = fmeta.get("size", 0)
        url  = fmeta.get("download_url", "")

        if not url:
            continue

        # Skip very large files (>50 MB) to stay Colab-friendly
        if size > 50_000_000:
            print(f"  Skipping {name} ({size // 1_000_000} MB — too large)")
            continue

        print(f"  Downloading {name} ({size // 1000} kB)...")
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"    Failed: {e}")
            continue

        if name.endswith(".csv"):
            failures, working = _parse_a3d_csv(r.text, max_sequences)
        elif name.endswith(".zip"):
            failures, working = _try_zip_csv(r.content, max_sequences)
        else:
            continue

        print(f"    {name}: {len(failures)} failures, {len(working)} working")
        all_failures.extend(failures)
        all_working.extend(working)

        if len(all_failures) + len(all_working) >= max_sequences:
            break

    if not all_failures and not all_working:
        print("  Figshare A3D data unavailable or contained no parseable sequences. Skipping.")
        return [], []

    n = min(len(all_failures), len(all_working), max_sequences // 2)
    print(f"  Figshare A3D: {len(all_failures)} failures, {len(all_working)} working loaded")
    return all_failures[:n], all_working[:n]
