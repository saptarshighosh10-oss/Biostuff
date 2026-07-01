"""
CANYA — nucleation kinetics dataset from orozco-lab.
111,000+ peptide sequences: 21,936 nucleators + 88,470 confirmed non-nucleators.
One of the only datasets with explicit balanced negative results.

GitHub: https://github.com/orozco-lab/CANYA
Data:   Zenodo (linked from repo) or directly in repo data/ directory.
"""

import csv
import io
import re
import requests

GITHUB_API_URL  = "https://api.github.com/repos/orozco-lab/CANYA/contents"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/orozco-lab/CANYA/main"

AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{4,}$", re.IGNORECASE)

# Zenodo fallback — search by concept DOI if GitHub data not found
ZENODO_SEARCH_URL = "https://zenodo.org/api/records?q=CANYA+nucleation+aggregation&sort=mostrecent&size=3"


def _list_repo_csvs(subdir: str = "") -> list[dict]:
    """List CSV files in the CANYA repo (root or subdirectory)."""
    url = f"{GITHUB_API_URL}/{subdir}" if subdir else GITHUB_API_URL
    try:
        r = requests.get(url, timeout=15,
                         headers={"Accept": "application/vnd.github.v3+json"})
        r.raise_for_status()
        items = r.json()
        # recurse into data/ subdirectory if present
        files = []
        for item in items:
            if item["type"] == "file" and item["name"].endswith(".csv"):
                files.append({"name": item["name"], "download_url": item["download_url"]})
            elif item["type"] == "dir" and item["name"].lower() in ("data", "datasets", "sequences"):
                files.extend(_list_repo_csvs(item["name"]))
        return files
    except requests.RequestException:
        return []


def _try_zenodo(max_sequences: int) -> tuple[list[dict], list[dict]]:
    """Try to find CANYA data on Zenodo if GitHub data is unavailable."""
    try:
        r = requests.get(ZENODO_SEARCH_URL, timeout=15)
        r.raise_for_status()
        records = r.json().get("hits", {}).get("hits", [])
        for record in records:
            for f in record.get("files", []):
                if f.get("key", "").endswith(".csv"):
                    url = f.get("links", {}).get("self", "")
                    if url:
                        r2 = requests.get(url, timeout=60)
                        r2.raise_for_status()
                        return _parse_canya_csv(r2.text, max_sequences)
    except requests.RequestException:
        pass
    return [], []


def _parse_canya_csv(
    text: str, max_sequences: int
) -> tuple[list[dict], list[dict]]:
    """
    Parse a CANYA CSV into (nucleators, non_nucleators).
    Auto-detects sequence column and label/score column.
    Label conventions vary: 'nucleator'/'non-nucleator', 1/0, True/False, score threshold.
    """
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []

    header = list(rows[0].keys())

    # find sequence column
    seq_col = next(
        (c for c in header if any(k in c.lower() for k in ("seq", "peptide", "sequence"))),
        None,
    )
    if not seq_col:
        for c in header:
            vals = [rows[i].get(c, "") for i in range(min(5, len(rows)))]
            if all(AA_PATTERN.match(v.strip()) for v in vals if v.strip()):
                seq_col = c
                break
    if not seq_col:
        return [], []

    # find label/score column
    label_col = next(
        (c for c in header if any(k in c.lower() for k in
         ("label", "nucleat", "class", "aggreg", "amyloid", "positive", "score"))),
        None,
    )
    if not label_col and len(header) >= 2:
        # guess: last column
        label_col = header[-1]

    nucleators, non_nucleators = [], []
    for row in rows:
        seq = row.get(seq_col, "").strip().upper()
        if not seq or not AA_PATTERN.match(seq):
            continue

        raw = row.get(label_col, "").strip().lower() if label_col else ""
        # interpret label
        if raw in ("1", "true", "yes", "positive", "nucleator", "aggregator"):
            label = "confirmed_failure"
        elif raw in ("0", "false", "no", "negative", "non-nucleator", "non_nucleator", "inert"):
            label = "working"
        else:
            try:
                val = float(raw)
                label = "confirmed_failure" if val >= 0.5 else "working"
            except ValueError:
                continue

        entry = {"variant_sequence": seq, "label": label, "source": "canya"}
        if label == "confirmed_failure":
            if len(nucleators) < max_sequences // 2:
                nucleators.append(entry)
        else:
            if len(non_nucleators) < max_sequences // 2:
                non_nucleators.append(entry)

        if len(nucleators) + len(non_nucleators) >= max_sequences:
            break

    return nucleators, non_nucleators


def load_canya_data(max_sequences: int = 2000) -> tuple[list[dict], list[dict]]:
    """
    Load CANYA nucleation data.
    Returns (nucleators→failures, non_nucleators→working).
    """
    print("Fetching CANYA data from GitHub...")
    csv_files = _list_repo_csvs()

    all_failures, all_working = [], []

    if csv_files:
        for f in csv_files:
            url = f.get("download_url", "")
            if not url:
                continue
            print(f"  Downloading {f['name']}...")
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                failures, working = _parse_canya_csv(r.text, max_sequences)
                all_failures.extend(failures)
                all_working.extend(working)
                print(f"    {f['name']}: {len(failures)} nucleators, {len(working)} non-nucleators")
                if len(all_failures) + len(all_working) >= max_sequences:
                    break
            except requests.RequestException as e:
                print(f"    Failed: {e}")
    else:
        print("  GitHub data not found — trying Zenodo...")
        all_failures, all_working = _try_zenodo(max_sequences)

    if not all_failures and not all_working:
        print("  CANYA data unavailable. Skipping.")

    # cap to max_sequences (balanced)
    n = min(len(all_failures), len(all_working), max_sequences // 2)
    print(f"  CANYA: {len(all_failures)} nucleators, {len(all_working)} non-nucleators loaded")
    return all_failures[:n * 2], all_working[:n * 2]
