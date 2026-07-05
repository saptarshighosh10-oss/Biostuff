"""
ProteinGym — large-scale deep mutational scanning failures.

ProteinGym aggregates 200+ DMS assays (~2.7M mutant sequences). The
"Stability" and "Expression" assays measure whether a mutant protein still
folds / expresses — low fitness = destabilized / aggregation-prone = a
confirmed FAILURE. This is the largest accessible source of "proteins that
don't work", and unlike FLAb/AbDev these are MUTANTS of a known wild-type,
so they populate the mutation-delta features (charge/hydrophobicity shifts).

Flagship assay: A4_HUMAN_Seuma_2022 — a 14,811-variant deep mutational scan
of amyloid-beta aggregation/nucleation itself.

The assay INDEX lives on GitHub (raw.githubusercontent, always reachable).
The per-assay score files live on HuggingFace / Zenodo (reachable from Colab;
may be blocked on locked-down networks — the loader fails gracefully).

Label convention (fitness, higher = more functional/stable):
  bottom `percentile`  → confirmed_failure
  top    `percentile`  → working
"""

import csv
import io
import re
import requests

INDEX_URL = ("https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/"
             "main/reference_files/DMS_substitutions.csv")

# Per-assay data file locations, tried in order. HuggingFace first (most
# reliable in Colab), then the Harvard mirror.
DATA_BASES = [
    "https://huggingface.co/datasets/OATML-Markslab/ProteinGym/resolve/main/DMS_ProteinGym_substitutions",
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/DMS_ProteinGym_substitutions",
]

AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{10,}$", re.IGNORECASE)

# Curated high-value assays: amyloid-beta first, then compact folding-stability
# scans (Tsuboyama 2023 mega-scale). Kept short so Colab runs stay quick.
PRIORITY_ASSAYS = [
    "A4_HUMAN_Seuma_2022",              # amyloid-beta aggregation DMS (on-topic)
    "AMFR_HUMAN_Tsuboyama_2023_4G3O",
    "ARGR_ECOLI_Tsuboyama_2023_1AOY",
    "BBC1_YEAST_Tsuboyama_2023_1TG0",
    "BCHB_CHLTE_Tsuboyama_2023_2KRU",
]


def _get_text(url: str, timeout: int = 60) -> str | None:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        return None


def list_stability_assays(max_assays: int = 8) -> list[dict]:
    """
    Read the ProteinGym index and return metadata for Stability/Expression
    assays, prioritising the curated on-topic ones.
    """
    text = _get_text(INDEX_URL, timeout=30)
    if not text:
        print("  Could not fetch ProteinGym index.")
        return []

    rows = list(csv.DictReader(io.StringIO(text)))
    relevant = [r for r in rows
                if r.get("coarse_selection_type") in ("Stability", "Expression")]

    # sort: priority assays first, then by fewest mutants (faster downloads)
    def sort_key(r):
        did = r.get("DMS_id", "")
        pri = PRIORITY_ASSAYS.index(did) if did in PRIORITY_ASSAYS else 999
        try:
            n = int(r.get("DMS_total_number_mutants", "0"))
        except ValueError:
            n = 10**9
        return (pri, n)

    relevant.sort(key=sort_key)
    return relevant[:max_assays]


def _mutations_from_field(mutant_field: str) -> list:
    """
    Parse a ProteinGym mutant string like 'G12A:K45R' into
    [(pos0, orig, mut), ...] (0-indexed positions).
    """
    mutations = []
    for token in mutant_field.split(":"):
        token = token.strip()
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", token)
        if m:
            orig, pos, mut = m.group(1), int(m.group(2)), m.group(3)
            mutations.append((pos - 1, orig, mut))  # to 0-indexed
    return mutations


def _download_assay(filename: str) -> str | None:
    for base in DATA_BASES:
        text = _get_text(f"{base}/{filename}")
        if text:
            return text
    return None


def _parse_assay(text: str, percentile: float, max_keep: int) -> tuple[list[dict], list[dict]]:
    """
    Parse one assay. Thresholds are computed over the FULL fitness distribution
    (not a truncated head), then `max_keep` most-extreme failures and workings
    are kept — so labels stay correct and the cap only limits redundancy.
    """
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []

    header = list(rows[0].keys())
    seq_col   = next((c for c in header if c.lower() in ("mutated_sequence", "sequence")), None)
    score_col = next((c for c in header if c.lower() in ("dms_score", "score")), None)
    mut_col   = next((c for c in header if c.lower() == "mutant"), None)
    if not seq_col or not score_col:
        return [], []

    entries = []
    for row in rows:  # read the whole file — thresholds need the true distribution
        seq = row.get(seq_col, "").strip().upper()
        if not seq or not AA_PATTERN.match(seq):
            continue
        try:
            score = float(row[score_col])
        except (ValueError, TypeError, KeyError):
            continue
        muts = _mutations_from_field(row.get(mut_col, "")) if mut_col else []
        entries.append((seq, score, muts))

    if len(entries) < 8:
        return [], []

    scores = sorted(e[1] for e in entries)
    n = len(scores)
    low_t  = scores[int(n * percentile)]
    high_t = scores[int(n * (1 - percentile))]

    fail_pool = [e for e in entries if e[1] <= low_t]
    work_pool = [e for e in entries if e[1] >= high_t]
    # keep the most extreme examples (clearest failures / clearest working)
    fail_pool.sort(key=lambda e: e[1])                # lowest fitness first
    work_pool.sort(key=lambda e: e[1], reverse=True)  # highest fitness first

    failures = [{"variant_sequence": s, "label": "confirmed_failure",
                 "mutations": m, "source": "proteingym", "dms_score": sc}
                for s, sc, m in fail_pool[:max_keep]]
    working  = [{"variant_sequence": s, "label": "working",
                 "mutations": m, "source": "proteingym", "dms_score": sc}
                for s, sc, m in work_pool[:max_keep]]
    return failures, working


def load_proteingym_data(
    percentile: float = 0.25,
    max_assays: int = 12,
    max_per_assay: int = 1200,
) -> tuple[list[dict], list[dict]]:
    """
    Load DMS stability/expression failures from ProteinGym.
    Returns (failures, working) — mutant sequences with populated mutation lists.
    """
    print("Fetching ProteinGym assay index...")
    assays = list_stability_assays(max_assays=max_assays)
    if not assays:
        print("  No ProteinGym assays available. Skipping.")
        return [], []

    print(f"Selected {len(assays)} stability/expression assays. Downloading...")
    all_failures, all_working = [], []
    seen: set[str] = set()

    for a in assays:
        fname = a.get("DMS_filename") or f"{a['DMS_id']}.csv"
        text = _download_assay(fname)
        if not text:
            print(f"  {a['DMS_id']}: data file unreachable (Colab/HF needed) — skipped")
            continue
        failures, working = _parse_assay(text, percentile, max_per_assay)
        # dedup across assays
        nf = nw = 0
        for e in failures:
            if e["variant_sequence"] not in seen:
                seen.add(e["variant_sequence"]); all_failures.append(e); nf += 1
        for e in working:
            if e["variant_sequence"] not in seen:
                seen.add(e["variant_sequence"]); all_working.append(e); nw += 1
        print(f"  {a['DMS_id']}: {nf} failures, {nw} working")

    print(f"\nProteinGym total: {len(all_failures)} failures, {len(all_working)} working")
    return all_failures, all_working
