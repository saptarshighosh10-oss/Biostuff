"""
OAS (Observed Antibody Space) natural-repertoire loader.

Reads LOCAL OAS "data-unit" files (gzipped AIRR-style CSVs) and emits row dicts
tagged as **background / OOD reference** — NOT supervised. OAS carries no
developability assay labels; it is natural antibody repertoire background used
for out-of-distribution detection and as a negative reference. Supervision role
matches data/contract.py::SOURCE_ROLES for other background sources
(``supervision_status="background_ood"``).

OAS file quirk (verified, see docs/research/oas-notes.md): each ``.csv.gz`` has
the data-unit METADATA as its FIRST line (a JSON object), and the real AIRR
column header on the SECOND line. The official read idiom is
``pd.read_csv(f, header=1)`` for rows and
``json.loads(','.join(pd.read_csv(f, nrows=0).columns))`` for metadata. We do the
stdlib equivalent: consume line 1 as metadata, then ``csv.DictReader`` reads
line 2 as the header.

Confirmed AIRR columns:
  unpaired: sequence_alignment_aa, v_call, j_call, locus  (IGH heavy / IGK,IGL light)
  paired:   sequence_alignment_aa_heavy / _light, v_call_heavy / _light, ...

stdlib-only, NEVER downloads: place OAS unit files locally first.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

LOCAL_OAS_DIR = "data/external/oas"

# AIRR sequence_alignment_aa can carry IMGT gap characters; strip them so we keep
# the real amino-acid sequence.
_GAP = str.maketrans("", "", ".-*")


def _norm(col: str) -> str:
    """Normalize a header for alias matching: lowercase, strip, spaces→_."""
    return col.strip().lower().replace(" ", "_").strip("_")


# Normalized column → canonical field. Covers paired (_heavy/_light) and unpaired
# unit schemas. Order within each set does not matter; first present wins.
_VH_ALIASES = {"sequence_alignment_aa_heavy", "vh_sequence", "vh_aa", "heavy_sequence"}
_VL_ALIASES = {"sequence_alignment_aa_light", "vl_sequence", "vl_aa", "light_sequence"}
_SINGLE_ALIASES = {"sequence_alignment_aa"}  # unpaired: one chain per file
_LOCUS_ALIASES = {"locus", "chain", "locus_heavy"}
_VCALL_ALIASES = {"v_call", "v_call_heavy", "v_gene"}
_JCALL_ALIASES = {"j_call", "j_call_heavy", "j_gene"}


def _pick(header_map: dict[str, str], aliases: set[str]) -> str | None:
    """Return the first real header whose normalized form is in ``aliases``."""
    for norm, raw in header_map.items():
        if norm in aliases:
            return raw
    return None


def _clean_aa(value: str | None) -> str:
    """Uppercase an amino-acid string with IMGT gaps/whitespace removed."""
    if not value:
        return ""
    return value.strip().upper().translate(_GAP)


def _open(path: Path):
    """Open an OAS unit file transparently (.csv.gz via gzip, else plain)."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def _read_metadata_line(fh) -> dict:
    """Consume the OAS metadata (first) line and return it parsed, best-effort.

    The line is a JSON object. Some mirrors CSV-quote it (``"{""k"": ...}"``), so
    fall back to un-CSV-ing before giving up. Metadata is optional context, never
    fatal: on any parse failure return {} and let study_id fall back to filename.
    """
    first = fh.readline()
    if not first:
        return {}
    first = first.strip()
    try:
        return json.loads(first)
    except (ValueError, TypeError):
        pass
    try:
        unquoted = next(csv.reader([first]))[0]  # strip CSV quoting, unescape ""
        return json.loads(unquoted)
    except (ValueError, TypeError, StopIteration, IndexError):
        return {}


def _study_id(metadata: dict, path: Path) -> str:
    """Best study/group id: metadata Author/Study, else the filename stem."""
    for key in ("Author", "Study", "study", "Run", "run"):
        val = metadata.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    stem = path.name
    for ext in (".csv.gz", ".gz", ".csv"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem


def load_oas(local_dir: str | Path = LOCAL_OAS_DIR, max_sequences: int = 2000) -> list[dict]:
    """
    Load local OAS unit files into background/OOD reference row dicts.

    Reads every ``*.csv.gz`` / ``*.csv`` under ``local_dir`` (sorted, so output is
    deterministic). Each file's first line is the data-unit metadata; the second
    line is the AIRR header. Paired and unpaired unit schemas are both tolerated
    via an alias map. Output is capped at ``max_sequences`` rows total.

    Each row carries:
      variant_sequence     heavy-chain aa (falls back to light when a unit is
                           light-only, so no sequence is dropped)
      vh_sequence          heavy aa or None
      vl_sequence          light aa or None (None for unpaired heavy units)
      source               "oas"
      supervision_status   "background_ood"  (NOT a supervised label)
      study_id             metadata Author/Run, else filename stem
      group_id             == study_id (keeps a unit together under grouped CV)
      v_call / j_call      germline gene calls when present

    Rows with no usable amino-acid sequence are skipped. Raises FileNotFoundError
    (with placement instructions) when ``local_dir`` holds no OAS unit files — the
    loader NEVER downloads.
    """
    root = Path(local_dir)
    files = sorted(p for p in root.glob("*.csv.gz")) + sorted(p for p in root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No OAS unit files found in {root}. Place OAS data-unit files "
            f"(*.csv.gz or *.csv) there first — the loader never downloads. "
            f"Download units from the Observed Antibody Space search/bulk pages: "
            f"https://opig.stats.ox.ac.uk/webapps/oas/ (each unit's first line is "
            f"JSON metadata, second line is the AIRR header)."
        )

    rows: list[dict] = []
    for path in files:
        if len(rows) >= max_sequences:
            break
        with _open(path) as fh:
            metadata = _read_metadata_line(fh)
            study = _study_id(metadata, path)
            reader = csv.DictReader(fh)  # header = second physical line
            header_map = {_norm(c): c for c in (reader.fieldnames or [])}

            vh_col = _pick(header_map, _VH_ALIASES)
            vl_col = _pick(header_map, _VL_ALIASES)
            single_col = _pick(header_map, _SINGLE_ALIASES)
            locus_col = _pick(header_map, _LOCUS_ALIASES)
            v_col = _pick(header_map, _VCALL_ALIASES)
            j_col = _pick(header_map, _JCALL_ALIASES)
            paired = vh_col is not None or vl_col is not None

            for record in reader:
                if len(rows) >= max_sequences:
                    break
                if paired:
                    vh = _clean_aa(record.get(vh_col)) if vh_col else ""
                    vl = _clean_aa(record.get(vl_col)) if vl_col else ""
                else:
                    # Unpaired: one chain per file. locus IGH → heavy, else light.
                    seq = _clean_aa(record.get(single_col)) if single_col else ""
                    locus = (record.get(locus_col, "") or "").strip().upper() if locus_col else ""
                    is_light = locus in ("IGK", "IGL", "K", "L", "LIGHT")
                    vh, vl = ("", seq) if is_light else (seq, "")

                variant = vh or vl  # prefer heavy; never emit an empty variant
                if not variant:
                    continue
                row = {
                    "variant_sequence": variant,
                    "vh_sequence": vh or None,
                    "vl_sequence": vl or None,
                    "source": "oas",
                    "supervision_status": "background_ood",
                    "study_id": study,
                    "group_id": study,
                }
                if v_col:
                    v = (record.get(v_col, "") or "").strip()
                    if v:
                        row["v_call"] = v
                if j_col:
                    j = (record.get(j_col, "") or "").strip()
                    if j:
                        row["j_call"] = j
                rows.append(row)
    return rows
