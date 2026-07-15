"""
Thera-SAbDab — Therapeutic Structural Antibody Database (Oxford Protein
Informatics Group) loader.

Thera-SAbDab lists every WHO-recognised antibody-/nanobody-derived therapeutic
(INN-assigned) with released variable-domain sequences. Each is a molecule that
reached clinical development — a KNOWN-DEVELOPABLE, paired VH/VL antibody. We
load them as a background / out-of-distribution reference of "working" antibodies
(``supervision_status="background_ood"``), NOT as assay-supervised rows.

Column names verified against the Thera-SAbDab bulk download
(``TheraSAbDab_SeqStruc_OnlineDownload.csv``); see
docs/research/thera-sabdab-notes.md for cited sources. The loader is stdlib-only
and NEVER downloads: place the file locally first.
"""

from __future__ import annotations

import csv
from pathlib import Path

LOCAL_THERA_DIR = "data/external/thera_sabdab"


def _norm(col: str) -> str:
    """Normalize a header for alias matching: drop any ``(...)`` suffix (the
    Highest_Clin_Trial date tag moves each release), lowercase, punct→_ ."""
    col = col.split("(")[0]  # "Highest_Clin_Trial (Feb '25)" -> "Highest_Clin_Trial "
    out = col.strip().lower()
    for ch in " -.%'":
        out = out.replace(ch, "_")
    return out.strip("_")


# Minor header variants → canonical field. Keys are _norm()-ed.
_ID_ALIASES = {"therapeutic", "therapeutic_name", "inn", "name"}
_VH_ALIASES = {"heavy_sequence", "vh_sequence", "vh", "heavy", "heavy_chain"}
_VL_ALIASES = {"light_sequence", "vl_sequence", "vl", "light", "light_chain"}
_PHASE_ALIASES = {"highest_clin_trial", "highest_clinical_trial", "clinical_phase", "phase"}
_FORMAT_ALIASES = {"format", "molecular_format"}
_TARGET_ALIASES = {"target", "targets", "intended_target"}


def _pick(header_map: dict[str, str], aliases: set[str]) -> str | None:
    """Return the first real header whose normalized form is in ``aliases``."""
    for norm, raw in header_map.items():
        if norm in aliases:
            return raw
    return None


def _resolve_csv(local_dir: str | Path) -> Path:
    """Locate the local Thera-SAbDab download or raise a placement error."""
    root = Path(local_dir)
    matches = sorted(p for p in (*root.glob("*.csv"), *root.glob("*.tsv")))
    if matches:
        # Prefer the canonical online-download name if present.
        for p in matches:
            if "therasabdab" in p.name.lower():
                return p
        return matches[0]
    raise FileNotFoundError(
        f"No Thera-SAbDab file found. Place the Thera-SAbDab bulk download at "
        f"{root}/TheraSAbDab_SeqStruc_OnlineDownload.csv "
        f"(Downloads tab of https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/search/). "
        f"The loader never downloads."
    )


def _reader(fh):
    """csv.DictReader with delimiter sniffed as comma-vs-tab (default comma)."""
    first = fh.readline()
    fh.seek(0)
    delim = "\t" if first.count("\t") > first.count(",") else ","
    return csv.DictReader(fh, delimiter=delim)


def load_thera_sabdab(
    local_dir: str | Path = LOCAL_THERA_DIR,
    max_sequences: int = 1000,
) -> list[dict]:
    """
    Load a local Thera-SAbDab download into background-OOD reference row dicts.

    Emits ONE ROW PER therapeutic with a usable VH. Each row carries:
    variant_sequence (==VH), vh_sequence, vl_sequence (None for VHH / single-
    domain), source="thera_sabdab", supervision_status="background_ood",
    study_id="thera_sabdab", group_id (==therapeutic name), plus clinical_phase,
    format, target when those columns are present.

    Rows with a blank VH (Heavy Sequence) are skipped. At most ``max_sequences``
    rows are returned. Raises FileNotFoundError (with placement instructions) if
    no file is present — the loader never downloads.
    """
    csv_path = _resolve_csv(local_dir)
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = _reader(fh)
        header = reader.fieldnames or []
        # setdefault keeps the FIRST column for a normalized name, so the primary
        # "Heavy Sequence" wins over the empty "Heavy Sequence (if bispec)" (both
        # normalize to heavy_sequence).
        header_map: dict[str, str] = {}
        for col in header:
            header_map.setdefault(_norm(col), col)

        id_col = _pick(header_map, _ID_ALIASES)
        vh_col = _pick(header_map, _VH_ALIASES)
        vl_col = _pick(header_map, _VL_ALIASES)
        phase_col = _pick(header_map, _PHASE_ALIASES)
        format_col = _pick(header_map, _FORMAT_ALIASES)
        target_col = _pick(header_map, _TARGET_ALIASES)

        rows: list[dict] = []
        for i, record in enumerate(reader):
            vh = (record.get(vh_col, "") or "").strip().upper() if vh_col else ""
            if not vh:
                continue  # no usable heavy chain (sequence not released yet)
            vl = (record.get(vl_col, "") or "").strip().upper() if vl_col else ""
            name = (record.get(id_col, "") or "").strip() if id_col else ""
            row = {
                "variant_sequence": vh,          # ==VH per repo convention
                "vh_sequence": vh,
                "vl_sequence": vl or None,        # VHH / single-domain: no VL
                "source": "thera_sabdab",
                "supervision_status": "background_ood",
                "study_id": "thera_sabdab",
                "group_id": name or f"thera_sabdab:{i}",
            }
            if phase_col:
                phase = (record.get(phase_col, "") or "").strip()
                if phase:
                    row["clinical_phase"] = phase
            if format_col:
                fmt = (record.get(format_col, "") or "").strip()
                if fmt:
                    row["format"] = fmt
            if target_col:
                target = (record.get(target_col, "") or "").strip()
                if target:
                    row["target"] = target
            rows.append(row)
            if len(rows) >= max_sequences:
                break
    return rows
