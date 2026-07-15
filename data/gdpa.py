"""
Ginkgo GDPa antibody developability loader (GDPa1 / GDPa2).

Parses a LOCAL Ginkgo GDPa CSV into one row per (antibody, assay), matching the
row schema emitted by ``data/flab.py::load_all_developability_data``. VH/VL are
paired; each measured developability endpoint (AC-SINS, HIC, Tm, PR, SEC, ...)
becomes its own row with a continuous ``endpoint_value``.

Column names verified against the Ginkgo abdev-benchmark ground-truth schema
(see docs/research/gdpa-benchmark-notes.md for cited sources). The loader is
stdlib-only and NEVER downloads: place the file locally first.

GDPa3 is a frozen single-shot holdout (design doc §11). ``load_gdpa3`` is a
tripwire that always raises — no GDPa3 download or inspection during development.
"""

from __future__ import annotations

import csv
from pathlib import Path

LOCAL_GDPA_DIR = "data/external/gdpa"


def _norm(col: str) -> str:
    """Normalize a header for alias matching: lowercase, spaces/%→_, strip."""
    return col.strip().lower().replace("%", "").replace(" ", "_").replace("-", "_").strip("_")


# Minor header variants → canonical field. Keys are _norm()-ed.
_ID_ALIASES = {"antibody_name", "antibody_id", "antibody", "name", "id"}
_VH_ALIASES = {"vh_protein_sequence", "vh_sequence", "vh", "heavy", "heavy_chain", "hc_protein_sequence"}
_VL_ALIASES = {"vl_protein_sequence", "vl_sequence", "vl", "light", "light_chain", "lc_protein_sequence"}

# Known GDPa developability assays: normalized column → (family, metric, direction).
# metric keeps the human-facing name; direction follows data/flab.py convention
# ("higher_bad" = high value is worse, "lower_bad" = low value is worse).
_ASSAY_SPEC = {
    "ac_sins_ph7.4":     ("self_association",    "AC-SINS_pH7.4", "higher_bad"),
    "ac_sins_ph6.0":     ("self_association",    "AC-SINS_pH6.0", "higher_bad"),
    "ac_sins":           ("self_association",    "AC-SINS",       "higher_bad"),
    "hic":               ("hydrophobicity",      "HIC",           "higher_bad"),
    "hac":               ("heparin_binding",     "HAC",           "higher_bad"),
    "smac":              ("colloidal_stability", "SMAC",          "higher_bad"),
    "pr_cho":            ("polyreactivity",      "PR_CHO",        "higher_bad"),
    "sec_monomer":       ("aggregation",         "SEC_%Monomer",  "lower_bad"),
    "sec":               ("aggregation",         "SEC",           "lower_bad"),
    "titer":             ("expression",          "Titer",         "lower_bad"),
    "purity":            ("purity",              "Purity",        "lower_bad"),
    "tonset":            ("thermostability",     "Tonset",        "lower_bad"),
    "tm1":               ("thermostability",     "Tm1",           "lower_bad"),
    "tm2":               ("thermostability",     "Tm2",           "lower_bad"),
    "tm3":               ("thermostability",     "Tm3",           "lower_bad"),
}


def _pick(header_map: dict[str, str], aliases: set[str]) -> str | None:
    """Return the first real header whose normalized form is in ``aliases``."""
    for norm, raw in header_map.items():
        if norm in aliases:
            return raw
    return None


def _resolve_csv(dataset: str, local_dir: str | Path) -> Path:
    """Locate the local GDPa CSV for ``dataset`` or raise a placement error."""
    root = Path(local_dir)
    # Case-insensitive prefix match (e.g. GDPa1_v1.2_20250814.csv, gdpa2.csv).
    matches = sorted(p for p in root.glob("*.csv") if p.stem.lower().startswith(dataset.lower()))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No {dataset} CSV found. Place the Ginkgo {dataset.upper()} ground-truth CSV at "
        f"{root}/{dataset.upper()}*.csv (e.g. {root}/GDPa1_v1.2_20250814.csv). "
        f"Download it from https://github.com/ginkgobioworks/abdev-benchmark (data/) "
        f"or https://datapoints.ginkgo.bio/dataset-access — the loader never downloads."
    )


def load_gdpa(dataset: str, local_dir: str | Path = LOCAL_GDPA_DIR) -> list[dict]:
    """
    Load a local Ginkgo GDPa CSV (dataset in {"gdpa1", "gdpa2"}) into row dicts.

    Emits ONE ROW PER (antibody, assay): an antibody measured on N developability
    assays yields N rows. Each row carries: variant_sequence (==VH), vh_sequence,
    vl_sequence, assay_family, assay_metric, endpoint_direction, endpoint_value
    (float), source="gdpa", study_id (==dataset), group_id (==antibody id, so all
    of one antibody's assay rows stay in the same CV group).

    Rows with a non-numeric / blank assay cell are skipped for that assay only.
    Raises FileNotFoundError (with placement instructions) if the CSV is absent.
    """
    if dataset.lower() not in ("gdpa1", "gdpa2"):
        raise ValueError(f"load_gdpa expects 'gdpa1' or 'gdpa2', got {dataset!r}")

    csv_path = _resolve_csv(dataset, local_dir)
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        header_map = {_norm(col): col for col in header}

        id_col = _pick(header_map, _ID_ALIASES)
        vh_col = _pick(header_map, _VH_ALIASES)
        vl_col = _pick(header_map, _VL_ALIASES)
        # Assay columns present in this file, in header order.
        assay_cols = [(raw, _ASSAY_SPEC[norm]) for norm, raw in header_map.items() if norm in _ASSAY_SPEC]

        rows: list[dict] = []
        for i, record in enumerate(reader):
            vh = (record.get(vh_col, "") or "").strip().upper() if vh_col else ""
            vl = (record.get(vl_col, "") or "").strip().upper() if vl_col else ""
            ab_id = (record.get(id_col, "") or "").strip() if id_col else ""
            group = ab_id or f"{dataset}:{i}"  # fall back to row index if unnamed
            for raw, (family, metric, direction) in assay_cols:
                cell = (record.get(raw, "") or "").strip()
                if not cell:
                    continue
                try:
                    value = float(cell)
                except ValueError:
                    continue  # QC flags / non-numeric cells: skip that assay
                rows.append({
                    "variant_sequence": vh,          # ==VH per repo convention
                    "vh_sequence": vh or None,
                    "vl_sequence": vl or None,        # VHH (GDPa2) may have no VL
                    "assay_family": family,
                    "assay_metric": metric,
                    "endpoint_direction": direction,
                    "endpoint_value": value,
                    "source": "gdpa",
                    "study_id": dataset.lower(),
                    "campaign_id": dataset.lower(),
                    "molecule_id": group,
                    "molecule_group_id": group,
                    "group_id": group,
                    "assay_id": f"{dataset.lower()}:{metric}:{group}",
                    "endpoint_unit": "native",
                    "source_version": csv_path.name,
                })
    return rows


def load_gdpa3(*args, **kwargs):
    """Tripwire: GDPa3 is a frozen holdout and must never be touched in dev."""
    raise RuntimeError(
        "GDPa3 is a frozen single-shot holdout; download/inspection during "
        "development is prohibited (design doc section 11)."
    )
