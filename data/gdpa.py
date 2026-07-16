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
from zipfile import ZipFile
from xml.etree import ElementTree as ET

LOCAL_GDPA_DIR = "data/external/gdpa"


def _norm(col: str) -> str:
    """Normalize a header for alias matching: lowercase, spaces/%→_, strip."""
    return col.strip().lower().replace("%", "").replace(" ", "_").replace("-", "_").strip("_")


# Minor header variants → canonical field. Keys are _norm()-ed.
_ID_ALIASES = {"antibody_name", "antibody_id", "antibody", "name", "id"}
_XLSX_ID_ALIASES = ("antibody_id", "antibody_name", "antibody", "name", "id")
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
    "pr_ova":            ("polyreactivity",      "PR_Ova",        "higher_bad"),
    "sec_monomer":       ("aggregation",         "SEC_%Monomer",  "lower_bad"),
    "sec":               ("aggregation",         "SEC",           "lower_bad"),
    "titer":             ("expression",          "Titer",         "lower_bad"),
    "purity":            ("purity",              "Purity",        "lower_bad"),
    "tonset":            ("thermostability",     "Tonset",        "lower_bad"),
    "tm1":               ("thermostability",     "Tm1",           "lower_bad"),
    "tm2":               ("thermostability",     "Tm2",           "lower_bad"),
    "tm3":               ("thermostability",     "Tm3",           "lower_bad"),
}


def _xlsx_rows(path: str | Path):
    """Read simple tabular XLSX sheets without adding an Excel dependency."""
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.findall(".//m:t", ns))
                      for item in root.findall("m:si", ns)]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            target = relmap[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            member = target.lstrip("/")
            if not member.startswith("xl/"):
                member = f"xl/{member}"
            root = ET.fromstring(archive.read(member))
            table = []
            for row in root.findall(".//m:sheetData/m:row", ns):
                values: dict[int, str] = {}
                for cell in row.findall("m:c", ns):
                    ref = cell.attrib.get("r", "A1")
                    column = 0
                    for char in ref:
                        if not char.isalpha():
                            break
                        column = column * 26 + ord(char.upper()) - 64
                    column -= 1
                    value = cell.find("m:v", ns)
                    text = "" if value is None else value.text or ""
                    if cell.attrib.get("t") == "s" and text.isdigit():
                        text = shared[int(text)]
                    elif cell.attrib.get("t") == "inlineStr":
                        text = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                    values[column] = text
                if values:
                    table.append([values.get(i, "") for i in range(max(values) + 1)])
            yield sheet.attrib["name"], table


def _xlsx_assay_spec(header: str) -> tuple[str, str, str] | None:
    key = _norm(header).replace("_avg", "").replace("_mean", "").replace("_median", "")
    if key.endswith(("_stddev", "_std", "_count", "_raw")):
        return None
    if "ac_sins" in key or "acsins" in key:
        return "self_association", header, "higher_bad"
    if key.startswith("hic") or key.startswith("hihplc"):
        return "hydrophobicity", header, "higher_bad"
    if key.startswith("pr_") or key.startswith("polyreactivity"):
        return "polyreactivity", header, "higher_bad"
    if key.startswith("sec") or key.startswith("sehplc"):
        return "aggregation", header, "lower_bad"
    if key.startswith("thermostability_tm") or key.startswith("thermostability_tonset"):
        return "thermostability", header, "lower_bad"
    if key.startswith("production_concentration") or key.startswith("production_amount"):
        return "expression", header, "lower_bad"
    if key.startswith("smac") or key.startswith("hac"):
        return None
    if key.startswith("tm") or key.startswith("tonset"):
        return "thermostability", header, "lower_bad"
    return None


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
    matches = sorted(p for p in root.iterdir()
                     if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}
                     and p.stem.lower().startswith(dataset.lower()))
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
    if dataset.lower() not in ("gdpa1", "gdpa2", "gdpa4.1"):
        raise ValueError(f"load_gdpa expects 'gdpa1' or 'gdpa2', got {dataset!r}")

    csv_path = _resolve_csv(dataset, local_dir)
    if csv_path.suffix.lower() == ".xlsx":
        return _load_gdpa_xlsx(dataset, csv_path)
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
                    # One assay bucket contains all antibodies measured on the
                    # metric; molecule_group_id remains the disjoint CV group.
                    "assay_id": f"{dataset.lower()}:{metric}",
                    "endpoint_unit": "native",
                    "source_version": csv_path.name,
                })
    return rows


def _load_gdpa_xlsx(dataset: str, path: str | Path) -> list[dict]:
    sheets = dict(_xlsx_rows(path))
    assay_name = next((n for n in sheets if "average" in n.lower()), None)
    if assay_name is None:
        assay_name = next((n for n in sheets if "tidy" in n.lower() or "median" in n.lower()), None)
    sequence_names = [n for n in sheets if "sequence" in n.lower()]
    if not sequence_names or not assay_name:
        raise ValueError(f"{path} lacks sequence or assay sheet")

    assay_rows = sheets[assay_name]
    assay_headers = [str(x).strip() for x in assay_rows[0]]
    assay_map = {_norm(h): i for i, h in enumerate(assay_headers) if h}
    id_index = next((assay_map[k] for k in _XLSX_ID_ALIASES if k in assay_map), None)
    if id_index is None:
        raise ValueError(f"{path} has no antibody identifier column")
    sequences: dict[str, tuple[str, str, str]] = {}
    for sequence_name in sequence_names:
        sequence_rows = sheets[sequence_name]
        seq_headers = [str(x).strip() for x in sequence_rows[0]]
        seq_map = {_norm(h): i for i, h in enumerate(seq_headers) if h}
        seq_id_index = next((seq_map[k] for k in _XLSX_ID_ALIASES if k in seq_map), None)
        vh_index = next((seq_map[k] for k in _VH_ALIASES if k in seq_map), None)
        vl_index = next((seq_map[k] for k in _VL_ALIASES if k in seq_map), None)
        seq_index = next((seq_map[k] for k in ("protein_sequence", "sequence") if k in seq_map), None)
        if seq_index is None and vh_index is None:
            seq_index = next((i for k, i in seq_map.items() if "sequence" in k and "dna" not in k), None)
        if seq_id_index is None or (seq_index is None and vh_index is None):
            continue
        for row in sequence_rows[1:]:
            if len(row) <= seq_id_index or not row[seq_id_index]:
                continue
            vh = str(row[vh_index]).strip().upper() if vh_index is not None and len(row) > vh_index else ""
            vl = str(row[vl_index]).strip().upper() if vl_index is not None and len(row) > vl_index else ""
            if not vh and seq_index is not None and len(row) > seq_index:
                vh = str(row[seq_index]).strip().upper()
            if vh:
                sequences[str(row[seq_id_index]).strip()] = (vh, vl, sequence_name)
    rows: list[dict] = []
    for record in assay_rows[1:]:
        if len(record) <= id_index:
            continue
        ab_id = str(record[id_index]).strip()
        sequence_info = sequences.get(ab_id)
        vh, vl, construct_type = sequence_info if sequence_info else ("", "", "")
        if not vh:
            continue
        for header, index in zip(assay_headers, range(len(assay_headers))):
            spec = _xlsx_assay_spec(header)
            if spec is None or index >= len(record):
                continue
            try:
                value = float(record[index])
            except (TypeError, ValueError):
                continue
            family, metric, direction = spec
            rows.append({
                "variant_sequence": vh,
                "vh_sequence": vh,
                "vl_sequence": vl or None,
                "assay_family": family,
                "assay_metric": metric,
                "endpoint_direction": direction,
                "endpoint_value": value,
                "source": "gdpa",
                "study_id": dataset.lower(),
                "campaign_id": dataset.lower(),
                "molecule_id": ab_id,
                "molecule_group_id": ab_id,
                "group_id": ab_id,
                "assay_id": f"{dataset.lower()}:{metric}",
                "endpoint_unit": "native",
                "source_version": Path(path).name,
                "construct_type": construct_type,
            })
    return rows


def load_gdpa3(*args, **kwargs):
    """Tripwire: GDPa3 is a frozen holdout and must never be touched in dev."""
    raise RuntimeError(
        "GDPa3 is a frozen single-shot holdout; download/inspection during "
        "development is prohibited (design doc section 11)."
    )
