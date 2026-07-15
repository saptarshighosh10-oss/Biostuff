"""Local public aggregation snapshots with explicit provenance.

This adapter intentionally loads only sequence-linked, measured aggregation
endpoints.  It does not turn descriptors, predicted columns, or sequence-only
repertoires into supervised Head B rows.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from data.contract import sha256_hex


_SEQ_RE = re.compile(r"[^A-Za-z]")
_AGGREGATION_TOKENS = ("hic", "acsins", "smac", "csi", "agg", "sas", "sec", "sgac")


def _sequence(value: object) -> str:
    return _SEQ_RE.sub("", str(value or "")).upper()


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _number(value: object) -> float | None:
    try:
        result = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _direction(metric: str) -> str:
    lower = metric.lower()
    if "sec" in lower or "monomer" in lower or "purity" in lower or "main" in lower:
        return "lower_bad"
    return "higher_bad"


def _flab_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("*.csv")):
        metric = path.stem.rsplit("_", 1)[-1]
        if not any(token in metric.lower() for token in _AGGREGATION_TOKENS):
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for index, raw in enumerate(csv.DictReader(handle), start=2):
                vh = _sequence(raw.get("heavy"))
                vl = _sequence(raw.get("light"))
                if not vh:
                    continue
                value = _number(raw.get("fitness"))
                if value is None:
                    for key, candidate in raw.items():
                        if key.lower() not in {"heavy", "light", "fitness"}:
                            value = _number(candidate)
                            if value is not None:
                                break
                if value is None:
                    continue
                pair_key = f"{vh}|{vl}" if vl else vh
                rows.append({
                    "variant_sequence": vh + (vl or ""),
                    "vh_sequence": vh,
                    "vl_sequence": vl or None,
                    "assay_family": "aggregation",
                    "assay_metric": metric,
                    "endpoint_direction": _direction(metric),
                    "endpoint_value": value,
                    "endpoint_unit": "native",
                    "source": "flab",
                    "source_version": f"public_snapshot/{path.name}",
                    "study_id": path.stem.rsplit("_", 1)[0],
                    "assay_id": path.name,
                    "molecule_id": f"flab:{sha256_hex(pair_key.upper())[:24]}",
                    "molecule_group_id": f"flab_pair:{sha256_hex(pair_key.upper())[:24]}",
                    "source_row": index,
                })
    return rows


def _dotad_rows(root: Path) -> list[dict]:
    from data.gdpa import _xlsx_rows

    def tables(path: Path):
        for sheet_name, table in _xlsx_rows(path):
            if not table:
                continue
            header_index = 0
            if path.name == "experimental_data.xlsx":
                header_index = next(
                    (i for i, row in enumerate(table)
                     if any(str(value or "").strip().lower() in {"name", "full name", "inn"} for value in row)),
                    0,
                )
            headers = [str(value or "").strip() for value in table[header_index]]
            yield sheet_name, [dict(zip(headers, values)) for values in table[header_index + 1:]]

    sequence_rows = next((rows for name, rows in tables(root / "sequence.xlsx") if name == "seq1"), [])
    by_inn = {}
    for row in sequence_rows:
        key = _slug(row.get("INN"))
        vh, vl = _sequence(row.get("VH")), _sequence(row.get("VL"))
        if key and vh:
            by_inn[key] = (row.get("ID") or key, vh, vl or None)

    output: list[dict] = []
    workbook = root / "experimental_data.xlsx"
    for study, experiment_rows in tables(workbook):
        if _slug(study) == "link":
            continue
        for row_index, row in enumerate(experiment_rows, start=2):
            name = row.get("Name") or row.get("Full Name") or row.get("INN")
            joined = by_inn.get(_slug(name))
            if not joined:
                continue
            molecule_id, vh, vl = joined
            for header, raw_value in row.items():
                header_text = str(header or "").strip()
                header_lower = header_text.lower()
                if not header_text or "pred" in header_lower or "negated" in header_lower:
                    continue
                if not any(token in header_lower for token in _AGGREGATION_TOKENS):
                    continue
                value = _number(raw_value)
                if value is None:
                    continue
                metric = f"DOTAD:{study}:{header_text}"
                study_slug = _slug(study)
                output.append({
                    "variant_sequence": vh + (vl or ""),
                    "vh_sequence": vh,
                    "vl_sequence": vl,
                    "assay_family": "aggregation",
                    "assay_metric": metric,
                    "endpoint_direction": _direction(header_text),
                    "endpoint_value": value,
                    "endpoint_unit": "native",
                    "source": "dotad",
                    "source_version": "DOTAD_local_snapshot",
                    "study_id": f"dotad_{study_slug}",
                    "assay_id": f"dotad:{study_slug}:{_slug(header_text)}",
                    "molecule_id": f"dotad:{molecule_id}",
                    "molecule_group_id": f"dotad:{molecule_id}",
                    "source_row": row_index,
                })
    return output


def load_public_aggregation_rows(public_dir: str = "data/external/public_sources") -> list[dict]:
    """Load staged public aggregation snapshots; missing snapshots are non-fatal."""
    root = Path(public_dir)
    rows: list[dict] = []
    flab_root = root / "flab_aggregation"
    dotad_root = root / "dotad"
    if flab_root.is_dir():
        rows.extend(_flab_rows(flab_root))
    if (dotad_root / "sequence.xlsx").is_file() and (dotad_root / "experimental_data.xlsx").is_file():
        rows.extend(_dotad_rows(dotad_root))
    return rows


def dedupe_aggregation_rows(rows: list[dict]) -> list[dict]:
    """Drop exact repeated observations while retaining independent assays."""
    seen: set[tuple] = set()
    kept: list[dict] = []
    for row in rows:
        sequence = _sequence(row.get("variant_sequence"))
        if not sequence:
            sequence = _sequence(row.get("vh_sequence")) + _sequence(row.get("vl_sequence"))
        metric = _slug(row.get("assay_metric"))
        direction = str(row.get("endpoint_direction") or "")
        value = _number(row.get("endpoint_value"))
        key = (sequence, metric, direction, None if value is None else round(value, 8))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept
