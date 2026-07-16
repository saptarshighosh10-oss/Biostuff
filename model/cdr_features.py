"""Small, dependency-free CDR descriptors for honest feature ablations."""

from __future__ import annotations

import re

from model.head_b_acsins_charge import net_charge


_AA = set("ACDEFGHIKLMNPQRSTVWY")
_KD = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9,
       "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3,
       "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
       "K": -3.9, "R": -4.5}


def _seq(value: object) -> str:
    value = str(value or "").strip().upper()
    return value if value and set(value) <= _AA else ""


def _provided(row: dict, cdr: str) -> str:
    """Return a supplied heavy CDR before attempting any sequence heuristic."""
    number = cdr[-1]
    for key in (f"vh_cdr{number}", f"cdrh{number}", f"h_cdr{number}"):
        sequence = _seq(row.get(key))
        if sequence:
            return sequence
    return ""


def _h3_from_vh(vh: str) -> str:
    """Extract between the last plausible FR3 Cys and downstream FR4 WGxG motif."""
    vh = _seq(vh)
    candidates: list[str] = []
    for motif in re.finditer(r"WG.G", vh):
        before = vh[:motif.start()]
        cys = before.rfind("C")
        if cys >= 0:
            cdr = before[cys + 1:]
            if 3 <= len(cdr) <= 40:
                candidates.append(cdr)
    return candidates[-1] if candidates else ""


def heavy_cdrs(row: dict) -> dict[str, str]:
    """Supplied H1/H2/H3 take precedence; only H3 has a fallback parser."""
    h3 = _provided(row, "h3") or _h3_from_vh(row.get("vh_sequence") or row.get("variant_sequence") or "")
    return {"h1": _provided(row, "h1"), "h2": _provided(row, "h2"), "h3": h3}


def _descriptor(sequence: str) -> list[float]:
    if not sequence:
        return [0.0] * 5
    length = len(sequence)
    return [float(length), net_charge(sequence, 7.4),
            sum(_KD[aa] for aa in sequence) / length,
            sum(aa in "FWY" for aa in sequence) / length, 1.0]


def cdr_feature_vector(row: dict, mode: str = "h3") -> list[float]:
    """Return five descriptors per CDR: length, charge, KD, aromatic fraction, valid."""
    names = ("h3",) if mode == "h3" else ("h1", "h2", "h3")
    if mode not in {"h3", "heavy_all"}:
        raise ValueError("mode must be 'h3' or 'heavy_all'")
    cdrs = heavy_cdrs(row)
    return [value for name in names for value in _descriptor(cdrs[name])]


def parse_rate(rows: list[dict], mode: str = "h3") -> dict[str, float]:
    """Fraction of rows with each requested CDR available from supplied values or H3 parsing."""
    names = ("h3",) if mode == "h3" else ("h1", "h2", "h3")
    counts = {name: 0 for name in names}
    for row in rows:
        cdrs = heavy_cdrs(row)
        for name in names:
            counts[name] += bool(cdrs[name])
    n = len(rows)
    return {name: counts[name] / n if n else 0.0 for name in names}
