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


def _h1_from_vh(vh: str) -> str:
    """Extract H1 between the FR1 Cys and its downstream conserved Trp."""
    vh = _seq(vh)
    for match in re.finditer(r"C([A-Z]{5,18})W[IV]", vh[:80]):
        return match.group(1)
    return ""


def _h2_from_vh(vh: str) -> str:
    """Extract H2 from the FR2 Trp through the next FR3 R[FT] boundary."""
    vh = _seq(vh)
    h1 = next(iter(re.finditer(r"C[A-Z]{5,18}W[IV]", vh[:80])), None)
    if not h1:
        return ""
    fr2 = re.search(r"W[IV]", vh[h1.end():])
    if not fr2:
        return ""
    start = h1.end() + fr2.end()
    end = re.search(r"R[FT]", vh[start:])
    if not end:
        return ""
    cdr = vh[start:start + end.start()]
    return cdr if 5 <= len(cdr) <= 35 else ""


def heavy_cdrs(row: dict) -> dict[str, str]:
    """Supplied heavy CDRs take precedence over dependency-free VH motif fallbacks."""
    vh = row.get("vh_sequence") or row.get("variant_sequence") or ""
    return {"h1": _provided(row, "h1") or _h1_from_vh(vh),
            "h2": _provided(row, "h2") or _h2_from_vh(vh),
            "h3": _provided(row, "h3") or _h3_from_vh(vh)}


def _descriptor(sequence: str) -> list[float]:
    if not sequence:
        return [0.0] * 5
    length = len(sequence)
    return [float(length), net_charge(sequence, 7.4),
            sum(_KD[aa] for aa in sequence) / length,
            sum(aa in "FWY" for aa in sequence) / length, 1.0]


def cdr_feature_vector(row: dict, mode: str = "h3") -> list[float]:
    """Return five descriptors per CDR: length, charge, KD, aromatic fraction, valid."""
    if mode not in {"h3", "h1_h3", "heavy_all"}:
        raise ValueError("mode must be 'h3', 'h1_h3', or 'heavy_all'")
    names = {
        "h3": ("h3",),
        "h1_h3": ("h1", "h3"),
        "heavy_all": ("h1", "h2", "h3"),
    }[mode]
    cdrs = heavy_cdrs(row)
    return [value for name in names for value in _descriptor(cdrs[name])]


def parse_rate(rows: list[dict], mode: str = "h3") -> dict[str, float]:
    """Fraction of rows with each requested CDR available from supplied values or H3 parsing."""
    if mode not in {"h3", "h1_h3", "heavy_all"}:
        raise ValueError("mode must be 'h3', 'h1_h3', or 'heavy_all'")
    names = {
        "h3": ("h3",),
        "h1_h3": ("h1", "h3"),
        "heavy_all": ("h1", "h2", "h3"),
    }[mode]
    counts = {name: 0 for name in names}
    for row in rows:
        cdrs = heavy_cdrs(row)
        for name in names:
            counts[name] += bool(cdrs[name])
    n = len(rows)
    return {name: counts[name] / n if n else 0.0 for name in names}
