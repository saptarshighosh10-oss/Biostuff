"""VH/VL chain typing and deterministic pairing (design section 5).

Stdlib-only, no ANARCI / no network. `features/cdr.py` only *extracts* CDR
loops for a chain whose type is already known, so it cannot classify VH vs VL;
the classification heuristic below is new. Deterministic pair_id reuses the
canonical hasher from `data.contract`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from data.contract import sha256_hex

logger = logging.getLogger(__name__)

_AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")

# ponytail: chain typing rides the FR4 J-segment motif, the single most
# reliable VH/VL discriminator without ANARCI. Heavy J = W-G-x-G ("WGQGTLVTVSS"),
# light J = F-G-x-G ("FGGGTKLTVL" / "FGQGTKVEIK"); the W-vs-F first residue is
# the whole signal. Fallbacks: C-terminal signature, then length window.
# KNOWN FAILURE MODES (this is the calibration knob — tune motifs on real data):
#   - truncated / CDR-only fragments missing FR4 -> "unknown"
#   - non-canonical or engineered J segments (nanobody VHH, scFv linkers)
#   - germlines with atypical FR4 (rare WG.G in a light chain, vice versa)
# Upgrade path: swap this function body for an ANARCI/HMMER call when available.
_HEAVY_J = re.compile(r"WG.G")
_LIGHT_J = re.compile(r"FG.G")
_HEAVY_TAIL = re.compile(r"(VSS|VTVSS)$")
_LIGHT_TAIL = re.compile(r"(EIK|EIR|VEIK|TVL|KLTV|TKL)$")


def _clean(sequence: str) -> str:
    if not isinstance(sequence, str):
        return ""
    return _AA_RE.sub("", sequence.strip().upper())


def classify_chain(sequence: str) -> str:
    """Return "vh" | "vl" | "unknown" from lightweight sequence heuristics."""
    seq = _clean(sequence)
    if len(seq) < 60:
        return "unknown"

    # Primary: FR4 J-segment motif in the C-terminal third (where FR4 lives).
    tail = seq[len(seq) * 2 // 3:]
    heavy_j = bool(_HEAVY_J.search(tail))
    light_j = bool(_LIGHT_J.search(tail))
    if heavy_j and not light_j:
        return "vh"
    if light_j and not heavy_j:
        return "vl"

    # Fallback: terminal signature residues.
    if _HEAVY_TAIL.search(seq) and not _LIGHT_TAIL.search(seq):
        return "vh"
    if _LIGHT_TAIL.search(seq) and not _HEAVY_TAIL.search(seq):
        return "vl"

    # Last resort: length window (VH ~118-128, VL ~107-115). Weak — only used
    # when motif/tail gave nothing.
    if not heavy_j and not light_j:
        if 116 <= len(seq) <= 135:
            return "vh"
        if 100 <= len(seq) <= 115:
            return "vl"
    return "unknown"


def _row_chain(row: dict[str, Any]) -> str:
    """Best-effort chain type for a row: explicit chain_id wins, else infer."""
    cid = row.get("chain_id")
    if isinstance(cid, str) and cid.strip().lower() in {"vh", "vl"}:
        return cid.strip().lower()
    if row.get("vh_sequence") and not row.get("vl_sequence"):
        return "vh"
    if row.get("vl_sequence") and not row.get("vh_sequence"):
        return "vl"
    seq = row.get("sequence") or row.get("vh_sequence") or row.get("vl_sequence")
    return classify_chain(seq) if seq else "unknown"


def _chain_seq(row: dict[str, Any], chain: str) -> str | None:
    if chain == "vh":
        return row.get("vh_sequence") or row.get("sequence")
    if chain == "vl":
        return row.get("vl_sequence") or row.get("sequence")
    return row.get("sequence")


def _flags(row: dict[str, Any]) -> set[str]:
    return set(str(f) for f in row.get("feature_flags", []) or [])


def pair_chains(records: list[dict]) -> list[dict]:
    """Group rows by molecule and pair a lone VH with a lone VL.

    When a molecule has exactly one VH and one VL, both rows get a shared
    deterministic pair_id = sha256("vh|vl")[:16] and both vh_sequence/
    vl_sequence populated. Otherwise pair_id stays None and rows get the
    "single_chain_only" feature flag. Pairing never crosses molecule_id.
    """
    out = [dict(r) for r in records]

    # Group by molecule_id; rows without one are isolated so they never pair.
    groups: dict[Any, list[int]] = {}
    for i, row in enumerate(out):
        mol = row.get("molecule_id")
        key = mol if mol is not None else ("__no_molecule__", i)
        groups.setdefault(key, []).append(i)

    for idxs in groups.values():
        vh = [i for i in idxs if _row_chain(out[i]) == "vh"]
        vl = [i for i in idxs if _row_chain(out[i]) == "vl"]

        if len(vh) == 1 and len(vl) == 1:
            vh_seq = _chain_seq(out[vh[0]], "vh")
            vl_seq = _chain_seq(out[vl[0]], "vl")
            if vh_seq and vl_seq:
                pair_id = sha256_hex(f"{vh_seq}|{vl_seq}")[:16]
                for i in idxs:
                    out[i]["pair_id"] = pair_id
                    out[i]["vh_sequence"] = vh_seq
                    out[i]["vl_sequence"] = vl_seq
                    out[i]["feature_flags"] = sorted(_flags(out[i]) - {"single_chain_only"})
                continue

        # Unpaired: single chain, malformed, or missing sequence.
        for i in idxs:
            out[i]["pair_id"] = None
            out[i]["feature_flags"] = sorted(_flags(out[i]) | {"single_chain_only"})

    return out


def assert_pairing_invariants(rows: list[dict]) -> None:
    """Raise if a pair_id lacks both chains, or a molecule has >2 same-type chains."""
    for row in rows:
        if row.get("pair_id") and not (row.get("vh_sequence") and row.get("vl_sequence")):
            raise ValueError(
                f"pair_id set but missing vh/vl on molecule "
                f"{row.get('molecule_id')!r}"
            )

    counts: dict[Any, dict[str, int]] = {}
    for row in rows:
        mol = row.get("molecule_id")
        chain = _row_chain(row)
        if chain in {"vh", "vl"}:
            counts.setdefault(mol, {"vh": 0, "vl": 0})[chain] += 1

    # ponytail: one molecule = one antibody = at most one VH and one VL, so
    # >1 of a type is malformed. (Spec prose said ">2" but a two-VH molecule
    # must fail — a duplicate chain is exactly the leak this guard exists for.)
    for mol, c in counts.items():
        for chain in ("vh", "vl"):
            if c[chain] > 1:
                logger.error("molecule %r has %d %s chains", mol, c[chain], chain)
                raise ValueError(
                    f"molecule {mol!r} has {c[chain]} {chain} chains (>1 of same type)"
                )
