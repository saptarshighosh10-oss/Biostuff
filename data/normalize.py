"""Thin loader-row -> contract normalizer.

Adapts a loose loader row dict onto the canonical ``NormalizedRow`` by
delegating all hashing/mapping/supervision logic to ``data.contract``. This
module adds nothing but the glue: inject ``source`` and supply deterministic
fallbacks for the few fields the contract requires that loaders omit.
"""

from __future__ import annotations

from data.contract import (
    ContractError,
    NormalizedRow,
    canonical_sequence,
    normalize_legacy_row,
    source_supervision_role,
)

# ponytail: supervised sources need endpoint_unit (+label_threshold when a label
# is present); loaders don't emit them. Deterministic placeholders keep the
# source_row_hash stable. Swap for real units/thresholds if a loader learns them.
_SUPERVISED_UNIT_FALLBACK = "unknown"
_SUPERVISED_THRESHOLD_FALLBACK = 0.0


def normalize_row(raw: dict, source: str) -> NormalizedRow:
    """Map a loader row + explicit ``source`` onto a contract ``NormalizedRow``."""
    row = dict(raw)
    row["source"] = source
    if source_supervision_role(source) == "supervised":
        row.setdefault("endpoint_unit", _SUPERVISED_UNIT_FALLBACK)
        label = row.get("label")
        if label is not None and str(label).strip():
            row.setdefault("label_threshold", _SUPERVISED_THRESHOLD_FALLBACK)
    return normalize_legacy_row(row)


def normalize_rows(raws: list[dict], source: str) -> tuple[list, int]:
    """Normalize a batch, skipping rows with an empty/invalid sequence.

    Returns ``(normalized_rows, n_skipped)``. Non-sequence contract violations
    still propagate — only bad sequences are silently dropped.
    """
    rows: list[NormalizedRow] = []
    skipped = 0
    for raw in raws:
        try:
            canonical_sequence(raw.get("variant_sequence") or raw.get("sequence") or "")
        except ContractError:
            skipped += 1
            continue
        rows.append(normalize_row(raw, source))
    return rows, skipped
