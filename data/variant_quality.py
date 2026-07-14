"""Small quality gate for variant rows before they enter a training cohort."""

from __future__ import annotations

from collections import defaultdict


LABELS = {"confirmed_failure", "working"}
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def _rebuild(parent: str, mutations: list) -> str | None:
    sequence = list(parent)
    for mutation in mutations:
        if not isinstance(mutation, (list, tuple)) or len(mutation) != 3:
            return None
        position, original, mutant = mutation
        if not isinstance(position, int) or not 0 <= position < len(sequence):
            return None
        if sequence[position] != original or original not in AMINO_ACIDS or mutant not in AMINO_ACIDS:
            return None
        sequence[position] = mutant
    return "".join(sequence)


def validate_rows(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Return deduplicated valid rows, quarantined rows, and audit counters."""
    valid: list[dict] = []
    quarantine: list[dict] = []
    seen: dict[tuple[str, str], str] = {}
    counters = defaultdict(int)

    for row in rows:
        reason = None
        label = row.get("label")
        sequence = str(row.get("variant_sequence", "")).upper()
        group = str(row.get("group_id", ""))
        if label not in LABELS:
            reason = "invalid_label"
        elif not sequence or not set(sequence) <= AMINO_ACIDS:
            reason = "invalid_variant_sequence"
        elif not group or group == "unknown":
            reason = "missing_assay_group"
        else:
            parent = str(row.get("wild_type_sequence", "")).upper()
            rebuilt = _rebuild(parent, row.get("mutations", [])) if parent else None
            if parent and rebuilt != sequence:
                reason = "mutation_parent_mismatch"
            elif not parent:
                row["parent_context"] = "missing"
                counters["parent_context_missing"] += 1

        key = (group, sequence)
        if reason is None and key in seen:
            if seen[key] != label:
                reason = "conflicting_duplicate_label"
            else:
                counters["duplicate_same_label"] += 1
                continue

        if reason:
            counters[reason] += 1
            quarantine.append({**row, "quarantine_reason": reason})
            continue
        seen[key] = label
        if "parent_context" not in row:
            row["parent_context"] = "complete"
        valid.append(row)

    counters["accepted"] = len(valid)
    counters["quarantined"] = len(quarantine)
    return valid, quarantine, dict(counters)
