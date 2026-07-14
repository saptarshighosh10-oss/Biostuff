"""Provenance preprocessing helpers for Plan C data contracts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from data.contract import NormalizedRow


def build_conflict_and_exclusion_ledgers(
    rows: list[NormalizedRow],
) -> tuple[list[NormalizedRow], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build conflict and exclusion ledgers without mutating row provenance.

    Rows with duplicate record ids, label flips, identity overlaps, and
    chain/pairing issues are moved to exclusion ledgers and removed from
    training rows.
    """
    rows_with_index: list[tuple[int, NormalizedRow]] = list(enumerate(rows))
    rows_by_sequence: dict[str, list[tuple[int, NormalizedRow]]] = defaultdict(list)
    rows_by_pair: dict[str, list[tuple[int, NormalizedRow]]] = defaultdict(list)
    rows_by_id: dict[str, list[tuple[int, NormalizedRow]]] = defaultdict(list)
    exclusion_reasons: dict[tuple[str, int], set[str]] = defaultdict(set)
    conflicts: list[dict[str, Any]] = []

    for idx, row in rows_with_index:
        rows_by_id[row.record_id].append((idx, row))
        rows_by_sequence[row.sequence_hash].append((idx, row))
        if row.pair_id:
            rows_by_pair[row.pair_id].append((idx, row))

    for record_id, members in sorted(rows_by_id.items()):
        if len(members) <= 1:
            continue
        for idx, _ in members:
            exclusion_reasons[(record_id, idx)].add("duplicate_record_id")

    for seq_hash in sorted(rows_by_sequence):
        group = rows_by_sequence[seq_hash]
        labels = sorted({row.binary_label for _, row in group if row.binary_label in (0, 1)})
        if len(labels) > 1:
            for idx, row in group:
                exclusion_reasons[(row.record_id, idx)].add("label_flip")
            conflict_records = [
                {
                    "record_id": row.record_id,
                    "source": row.source,
                    "binary_label": row.binary_label,
                    "supervision_status": row.supervision_status,
                }
                for _, row in sorted(group, key=lambda item: item[1].record_id)
            ]
            conflicts.append(
                {
                    "reason": "label_flip",
                    "sequence_hash": seq_hash,
                    "binary_labels": labels,
                    "records": conflict_records,
                    "n_records": len(group),
                }
            )

        if len({row.pair_id for _, row in group if row.pair_id}) > 1:
            for idx, row in group:
                exclusion_reasons[(row.record_id, idx)].add("identity_overlap")

    for pair_id in sorted(rows_by_pair):
        chain_rows = rows_by_pair[pair_id]
        has_explicit_chain_rows = any(row.chain_id is not None for _, row in chain_rows)
        if not has_explicit_chain_rows:
            continue
        chain_values = {row.chain_id for _, row in chain_rows if row.chain_id}
        has_pair_level_rows = any(row.chain_id is None for _, row in chain_rows)
        if has_pair_level_rows:
            for idx, row in chain_rows:
                exclusion_reasons[(row.record_id, idx)].add("chain_mismatch")
            continue
        if len(chain_rows) > 1 and (len(chain_values) != 1 and len(chain_values) != 2):
            for idx, row in chain_rows:
                exclusion_reasons[(row.record_id, idx)].add("chain_mismatch")
            continue
        if len(chain_values) == 2 and any(
            row.chain_id not in {"vh", "vl"} for _, row in chain_rows
        ):
            for idx, row in chain_rows:
                exclusion_reasons[(row.record_id, idx)].add("chain_mismatch")

    for idx, row in rows_with_index:
        if row.chain_id is not None and row.chain_id not in {"vh", "vl"}:
            exclusion_reasons[(row.record_id, idx)].add("invalid_chain_id")
        if row.pair_id is not None and row.chain_id in {"vh", "vl"}:
            exclusion_reasons[(row.record_id, idx)].add("single_chain_pair_id")
        if row.pair_id is not None and not (row.vh_sequence and row.vl_sequence):
            exclusion_reasons[(row.record_id, idx)].add("single_chain_pair_id")

    exclusions: list[dict[str, Any]] = []
    for (record_id, idx), reasons in sorted(
        exclusion_reasons.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        row = rows_with_index[idx][1]
        all_reasons = sorted(set(row.exclusions + list(reasons)))
        exclusions.append(
            {
                "reason": ";".join(all_reasons) if all_reasons else "excluded",
                "sequence_hash": row.sequence_hash,
                "record_id": row.record_id,
                "source": row.source,
                "study_id": row.study_id,
                "campaign_id": row.campaign_id,
                "chain_id": row.chain_id,
                "pair_id": row.pair_id,
                "binary_label": row.binary_label,
                "supervision_status": row.supervision_status,
            }
        )

    keep_records = [
        row
        for idx, row in rows_with_index
        if (row.record_id, idx) not in exclusion_reasons
    ]
    keep_records.sort(key=lambda row: (row.record_id, row.sequence_hash))

    return keep_records, conflicts, exclusions


def write_conflicts_tsv(conflicts: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "reason",
                "sequence_hash",
                "binary_labels",
                "n_records",
                "records",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in sorted(conflicts, key=lambda x: (x["reason"], x["sequence_hash"])):
            writer.writerow(
                {
                    "reason": row["reason"],
                    "sequence_hash": row["sequence_hash"],
                    "binary_labels": ",".join(str(v) for v in row["binary_labels"]),
                    "n_records": row["n_records"],
                    "records": "|".join(
                        f"{r['record_id']}:{r['source']}:{r['binary_label']}"
                        for r in row["records"]
                    ),
                }
            )


def write_exclusions_tsv(exclusions: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "reason",
                "record_id",
                "sequence_hash",
                "source",
                "study_id",
                "campaign_id",
                "pair_id",
                "chain_id",
                "binary_label",
                "supervision_status",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in sorted(exclusions, key=lambda x: x["record_id"]):
            writer.writerow(row)
