"""Build deterministic protein-disjoint ProteinGym auxiliary cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.contract import hash_payload
from data.proteingym import load_proteingym_data
from data.variant_quality import validate_rows


def _group_order(groups: dict[str, list[dict]]) -> list[str]:
    return sorted(
        groups,
        key=lambda group: (len(groups[group]), hashlib.sha256(group.encode()).hexdigest()),
    )


def split_by_protein(rows: list[dict], train_min: int = 200_000, test_min: int = 100_000) -> tuple[list[dict], list[dict], list[str], list[str], int]:
    by_group: dict[str, list[dict]] = {}
    fallback_count = 0
    for row in rows:
        group = str(row.get("protein_group_id") or row.get("group_id", "unknown"))
        if row.get("protein_group_fallback"):
            fallback_count += 1
        by_group.setdefault(group, []).append(row)

    ordered = _group_order(by_group)
    test_groups: list[str] = []
    test_rows: list[dict] = []
    for group in ordered:
        test_groups.append(group)
        test_rows.extend(by_group[group])
        if len(test_rows) >= test_min:
            break

    train_groups = [group for group in ordered if group not in test_groups]
    selected_train_groups = list(train_groups)
    train_rows = [row for group in selected_train_groups for row in by_group[group]]

    if len(train_rows) < train_min or len(test_rows) < test_min:
        raise RuntimeError(
            f"ProteinGym cohort too small for requested partition: "
            f"train={len(train_rows)}, test={len(test_rows)}"
        )
    if set(selected_train_groups) & set(test_groups):
        raise RuntimeError("Protein-disjoint partition construction produced overlapping groups")
    return train_rows, test_rows, selected_train_groups, test_groups, fallback_count


def build(output_dir: str | Path = "data/external/proteingym", train_min: int = 1_000_000, test_min: int = 300_000, max_per_assay: int = 10_000, max_assays: int = 217) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failures, working = load_proteingym_data(max_assays=max_assays, max_per_assay=max_per_assay, seed=None, keep_all=True)
    rows = failures + working
    for row in rows:
        row["supervision_status"] = "auxiliary"
        row["endpoint_family"] = "protein_mutation_fitness"
        row["source_role"] = "auxiliary_non_antibody"

    rows, quarantine, quality = validate_rows(rows)
    (output / "quarantine.json").write_text(
        json.dumps(quarantine, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    train, test, train_groups, test_groups, fallback_count = split_by_protein(
        rows, train_min=train_min, test_min=test_min
    )
    for name, payload in (("train.json", train), ("test.json", test)):
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "2",
        "source": "ProteinGym",
        "source_url": "https://github.com/OATML-Markslab/ProteinGym",
        "endpoint_family": "protein_mutation_fitness",
        "source_role": "auxiliary_non_antibody",
        "split_unit": "protein_group_id",
        "protein_group_fallback_count": fallback_count,
        "max_per_assay": max_per_assay,
        "max_assays": max_assays,
        "keep_all_labeled_rows": True,
        "quality_gate": quality,
        "train_count": len(train),
        "test_count": len(test),
        "train_protein_groups": train_groups,
        "test_protein_groups": test_groups,
        "train_sha256": hashlib.sha256((output / "train.json").read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256((output / "test.json").read_bytes()).hexdigest(),
    }
    manifest["manifest_hash"] = hash_payload(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ProteinGym protein-disjoint partition: train={len(train)} test={len(test)}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/external/proteingym")
    parser.add_argument("--train-min", type=int, default=1_000_000)
    parser.add_argument("--test-min", type=int, default=300_000)
    parser.add_argument("--max-per-assay", type=int, default=10_000)
    parser.add_argument("--max-assays", type=int, default=217)
    args = parser.parse_args()
    build(args.output_dir, args.train_min, args.test_min, args.max_per_assay, args.max_assays)
