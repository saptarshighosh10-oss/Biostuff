"""Build the local provenance registry from the consolidated antibody tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.developability_registry import build_registry, default_inputs, default_model_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("results/developability_registry.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("results/developability_registry.json"))
    args = parser.parse_args(argv)
    report = build_registry(args.database, default_inputs(), model_rows=default_model_rows())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Unified developability registry")
    print(f"  input rows: {report['input_rows']}")
    print(f"  accepted: {report['accepted_rows']}")
    print(f"  conflicts quarantined: {report['quarantined_conflict_rows']}")
    print(f"  exact duplicates: {report['duplicate_rows']}")
    print(f"  rejected: {report['rejected_rows']}")
    print(f"  supervision tiers: {report['tier_counts']}")
    print(f"  evidence tiers: {report['evidence_counts']}")
    print(f"  supervised training eligible: {report['training_eligible_rows']}")
    print(f"  holdout eligible: {report['holdout_eligible_rows']}")
    print(f"  unique molecules: {report['unique_molecules']}")
    print(f"  database: {report['database']}")
    print(f"  report: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
