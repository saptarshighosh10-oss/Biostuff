"""Test paired-chain biophysical features on the fixed HIC development holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.benchmark_governance import load_policy, validate_policy
from model.developability_matrix import load_property_rows
from model.developability_multi_expert import (
    CandidateSpec,
    nested_generalist,
    summarize_generalists,
)


PROPERTY = "hydrophobicity"
CURRENT_BEST = 0.757146908247174
TOLERANCE = 0.005
SPECS = (
    CandidateSpec(
        "bio_mixed_lgbm",
        "biophysical_charge",
        "mixed",
        "lgbm",
        ("paired_fv_supervised", "single_chain_supervised"),
    ),
    CandidateSpec(
        "paired_bio_mixed_lgbm",
        "paired_biophysical_charge",
        "mixed",
        "lgbm",
        ("paired_fv_supervised", "single_chain_supervised"),
    ),
)


def run(database: Path, policy_path: Path) -> dict:
    policy = load_policy(policy_path)
    validate_policy(policy)
    rows, manifest = load_property_rows(
        database, PROPERTY, prefetch_per_binary_class=40000
    )
    if any(str(row["source"]).lower() == "gdpa3" for row in rows):
        raise RuntimeError("GDPa3 appeared in development rows; refusing to continue")
    held_sources = [
        item["name"]
        for item in policy["properties"][PROPERTY]["source_roles"]["development"]
    ]
    runs = [
        nested_generalist(
            rows,
            held_sources,
            PROPERTY,
            seed,
            candidates=SPECS,
        )
        for seed in (42, 43, 44)
    ]
    summary = summarize_generalists(runs)[PROPERTY]
    for run_result in summary["per_seed"]:
        for source_result in run_result["per_source"].values():
            source_result.pop("record_predictions", None)
    trial_auc = float(summary["trial"]["mean_macro_source_auc"])
    return {
        "schema_version": "1",
        "experiment": "hic_paired_chain_biophysical",
        "hypothesis": (
            "VL biophysical descriptors add HIC signal missing from the VH-only baseline"
        ),
        "benchmark_role": "development_only",
        "seeds": [42, 43, 44],
        "current_best_auc": CURRENT_BEST,
        "gate_floor_auc": CURRENT_BEST - TOLERANCE,
        "experimental_admit": trial_auc >= CURRENT_BEST - TOLERANCE,
        "summary": summary,
        "registry_hash": manifest.get("registry_hash"),
        "limitations": [
            "No locked validation source is assigned.",
            "GDPa3 was not loaded or inspected.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("results/developability_registry.sqlite"),
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("tools/benchmark_policy.json")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/research_hic_paired_features.json"),
    )
    args = parser.parse_args(argv)
    report = run(args.database, args.policy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    baseline = summary["baseline"]["mean_macro_source_auc"]
    trial = summary["trial"]["mean_macro_source_auc"]
    decision = "ACCEPTED" if report["experimental_admit"] else "QUARANTINED"
    print(f"HIC paired-chain experiment: {baseline:.3f} -> {trial:.3f} {decision}")
    for source, auc in summary["trial"]["source_mean_auc"].items():
        print(f"  {source:<22} AUC={auc:.3f}")
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
