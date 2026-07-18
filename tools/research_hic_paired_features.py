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
ENSEMBLE_BEST = 0.757651292206039
ENSEMBLE_SPECS = SPECS + (
    CandidateSpec(
        "paired_bio_mixed_histgb",
        "paired_biophysical_charge",
        "mixed",
        "histgb",
        ("paired_fv_supervised", "single_chain_supervised"),
    ),
    CandidateSpec(
        "paired_bio_mixed_ridge",
        "paired_biophysical_charge",
        "mixed",
        "ridge",
        ("paired_fv_supervised", "single_chain_supervised"),
    ),
)
RANK_BEST = 0.7601431854319337
RANK_SPECS = ENSEMBLE_SPECS + (
    CandidateSpec(
        "paired_bio_mixed_rank_lgbm",
        "paired_biophysical_charge",
        "mixed",
        "rank_lgbm",
        ("paired_fv_supervised", "single_chain_supervised"),
    ),
)


def run(
    database: Path,
    policy_path: Path,
    ensemble: bool = False,
    rank: bool = False,
) -> dict:
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
    specs = RANK_SPECS if rank else ENSEMBLE_SPECS if ensemble else SPECS
    current_best = RANK_BEST if rank else ENSEMBLE_BEST if ensemble else CURRENT_BEST
    runs = [
        nested_generalist(
            rows,
            held_sources,
            PROPERTY,
            seed,
            candidates=specs,
        )
        for seed in (42, 43, 44)
    ]
    summary = summarize_generalists(runs)[PROPERTY]
    for run_result in summary["per_seed"]:
        for source_result in run_result["per_source"].values():
            source_result.pop("record_predictions", None)
    trial_auc = float(summary["trial"]["mean_macro_source_auc"])
    experiment = (
        "hic_continuous_rank_ensemble"
        if rank
        else "hic_paired_chain_learner_ensemble"
        if ensemble
        else "hic_paired_chain_biophysical"
    )
    hypothesis = (
        "Within-assay HIC percentile regression uses numeric label ordering discarded by binary training"
        if rank
        else "Nested LGBM, HistGB, and ridge blends stabilize paired-chain HIC transfer"
        if ensemble
        else "VL biophysical descriptors add HIC signal missing from the VH-only baseline"
    )
    return {
        "schema_version": "1",
        "experiment": experiment,
        "hypothesis": hypothesis,
        "benchmark_role": "development_only",
        "seeds": [42, 43, 44],
        "current_best_auc": current_best,
        "gate_floor_auc": current_best - TOLERANCE,
        "experimental_admit": trial_auc >= current_best - TOLERANCE,
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
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="also test paired-chain HistGB and ridge candidates",
    )
    parser.add_argument(
        "--rank",
        action="store_true",
        help="also test continuous within-assay percentile regression",
    )
    args = parser.parse_args(argv)
    report = run(args.database, args.policy, args.ensemble, args.rank)
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
