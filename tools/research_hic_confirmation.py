"""Confirm the frozen HIC interaction family on one selection-isolated source."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from model.benchmark_governance import load_policy, validate_policy
from model.developability_matrix import load_property_rows
from model.developability_multi_expert import nested_generalist, summarize_generalists
from tools.research_hic_paired_features import (
    HYBRID_WEIGHT_SPECS,
    INTERACTION_BEST,
    INTERACTION_SPECS,
)


PROPERTY = "hydrophobicity"
SEEDS = (42, 43, 44)
HOLD_TOLERANCE = 0.005


def _strip_predictions(runs: list[dict]) -> None:
    for run in runs:
        for result in run["per_source"].values():
            result.pop("record_predictions", None)


def _evaluate(
    rows: list[dict], held_sources: list[str], outer_sources: list[str], specs
) -> list[dict]:
    return [
        nested_generalist(
            rows,
            held_sources,
            PROPERTY,
            seed,
            candidates=specs,
            outer_sources=outer_sources,
        )
        for seed in SEEDS
    ]


def run(database: Path, policy_path: Path, confirmation_source: str) -> dict:
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
    if confirmation_source not in held_sources:
        raise ValueError(f"{confirmation_source} is not a declared HIC development source")
    selection_sources = [
        source for source in held_sources if source != confirmation_source
    ]
    selection_rows = [
        row for row in rows if row["source"] != confirmation_source
    ]

    baseline_selection = _evaluate(
        selection_rows, selection_sources, selection_sources, HYBRID_WEIGHT_SPECS
    )
    trial_selection = _evaluate(
        selection_rows, selection_sources, selection_sources, INTERACTION_SPECS
    )
    baseline_confirmation = _evaluate(
        rows, held_sources, [confirmation_source], HYBRID_WEIGHT_SPECS
    )
    trial_confirmation = _evaluate(
        rows, held_sources, [confirmation_source], INTERACTION_SPECS
    )
    all_runs = (
        baseline_selection + trial_selection
        + baseline_confirmation + trial_confirmation
    )
    _strip_predictions(all_runs)

    def summary(runs):
        return summarize_generalists(runs)[PROPERTY]["trial"]

    base_select = summary(baseline_selection)
    trial_select = summary(trial_selection)
    base_confirm = summary(baseline_confirmation)
    trial_confirm = summary(trial_confirmation)
    combined_seed_auc = [
        statistics.fmean(
            [
                float(result["trial"]["roc_auc"])
                for result in selection_run["per_source"].values()
            ]
            + [
                float(
                    confirmation_run["per_source"][confirmation_source]
                    ["trial"]["roc_auc"]
                )
            ]
        )
        for selection_run, confirmation_run in zip(
            trial_selection, trial_confirmation
        )
    ]
    combined_macro = statistics.fmean(combined_seed_auc)
    calibration_methods = [
        result["trial"]["calibration_method"]
        for run in trial_selection + trial_confirmation
        for result in run["per_source"].values()
    ]
    checks = {
        "combined_macro_within_0.005_of_0.809": (
            combined_macro >= INTERACTION_BEST - HOLD_TOLERANCE
        ),
        "confirmation_beats_no_interaction": (
            trial_confirm["mean_macro_source_auc"]
            >= base_confirm["mean_macro_source_auc"]
        ),
        "confirmation_auc_at_least_0.65": (
            trial_confirm["mean_macro_source_auc"] >= 0.65
        ),
        "combined_seed_std_at_most_0.03": (
            statistics.pstdev(combined_seed_auc) <= 0.03
        ),
        "confirmation_seed_std_at_most_0.03": (
            trial_confirm["seed_std_auc"] <= 0.03
        ),
        "train_only_platt_calibration": bool(calibration_methods)
        and all(method.startswith("platt_") for method in calibration_methods),
        "confirmation_excluded_from_selection_rows": all(
            row["source"] != confirmation_source for row in selection_rows
        ),
        "gdpa3_unspent": True,
    }
    return {
        "schema_version": "1",
        "experiment": "hic_interactions_platt_nested_confirmation",
        "benchmark_role": "development_confirmation",
        "confirmation_source": confirmation_source,
        "confirmation_prior_use_caveat": (
            "This source is isolated from this run's selection, but historical "
            "development results were already known; it is not locked validation."
        ),
        "seeds": list(SEEDS),
        "current_best_auc": INTERACTION_BEST,
        "selection_sources": selection_sources,
        "baseline_selection": base_select,
        "trial_selection": trial_select,
        "baseline_confirmation": base_confirm,
        "trial_confirmation": trial_confirm,
        "combined_seed_auc": combined_seed_auc,
        "combined_macro_source_auc": combined_macro,
        "combined_seed_std_auc": statistics.pstdev(combined_seed_auc),
        "checks": checks,
        "holds": all(checks.values()),
        "registry_hash": manifest.get("registry_hash"),
        "runs": {
            "baseline_selection": baseline_selection,
            "trial_selection": trial_selection,
            "baseline_confirmation": baseline_confirmation,
            "trial_confirmation": trial_confirmation,
        },
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
    parser.add_argument("--confirmation-source", default="shehata2019")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/research_hic_confirmation.json"),
    )
    args = parser.parse_args(argv)
    report = run(args.database, args.policy, args.confirmation_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = "HOLDS" if report["holds"] else "DOES NOT HOLD"
    print(
        f"HIC nested confirmation: {report['combined_macro_source_auc']:.3f}; "
        f"{args.confirmation_source} "
        f"{report['baseline_confirmation']['mean_macro_source_auc']:.3f} -> "
        f"{report['trial_confirmation']['mean_macro_source_auc']:.3f}; {status}"
    )
    for name, passed in report["checks"].items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
