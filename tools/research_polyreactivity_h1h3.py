"""Test reliable H1+H3 descriptors on diverse polyreactivity holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.benchmark_governance import load_policy, validate_policy
from model.cdr_features import parse_rate
from model.developability_matrix import _feature_row, load_property_rows
from model.developability_multi_expert import (
    CANDIDATES,
    CandidateSpec,
    nested_generalist,
    summarize_generalists,
)


PROPERTY = "polyreactivity"
SEEDS = (42, 43, 44)
CURRENT_BEST = 0.6714802054330579
TOLERANCE = 0.005
BASELINE_SPECS = CANDIDATES[PROPERTY]
TRIAL_SPECS = BASELINE_SPECS + (
    CandidateSpec(
        "cdr_h1h3_library_lgbm",
        "cdr_h1_h3",
        "library_only",
        "lgbm",
    ),
)


def _evaluate(rows: list[dict], held_sources: list[str], specs) -> list[dict]:
    runs = [
        nested_generalist(rows, held_sources, PROPERTY, seed, candidates=specs)
        for seed in SEEDS
    ]
    for run in runs:
        for result in run["per_source"].values():
            result.pop("record_predictions", None)
    return runs


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
    baseline_runs = _evaluate(rows, held_sources, BASELINE_SPECS)
    trial_runs = _evaluate(rows, held_sources, TRIAL_SPECS)
    baseline = summarize_generalists(baseline_runs)[PROPERTY]["trial"]
    trial = summarize_generalists(trial_runs)[PROPERTY]["trial"]
    return {
        "schema_version": "1",
        "experiment": "polyreactivity_h1_h3",
        "hypothesis": (
            "High-coverage H1 and H3 descriptors retain heavy-CDR signal without "
            "the unreliable H2 heuristic"
        ),
        "benchmark_role": "development_only",
        "seeds": list(SEEDS),
        "current_best_auc": CURRENT_BEST,
        "gate_floor_auc": CURRENT_BEST - TOLERANCE,
        "experimental_admit": (
            trial["mean_macro_source_auc"] >= CURRENT_BEST - TOLERANCE
        ),
        "baseline": baseline,
        "trial": trial,
        "held_source_parse_rate": {
            source: parse_rate(
                [_feature_row(row) for row in rows if row["source"] == source],
                "h1_h3",
            )
            for source in held_sources
        },
        "registry_hash": manifest.get("registry_hash"),
        "limitations": [
            "AIMS Flu has only 97 held-out rows.",
            "No locked validation source is assigned.",
            "GDPa3 was not loaded or inspected.",
        ],
        "runs": {"baseline": baseline_runs, "trial": trial_runs},
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
        default=Path("results/research_polyreactivity_h1h3.json"),
    )
    args = parser.parse_args(argv)
    report = run(args.database, args.policy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision = "ACCEPTED" if report["experimental_admit"] else "QUARANTINED"
    print(
        f"Polyreactivity H1+H3 experiment: "
        f"{report['baseline']['mean_macro_source_auc']:.3f} -> "
        f"{report['trial']['mean_macro_source_auc']:.3f} {decision}"
    )
    for source, auc in report["trial"]["source_mean_auc"].items():
        print(f"  {source:<22} AUC={auc:.3f}")
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
