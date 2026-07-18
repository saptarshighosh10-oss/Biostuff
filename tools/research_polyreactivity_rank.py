"""Test whether continuous PSR ranks improve polyreactivity transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.benchmark_governance import load_policy, validate_policy
from model.developability_matrix import load_property_rows
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
TRIAL_SPECS = CANDIDATES[PROPERTY] + (
    CandidateSpec(
        "cdr_mixed_rank_lgbm_hybrid",
        "cdr_aware",
        "mixed",
        "rank_lgbm_hybrid",
    ),
)


def _load_baseline(path: Path, registry_hash: str, held_sources: list[str]) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("registry_hash") != registry_hash:
        raise RuntimeError("baseline registry hash differs from current data")
    if report.get("seeds") != list(SEEDS):
        raise RuntimeError("baseline seeds differ from declared seeds")
    runs = report.get("runs", {}).get("baseline", [])
    if len(runs) != len(SEEDS) or any(
        set(run.get("per_source", {})) != set(held_sources) for run in runs
    ):
        raise RuntimeError("baseline lacks the declared seed/source folds")
    return report["baseline"]


def run(database: Path, policy_path: Path, baseline_path: Path) -> dict:
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
    baseline = _load_baseline(
        baseline_path, str(manifest.get("registry_hash")), held_sources
    )
    runs = [
        nested_generalist(
            rows, held_sources, PROPERTY, seed, candidates=TRIAL_SPECS
        )
        for seed in SEEDS
    ]
    for seed_run in runs:
        for source_result in seed_run["per_source"].values():
            source_result.pop("record_predictions", None)
    trial = summarize_generalists(runs)[PROPERTY]["trial"]
    return {
        "schema_version": "1",
        "experiment": "polyreactivity_continuous_psr_rank",
        "hypothesis": (
            "Source-normalized continuous PSR ranks retain information lost by "
            "binary polyreactivity targets"
        ),
        "benchmark_role": "development_only",
        "seeds": list(SEEDS),
        "current_best_auc": CURRENT_BEST,
        "gate_floor_auc": CURRENT_BEST - TOLERANCE,
        "experimental_admit": (
            trial["mean_macro_source_auc"] >= CURRENT_BEST - TOLERANCE
        ),
        "baseline_source": str(baseline_path),
        "baseline": baseline,
        "trial": trial,
        "registry_hash": manifest.get("registry_hash"),
        "limitations": [
            "AIMS Flu has only 97 held-out rows.",
            "No locked validation source is assigned.",
            "GDPa3 was not loaded or inspected.",
            "Binary sources retain binary rank targets; only continuous assays add resolution.",
        ],
        "runs": runs,
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
        "--baseline",
        type=Path,
        default=Path("results/research_polyreactivity_domain.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/research_polyreactivity_rank.json"),
    )
    args = parser.parse_args(argv)
    report = run(args.database, args.policy, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision = "ACCEPTED" if report["experimental_admit"] else "QUARANTINED"
    print(
        "Polyreactivity continuous-rank experiment: "
        f"{report['baseline']['mean_macro_source_auc']:.3f} -> "
        f"{report['trial']['mean_macro_source_auc']:.3f} {decision}"
    )
    for source, auc in report["trial"]["source_mean_auc"].items():
        print(f"  {source:<22} AUC={auc:.3f}")
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
