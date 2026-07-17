"""Gate multi-expert trials against identical held-source baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.benchmark_governance import load_policy
from tools.promotion_gate import evaluate as evaluate_serving


MIN_MACRO_DELTA = 0.010
MAX_SOURCE_DROP = 0.030
MAX_SEED_STD = 0.030
MAX_SPECIALIST_DROP = 0.005
MIN_SPECIALIST_ROUTE_COVERAGE = 0.80
EPSILON = 1e-12


def evaluate(
    baseline: dict,
    trial: dict,
    policy: dict,
    promotion_matrix: dict,
) -> dict:
    """Apply the relative experiment gate, then defer serving to the strict gate."""
    serving_report = evaluate_serving(promotion_matrix, policy)
    decisions = {}
    for property_name in sorted(trial):
        if property_name not in baseline:
            raise ValueError(f"missing baseline for {property_name}")
        base = baseline[property_name]
        candidate = trial[property_name]
        if set(base["source_mean_auc"]) != set(candidate["source_mean_auc"]):
            raise ValueError(f"baseline/trial source mismatch for {property_name}")
        source_delta = {
            source: float(candidate["source_mean_auc"][source]) - float(value)
            for source, value in base["source_mean_auc"].items()
        }
        macro_delta = (
            float(candidate["mean_macro_source_auc"])
            - float(base["mean_macro_source_auc"])
        )
        enrichment_delta = (
            float(candidate["mean_enrichment_at_15pct"])
            - float(base["mean_enrichment_at_15pct"])
        )
        checks = {
            "macro_auc_delta_at_least_0.010": macro_delta >= MIN_MACRO_DELTA - EPSILON,
            "no_source_drop_over_0.030": (
                min(source_delta.values()) >= -MAX_SOURCE_DROP - EPSILON
            ),
            "enrichment_at_15pct_holds": enrichment_delta >= -EPSILON,
            "seed_std_at_most_0.030": (
                float(candidate["seed_std_auc"]) <= MAX_SEED_STD
            ),
            "audits_present": bool(candidate.get("audits_present")),
        }
        serving = serving_report["decisions"].get(
            property_name, {"promote": False, "reason": "no strict serving decision"}
        )
        failed_serving = [
            name for name, passed in serving.get("checks", {}).items() if not passed
        ]
        decisions[property_name] = {
            "experimental_admit": all(checks.values()),
            "checks": checks,
            "baseline_macro_auc": float(base["mean_macro_source_auc"]),
            "trial_macro_auc": float(candidate["mean_macro_source_auc"]),
            "macro_delta_auc": macro_delta,
            "source_delta_auc": source_delta,
            "enrichment_delta_at_15pct": enrichment_delta,
            "serving_eligible": bool(serving.get("promote")),
            "serving_reason": (
                "strict promotion gate passed"
                if serving.get("promote")
                else ", ".join(failed_serving)
                or str(serving.get("reason") or "strict promotion gate failed")
            ),
        }
    return decisions


def evaluate_specialists(specialists: dict, comparable: dict) -> dict:
    """Gate specialists separately so they can never inflate generalist evidence."""
    decisions = {}
    for property_name, sources in specialists.items():
        property_decisions = {}
        for source, result in sources.items():
            prior = comparable.get(property_name, {}).get(source)
            enough_rows = (
                int(result.get("n", 0)) >= 40
                and min(
                    int(result.get("positives", 0)),
                    int(result.get("n", 0)) - int(result.get("positives", 0)),
                ) >= 10
            )
            route_coverage = float(result.get("mean_router_coverage", 0.0))
            delta = float(result["mean_auc"]) - float(prior) if prior is not None else None
            property_decisions[source] = {
                "experimental_admit": bool(
                    enough_rows
                    and delta is not None
                    and delta >= -MAX_SPECIALIST_DROP - EPSILON
                    and route_coverage >= MIN_SPECIALIST_ROUTE_COVERAGE - EPSILON
                ),
                "n": int(result.get("n", 0)),
                "mean_auc": float(result["mean_auc"]),
                "mean_router_coverage": route_coverage,
                "best_comparable_auc": float(prior) if prior is not None else None,
                "delta_auc": delta,
                "reason": (
                    "held comparable AUC with at least 80% route coverage"
                    if enough_rows
                    and delta is not None
                    and delta >= -MAX_SPECIALIST_DROP - EPSILON
                    and route_coverage >= MIN_SPECIALIST_ROUTE_COVERAGE - EPSILON
                    else (
                        "route coverage below 80%"
                        if route_coverage < MIN_SPECIALIST_ROUTE_COVERAGE - EPSILON
                        else "insufficient rows or did not hold comparable specialist AUC"
                    )
                ),
            }
        decisions[property_name] = property_decisions
    return decisions


def _comparable_specialists(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"specialist baseline report not found: {path}; "
            "pass --specialist-baseline with an existing head_c report"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "polyreactivity": {
            source: float(result["auc"])
            for source, result in report.get("per_source", {}).items()
            if "auc" in result
        }
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=Path("results/developability_multi_expert.json"),
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("tools/benchmark_policy.json")
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/developability_multi_expert_gate.json"),
    )
    parser.add_argument(
        "--specialist-baseline", type=Path,
        default=Path("results/head_c_polyreactivity.json"),
    )
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    generalists = report["generalists"]
    baseline = {
        property_name: summary["baseline"]
        for property_name, summary in generalists.items()
    }
    trial = {
        property_name: summary["trial"]
        for property_name, summary in generalists.items()
    }
    policy = load_policy(args.policy)
    decisions = evaluate(baseline, trial, policy, report["promotion_matrix"])
    specialist_decisions = evaluate_specialists(
        report.get("specialists", {}),
        _comparable_specialists(args.specialist_baseline),
    )
    output = {
        "schema_version": "1",
        "decisions": decisions,
        "specialists": specialist_decisions,
        "serving_modified": False,
        "gdpa3_touched": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Multi-expert development gate")
    for property_name, decision in decisions.items():
        status = "ACCEPTED" if decision["experimental_admit"] else "QUARANTINED"
        print(
            f"  {property_name:<18} {decision['baseline_macro_auc']:.3f} -> "
            f"{decision['trial_macro_auc']:.3f} "
            f"({decision['macro_delta_auc']:+.3f}) {status}"
        )
        print(
            f"    serving_eligible={decision['serving_eligible']} "
            f"({decision['serving_reason']})"
        )
    for property_name, sources in specialist_decisions.items():
        for source, decision in sources.items():
            status = "ACCEPTED" if decision["experimental_admit"] else "QUARANTINED"
            prior = decision["best_comparable_auc"]
            prior_text = "unavailable" if prior is None else f"{prior:.3f}"
            print(
                f"  specialist {property_name}/{source}: "
                f"{prior_text} -> {decision['mean_auc']:.3f} "
                f"route={decision['mean_router_coverage']:.3f} {status}"
            )
    print("  serving code and GDPa3 were untouched")
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
