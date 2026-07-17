"""Apply the preregistered lab-readiness gate without modifying serving code."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from model.benchmark_governance import load_policy, validate_policy


MIN_MACRO_AUC = 0.80
MIN_ENRICHMENT_15 = 2.0
MAX_SEED_STD = 0.03
MIN_SOURCE_AUC = 0.65


def evaluate(matrix: dict, policy: dict) -> dict:
    validate_policy(policy)
    decisions = {}
    for property_name in ("polyreactivity", "aggregation", "hydrophobicity"):
        candidates = [
            summary for summary in matrix.get("summaries", [])
            if summary.get("property") == property_name
            and summary.get("test_tier") == "paired_fv_supervised"
            and summary.get("mean_macro_source_auc") is not None
            and len(summary.get("source_mean_auc", {})) >= 2
        ]
        if not candidates:
            decisions[property_name] = {"promote": False, "reason": "no scorable paired-Fv candidate"}
            continue
        candidate = max(candidates, key=lambda item: float(item["mean_macro_source_auc"]))
        runs = [run for run in matrix.get("experiments", []) if run.get("experiment") == candidate["experiment"]]
        source_results = [source for run in runs for source in run.get("per_source", []) if "roc_auc" in source]
        enrichments = [float(source["enrichment_at_15pct"]) for source in source_results]
        calibration_methods = [source.get("calibration_method", "") for source in source_results]
        locked_sources = policy["properties"][property_name]["source_roles"]["locked_validation"]
        checks = {
            "macro_source_auc_at_least_0.80": float(candidate["mean_macro_source_auc"]) >= MIN_MACRO_AUC,
            "three_declared_seeds": len(candidate.get("seed_macro_source_auc", [])) >= 3,
            "seed_std_at_most_0.03": float(candidate.get("seed_std_auc") or 0.0) <= MAX_SEED_STD,
            "at_least_two_diverse_sources": len(candidate.get("source_mean_auc", {})) >= 2,
            "all_source_auc_at_least_0.65": min(candidate["source_mean_auc"].values()) >= MIN_SOURCE_AUC,
            "mean_enrichment_at_15pct_at_least_2x": (
                statistics.fmean(enrichments) >= MIN_ENRICHMENT_15 if enrichments else False
            ),
            "train_only_calibration_present": bool(calibration_methods)
                and all(method.startswith("platt_") for method in calibration_methods),
            "source_bootstrap_ci_present": candidate.get("source_bootstrap_ci95") is not None,
            "locked_validation_source_assigned": bool(locked_sources),
            "gdpa3_unspent": all(
                not entry.get("prior_use")
                for entry in policy["properties"][property_name]["source_roles"]["final"]
                if entry["name"] == "gdpa3"
            ),
        }
        promote = all(checks.values())
        decisions[property_name] = {
            "promote": promote,
            "candidate": candidate["experiment"],
            "macro_source_auc": candidate["mean_macro_source_auc"],
            "source_auc": candidate["source_mean_auc"],
            "seed_std_auc": candidate["seed_std_auc"],
            "mean_enrichment_at_15pct": statistics.fmean(enrichments) if enrichments else None,
            "checks": checks,
            "serving_action": "eligible_for_scoped_integration_review" if promote else "keep_current_serving_head",
        }
    return {
        "schema_version": "1",
        "thresholds": {
            "macro_source_auc": MIN_MACRO_AUC,
            "enrichment_at_15pct": MIN_ENRICHMENT_15,
            "max_seed_std": MAX_SEED_STD,
            "minimum_source_auc": MIN_SOURCE_AUC,
        },
        "decisions": decisions,
        "serving_modified": False,
        "gdpa3_touched": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("results/developability_baseline_matrix.json"))
    parser.add_argument("--policy", type=Path, default=Path("tools/benchmark_policy.json"))
    parser.add_argument("--out", type=Path, default=Path("results/developability_promotion_gate.json"))
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    report = evaluate(matrix, load_policy(args.policy))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Developability promotion gate")
    for property_name, decision in report["decisions"].items():
        status = "PASS" if decision["promote"] else "FAIL"
        print(f"  {property_name}: {status} — {decision.get('macro_source_auc', 'no candidate')}")
        if not decision["promote"] and "checks" in decision:
            failed = [name for name, passed in decision["checks"].items() if not passed]
            print(f"    failed: {', '.join(failed)}")
    print("  serving code was not modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
