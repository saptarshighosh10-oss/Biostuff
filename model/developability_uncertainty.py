"""Consolidate multi-seed held-out predictions into uncertainty records."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from data.run_manifest import file_sha256


DOMAIN_ORDER = {"in_domain": 0, "borderline": 1, "out_of_domain": 2}


def consolidate(predictions: str | Path, output: str | Path) -> dict:
    source = Path(predictions)
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["experiment"], row["property"], row["feature_mode"],
                row["held_source"], row["assay"], row["record_id"], row["label"],
            )
            groups[key].append(row)

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "experiment", "property", "feature_mode", "held_source", "assay",
        "record_id", "label", "seed_count", "risk_score", "raw_risk_score",
        "ensemble_disagreement", "ood_distance", "applicability",
        "nearest_training_sources", "abstain", "abstention_reason",
    )
    domain_counts: Counter[str] = Counter()
    disagreements = []
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            scores = [float(row["score"]) for row in rows]
            raw_scores = [float(row["raw_score"]) for row in rows]
            distances = [float(row["ood_distance"]) for row in rows]
            domains = [row["applicability"] for row in rows]
            domain = max(domains, key=lambda value: DOMAIN_ORDER[value])
            disagreement = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            nearest = sorted({row["nearest_training_source"] for row in rows})
            reasons = []
            if len(rows) < 3:
                reasons.append("fewer_than_three_seeds")
            if domain == "out_of_domain":
                reasons.append("out_of_domain")
            if disagreement > 0.15:
                reasons.append("high_seed_disagreement")
            writer.writerow({
                "experiment": key[0], "property": key[1], "feature_mode": key[2],
                "held_source": key[3], "assay": key[4], "record_id": key[5],
                "label": key[6], "seed_count": len(rows),
                "risk_score": statistics.fmean(scores),
                "raw_risk_score": statistics.fmean(raw_scores),
                "ensemble_disagreement": disagreement,
                "ood_distance": statistics.fmean(distances),
                "applicability": domain,
                "nearest_training_sources": ";".join(nearest),
                "abstain": bool(reasons),
                "abstention_reason": ";".join(reasons),
            })
            domain_counts[domain] += 1
            disagreements.append(disagreement)
    sorted_disagreement = sorted(disagreements)
    return {
        "schema_version": "1",
        "input_predictions": str(source.resolve()),
        "input_sha256": file_sha256(source),
        "output": str(target.resolve()),
        "output_sha256": file_sha256(target),
        "unique_prediction_records": len(groups),
        "domain_counts": dict(sorted(domain_counts.items())),
        "mean_seed_disagreement": statistics.fmean(disagreements) if disagreements else None,
        "p95_seed_disagreement": (
            sorted_disagreement[int(0.95 * (len(sorted_disagreement) - 1))]
            if sorted_disagreement else None
        ),
        "abstention_policy": {
            "out_of_domain": True,
            "seed_disagreement_above": 0.15,
            "minimum_seeds": 3,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("results/developability_baseline_predictions.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/developability_uncertainty_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/developability_uncertainty.json"))
    args = parser.parse_args(argv)
    report = consolidate(args.predictions, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Developability uncertainty consolidation")
    print(f"  unique prediction records: {report['unique_prediction_records']}")
    print(f"  applicability: {report['domain_counts']}")
    print(f"  mean seed disagreement: {report['mean_seed_disagreement']}")
    print(f"  p95 seed disagreement: {report['p95_seed_disagreement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
