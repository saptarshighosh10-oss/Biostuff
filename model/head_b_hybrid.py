"""Hybrid Head B (antibody aggregation) — route each assay to its best model.

Two evaluators exist:
  - independent per-assay (model/head_b_gbm.evaluate_assay): strong on assays with
    enough rows, but collapses to noise on tiny assays (n < ~50).
  - pooled multi-task (model/head_b_pooled.run_pooled): rescues tiny assays by
    sharing strength across the whole pool, at a small cost to a few already-strong
    large assays (negative transfer).

The hybrid keeps the winner of each: assays with n < ROUTE_THRESHOLD are served by
the pooled model; larger assays keep their own independent model. This captures the
big small-assay gain without the large-assay regression.

Run:
  python -m model.head_b_hybrid --use-plm --out results/head_b_hybrid_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from model.head_b_gbm import _assay_key, evaluate_assay, MIN_ROWS_PER_ASSAY
from model.head_b_pooled import _pool_rows, run_pooled

# Below this row count an assay is better served by the pooled model than by its own.
ROUTE_THRESHOLD = 50


def run_hybrid(use_plm: bool = True, route_threshold: int = ROUTE_THRESHOLD,
               n_splits: int = 5, seed: int = 42, include_gdpa: bool = True) -> dict:
    rows = _pool_rows(include_gdpa)
    by_assay: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_assay[_assay_key(r)].append(r)

    # Pooled per-assay Spearman (one shared run), keyed by "study/metric".
    pooled = run_pooled(use_plm=use_plm, n_splits=n_splits, seed=seed, include_gdpa=include_gdpa)
    pooled_assays = pooled["assays"]

    assays_report: dict[str, dict] = {}
    for key, bucket in by_assay.items():
        r0 = bucket[0]
        study = str(r0.get("study_id") or "?")
        metric = key[2]
        report_key = f"{study}/{metric}"
        n = len(bucket)

        if n < route_threshold:
            # pooled model owns small assays
            p = pooled_assays.get(report_key, {})
            spearman = p.get("spearman")
            source = "pooled"
            n_groups = p.get("n_groups")
        else:
            res = evaluate_assay(bucket, use_plm=use_plm, n_splits=n_splits, seed=seed)
            if res.get("abstain"):
                spearman, n_groups = None, None
            else:
                spearman = res.get("spearman")
                n_groups = res.get("n_groups")
            source = "independent"

        assays_report[report_key] = {
            "assay_metric": metric,
            "assay_family": r0.get("assay_family"),
            "n": n,
            "n_groups": n_groups,
            "model": source,
            "spearman": spearman,
        }

    # Serving view: an assay must be honestly evaluable (>= MIN_ROWS) and positive.
    def _served(a: dict) -> bool:
        return (a["spearman"] is not None and a["n"] >= MIN_ROWS_PER_ASSAY
                and a["spearman"] >= 0)

    scored = [a["spearman"] for a in assays_report.values() if a["spearman"] is not None]
    served = [a["spearman"] for a in assays_report.values() if _served(a)]
    small_served = [a["spearman"] for a in assays_report.values()
                    if _served(a) and a["model"] == "pooled"]

    return {
        "head": "antibody_aggregation_hybrid",
        "features": "plm+biophysical" if use_plm else "biophysical_only",
        "route_threshold": route_threshold,
        "pool_rows": len(rows),
        "n_assays": len(assays_report),
        "mean_spearman_all_scored": round(mean(scored), 4) if scored else None,
        "mean_spearman_served": round(mean(served), 4) if served else None,
        "served_assays": len(served),
        "small_assays_rescued": len(small_served),
        "assays": assays_report,
    }


def _print(report: dict) -> None:
    print(f"Hybrid Head B  ({report['features']}, pool={report['pool_rows']} rows, "
          f"route<{report['route_threshold']} -> pooled)")
    print(f"  mean Spearman (served)     : {report['mean_spearman_served']}  "
          f"over {report['served_assays']} assays")
    print(f"  mean Spearman (all scored) : {report['mean_spearman_all_scored']}")
    print(f"  small assays rescued       : {report['small_assays_rescued']}")
    print()
    print(f"  {'assay':<40}{'n':>5}{'model':>13}{'rho':>8}")
    for k, a in sorted(report["assays"].items(),
                       key=lambda kv: -(kv[1]["spearman"] if kv[1]["spearman"] is not None else -9)):
        rho = f"{a['spearman']:+.3f}" if a["spearman"] is not None else "  n/a"
        print(f"  {k[:39]:<40}{a['n']:>5}{a['model']:>13}{rho:>8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hybrid (pooled-for-small, independent-for-large) Head B eval")
    p.add_argument("--use-plm", action="store_true")
    p.add_argument("--route-threshold", type=int, default=ROUTE_THRESHOLD)
    p.add_argument("--no-gdpa", action="store_true")
    p.add_argument("--out", default="results/head_b_hybrid_report.json")
    args = p.parse_args()

    report = run_hybrid(use_plm=args.use_plm, route_threshold=args.route_threshold,
                        include_gdpa=not args.no_gdpa)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(report)
    print(f"\nwrote {args.out}")
