"""Test whether CDR-aware dipeptide composition improves polyreactivity transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model import developability_multi_expert as multi
from model.benchmark_governance import load_policy, validate_policy
from model.developability_matrix import load_property_rows, make_featurizer


PROPERTY = "polyreactivity"
SEEDS = (42, 43, 44)
CURRENT_BEST = 0.6714802054330579
TOLERANCE = 0.005
FEATURE_MODE = "cdr_dipeptide"
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = tuple(a + b for a in STANDARD_AA for b in STANDARD_AA)
DIPEPTIDE_INDEX = {pair: index for index, pair in enumerate(DIPEPTIDES)}
TRIAL_SPECS = multi.CANDIDATES[PROPERTY] + (
    multi.CandidateSpec(
        "cdr_dipeptide_mixed_lgbm",
        FEATURE_MODE,
        "mixed",
        "lgbm",
    ),
)


def _normalized_dipeptides(sequence: str) -> list[float]:
    """Return deterministic normalized counts for all 400 standard dipeptides."""
    sequence = str(sequence or "").strip().upper()
    counts = [0.0] * len(DIPEPTIDES)
    denominator = max(len(sequence) - 1, 1)
    for index in range(len(sequence) - 1):
        pair_index = DIPEPTIDE_INDEX.get(sequence[index : index + 2])
        if pair_index is not None:
            counts[pair_index] += 1.0 / denominator
    return counts


def _patched_featurizer(mode: str, baseline_cache=None):
    """Extend the experiment process only; production feature modes stay unchanged."""
    if mode != FEATURE_MODE:
        return make_featurizer(mode, baseline_cache)
    cdr_featurize = make_featurizer("cdr_aware", baseline_cache)
    cache: dict[str, list[float]] = {}

    def feature(row: dict) -> list[float]:
        record_id = str(row["record_id"])
        if record_id not in cache:
            cdr = cdr_featurize(row)
            if cdr is None:
                raise RuntimeError("CDR-aware baseline unexpectedly returned no features")
            vh = str(row.get("vh") or "")
            vl = str(row.get("vl") or "")
            cache[record_id] = (
                list(cdr)
                + _normalized_dipeptides(vh)
                + _normalized_dipeptides(vl)
                + [float(bool(vl))]
            )
        return cache[record_id]

    return feature


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

    original_featurizer = multi.make_featurizer
    multi.make_featurizer = _patched_featurizer
    try:
        runs = [
            multi.nested_generalist(
                rows, held_sources, PROPERTY, seed, candidates=TRIAL_SPECS
            )
            for seed in SEEDS
        ]
    finally:
        multi.make_featurizer = original_featurizer
    for seed_run in runs:
        for source_result in seed_run["per_source"].values():
            source_result.pop("record_predictions", None)
    trial = multi.summarize_generalists(runs)[PROPERTY]["trial"]
    return {
        "schema_version": "1",
        "experiment": "polyreactivity_cdr_dipeptide",
        "hypothesis": (
            "CDR descriptors plus exact whole-chain dipeptide composition capture "
            "transferable local sequence motifs missed by global biophysics"
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
        "feature_definition": {
            "base": "cdr_aware",
            "vh_dipeptides": len(DIPEPTIDES),
            "vl_dipeptides": len(DIPEPTIDES),
            "normalization": "count divided by chain length minus one",
        },
        "limitations": [
            "AIMS Flu has only 97 held-out rows.",
            "No locked validation source is assigned.",
            "GDPa3 was not loaded or inspected.",
            "Dipeptide composition discards residue position and longer motif context.",
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
        default=Path("results/research_polyreactivity_dipeptide.json"),
    )
    args = parser.parse_args(argv)
    report = run(args.database, args.policy, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision = "ACCEPTED" if report["experimental_admit"] else "QUARANTINED"
    print(
        "Polyreactivity CDR+dipeptide experiment: "
        f"{report['baseline']['mean_macro_source_auc']:.3f} -> "
        f"{report['trial']['mean_macro_source_auc']:.3f} {decision}"
    )
    for source, auc in report["trial"]["source_mean_auc"].items():
        print(f"  {source:<22} AUC={auc:.3f}")
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
