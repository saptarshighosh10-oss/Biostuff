"""Leave-one-study-out generalization test for Head B.

Cross-validation (even molecule-grouped) trains and tests within the SAME studies,
so it measures in-distribution ranking. The honest question — "how well does it
work on a random unseen set from a new lab?" — needs a harder test: hold out an
ENTIRE study, train on all the others, and predict the held-out study cold.

Design choices that make it honest:
  - Features ONLY (no assay-identity one-hot). A brand-new study's assay is unknown
    at predict time, so the model must generalise from sequence/biophysics alone —
    exactly the real deployment condition.
  - Targets are per-assay percentile ranks; Spearman is rank-based, so the held-out
    study's raw scale never needs to match training.
  - A held-out study's assay TYPE must also appear in training (else there is no
    signal to transfer and the test is meaningless) — reported as `train_support`.

This is the free, repeatable proxy for the frozen GDPa3 single-shot holdout.

Run:
  python -m model.head_b_generalization --use-plm --out results/head_b_generalization.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from model.head_b_gbm import _assay_key, build_feature_vector, MIN_ROWS_PER_ASSAY
from model.head_b_pooled import _pool_rows, _percentile_targets
from model.metrics import spearman_rho


def _study(row: dict) -> str:
    return str(row.get("study_id") or row.get("source") or "unknown")


def _canonical_metric(key: tuple) -> str:
    """Assay metric name, lowercased, for cross-study 'same assay type' matching."""
    return str(key[2] or "").lower()


def _regressor(use_plm: bool):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                         subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)


def run_generalization(use_plm: bool = True, include_gdpa: bool = True,
                       min_test_rows: int = MIN_ROWS_PER_ASSAY) -> dict:
    rows = _pool_rows(include_gdpa)
    studies = sorted({_study(r) for r in rows})

    # per-assay percentile targets computed WITHIN each (study, assay) so scales are comparable
    by_assay: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_assay[_assay_key(r)].append(i)
    y = [0.0] * len(rows)
    for key, idxs in by_assay.items():
        for i, p in zip(idxs, _percentile_targets([float(rows[i]["endpoint_value"]) for i in idxs])):
            y[i] = p

    # cache feature vectors
    base_cache: dict[str, list[float]] = {}

    def feat(i: int) -> list[float]:
        r = rows[i]
        seq = r.get("variant_sequence") or r.get("vh_sequence") or r.get("sequence", "")
        ck = f"{seq}|{use_plm}"
        v = base_cache.get(ck)
        if v is None:
            v = build_feature_vector(r, use_plm)
            base_cache[ck] = v
        return v

    # canonical metric -> set of studies that contain it (for train_support check)
    metric_studies: dict[str, set] = defaultdict(set)
    for key, idxs in by_assay.items():
        for i in idxs:
            metric_studies[_canonical_metric(key)].add(_study(rows[i]))

    report_studies: dict[str, dict] = {}
    supported_scores: list[float] = []
    for held in studies:
        train_idx = [i for i in range(len(rows)) if _study(rows[i]) != held]
        test_idx = [i for i in range(len(rows)) if _study(rows[i]) == held]
        if len(train_idx) < 50 or len(test_idx) < min_test_rows:
            continue
        model = _regressor(use_plm)
        model.fit([feat(i) for i in train_idx], [y[i] for i in train_idx])
        pred = {i: float(p) for i, p in zip(test_idx, model.predict([feat(i) for i in test_idx]))}

        # per-assay Spearman within the held-out study
        held_assays: dict[tuple, list[int]] = defaultdict(list)
        for i in test_idx:
            held_assays[_assay_key(rows[i])].append(i)
        assay_out = {}
        for key, idxs in held_assays.items():
            if len(idxs) < min_test_rows:
                continue
            metric = _canonical_metric(key)
            # is this assay type present in any TRAINING study?
            support = sorted(metric_studies[metric] - {held})
            rho = spearman_rho([float(rows[i]["endpoint_value"]) for i in idxs],
                               [pred[i] for i in idxs])
            assay_out[key[2]] = {
                "n": len(idxs),
                "spearman": round(rho, 4),
                "train_support_studies": len(support),
            }
            if support:
                supported_scores.append(rho)
        if assay_out:
            supported = [a["spearman"] for a in assay_out.values() if a["train_support_studies"] > 0]
            report_studies[held] = {
                "n_test_rows": len(test_idx),
                "assays": assay_out,
                "mean_spearman_supported": round(mean(supported), 4) if supported else None,
            }

    return {
        "head": "antibody_aggregation_leave_one_study_out",
        "features": "plm+biophysical" if use_plm else "biophysical_only",
        "note": "Honest cross-study generalization. 'supported' = assay type also seen in training.",
        "overall_mean_spearman_supported": round(mean(supported_scores), 4) if supported_scores else None,
        "n_supported_assay_evals": len(supported_scores),
        "studies": report_studies,
    }


def _print(report: dict) -> None:
    print(f"Leave-one-study-out generalization  ({report['features']})")
    print(f"  OVERALL mean Spearman (supported assays): {report['overall_mean_spearman_supported']}  "
          f"over {report['n_supported_assay_evals']} evals")
    print("  (this is the honest 'random unseen set' number — compare to CV ~0.26)")
    print()
    for study, s in sorted(report["studies"].items(),
                           key=lambda kv: -(kv[1]["mean_spearman_supported"] or -9)):
        print(f"  {study:<28} held-out mean={s['mean_spearman_supported']}  ({s['n_test_rows']} rows)")
        for metric, a in sorted(s["assays"].items(), key=lambda kv: -kv[1]["spearman"]):
            tag = "" if a["train_support_studies"] else "  (no train support — ignore)"
            print(f"      {metric:<34}n={a['n']:<4}{a['spearman']:+.3f}{tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Leave-one-study-out generalization test for Head B")
    p.add_argument("--use-plm", action="store_true")
    p.add_argument("--no-gdpa", action="store_true")
    p.add_argument("--out", default="results/head_b_generalization.json")
    args = p.parse_args()

    report = run_generalization(use_plm=args.use_plm, include_gdpa=not args.no_gdpa)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(report)
    print(f"\nwrote {args.out}")
