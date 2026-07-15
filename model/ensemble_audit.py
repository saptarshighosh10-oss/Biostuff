"""Audit label quality, source contribution, and PLM/biophysical blending."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from data.cohort import build_antibody_cohort
from model.autonomous_lab import _profile_rows
from model.head_b_gbm import spearman_rho


def _report(root: Path, profile: str, variant: str) -> dict:
    return json.loads((root / f"{profile}_{variant}.json").read_text())


def _oof(item: dict) -> dict[int, dict]:
    return {int(row["row_index"]): row for row in item.get("oof_predictions", [])}


def _ensemble_profile(root: Path, profile: str) -> dict:
    plm = _report(root, profile, "plm")
    bio = _report(root, profile, "biophysical")
    by_key = {}
    for key, plm_assay in plm.get("assays", {}).items():
        bio_assay = bio.get("assays", {}).get(key)
        if not bio_assay:
            continue
        left, right = _oof(plm_assay), _oof(bio_assay)
        shared = sorted(set(left) & set(right))
        if len(shared) < 10:
            continue
        truth = [left[i]["endpoint_value"] for i in shared]
        plm_pred = [left[i]["prediction"] for i in shared]
        bio_pred = [right[i]["prediction"] for i in shared]
        scores = {}
        for step in range(11):
            weight = step / 10
            blended = [weight * a + (1 - weight) * b for a, b in zip(plm_pred, bio_pred)]
            scores[str(weight)] = round(spearman_rho(truth, blended), 6)
        best_weight = max(scores, key=scores.get)
        by_key[key] = {
            "active_both": plm_assay.get("serving_status") == "active" and bio_assay.get("serving_status") == "active",
            "plm_spearman": plm_assay.get("spearman"),
            "biophysical_spearman": bio_assay.get("spearman"),
            "best_blend_weight_plm": float(best_weight),
            "best_blend_spearman": scores[best_weight],
            "weights": scores,
        }
    active = [item for item in by_key.values() if item["active_both"]]
    return {
        "assays": by_key,
        "active_both_count": len(active),
        "mean_plm": round(sum(x["plm_spearman"] for x in active) / len(active), 6) if active else None,
        "mean_biophysical": round(sum(x["biophysical_spearman"] for x in active) / len(active), 6) if active else None,
        "mean_best_blend": round(sum(x["best_blend_spearman"] for x in active) / len(active), 6) if active else None,
        "warning": "blend weights are selected on OOF predictions and require nested validation before promotion",
    }


def _label_conflicts(rows: list[dict]) -> dict:
    within_assay = defaultdict(set)
    cross_assay = defaultdict(set)
    for row in rows:
        sequence = str(row.get("variant_sequence") or "").strip().upper()
        metric = str(row.get("assay_metric") or "")
        direction = str(row.get("endpoint_direction") or "")
        assay = str(row.get("assay_id") or "")
        value = row.get("endpoint_value")
        if sequence and metric and value is not None:
            numeric = round(float(value), 8)
            within_assay[(sequence, assay, metric, direction)].add(numeric)
            cross_assay[(sequence, metric, direction)].add(numeric)
    return {
        "within_assay_groups": len(within_assay),
        "within_assay_conflicts": sum(len(values) > 1 for values in within_assay.values()),
        "cross_assay_groups": len(cross_assay),
        "cross_assay_variation": sum(len(values) > 1 for values in cross_assay.values()),
    }


def run(out: str | Path = "results/ensemble_audit.json") -> dict:
    root = Path("results/autonomous_lab")
    rows = build_antibody_cohort()
    payload = {
        "label_quality": _label_conflicts(rows),
        "profiles": {profile: _ensemble_profile(root, profile) for profile in ("igg", "gdp4_mab", "gdp4_bispecific")},
        "source_quality": {},
    }
    for profile in ("igg", "gdp4_mab", "gdp4_bispecific"):
        report = _report(root, profile, "plm")
        by_source = defaultdict(list)
        for assay in report.get("assays", {}).values():
            if assay.get("serving_status") == "active" and assay.get("spearman") is not None:
                by_source[assay.get("source", "unknown")].append(assay["spearman"])
        payload["source_quality"][profile] = {
            source: {"active_assays": len(values), "mean_spearman": round(sum(values) / len(values), 6)}
            for source, values in by_source.items()
        }
    Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
