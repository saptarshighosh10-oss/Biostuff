"""Rank candidate sequences for the next informative assay round.

This uses disagreement between the promoted PLM and biophysical artifacts as
an uncertainty proxy. It is a prioritization tool, not a probability claim.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from model.head_b_gbm import build_feature_vector


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_report(root: Path, profile: str, variant: str) -> dict:
    path = root / f"{profile}_{variant}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _risk_baselines(report: dict) -> dict[tuple[str, str], list[float]]:
    baselines: dict[tuple[str, str], list[float]] = {}
    for assay in report.get("assays", {}).values():
        key = (assay.get("assay_id"), assay.get("assay_metric"))
        values = []
        for row in assay.get("oof_predictions", []):
            value = row.get("risk_score", row.get("prediction"))
            if value is not None:
                values.append(float(value))
        if values:
            baselines[key] = sorted(values)
    return baselines


def _percentile(value: float, baseline: list[float]) -> float:
    if not baseline:
        return 0.5
    return sum(item <= value for item in baseline) / len(baseline)


def _predict(model, vector: list[float]) -> float:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return float(booster.predict([vector])[0])
    return float(model.predict([vector])[0])


def rank_candidates(candidates: list[dict], profile: str = "igg") -> list[dict]:
    root = Path("results/autonomous_lab")
    plm = _load(root / f"{profile}_plm.pkl")
    bio = _load(root / f"{profile}_biophysical.pkl")
    plm_baselines = _risk_baselines(_load_report(root, profile, "plm"))
    bio_baselines = _risk_baselines(_load_report(root, profile, "biophysical"))
    plm_by_key = {(v["assay_id"], v["assay_metric"]): v for v in plm.get("assays", {}).values()}
    bio_by_key = {(v["assay_id"], v["assay_metric"]): v for v in bio.get("assays", {}).values()}
    scored: list[dict] = []
    for candidate in candidates:
        sequence = str(candidate.get("variant_sequence") or "").strip().upper()
        if not sequence:
            sequence = (str(candidate.get("vh_sequence") or "") +
                        str(candidate.get("vl_sequence") or "")).strip().upper()
        if not sequence:
            continue
        predictions = []
        for key, model_info in plm_by_key.items():
            other = bio_by_key.get(key)
            if other is None:
                continue
            vec_plm = build_feature_vector({"variant_sequence": sequence}, use_plm=True)
            vec_bio = build_feature_vector({"variant_sequence": sequence}, use_plm=False)
            plm_prediction = _predict(model_info["model"], vec_plm)
            bio_prediction = _predict(other["model"], vec_bio)
            direction = model_info.get("endpoint_direction", "higher_bad")
            if direction != "higher_bad":
                plm_prediction, bio_prediction = -plm_prediction, -bio_prediction
            plm_percentile = _percentile(plm_prediction, plm_baselines.get(key, []))
            bio_percentile = _percentile(bio_prediction, bio_baselines.get(key, []))
            predictions.append({
                "assay_id": key[0], "assay_metric": key[1],
                "plm_risk": round(plm_prediction, 6),
                "biophysical_risk": round(bio_prediction, 6),
                "plm_risk_percentile": round(plm_percentile, 6),
                "biophysical_risk_percentile": round(bio_percentile, 6),
                "disagreement": round(abs(plm_percentile - bio_percentile), 6),
            })
        if not predictions:
            continue
        predictions.sort(key=lambda item: item["disagreement"], reverse=True)
        uncertainty = sum(item["disagreement"] for item in predictions) / len(predictions)
        high_risk = sum(item["plm_risk_percentile"] for item in predictions) / len(predictions)
        scored.append({
            **candidate,
            "sequence": sequence,
            "coverage_assays": len(predictions),
            "score_scale": "per_assay_empirical_percentile",
            "uncertainty_proxy": round(uncertainty, 6),
            "mean_plm_risk": round(high_risk, 6),
            "active_learning_score": round(uncertainty * (1.0 + max(high_risk, 0.0)), 6),
            "top_disagreements": predictions[:5],
        })
    scored.sort(key=lambda item: item["active_learning_score"], reverse=True)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Head B candidates for informative testing")
    parser.add_argument("--candidates", required=True, help="JSON list of candidate sequence objects")
    parser.add_argument("--profile", default="igg")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", default="results/active_learning_candidates.json")
    args = parser.parse_args()
    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    ranked = rank_candidates(candidates, profile=args.profile)[:args.top]
    Path(args.out).write_text(json.dumps(ranked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(ranked)} ranked candidates to {args.out}")


if __name__ == "__main__":
    main()
