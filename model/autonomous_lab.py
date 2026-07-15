"""Profile-specific, champion-gated Head B experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from data.cohort import build_antibody_cohort
from model.head_b_gbm import train_head_b


def _production_regressor_factory(profile: str):
    if profile not in {"igg", "gdp4_bispecific"}:
        return None

    def factory():
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1,
        )
    return factory


def _score(report: dict) -> tuple[float, int]:
    # ponytail: score only serving assays; quarantined/abstained assays are
    # diagnostic and must not influence champion selection.
    values = [float(item["spearman"]) for item in report.get("assays", {}).values()
              if item.get("serving_status") == "active" and item.get("spearman") is not None]
    return (sum(values) / len(values) if values else float("-inf"), len(values))


def _profile_rows(rows: list[dict], profile: str) -> list[dict]:
    if profile == "igg":
        return [r for r in rows if r.get("source") == "flab" or r.get("study_id") == "gdpa1"]
    if profile == "vhh":
        return [r for r in rows if r.get("study_id") == "gdpa2"]
    if profile == "gdp4_mab":
        return [r for r in rows if r.get("study_id") == "gdpa4.1"
                and r.get("construct_type") == "mAb Sequences"]
    if profile == "gdp4_bispecific":
        return [r for r in rows if r.get("study_id") == "gdpa4.1"
                and r.get("construct_type") == "bsAb Sequences"]
    raise ValueError(f"Unknown profile: {profile}")


def run_lab(run_dir: str | Path = "results/autonomous_lab") -> dict:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = build_antibody_cohort()
    summary = {"head": "antibody_aggregation", "profiles": {},
               "holdout": "GDPa3 excluded from all development profiles"}

    for profile in ("igg", "vhh", "gdp4_mab", "gdp4_bispecific"):
        profile_rows = _profile_rows(rows, profile)
        profile_result = {"rows": len(profile_rows), "candidates": {}}
        if len(profile_rows) < 20:
            profile_result["status"] = "abstain"
            profile_result["reason"] = "fewer than 20 aggregation rows"
            summary["profiles"][profile] = profile_result
            continue
        for name, use_plm in (("plm", True), ("biophysical", False)):
            artifact = root / f"{profile}_{name}.pkl"
            report = train_head_b(
                profile_rows, use_plm=use_plm, progress_every=0,
                regressor_factory=_production_regressor_factory(profile),
                progress_file=str(root / f"{profile}_{name}_progress.json"),
                artifact_file=str(artifact),
            )
            report_path = root / f"{profile}_{name}.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            profile_result["candidates"][name] = {
                "score": _score(report), "artifact": str(artifact), "report": str(report_path)
            }
        winner = max(profile_result["candidates"], key=lambda name: profile_result["candidates"][name]["score"])
        profile_result["winner"] = winner
        profile_result["winner_score"] = {
            "mean_spearman": profile_result["candidates"][winner]["score"][0],
            "fitted_assays": profile_result["candidates"][winner]["score"][1],
        }
        if profile_result["winner_score"]["mean_spearman"] <= 0:
            profile_result["status"] = "quarantined"
            profile_result["reason"] = "winner mean Spearman <= 0; not promoted for serving"
            summary["profiles"][profile] = profile_result
            continue
        if profile == "igg":
            shutil.copyfile(profile_result["candidates"][winner]["artifact"],
                            "model/saved/champion_head_b_aggregation.pkl")
            profile_result["promoted_to"] = "model/saved/champion_head_b_aggregation.pkl"
        else:
            destination = Path("model/saved") / f"{profile}_head_b_aggregation.pkl"
            shutil.copyfile(profile_result["candidates"][winner]["artifact"], destination)
            profile_result["promoted_to"] = str(destination)
        summary["profiles"][profile] = profile_result

    Path("results/autonomous_lab_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run profile-specific Head B experiments")
    parser.add_argument("--run-dir", default="results/autonomous_lab")
    args = parser.parse_args()
    print(json.dumps(run_lab(args.run_dir), indent=2, sort_keys=True))
