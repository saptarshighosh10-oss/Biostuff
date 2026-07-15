"""Bounded local Head B hyperparameter sweep.

Every run stays grouped-CV evaluated and writes separate artifacts. This is a
diagnostic comparison; promotion remains an explicit decision.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.cohort import build_antibody_cohort
from model.autonomous_lab import _profile_rows
from model.head_b_gbm import train_head_b


CONFIGS = {
    "regularized": {
        "n_estimators": 300,
        "num_leaves": 15,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
    },
    "smooth": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 20,
        "reg_lambda": 2.0,
    },
}


def _factory(parameters: dict):
    def make():
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=42, verbose=-1, **parameters)
    return make


def _active_score(report: dict) -> tuple[float, int]:
    values = [float(item["spearman"]) for item in report.get("assays", {}).values()
              if item.get("serving_status") == "active" and item.get("spearman") is not None]
    return (sum(values) / len(values) if values else float("-inf"), len(values))


def run(out_dir: str | Path = "results/optimization_sweep") -> dict:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = build_antibody_cohort()
    summary = {}
    for profile in ("igg", "gdp4_mab", "gdp4_bispecific"):
        profile_rows = _profile_rows(rows, profile)
        summary[profile] = {}
        for name, parameters in CONFIGS.items():
            stem = f"{profile}_{name}"
            report = train_head_b(
                profile_rows,
                use_plm=True,
                regressor_factory=_factory(parameters),
                progress_every=0,
                progress_file=str(root / f"{stem}_progress.json"),
                artifact_file=str(root / f"{stem}.pkl"),
            )
            report_path = root / f"{stem}.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            score, active = _active_score(report)
            summary[profile][name] = {
                "mean_active_spearman": score,
                "active_assays": active,
                "report": str(report_path),
                "artifact": str(root / f"{stem}.pkl"),
            }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
