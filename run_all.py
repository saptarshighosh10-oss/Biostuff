"""
One-command orchestrator — chains every phase of the pipeline end to end:
Phase 1 -> Phase 2 -> Phase 3 -> train -> predict -> significance ->
calibration -> unified report -> visualize.

Two presets:
  --mode quick   fast, small settings — good for checking the whole chain works
  --mode full    the real, maximum-data run — takes much longer (ESMFold +
                 ESM-2 + all 9 training data sources)

Usage:
    python run_all.py --mode quick
    python run_all.py --mode full
    python run_all.py --mode full --skip-phase1 --skip-phase2   # reuse existing results
"""

import argparse
import json
import time
from pathlib import Path

PRESETS = {
    "quick": {
        "entries": 100, "variants": 5, "mutations": 2, "esm2": False, "esmfold": False, "top1": 20,
        "phase2_top": 10, "phase2_md": False,
        "phase3_top": 10, "phase3_delay": 0.3,
        "train_sources": dict(use_flab=True, use_abdev=True, use_anchors=True,
                              use_proteingym=False, use_sabdab=False, flab_percentile=0.4),
        "n_permutations": 15,
        "report_top": 3,
    },
    "full": {
        "entries": 2000, "variants": 20, "mutations": 2, "esm2": True, "esmfold": True, "top1": 200,
        "phase2_top": 20, "phase2_md": False,
        "phase3_top": 50, "phase3_delay": 0.5,
        "train_sources": dict(use_flab=True, use_abdev=True, use_anchors=True, use_proteingym=True,
                              use_sabdab=True, proteingym_assays=84, flab_percentile=0.4),
        "n_permutations": 50,
        "report_top": 5,
    },
}


def _section(title: str) -> None:
    print("\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)


def _best_candidates_file() -> str | None:
    for f in ("results/phase3_candidates.json", "results/phase2_candidates.json", "results/phase1_candidates.json"):
        if Path(f).exists():
            return f
    return None


def run_all(
    mode: str = "quick",
    skip_phase1: bool = False,
    skip_phase2: bool = False,
    skip_phase3: bool = False,
    skip_train: bool = False,
    skip_predict: bool = False,
    skip_significance: bool = False,
    skip_calibration: bool = False,
    skip_report: bool = False,
    skip_visualize: bool = False,
) -> None:
    preset = PRESETS[mode]
    Path("results").mkdir(exist_ok=True)
    t0 = time.time()

    if not skip_phase1:
        _section("PHASE 1 — anchor, mutate, score, flag")
        from pipeline import run_phase1
        run_phase1(
            max_pdb_entries=preset["entries"],
            variants_per_sequence=preset["variants"],
            mutations_per_variant=preset["mutations"],
            use_esmfold=preset["esmfold"],
            use_esm2=preset["esm2"],
            top_n=preset["top1"],
        )
    else:
        print("Skipping Phase 1 (reusing existing results/phase1_candidates.json)")

    if not skip_phase2 and Path("results/phase1_candidates.json").exists():
        _section("PHASE 2 — multi-predictor consensus + explanation")
        from pipeline_phase2 import run_phase2
        run_phase2(top_n=preset["phase2_top"], run_md=preset["phase2_md"])
    elif not skip_phase2:
        print("Skipping Phase 2 — no Phase 1 output found")

    if not skip_phase3 and Path("results/phase2_candidates.json").exists():
        _section("PHASE 3 — database cross-reference")
        from pipeline_phase3 import run_phase3
        run_phase3(top_n=preset["phase3_top"], delay=preset["phase3_delay"])
    elif not skip_phase3:
        print("Skipping Phase 3 — no Phase 2 output found")

    candidates_file = _best_candidates_file()

    if not skip_train:
        _section("TRAIN — fit the failure model on public data")
        from model.train import train
        train(**preset["train_sources"])
    else:
        print("Skipping training (reusing existing model/saved/model.pkl)")

    model_exists = Path("model/saved/model.pkl").exists()

    if not skip_predict and model_exists and candidates_file:
        _section("PREDICT — re-score candidates with the trained model")
        from model.predict import score_file
        results = score_file(candidates_file, top_n=preset["phase3_top"])
        for i, r in enumerate(results[:5]):
            print(f"  {i + 1}. {r['anchor_pdb']} | prob={r['failure_probability']:.3f} ({r['risk_level']})")

    if not skip_significance and model_exists:
        _section("SIGNIFICANCE — is the AUC better than chance?")
        from model.significance import run_significance_test
        run_significance_test(n_permutations=preset["n_permutations"], **preset["train_sources"])

    if not skip_calibration and model_exists:
        _section("CALIBRATION — is 'X% risk' really X%?")
        from model.calibration_plot import run_calibration_check
        run_calibration_check(**preset["train_sources"])

    if not skip_report and model_exists and candidates_file:
        _section("REPORT — unified narrative for the top candidates")
        from model.report import build_report, render_report
        from model.failure_model import AggregationFailureModel
        with open(candidates_file) as f:
            candidates = json.load(f)
        model = AggregationFailureModel.load("model/saved/model.pkl")
        for i in range(min(preset["report_top"], len(candidates))):
            print("\n" + render_report(build_report(candidates[i], model=model)))

    if not skip_visualize and candidates_file:
        _section("VISUALIZE — results plots")
        from visualize import run_viz
        run_viz(file=candidates_file)

    elapsed_min = (time.time() - t0) / 60
    _section(f"DONE in {elapsed_min:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full antibody aggregation pipeline end-to-end")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-significance", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-visualize", action="store_true")
    args = parser.parse_args()

    run_all(
        mode=args.mode,
        skip_phase1=args.skip_phase1,
        skip_phase2=args.skip_phase2,
        skip_phase3=args.skip_phase3,
        skip_train=args.skip_train,
        skip_predict=args.skip_predict,
        skip_significance=args.skip_significance,
        skip_calibration=args.skip_calibration,
        skip_report=args.skip_report,
        skip_visualize=args.skip_visualize,
    )
