"""
Calibration reliability diagram.

Answers: "when the model says 70% likely to fail, is it actually right about
70% of the time?" Builds out-of-fold calibrated predictions across the same
grouped/stratified CV splits used elsewhere (mutants of the same protein
never leak across folds), buckets them into probability bins, and plots
predicted probability vs. observed failure rate per bin against the
diagonal (perfect calibration) line.

Uses the exact same data-loading path as model/train.py (via
build_training_data), so this reflects whatever training command you'd
actually run.

Usage:
    python -m model.calibration_plot --flab --abdev --anchors
"""

import argparse
import numpy as np


def _build_cv_splits(X_scaled, y, groups):
    """Mirrors the grouped-vs-standard CV choice in failure_model.py/significance.py."""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

    groups_arr = np.array(groups) if groups is not None else None
    n_failures = int(y.sum())
    n_splits = min(5, n_failures)
    n_unique_groups = len(np.unique(groups_arr)) if groups_arr is not None else 0

    if groups_arr is not None and n_unique_groups >= 2 and n_unique_groups >= n_splits:
        cv = StratifiedGroupKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
        return list(cv.split(X_scaled, y, groups=groups_arr)), f"grouped ({n_unique_groups} groups)"
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
    return list(cv.split(X_scaled, y)), "standard"


def compute_reliability_data(X: list, y: list, groups: list | None, n_bins: int = 10):
    """
    Returns (bin_mean_pred, bin_obs_rate, bin_counts, oof_probs, oof_labels).
    Every sample gets exactly one out-of-fold calibrated prediction (it's
    never part of the data used to fit or calibrate the model that scores it).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv_splits, cv_kind = _build_cv_splits(X_scaled, y, groups)
    print(f"CV kind: {cv_kind}, {len(cv_splits)} folds")

    oof_probs = np.zeros(len(y))
    oof_seen = np.zeros(len(y), dtype=bool)

    for fold_i, (train_idx, test_idx) in enumerate(cv_splits):
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)

        method = "isotonic" if len(train_idx) >= 1000 else "sigmoid"
        lr_cal = CalibratedClassifierCV(lr, method=method, cv=3)
        rf_cal = CalibratedClassifierCV(rf, method=method, cv=3)
        lr_cal.fit(X_scaled[train_idx], y[train_idx])
        rf_cal.fit(X[train_idx], y[train_idx])

        lr_p = lr_cal.predict_proba(X_scaled[test_idx])[:, 1]
        rf_p = rf_cal.predict_proba(X[test_idx])[:, 1]
        oof_probs[test_idx] = (lr_p + rf_p) / 2.0
        oof_seen[test_idx] = True
        print(f"  fold {fold_i + 1}/{len(cv_splits)} done")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mean_pred, bin_obs_rate, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        upper = (oof_probs <= hi) if i == n_bins - 1 else (oof_probs < hi)
        mask = oof_seen & (oof_probs >= lo) & upper
        if mask.sum() == 0:
            continue
        bin_mean_pred.append(float(oof_probs[mask].mean()))
        bin_obs_rate.append(float(y[mask].mean()))
        bin_counts.append(int(mask.sum()))

    return bin_mean_pred, bin_obs_rate, bin_counts, oof_probs[oof_seen], y[oof_seen]


def plot_reliability(bin_mean_pred, bin_obs_rate, bin_counts, out_path: str = "results/viz/calibration.png") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})

    ax1.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax1.plot(bin_mean_pred, bin_obs_rate, "o-", color="#3B4EA8", label="this model")
    ax1.set_xlabel("Predicted failure probability")
    ax1.set_ylabel("Observed failure rate")
    ax1.set_title("Calibration reliability diagram")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()

    ax2.bar(range(len(bin_counts)), bin_counts, color="#8B8FA0")
    ax2.set_xlabel("Probability bin")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Samples per bin")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved calibration plot to {out_path}")
    return out_path


def run_calibration_check(**train_kwargs) -> dict | None:
    from model.train import build_training_data

    print("=" * 60)
    print("CALIBRATION CHECK: is 'X% risk' actually right X% of the time?")
    print("=" * 60)

    data = build_training_data(**train_kwargs)
    if data is None:
        print("\nNot enough data to run the calibration check.")
        return None
    X, y, groups, _identities = data

    bin_mean_pred, bin_obs_rate, bin_counts, _oof_probs, _oof_labels = compute_reliability_data(X, y, groups)

    total = sum(bin_counts)
    mace = (sum(abs(p - o) * c for p, o, c in zip(bin_mean_pred, bin_obs_rate, bin_counts)) / total
            if total else None)

    print(f"\n{'=' * 60}")
    print("RESULT")
    print(f"{'=' * 60}")
    if mace is not None:
        print(f"  Mean calibration error (bin-size weighted): {mace:.3f}")
        if mace < 0.05:
            print("  Well calibrated — predicted probabilities closely match observed rates.")
        elif mace < 0.12:
            print("  Reasonably calibrated, with some drift in a few bins.")
        else:
            print("  Poorly calibrated — treat predicted probabilities as a ranking, not a percentage.")
    plot_reliability(bin_mean_pred, bin_obs_rate, bin_counts)

    return {
        "bin_mean_pred": bin_mean_pred,
        "bin_obs_rate": bin_obs_rate,
        "bin_counts": bin_counts,
        "mean_calibration_error": mace,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a calibration reliability diagram for the trained model")
    parser.add_argument("--input", default=None)
    parser.add_argument("--flab", action="store_true")
    parser.add_argument("--antiref", action="store_true")
    parser.add_argument("--canya", action="store_true")
    parser.add_argument("--figshare-agg", action="store_true")
    parser.add_argument("--abdev", action="store_true")
    parser.add_argument("--sabdab", action="store_true")
    parser.add_argument("--anchors", action="store_true")
    parser.add_argument("--proteingym", action="store_true")
    parser.add_argument("--proteingym-assays", type=int, default=84)
    parser.add_argument("--proteingym-seed", type=int, default=42)
    parser.add_argument("--anchor-file", default="results/phase1_candidates.json")
    parser.add_argument("--flab-percentile", type=float, default=0.25)
    args = parser.parse_args()

    run_calibration_check(
        input_file=args.input,
        use_flab=args.flab,
        use_antiref=args.antiref,
        use_canya=args.canya,
        use_figshare_agg=args.figshare_agg,
        use_abdev=args.abdev,
        use_sabdab=args.sabdab,
        use_anchors=args.anchors,
        use_proteingym=args.proteingym,
        proteingym_assays=args.proteingym_assays,
        proteingym_seed=args.proteingym_seed,
        anchor_file=args.anchor_file,
        flab_percentile=args.flab_percentile,
    )
