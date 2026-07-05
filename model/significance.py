"""
Permutation significance test for the model's cross-validation AUC.

Answers: "is the CV AUC significantly better than chance, or could random
labels produce a score this high just from feature-space geometry / class
imbalance?"

Method: shuffle the failure/working labels (keeping features and CV group
structure fixed), retrain, and measure CV AUC on the shuffled labels. Repeat
many times to build a null distribution. The p-value is the fraction of
shuffled runs that matched or beat the real AUC.

Uses the exact same data-loading path as model/train.py (via
build_training_data), so the test is apples-to-apples with whatever training
sources you'd actually use. Skips probability calibration during permutation
runs — calibration is a monotonic-ish transform that doesn't change ranking/
AUC, so leaving it out keeps repeated retraining fast.

Usage:
    python -m model.significance --flab --abdev --anchors --permutations 50
"""

import argparse
import numpy as np


def permutation_test(
    X: list,
    y: list,
    groups: list | None,
    n_permutations: int = 50,
    seed: int = 0,
) -> dict:
    """
    Returns {"real_auc", "null_aucs", "null_mean", "null_std", "p_value",
    "n_permutations"}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    groups_arr = np.array(groups) if groups is not None else None

    n_failures = int(y.sum())
    n_splits = min(5, n_failures)
    n_unique_groups = len(np.unique(groups_arr)) if groups_arr is not None else 0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if groups_arr is not None and n_unique_groups >= 2 and n_unique_groups >= n_splits:
        cv = StratifiedGroupKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
        cv_splits = list(cv.split(X_scaled, y, groups=groups_arr))
        cv_kind = f"grouped ({n_unique_groups} groups)"
    else:
        cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
        cv_splits = list(cv.split(X_scaled, y))
        cv_kind = "standard"

    def _cv_auc(labels: np.ndarray) -> float:
        aucs = []
        for train_idx, test_idx in cv_splits:
            if len(np.unique(labels[test_idx])) < 2:
                continue  # AUC undefined for a single-class fold
            lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
            rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
            lr.fit(X_scaled[train_idx], labels[train_idx])
            rf.fit(X[train_idx], labels[train_idx])
            lr_p = lr.predict_proba(X_scaled[test_idx])[:, 1]
            rf_p = rf.predict_proba(X[test_idx])[:, 1]
            ensemble = (lr_p + rf_p) / 2.0
            aucs.append(roc_auc_score(labels[test_idx], ensemble))
        return float(np.mean(aucs)) if aucs else 0.5

    print(f"CV kind: {cv_kind}, {len(cv_splits)} folds")
    print("Computing real CV AUC...")
    real_auc = _cv_auc(y)
    print(f"  Real AUC: {real_auc:.4f}")

    rng = np.random.RandomState(seed)
    null_aucs = []
    print(f"\nRunning {n_permutations} label-shuffled permutations "
          f"(same features/groups, random labels)...")
    for i in range(n_permutations):
        shuffled_y = rng.permutation(y)
        null_aucs.append(_cv_auc(shuffled_y))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_permutations} done...")

    null_aucs = np.array(null_aucs)
    n_beat = int((null_aucs >= real_auc).sum())
    # add-one smoothing avoids reporting p=0 from a finite permutation sample
    p_value = float(n_beat + 1) / (n_permutations + 1)

    return {
        "real_auc": real_auc,
        "null_aucs": null_aucs.tolist(),
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std()),
        "p_value": p_value,
        "n_beat": n_beat,
        "n_permutations": n_permutations,
        "cv_kind": cv_kind,
    }


def run_significance_test(n_permutations: int = 50, seed: int = 0, **train_kwargs) -> dict | None:
    from model.train import build_training_data

    print("=" * 60)
    print("SIGNIFICANCE TEST: Is the model's AUC better than chance?")
    print("=" * 60)

    data = build_training_data(**train_kwargs)
    if data is None:
        print("\nNot enough data to run the significance test.")
        return None
    X, y, groups, _identities = data

    result = permutation_test(X, y, groups, n_permutations=n_permutations, seed=seed)

    print(f"\n{'=' * 60}")
    print("RESULT")
    print(f"{'=' * 60}")
    print(f"  Real AUC:        {result['real_auc']:.4f}")
    print(f"  Null mean ± std: {result['null_mean']:.4f} ± {result['null_std']:.4f}  "
          f"(shuffled-label baseline, should hover near 0.5)")
    print(f"  p-value:         {result['p_value']:.4f}  "
          f"({result['n_beat']}/{result['n_permutations']} permutations matched or beat the real AUC)")

    if result["p_value"] < 0.01:
        print("\n  p < 0.01 — the real AUC is very unlikely to arise from chance.")
        print("  The model has learned genuine signal, not noise.")
    elif result["p_value"] < 0.05:
        print("\n  p < 0.05 — the real AUC is unlikely to arise from chance,")
        print("  though with more permutations this would be worth re-checking.")
    else:
        print("\n  p >= 0.05 — cannot rule out that the real AUC arose from chance")
        print("  given this data. Treat the trained model's ranking with caution.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Permutation significance test for the trained model's CV AUC")
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
    parser.add_argument("--permutations", type=int, default=50,
                        help="Number of label-shuffled retrains (default 50; more = tighter p-value, slower)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_significance_test(
        n_permutations=args.permutations,
        seed=args.seed,
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
