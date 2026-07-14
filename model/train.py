"""
Training orchestrator.

Data sources (any combination):
  --input        results/labeled_results.json   your own wet-lab labeled data
  --flab                                        FLAb experimental aggregation datasets
  --antiref                                     AntiRef human germline negatives
  --canya                                       CANYA nucleation peptides
  --figshare-agg                                Figshare Aggrescan3D structural scores
  --abdev                                       AbDev clinical-stage antibody negatives
  --sabdab                                      SAbDab structural antibody negatives
  --anchors                                     your own Phase 1 PDB anchor chains
  --proteingym                                  Use the trained ProteinGym head as one derived feature

Output: model/saved/model.pkl

Usage:
    # train the antibody model with the separate ProteinGym-derived feature
    python -m model.pretrain_proteingym_fitness
    python -m model.train --flab --abdev --anchors --proteingym

    # add your own wet-lab data on top
    python -m model.train --flab --abdev --anchors --input results/labeled_results.json
"""

import hashlib
import json
import argparse
from pathlib import Path

from model.failure_model import AggregationFailureModel, MIN_TRAINING_FAILURES
from model.features import extract_features, features_to_vector, FEATURE_NAMES

SAVE_DIR = Path("model/saved")
EXPERIMENT_LOG = Path("results/experiment_log.jsonl")


def _log_experiment(kwargs: dict, model: "AggregationFailureModel") -> None:
    """
    Append one line to results/experiment_log.jsonl recording this run's data
    sources and results, so AUC/calibration can be tracked as a trajectory
    across runs instead of a single snapshot number.
    """
    import datetime

    sources = {k: v for k, v in kwargs.items() if k.startswith("use_") and v}
    cv = model.cv_scores
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sources": sorted(sources.keys()),
        "n_failures": model.n_failures,
        "n_working": model.n_working,
        "n_groups": model.n_groups,
        "used_grouped_cv": model.used_grouped_cv,
        "lr_auc_mean": round(float(cv["logistic_regression_auc"].mean()), 4),
        "rf_auc_mean": round(float(cv["random_forest_auc"].mean()), 4),
        "best_auc": round(float(max(cv["logistic_regression_auc"].mean(), cv["random_forest_auc"].mean())), 4),
        "calibration_method": model.calibration_method,
    }
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nLogged this run to {EXPERIMENT_LOG}")


def print_experiment_log(log_path: str = str(EXPERIMENT_LOG)) -> None:
    """Print the training run history as a table, most recent last."""
    p = Path(log_path)
    if not p.exists():
        print(f"No experiment log yet at {log_path} — run training first.")
        return

    with open(p) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        print("Experiment log is empty.")
        return

    print(f"{'timestamp':20s} {'failures':>8s} {'working':>8s} {'groups':>7s} {'best_auc':>8s}  sources")
    print("-" * 90)
    for r in rows:
        print(f"{r['timestamp']:20s} {r['n_failures']:>8d} {r['n_working']:>8d} "
              f"{str(r.get('n_groups', '')):>7s} {r['best_auc']:>8.3f}  "
              f"{','.join(s.replace('use_', '') for s in r['sources'])}")


def _group_id_for(entry: dict) -> str:
    """
    Cross-validation group key: entries sharing a group_id (mutants of the same
    wild-type/assay, or rows from the same study) must never be split across
    train/test folds, or CV performance is inflated by near-duplicate leakage.
    Loaders that group mutants (proteingym, flab, anchors) set "group_id"
    explicitly; everything else falls back to a hash of its own sequence,
    which is safe — each such entry is an independently-sourced protein with
    no siblings to leak against.
    """
    gid = entry.get("group_id") or entry.get("dataset") or entry.get("anchor_pdb")
    if gid:
        return str(gid)
    seq = entry.get("variant_sequence", "")
    return hashlib.md5(seq.encode()).hexdigest() if seq else "unknown"


def _identity_for(entry: dict, idx: int) -> dict:
    """
    Human-readable identity for one training row, used to power
    nearest_neighbors() lookups at inference — "this candidate most
    resembles known-working protein X".
    """
    name = (entry.get("name") or entry.get("anchor_pdb") or entry.get("group_id")
            or f"{entry.get('source', 'entry')}_{idx}")
    return {
        "name": str(name),
        "source": entry.get("source", ""),
        "sequence": entry.get("variant_sequence", ""),
    }


def load_labeled_data(input_file: str) -> tuple[list, list]:
    if not Path(input_file).exists():
        return [], []
    with open(input_file) as f:
        data = json.load(f)
    failures, working, skipped = [], [], 0
    for entry in data:
        label = entry.get("label", "").strip().lower()
        if label == "confirmed_failure":
            failures.append(entry)
        elif label == "working":
            working.append(entry)
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} entries with missing/unknown label")
    return failures, working


def build_training_data(
    input_file: str | None = None,
    use_flab: bool = False,
    use_antiref: bool = False,
    use_canya: bool = False,
    use_figshare_agg: bool = False,
    use_abdev: bool = False,
    use_sabdab: bool = False,
    use_anchors: bool = False,
    use_proteingym: bool = False,
    anchor_file: str = "results/phase1_candidates.json",
    min_failures: int = MIN_TRAINING_FAILURES,
    flab_percentile: float = 0.25,
    antiref_max: int = 500,
    canya_max: int = 2000,
    abdev_max: int = 300,
    figshare_max: int = 2000,
    sabdab_max: int = 500,
    anchor_max: int = 500,
    proteingym_assays: int = 84,
    proteingym_seed: int | None = 42,
) -> tuple[list, list, list, list] | None:
    """
    Load every requested data source, deduplicate, and featurize.
    Returns (X, y, groups, identities), or None if there isn't enough data
    to train. Shared by train() and model/significance.py so the permutation
    test runs on the exact same data a training run would use.
    """
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    all_failures: list = []
    all_working:  list = []

    # ── Source 1: user-provided labeled data ────────────────────────
    if input_file and Path(input_file).exists():
        print(f"\n[1/3] Loading labeled data from {input_file}...")
        f, w = load_labeled_data(input_file)
        print(f"  confirmed_failure: {len(f)}  |  working: {len(w)}")
        all_failures.extend(f)
        all_working.extend(w)
    else:
        print("\n[1/3] No --input file provided (or file not found) — skipping")

    # ── Source 2: FLAb experimental datasets ─────────────────────────
    if use_flab:
        print("\n[2/3] Loading FLAb aggregation datasets...")
        from data.flab import load_all_flab_data
        f, w = load_all_flab_data(percentile=flab_percentile)
        all_failures.extend(f)
        all_working.extend(w)
    else:
        print("\n[2/3] FLAb skipped (pass --flab to include)")

    # ── Source 3: AntiRef negatives ───────────────────────────────────
    if use_antiref:
        print("\n[3/3] Loading AntiRef negative sequences...")
        from data.antiref import load_antiref_negatives
        negatives = load_antiref_negatives(max_sequences=antiref_max)
        all_working.extend(negatives)
        print(f"  Added {len(negatives)} AntiRef working sequences")
    else:
        print("\n[3/3] AntiRef skipped (pass --antiref to include)")

    # ── Source 4: CANYA nucleation dataset ──────────────────────────
    if use_canya:
        print("\n[4/4] Loading CANYA nucleation data...")
        from data.canya import load_canya_data
        f, w = load_canya_data(max_sequences=canya_max)
        all_failures.extend(f)
        all_working.extend(w)
        print(f"  CANYA: {len(f)} nucleators (failures), {len(w)} non-nucleators (working)")
    else:
        print("\n[4/4] CANYA skipped (pass --canya to include)")

    # ── Source 5: Figshare Aggrescan3D database ───────────────────────
    if use_figshare_agg:
        print("\n[5/5] Loading Figshare A3D data...")
        from data.figshare_agg import load_figshare_agg_data
        f, w = load_figshare_agg_data(max_sequences=figshare_max)
        all_failures.extend(f)
        all_working.extend(w)
        print(f"  Figshare A3D: {len(f)} failures, {len(w)} working")
    else:
        print("\n[5/5] Figshare A3D skipped (pass --figshare-agg to include)")

    # ── Source 6: AbDev clinical antibody negatives ───────────────────
    if use_abdev:
        print("\n[6/6] Loading AbDev clinical antibody negatives...")
        from data.abdev import load_abdev_negatives
        negatives = load_abdev_negatives(max_sequences=abdev_max)
        all_working.extend(negatives)
        print(f"  AbDev: {len(negatives)} clinical antibody working sequences")
    else:
        print("\n[6/6] AbDev skipped (pass --abdev to include)")

    # ── Source 7: SAbDab structural antibody database ─────────────────
    if use_sabdab:
        print("\n[7/7] Loading SAbDab sequences...")
        from data.sabdab import load_sabdab_negatives
        negatives = load_sabdab_negatives(max_sequences=sabdab_max)
        all_working.extend(negatives)
        print(f"  SAbDab: {len(negatives)} working sequences")
    else:
        print("\n[7/7] SAbDab skipped (pass --sabdab to include)")

    # ── Source 8: Phase 1 anchor sequences (already on disk) ─────────
    if use_anchors:
        print("\n[8/9] Loading Phase 1 anchor sequences as negatives...")
        from data.anchor_negatives import load_anchor_negatives
        negatives = load_anchor_negatives(anchor_file, max_sequences=anchor_max)
        all_working.extend(negatives)
    else:
        print("\n[8/9] Anchor sequences skipped (pass --anchors to include)")

    # ── Source 9: ProteinGym Head A derived feature ────────────────────
    if use_proteingym:
        print("\n[9/9] Loading the separate ProteinGym fitness head...")
        from model.pretrain_proteingym_fitness import load_general_fitness_model
        fitness_head = load_general_fitness_model()
        antibody_entries = all_failures + all_working
        for entry in antibody_entries:
            sequence = entry.get("variant_sequence", "")
            entry["proteingym_fitness_score"] = fitness_head.score_general_fitness(sequence) if sequence else 0.0
        print(f"  ProteinGym: derived feature added to {len(antibody_entries)} antibody entries")
    else:
        print("\n[9/9] ProteinGym head skipped (pass --proteingym to add its derived feature)")

    # ── Deduplicate across sources ───────────────────────────────────
    seen: set[str] = set()
    failures_dedup, working_dedup = [], []
    for entry in all_failures:
        seq = entry.get("variant_sequence", "")
        if seq and seq not in seen:
            seen.add(seq)
            failures_dedup.append(entry)
    for entry in all_working:
        seq = entry.get("variant_sequence", "")
        if seq and seq not in seen:
            seen.add(seq)
            working_dedup.append(entry)

    print(f"\nCombined (after dedup):")
    print(f"  confirmed_failure: {len(failures_dedup)}")
    print(f"  working:           {len(working_dedup)}")

    if len(failures_dedup) < min_failures:
        print(
            f"\nNot enough failures to train ({len(failures_dedup)} < {min_failures}).\n"
            f"Try: python -m model.train --flab --abdev --anchors"
        )
        return None

    if not working_dedup:
        print("\nNo working sequences — need both classes to train.")
        return None

    # ── Extract features ─────────────────────────────────────────────
    print("\nExtracting features...")
    X, y, groups, identities = [], [], [], []
    for i, entry in enumerate(failures_dedup):
        feats = extract_features(entry)
        X.append(features_to_vector(feats))
        y.append(1)
        groups.append(_group_id_for(entry))
        identities.append(_identity_for(entry, i))
        if (len(X)) % 200 == 0:
            print(f"  {len(X)} features extracted...")
    for i, entry in enumerate(working_dedup):
        feats = extract_features(entry)
        X.append(features_to_vector(feats))
        y.append(0)
        groups.append(_group_id_for(entry))
        identities.append(_identity_for(entry, i))

    n_groups = len(set(groups))
    print(f"  Feature vector size: {len(FEATURE_NAMES)}")
    print(f"  Total samples: {len(X)} ({sum(y)} failures, {len(y)-sum(y)} working)")
    print(f"  Distinct CV groups: {n_groups}")

    return X, y, groups, identities


def train(log_run: bool = True, **kwargs) -> AggregationFailureModel | None:
    """
    Build training data (see build_training_data for all kwargs) and fit
    the model. Prints CV results, feature importances, and saves model.pkl.
    Appends a row to results/experiment_log.jsonl unless log_run=False.
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING: Aggregation Failure Model")
    print("=" * 60)

    data = build_training_data(**kwargs)
    if data is None:
        return None
    X, y, groups, identities = data

    # ── Train ────────────────────────────────────────────────────────
    print("\nTraining (logistic regression + random forest, grouped 5-fold CV)...")
    model = AggregationFailureModel()
    model.train(X, y, FEATURE_NAMES, groups=groups, identities=identities)

    cv = model.cv_scores
    lr_auc = cv["logistic_regression_auc"]
    rf_auc = cv["random_forest_auc"]
    cv_kind = (f"grouped ({model.n_groups} groups) — mutants of the same protein "
               f"never split across folds" if model.used_grouped_cv else
               "standard (too few distinct groups to group-split)")
    print(f"\nCross-validation ROC-AUC [{cv_kind}]:")
    print(f"  Logistic Regression: {lr_auc.mean():.3f} ± {lr_auc.std():.3f}")
    print(f"  Random Forest:       {rf_auc.mean():.3f} ± {rf_auc.std():.3f}")
    print(f"  Probability calibration: {model.calibration_method}")

    if lr_auc.mean() < 0.6 and rf_auc.mean() < 0.6:
        print("\n  WARNING: AUC near 0.5 — model not learning well.")
        print("  If using only FLAb: the sequence features alone may not separate")
        print("  HIC/SEC scores cleanly. Add Phase 2 multi-predictor features via")
        print("  --input with Phase 2 annotated candidates for better signal.")

    print("\nTop 10 most predictive features:")
    for name, score in model.feature_importance()[:10]:
        bar = "█" * int(score * 100)
        print(f"  {name:35s} {bar} ({score:.4f})")

    save_path = str(SAVE_DIR / "model.pkl")
    model.save(save_path)

    if log_run:
        _log_experiment(kwargs, model)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"  Failures:   {model.n_failures}")
    print(f"  Working:    {model.n_working}")
    print(f"  Best AUC:   {max(lr_auc.mean(), rf_auc.mean()):.3f}")
    print(f"  Saved to:   {save_path}")
    print(f"{'=' * 60}")
    print(f"\nRe-score your candidates:")
    print(f"  python -m model.predict --file results/phase3_candidates.json")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the aggregation failure model")
    parser.add_argument("--input",    default=None,  help="Path to labeled_results.json")
    parser.add_argument("--flab",     action="store_true", help="Include FLAb experimental data")
    parser.add_argument("--antiref",  action="store_true", help="Include AntiRef negatives")
    parser.add_argument("--canya",    action="store_true", help="Include CANYA nucleation data")
    parser.add_argument("--figshare-agg", action="store_true", help="Include Figshare Aggrescan3D data")
    parser.add_argument("--abdev",    action="store_true", help="Include AbDev clinical antibody negatives")
    parser.add_argument("--sabdab",   action="store_true", help="Include SAbDab structural antibody negatives")
    parser.add_argument("--anchors",  action="store_true", help="Include Phase 1 anchor sequences as negatives")
    parser.add_argument("--proteingym", action="store_true", help="Include ProteinGym DMS stability failures (mutants)")
    parser.add_argument("--proteingym-assays", type=int, default=84, help="Number of ProteinGym assays to load (84 = all)")
    parser.add_argument("--proteingym-seed", type=int, default=42, help="Random seed for ProteinGym sampling (vary to test robustness)")
    parser.add_argument("--anchor-file", default="results/phase1_candidates.json")
    parser.add_argument("--min-failures", type=int, default=MIN_TRAINING_FAILURES)
    parser.add_argument("--flab-percentile", type=float, default=0.25,
                        help="Score percentile cutoff for FLAb labeling (default 0.25)")
    parser.add_argument("--antiref-max", type=int, default=500,
                        help="Max AntiRef negatives to load (default 500)")
    parser.add_argument("--canya-max", type=int, default=2000,
                        help="Max CANYA sequences to load (default 2000)")
    parser.add_argument("--figshare-max", type=int, default=2000,
                        help="Max Figshare A3D sequences to load (default 2000)")
    parser.add_argument("--abdev-max", type=int, default=300,
                        help="Max AbDev sequences to load (default 300)")
    parser.add_argument("--no-log", action="store_true", help="Don't append this run to results/experiment_log.jsonl")
    parser.add_argument("--history", action="store_true", help="Print the experiment log and exit (no training)")
    args = parser.parse_args()

    if args.history:
        print_experiment_log()
        raise SystemExit(0)

    train(
        log_run=not args.no_log,
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
        min_failures=args.min_failures,
        flab_percentile=args.flab_percentile,
        antiref_max=args.antiref_max,
        canya_max=args.canya_max,
        figshare_max=args.figshare_max,
        abdev_max=args.abdev_max,
    )
