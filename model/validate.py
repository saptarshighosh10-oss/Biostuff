"""
Validate the trained model against a held-out benchmark of textbook cases.

Runs the model on canonical aggregators vs canonical soluble proteins —
none of which were in the training data — and reports how well it separates
them: ROC-AUC, per-class mean probability, and a full ranked table.

Usage:
    python -m model.validate
    python -m model.validate --save results/benchmark_scored.json
"""

import json
import argparse
from pathlib import Path

MODEL_PATH = "model/saved/model.pkl"


def run_validation(save_path: str | None = None) -> dict:
    from model.failure_model import AggregationFailureModel
    from model.features import extract_from_sequence, features_to_vector
    from data.benchmark import load_benchmark

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python -m model.train --flab --abdev --anchors` first."
        )

    model = AggregationFailureModel.load(MODEL_PATH)
    benchmark = load_benchmark()

    print("=" * 68)
    print("BENCHMARK VALIDATION — textbook aggregators vs soluble proteins")
    print("(none of these were in the training data)")
    print("=" * 68)

    scored = []
    for entry in benchmark:
        seq = entry["variant_sequence"]
        feat = extract_from_sequence(seq)
        vec = features_to_vector(feat)
        probs, gaps = model.predict_with_uncertainty([vec])
        scored.append({
            "name":  entry["name"],
            "label": entry["label"],
            "note":  entry["note"],
            "prob":  float(probs[0]),
            "gap":   float(gaps[0]),
            "length": len(seq),
        })

    # ── Metrics ──────────────────────────────────────────────────────────
    fail = [s["prob"] for s in scored if s["label"] == "confirmed_failure"]
    work = [s["prob"] for s in scored if s["label"] == "working"]

    mean_fail = sum(fail) / len(fail) if fail else 0.0
    mean_work = sum(work) / len(work) if work else 0.0

    # ROC-AUC via rank statistic (Mann-Whitney U) — no sklearn dependency needed
    auc = _rank_auc(fail, work)

    # ── Ranked table ─────────────────────────────────────────────────────
    scored.sort(key=lambda s: s["prob"], reverse=True)
    print(f"\n{'rank':>4}  {'protein':22}  {'true':16}  {'risk':>5}  {'gap':>5}  verdict")
    print("-" * 68)
    for i, s in enumerate(scored):
        true_lbl = "AGGREGATOR" if s["label"] == "confirmed_failure" else "soluble"
        # model's call at 0.5 threshold
        called_fail = s["prob"] >= 0.5
        correct = called_fail == (s["label"] == "confirmed_failure")
        verdict = "OK" if correct else "miss"
        print(f"{i+1:>4}  {s['name']:22}  {true_lbl:16}  "
              f"{s['prob']:>5.3f}  {s['gap']:>5.3f}  {verdict}")

    print("-" * 68)
    print(f"\nMean risk — aggregators: {mean_fail:.3f}   soluble: {mean_work:.3f}"
          f"   separation: {mean_fail - mean_work:+.3f}")
    print(f"Ranking AUC (aggregator ranked above soluble): {auc:.3f}")
    print(f"  1.0 = perfect separation, 0.5 = random, <0.5 = inverted")

    # honest interpretation
    print("\nInterpretation:")
    if auc >= 0.8:
        print("  Strong separation. The antibody-trained features transfer to")
        print("  general aggregation physics — the model ranks textbook amyloids")
        print("  well above textbook-soluble proteins.")
    elif auc >= 0.65:
        print("  Moderate separation. The model captures real signal but the")
        print("  antibody-only training limits cross-domain transfer. Expected")
        print("  to improve once real mutant failures enter via the feedback loop.")
    elif auc >= 0.5:
        print("  Weak separation. Features carry limited signal on non-antibody")
        print("  proteins — unsurprising given antibody-only training. This")
        print("  quantifies exactly why real wet-lab failures are the missing piece.")
    else:
        print("  Inverted ranking. The antibody-trained features do NOT transfer")
        print("  to these extreme cases — a concrete, honest limitation to report.")

    result = {
        "auc": auc,
        "mean_aggregator_risk": mean_fail,
        "mean_soluble_risk": mean_work,
        "separation": mean_fail - mean_work,
        "n_aggregators": len(fail),
        "n_soluble": len(work),
        "scored": scored,
    }

    if save_path:
        # write in candidate format so visualize.py can read it
        viz_ready = [{
            "anchor_pdb": s["name"],
            "chain_type": "benchmark",
            "variant_sequence": "",
            "mutations": [],
            "failure_probability": s["prob"],
            "confidence_gap": s["gap"],
            "risk": {"combined_risk": s["prob"], "disagreement_score": s["gap"]},
            "label": s["label"],
        } for s in scored]
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(viz_ready, f, indent=2)
        print(f"\nSaved scored benchmark to {save_path}")

    return result


def plot_benchmark(scored: list[dict], auc: float, out_path: str = "results/viz/benchmark_validation.png"):
    """
    Demo plot: horizontal bars sorted by risk, colored by TRUE class.
    Clean separation = aggregators (red) cluster high, soluble (blue) cluster low.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path

    ordered = sorted(scored, key=lambda s: s["prob"])
    names  = [s["name"] for s in ordered]
    probs  = [s["prob"] for s in ordered]
    colors = ["#e84545" if s["label"] == "confirmed_failure" else "#4a90d9"
              for s in ordered]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # ── Left: sorted bars colored by true class ──────────────────────────
    y = range(len(names))
    ax1.barh(list(y), probs, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("Model failure probability", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_title(f"Held-out benchmark — model never saw these\nRanking AUC = {auc:.3f}",
                  fontsize=12, fontweight="bold")
    ax1.legend(handles=[
        mpatches.Patch(color="#e84545", label="known aggregator (should score high)"),
        mpatches.Patch(color="#4a90d9", label="soluble protein (should score low)"),
    ], fontsize=9, loc="lower right")

    # ── Right: class distributions ───────────────────────────────────────
    fail = [s["prob"] for s in scored if s["label"] == "confirmed_failure"]
    work = [s["prob"] for s in scored if s["label"] == "working"]
    ax2.hist(fail, bins=8, range=(0, 1), alpha=0.6, color="#e84545",
             label=f"aggregators (mean {sum(fail)/len(fail):.2f})", orientation="horizontal")
    ax2.hist(work, bins=8, range=(0, 1), alpha=0.6, color="#4a90d9",
             label=f"soluble (mean {sum(work)/len(work):.2f})", orientation="horizontal")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Model failure probability", fontsize=11)
    ax2.set_xlabel("Count", fontsize=11)
    ax2.set_title("Score distributions by true class", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved benchmark plot to {out_path}")
    return out_path


def _rank_auc(positives: list[float], negatives: list[float]) -> float:
    """
    ROC-AUC computed as the probability a random positive outranks a random
    negative (Mann-Whitney U / 2), with 0.5 credit for ties.
    """
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model on textbook benchmark")
    parser.add_argument("--save", default=None,
                        help="Save scored benchmark JSON (for visualize.py)")
    parser.add_argument("--plot", action="store_true",
                        help="Save a benchmark separation plot to results/viz/")
    args = parser.parse_args()
    result = run_validation(save_path=args.save)
    if args.plot:
        plot_benchmark(result["scored"], result["auc"])
