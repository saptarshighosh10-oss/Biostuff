"""
Visualize pipeline results.
Works with phase1, phase2, phase3, or model-scored candidates.

Usage:
    python visualize.py                                     # auto-finds best file
    python visualize.py --file results/phase3_candidates.json
    python visualize.py --file results/phase3_candidates.json --show
"""

import json
import argparse
import collections
from pathlib import Path

RESULTS_DIR = Path("results")
VIZ_DIR     = Path("results/viz")


def _load(file: str | None) -> tuple[list[dict], str]:
    """Load candidates from the best available file."""
    candidates_files = [
        "results/phase3_candidates.json",
        "results/phase2_candidates.json",
        "results/phase1_candidates.json",
    ]
    if file:
        path = file
    else:
        path = next((f for f in candidates_files if Path(f).exists()), None)
        if not path:
            raise FileNotFoundError("No candidates file found. Run the pipeline first.")
    with open(path) as f:
        return json.load(f), path


def _risk_and_uncertainty(c: dict) -> tuple[float, float | None]:
    """Extract best available risk score and confidence gap from a candidate."""
    # model-scored
    if "failure_probability" in c:
        return c["failure_probability"], c.get("confidence_gap")
    # phase3
    risk = c.get("risk", {})
    score = risk.get("db_boosted_risk", risk.get("combined_risk", 0.0))
    gap   = risk.get("disagreement_score")
    return score, gap


def plot_quadrant(candidates: list[dict], ax, source_label: str):
    """Risk vs uncertainty four-quadrant scatter."""
    import matplotlib.patches as mpatches

    xs, ys, colors, labels = [], [], [], []
    for c in candidates:
        risk, gap = _risk_and_uncertainty(c)
        if gap is None:
            gap = 0.0
        xs.append(risk)
        ys.append(gap)
        # color by quadrant
        if risk >= 0.5 and gap >= 0.25:
            colors.append("#e84545")   # high risk + uncertain → test first
        elif risk >= 0.5 and gap < 0.25:
            colors.append("#f0a500")   # high risk + confident
        elif risk < 0.5 and gap >= 0.25:
            colors.append("#2196f3")   # low risk + uncertain → edge case
        else:
            colors.append("#aaaaaa")   # low risk + confident → skip

    ax.scatter(xs, ys, c=colors, alpha=0.7, s=40, linewidths=0)
    ax.axvline(0.5,  color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.25, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_xlabel("Risk score", fontsize=11)
    ax.set_ylabel("Uncertainty (gap)", fontsize=11)
    ax.set_title(f"Risk vs Uncertainty  [{source_label}]", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    legend_items = [
        mpatches.Patch(color="#e84545", label="High risk + uncertain  ← test first"),
        mpatches.Patch(color="#f0a500", label="High risk + confident"),
        mpatches.Patch(color="#2196f3", label="Low risk + uncertain  ← edge case"),
        mpatches.Patch(color="#aaaaaa", label="Low risk + confident  ← skip"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="upper left")

    # label top 5 by combined score
    scored = sorted(
        zip(xs, ys, [c.get("anchor_pdb", "?") for c in candidates]),
        key=lambda t: t[0] * (1 + t[1]) / 2,
        reverse=True,
    )
    for x, y, pdb in scored[:5]:
        ax.annotate(pdb, (x, y), fontsize=7, ha="left",
                    xytext=(4, 4), textcoords="offset points", alpha=0.9)


def plot_risk_histogram(candidates: list[dict], ax, source_label: str):
    """Risk score distribution."""
    risks = [_risk_and_uncertainty(c)[0] for c in candidates]

    n_high   = sum(1 for r in risks if r >= 0.6)
    n_medium = sum(1 for r in risks if 0.4 <= r < 0.6)
    n_low    = sum(1 for r in risks if r < 0.4)

    ax.hist(risks, bins=30, color="#4a90d9", edgecolor="white", linewidth=0.5)
    ax.axvline(0.6, color="#e84545", linewidth=1.5, linestyle="--", label=f"high (≥0.6): {n_high}")
    ax.axvline(0.4, color="#f0a500", linewidth=1.5, linestyle="--", label=f"medium (0.4–0.6): {n_medium}")
    ax.set_xlabel("Risk score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Risk Score Distribution  [{source_label}]", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.02, 0.97, f"low (<0.4): {n_low}", transform=ax.transAxes,
            fontsize=9, va="top", color="#888888")


def plot_mutation_positions(candidates: list[dict], ax):
    """Top 15 mutated positions across all candidates."""
    pos_counts: dict[int, int] = collections.Counter()
    for c in candidates:
        for mut in c.get("mutations", []):
            pos_counts[mut[0]] += 1

    if not pos_counts:
        ax.text(0.5, 0.5, "No mutation data", ha="center", va="center",
                transform=ax.transAxes)
        return

    top = pos_counts.most_common(15)
    positions = [str(p) for p, _ in top]
    counts    = [n for _, n in top]

    bars = ax.bar(positions, counts, color="#5c6bc0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Sequence position", fontsize=11)
    ax.set_ylabel("Times mutated", fontsize=11)
    ax.set_title("Top 15 Mutated Positions", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(count), ha="center", va="bottom", fontsize=8)


def plot_substitution_types(candidates: list[dict], ax):
    """Top 12 amino acid substitution types."""
    sub_counts: dict[str, int] = collections.Counter()
    for c in candidates:
        for mut in c.get("mutations", []):
            if len(mut) == 3:
                sub_counts[f"{mut[1]}→{mut[2]}"] += 1

    if not sub_counts:
        ax.text(0.5, 0.5, "No substitution data", ha="center", va="center",
                transform=ax.transAxes)
        return

    top = sub_counts.most_common(12)
    subs   = [s for s, _ in top]
    counts = [n for _, n in top]

    # color by type: charge-changing = red, hydrophobic = blue, other = grey
    def sub_color(s):
        charged = set("DEKRH")
        hydro   = set("VILMFYW")
        orig, new = s[0], s[-1]
        if orig in charged or new in charged:
            return "#e84545"
        if orig in hydro or new in hydro:
            return "#4a90d9"
        return "#888888"

    colors = [sub_color(s) for s in subs]
    bars = ax.bar(subs, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Substitution", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Top Substitution Types", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#e84545", label="charge-changing"),
        mpatches.Patch(color="#4a90d9", label="hydrophobic"),
        mpatches.Patch(color="#888888", label="other"),
    ], fontsize=8)


def run_viz(file: str | None = None, show: bool = False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates, source_path = _load(file)
    source_label = Path(source_path).stem
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(candidates)} candidates from {source_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Antibody Aggregation Pipeline — Results", fontsize=14, fontweight="bold", y=1.01)

    plot_quadrant(candidates, axes[0][0], source_label)
    plot_risk_histogram(candidates, axes[0][1], source_label)
    plot_mutation_positions(candidates, axes[1][0])
    plot_substitution_types(candidates, axes[1][1])

    plt.tight_layout()
    out = VIZ_DIR / f"{source_label}_viz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")

    if show:
        plt.show()

    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pipeline results")
    parser.add_argument("--file", default=None, help="Candidates JSON (auto-detected if omitted)")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()
    run_viz(file=args.file, show=args.show)
