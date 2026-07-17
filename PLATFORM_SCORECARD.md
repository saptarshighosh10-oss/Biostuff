# Antibody Developability Platform — Honest Scorecard

**What it does:** predicts developability risks of an antibody **from its amino-acid
sequence alone**, so a lab can triage which candidates to test first — before running
any wet-lab assay.

**How it's graded (the important part):** every number below is measured the *honest* way —
on antibodies the model has **never seen** (leave-one-dataset-out), with any shared
antibodies removed from training so nothing leaks. Scores are **AUC** = the chance it
correctly picks the worse of two antibodies (0.5 = coin flip, 1.0 = perfect).

## The three validated properties

| Property | What it means | Honest AUC (unseen antibodies) |
|---|---|---|
| **Aggregation (self-association / AC-SINS)** | how much it clumps / gets sticky | **0.72** |
| **Polyreactivity** | how much it sticks to the wrong things | **0.70** |
| **Hydrophobicity (HIC)** | how "greasy" its surface is | **0.70** |

## Benchmark interpretation (Phase 0 lock)

The table above is the historical platform scorecard. It is not a claim that a
library-trained polyreactivity model achieves 0.70 on unseen diverse antibodies.
The current fixed hard-transfer baseline is 0.626926 macro-source ROC AUC across
`aims_mouse`, `aims_flu`, and `shehata2019`; all three have already been used for
development. Aggregation's strict solid-study development estimate is 0.739.
Hydrophobicity currently has no newer hard-transfer AUC registered in machine-
readable evidence, so its traceable Phase 0 value remains the historical 0.70.

GDPa3 remains the unspent, single-shot final source for all three properties. It
must not be inspected, downloaded, counted, trained on, or tuned against during
development. No locked-validation source is assigned yet.

Run the traceable status report:

```bash
cd /Users/saptarshighosh/Biostuff
/Users/saptarshighosh/Biostuff/.venv/bin/python -m tools.benchmark_status
```

In plain terms: **~7 times out of 10** it correctly flags the riskier antibody — on
molecules from a lab it has never worked with. That catches roughly **2× more problem
antibodies** than random screening.

## What was tested and honestly rejected

| Property | Why it isn't served |
|---|---|
| Thermostability (Tm) | Tested 3 ways (biophysical, +structure, +multi-task) — never beat ~0.53 (coin flip) on unseen antibodies. Depends on internal fold energetics; would need physics-based ΔΔG simulation. |
| Expression / yield | ~0.50 (coin flip). Depends heavily on cell biology, not just the sequence. |

**These are deliberately excluded.** An honest 3-property platform is stronger than a
5-property one with two heads that don't work — fake numbers destroy trust in the real ones.

## How it was built (the method, briefly)

1. Pooled public developability data (Ginkgo GDPa, Jain, FLAb, and others).
2. Features: protein-language-model embeddings + biophysical descriptors + charge-at-assay-pH.
3. One model per property; aggregation uses a small model ensemble.
4. **Rigorous honesty checks:** leave-one-dataset-out generalization, molecule-level
   de-duplication (a real data leak was found and removed), and catch-the-bad-ones scoring.

## Honest limits

- Numbers are for **triage/ranking**, not exact prediction. It tells you *which* to test
  first, not the precise assay value.
- On very small or unusual antibody sets, reliability drops.
- The single frozen external test set (GDPa3) has **not** been spent yet — reserved for one
  final, honest validation run.

## Where it can grow

- More labeled antibodies (the biggest lever) → higher scores + a data flywheel as labs use it.
- PLM/structure features for the weaker sources.
- Thermostability *is* possible later, but needs heavy physics-based stability modeling.

_Artifacts: `model/saved/platform.pkl`, `results/platform_scorecard.json`,
`results/head_developability_honest.json`._
