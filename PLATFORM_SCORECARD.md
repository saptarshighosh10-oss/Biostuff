# Antibody Developability Platform — Honest Scorecard

**What it does:** predicts developability risks of an antibody **from its amino-acid
sequence alone**, so a lab can triage which candidates to test first — before running
any wet-lab assay.

**How it's graded (the important part):** every number below is measured the *honest* way —
on antibodies the model has **never seen** (leave-one-dataset-out), with any shared
antibodies removed from training so nothing leaks. Scores are **AUC** = the chance it
correctly picks the worse of two antibodies (0.5 = coin flip, 1.0 = perfect).

## Historical served estimates

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

## Current three-seed development matrix (2026-07-17)

The new matrix uses the unified registry, source/class caps, exact VH/VL removal,
90% positional VH identity filtering for equal-length chains, train-only Platt
calibration, seeds 42/43/44, and macro source AUC. These are development results,
not locked validation.

| Property | Best development arm | Macro source AUC | Held-out source AUCs | Promotion |
|---|---|---:|---|---|
| Polyreactivity | CDR-aware, library-only, paired Fv | **0.671** | AIMS flu 0.611; AIMS mouse 0.666; Shehata 0.737 | **Fail** |
| Aggregation | Frozen PLM, diverse-only, paired Fv | **0.526** | GDPa4.1 0.546; Jain 2017 0.507 | **Fail** |
| Hydrophobicity | Biophysical + charge, mixed paired/VH training | **0.757** | GDPa1 0.769; GDPa4.1 0.811; Jain 2017 0.828; Jetha 2019 0.713; Shehata 0.665 | **Fail** |

Hydrophobicity is the closest to lab-ready and averages 2.53x enrichment at a
15% screening budget, but it is still below the preregistered 0.80 macro AUC and
has no locked-validation source. Polyreactivity also fails the minimum per-source
AUC and 2x enrichment checks. Aggregation fails the transfer bar in this comparable
all-assay matrix. Therefore `model/head_platform.py` and the serving artifact were
not changed.

The saved 66,558 held-out prediction rows were independently recomputed: 102
experiment-seed runs, 255 source evaluations, and 450 assay evaluations matched
the report and SHA-256 exactly. Across the 22,186 seed-consolidated records,
20,095 are in-domain, 1,127 borderline, and 964 out-of-domain; mean seed
disagreement is 0.0315 (95th percentile 0.0857).

## Routed multi-expert experiment (2026-07-17)

This experiment tested whether nested learner selection and sequence-only family
routing could preserve the high library-specialist results while improving
unseen-source generalization. Baseline and trial below use the exact same outer
records. They are directly comparable to each other, but not interchangeable
with differently filtered historical rows elsewhere in this scorecard.

| Property | Fixed-record baseline | Routed trial | Delta | Development decision | Serving decision |
|---|---:|---:|---:|---|---|
| Aggregation | 0.508 | 0.508 | +0.000 | Quarantined | Not eligible |
| Hydrophobicity | 0.744 | **0.756** | **+0.012** | Accepted experimentally | Not eligible |
| Polyreactivity | 0.668 | 0.666 | -0.002 | Quarantined | Not eligible |

Hydrophobicity improved on four of five held sources and held on the fifth:
GDPa1 0.706→0.706, GDPa4.1 0.810→0.815, Jain 2017 HIC
0.800→0.820, Jetha 2019 HIC 0.717→0.735, and Shehata 2019
0.685→0.703. Mean enrichment at 15% improved from 2.417x to 2.511x,
and seed-to-seed macro AUC standard deviation was 0.0072. This is a real
development gain, but it remains below the preregistered 0.80 serving bar.

Polyreactivity did not improve: AIMS flu held at 0.605, AIMS mouse declined
0.664→0.658, and Shehata held at 0.735. Aggregation selected the same
biophysical LightGBM baseline in every outer fold, so the apparent ensemble
added no value and was quarantined.

Known-family specialists were gated separately from generalists:

| Specialist | Historical AUC | Trial AUC | Conservative route coverage | Decision |
|---|---:|---:|---:|---|
| EMI | 0.948 | 0.948 | 100.0% | Accepted experimentally |
| ISO | 0.699 | 0.660 | 100.0% | Quarantined: AUC regressed |
| Tessier library 1 | 0.984 | 0.980 | 1.2% | Quarantined: router mostly dormant |
| Tessier library 2 | 0.982 | 0.980 | 2.1% | Quarantined: router mostly dormant |

IGG was not evaluated as a specialist because the deduplicated registry slice
has only 57 rows and 9 positive labels, below the preregistered minimum of 10
per class. A separate routing audit found that even 64 conservative Tessier VH
references cover only about 10–13% of those libraries at 90% identity. The
threshold was not relaxed because doing so could misroute unrelated diverse
antibodies.

The 10,428-row prediction ledger was independently recomputed from scratch:
78 source/mode and 132 assay/mode metric checks matched the JSON report exactly.
Its SHA-256 is
`98e1f5caf62217dc29666ad2970a3939384b0c64e06b78dad945e932f53c75f5`.
The experimental artifact loaded successfully and a real VH-only EMI query
routed polyreactivity to the EMI specialist while aggregation and HIC used the
diverse generalists. Across 3,476 unique held-out records, 3,321 were in-domain,
102 borderline, and 53 out-of-domain; mean seed disagreement was 0.0326 and the
95th percentile was 0.0885.

Serving unchanged: no genuinely locked validation source is assigned; GDPa3
remains untouched. The research-only outputs are
`model/saved/platform_experimental.pkl`,
`results/developability_multi_expert.json`,
`results/developability_multi_expert_predictions.csv`, and
`results/developability_multi_expert_gate.json`.

The Colab runner is ready for frozen, adapter, and full fine-tuning, but no new
GPU result is claimed here. Local preflight found 519,711 raw rows, rejected 11
invalid-sequence rows, and prepared a 24,165-row balanced library arm. AIMS flu
is explicitly reported as a tiny 97-row holdout.

Run the traceable status report:

```bash
cd /Users/saptarshighosh/Biostuff
/Users/saptarshighosh/Biostuff/.venv/bin/python -m tools.benchmark_status
```

The historical served estimates remain useful context, but promotion decisions now
use the stricter current matrix above rather than the older three-number mean.

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
`results/developability_baseline_matrix.json`,
`results/developability_baseline_predictions.csv`,
`results/developability_uncertainty.json`, and
`results/developability_promotion_gate.json`._
