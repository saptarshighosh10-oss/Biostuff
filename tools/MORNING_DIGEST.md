# Morning digest — 2026-07-16

## Decisions

- **ACCEPTED:** none. No property head or `model/saved/platform.pkl` was retrained.
- **QUARANTINED:**
  - **AbSci unlocking-de-novo** — skipped after the repository-inspection command failed before the source could be inspected. Exact error: `zsh:1: no matches found: repos/AbSciBio/unlocking-de-novo-antibody-design/git/trees/main?recursive=1`.
  - **ANDD (Zenodo)** — 48,800 antibody/nanobody sequences, but the measured endpoint is antigen binding affinity, not AC-SINS, PSR/polyreactivity, or HIC.
  - **Roche-34 relative HIC** — direct Frontiers workbook verified; 17 paired, standard-AA VH/VL HIC rows were retained (17 of 34 rows; relative-HIC range `-0.050` to `0.550`, higher = more hydrophobic). It remains excluded from training because there is no accepted HIC training source independent of the diverse Shehata holdout; a baseline→trial generalization AUC would require training on that holdout. Baseline/trial: not computable.

## Current platform artifact

`model/saved/platform.pkl` was unchanged. Its recorded leakage-free scorecard remains: aggregation AUC `0.72`, hydrophobicity AUC `0.70`, polyreactivity AUC `0.70`.

## Queues

- **Auto queue:** no rows remain with status `new` or `downloaded`.
- **Human queue:**
  - Borowska et al. 2023 — Cell Reports supplementary for polyreactivity.
  - SEC developability models (bioRxiv 2025) — supplementary URL returned HTTP 403 to curl; exact URL is in `tools/human_queue.csv`.

## Other checked lead

- The already-downloaded supplement for *Reduction of therapeutic antibody self-association using yeast-display selections and machine learning* was unusable: the machine-readable tables only contain donor characteristics and the AC-SINS results are figures.

**Biggest real gain of the night:** no model gain — the useful outcome was a verified, additive 17-row Roche HIC dataset plus a clear holdout-safe reason not to train it yet.
