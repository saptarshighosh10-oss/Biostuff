# Morning digest — 2026-07-16

## Decisions

- **ACCEPTED:** none. No property head or `model/saved/platform.pkl` was retrained.
- **QUARANTINED:**
  - **AbSci unlocking-de-novo** — the mechanical shell-quoting issue was corrected and the CSV was inspected. It has numeric AC-SINS, polyspecificity, and HIC endpoints but only HCDR3 strings, not full VH/VL sequences, so it is unusable.
  - **ANDD (Zenodo)** — 48,800 antibody/nanobody sequences, but the measured endpoint is antigen binding affinity, not AC-SINS, PSR/polyreactivity, or HIC.
  - **Roche-34 relative HIC** — moved back to `downloaded`/pending after the runbook self-fix clause was rechecked. The direct Frontiers workbook verified 17 paired, standard-AA VH/VL HIC rows (17 of 34; relative-HIC range `-0.050` to `0.550`, higher = more hydrophobic). An honest HIC baseline must be constructed before a baseline→trial decision; it must not train on the diverse Shehata holdout.

## Current platform artifact

`model/saved/platform.pkl` was unchanged. Its recorded leakage-free scorecard remains: aggregation AUC `0.72`, hydrophobicity AUC `0.70`, polyreactivity AUC `0.70`.

## Queues

- **Auto queue:** Roche-34 relative HIC remains `downloaded` and pending an honest HIC baseline/gate. No other row is `new` or `downloaded`.
- **Human queue:**
  - Borowska et al. 2023 — Cell Reports supplementary for polyreactivity.
  - SEC developability models (bioRxiv 2025) — supplementary URL returned HTTP 403 to curl; exact URL is in `tools/human_queue.csv`.

## Other checked lead

- The already-downloaded supplement for *Reduction of therapeutic antibody self-association using yeast-display selections and machine learning* was unusable: the machine-readable tables only contain donor characteristics and the AC-SINS results are figures.

**Biggest real gain of the night:** a verified, additive 17-row Roche HIC dataset; no model gain has been claimed because its holdout-safe gate is still pending.
