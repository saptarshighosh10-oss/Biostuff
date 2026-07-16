# Antibody Developability Panel

> 7/19 mastered · 37% complete

## Assay Semantics

- 🔵 **Endpoint Direction** (in progress)
  - higher\_bad
  - lower\_bad
  - per-assay scales
- 🔵 **Tm and Tonset** (in progress)
  - thermal stability
  - assay-specific interpretation
- 🔵 **HIC and SEC** (in progress)
  - hydrophobicity
  - monomer fraction
- 🔵 **Polyreactivity** (in progress)
  - PR\_CHO
  - PR\_Ova
- 🔵 **Expression Measurements** (in progress)
  - titer
  - production amount

## Data Integrity

- 🟢 **GDPa1 CSV Schema** (mastered)
  - paired VH/VL
  - native endpoints
- 🔵 **GDPa2 Spreadsheet Schema** (in progress)
  - average assay sheet
  - single sequence
- 🟢 **GDPa4 Spreadsheet Schema** (mastered)
  - mAb VH/VL sequences
  - median endpoints
- 🟢 **Sequence Pairing** (mastered)
  - VH required
  - VL optional
- ⚪ **Frozen External Validation** (unexplored)
  - GDPa3 tripwire
  - single-shot evaluation

## Model Evaluation

- 🟢 **Grouped Cross-Validation** (mastered)
  - molecule grouping
  - leakage prevention
- 🔵 **Spearman Rank Correlation** (in progress)
  - out-of-fold ranking
  - assay-local scale
- 🟢 **Serving Gate** (mastered)
  - Spearman >= 0.20
  - quarantine weak endpoints
- 🟢 **PLM Embeddings** (mastered)
  - 300 of 300 GDPa VH sequences cached
  - 15 evidence-gated active assays
- ⚪ **Calibration and Thresholds** (unexplored)
  - decision thresholds
  - prospective use

## Iteration Plan

- 🟢 **Prioritize Strong Assays** (mastered)
  - Tm2
  - HIC
  - PR
  - production amount
- 🔵 **Improve Weak Tm and SEC** (in progress)
  - PLM rescued selected Tm and SEC assays
  - other SEC endpoints remain quarantined
- ⚪ **External Validation** (unexplored)
  - GDPa3 last
  - holdout discipline
- ⚪ **Viscosity Dataset Integration** (unexplored)
  - separate local datasets
  - schema inspection
