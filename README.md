# Antibody Aggregation Failure Pipeline

A lab-in-the-loop pipeline for predicting and understanding antibody aggregation failures.

## Structure

```
data/           fetch real antibody sequences from PDB (anchor step)
features/       compute physical properties: hydrophobicity, charge, pLDDT variance
predictors/     run CamSol + ESMFold; compute disagreement score between them
labeling/       auto-label failures using two-signal requirement (diff + physical plausibility)
model/          aggregation failure predictor (trainable only after 30+ labeled failures)
pipeline.py     orchestrates all steps end-to-end
```

## Phases

**Phase 1 (now, solo, free):**
- Pull working antibody sequences from PDB
- Compute sequence + structure features
- Run CamSol and ESMFold API (no local GPU needed)
- Flag variants where predictors disagree — these are candidates for testing

**Phase 2 (needs lab access):**
- Test ~10–20 high-disagreement variants per round
- Auto-label confirmed failures using diff + physical plausibility checks
- Retrain failure model once 30+ labeled failures are collected

## Running

```bash
# Fetch PDB entries
python data/fetch_pdb.py

# Run full Phase 1 pipeline
python pipeline.py
```

## Requirements

- Python 3.10+
- `requests` (for PDB + ESMFold API calls)
- No GPU needed for Phase 1 — ESMFold runs via free API

## Design Constraints

- Auto-labels are only trusted within ≤3 mutations of a working sequence
- Two independent signals required before confirming a failure label: mutation diff + physical property shift
- Model training is blocked until ≥30 real labeled failures exist
- Held-out split is never used during training — reserved for spot-checking label quality
