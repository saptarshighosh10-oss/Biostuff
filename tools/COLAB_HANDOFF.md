# Colab folding task — brief for the assistant

**Goal:** Fold 606 antibody sequences with ESMFold on a Colab GPU and compute
structural aggregation features. Success = a `structure_features.csv` with one row
per `seq_id` and no crashes.

**Inputs (upload both to Colab):**
- `acsins_sequences_to_fold.csv` — columns `seq_id, sequence` (606 rows)
- `tools/fold_structures_colab.py` — the ready-to-run script (has step-by-step cells in its docstring)

**Environment / constraints:**
- Colab, GPU runtime (T4 is fine). Use ESMFold via HuggingFace
  (`transformers.EsmForProteinFolding`, model `facebook/esmfold_v1`) — do NOT try to
  install `openfold` or `igfold` (they break).
- The script is resumable (appends to `structure_features.csv`, skips done ids). If the
  runtime disconnects, just rerun the last cell.
- ~15–30 s per sequence → ~2–4 h total. That's expected; let it run.

**Output (bring back to the local machine):**
- `structure_features.csv` with columns: `seq_id, plddt_mean, plddt_variance,
  low_confidence_fraction, sap_score, sap_per_residue, max_hydrophobic_patch,
  surface_hydrophobic_fraction, n_res`.
- Place it locally at `Biostuff/data/external/structures/structure_features.csv`.

**Do not:** change the feature formulas or column names, split the sequence into chains,
or "improve" the model choice — the local pipeline expects exactly this schema.

**Report back:** how many of the 606 folded successfully, any that failed, and the file.
