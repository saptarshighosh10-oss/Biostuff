You are adding hydrophobicity (HIC) data to an antibody model in ~/Biostuff, run with
~/Biostuff/.venv/bin/python. Assume I'm a beginner. Be rigorously honest — real numbers only,
never fabricate, if it errors paste the exact error and stop.

## READ FIRST — rules (already-made mistakes; don't repeat)
- Higher HIC retention time = MORE hydrophobic = worse (label direction "higher_bad").
- Keep properties separate: this is HYDROPHOBICITY only. Ignore PSR and Tm columns.
- Judge by the honest per-source AUC, never a padded mean.
- Do NOT modify files under ~/Biostuff/model/. Only add your own script + a new data CSV.
- This is a DIVERSE dataset (400 different antibodies) — the good kind.

## The data (already downloaded)
~/Downloads/mmc2.xlsx  (Shehata et al. 2019 Data S1, 400 B-cell-derived mAbs).
Sheet1, row 1 = headers. Relevant columns (match by header NAME, not index):
  "HIC retention time (min)"  -> the hydrophobicity value (label)
  "VH Protein"                -> full VH amino-acid sequence
  "VL Protein"                -> full VL amino-acid sequence
Keep only rows with a numeric HIC and standard-20-AA VH length >= 20. De-dup by VH.

## Build ~/Downloads/antibody_sources/consolidate_shehata_hic.py
1. Read mmc2.xlsx, extract vh, vl, hic per antibody.
2. Write ~/Downloads/antibody_sources/hydrophobicity_dataset.csv with columns:
     source, vh, vl, label, label_type
   rows: source="shehata2019", vh, vl, label=HIC value, label_type="hic_retention_min".
   (Create the file if missing; if it exists, append and preserve existing rows.)
3. Print how many antibodies added and the HIC range.

## Then evaluate it honestly (add model/eval_hydrophobicity.py)
Reuse the SAME featurization and CV as the polyreactivity head:
  from model.head_b_gbm import build_feature_vector
  from model.head_b_acsins_charge import charge_features
  # per-row features = build_feature_vector(row, use_plm=False) + charge_features(row)
  # row dict keys: variant_sequence (=vh), vh_sequence, vl_sequence
For the shehata2019 source: within-source StratifiedKFold CV. Binarize "bad" = top 15% highest
HIC (most hydrophobic). Train LightGBM (regressor or classifier), compute ROC AUC for flagging
the most-hydrophobic ones, and catch-rate@15%.
Print: "shehata2019 hydrophobicity (400 diverse antibodies): AUC=X.XXX".

## Why we expect this to WORK (unlike polyreactivity)
Hydrophobicity is driven by whole-sequence hydrophobic content — which our biophysical features
DO capture (unlike polyreactivity, which lives in the CDRs and scored ~0.50). So this diverse
set has a real chance of a good number. But report honestly whatever it is.

## Report back
- antibodies added + HIC range
- the shehata2019 hydrophobicity AUC
