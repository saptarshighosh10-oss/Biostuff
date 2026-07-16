You are building the "designer" (inverse / optimization model) for an antibody
developability project, in the existing repo at ~/Biostuff. Assume the person is a beginner.
Write clean, documented code they run locally. Be scientifically honest: the designer PROPOSES
variants and RANKS them with an existing predictor — it does NOT prove they work; only a lab
does. Say that clearly in the output.

## The idea
We already have a PREDICTOR (sequence -> developability risk). You build the OPPOSITE: take a
lead antibody and propose sequence changes that the predictor says REDUCE risk (stickiness /
polyreactivity / hydrophobicity) while changing as little as possible. This works well because
the predictor is strong at ranking VARIANTS OF ONE antibody (AUC ~0.95), even though it's only
~0.72 across diverse antibodies. Optimization lives in the strong regime.

## What already exists in the repo (USE these, do not reinvent)
- Trained platform: `model/saved/platform.pkl`. Structure:
    {"head":"developability_platform", "features":"plm+biophysical+charge",
     "heads": { prop: {"family": str,
                       "assays": { assay_metric: {"model": <fitted LGBMRegressor>,
                                                  "direction": "higher_bad"|"lower_bad",
                                                  "n_train": int} } } } }
  Served props: "aggregation", "polyreactivity", "hydrophobicity".
- Feature builder (import these):
    from model.head_b_gbm import build_feature_vector      # build_feature_vector(row, use_plm=True)
    from model.head_b_acsins_charge import charge_features  # charge_features(row)
  A "row" is a dict with keys: variant_sequence (=VH), vh_sequence, vl_sequence.
  Full feature vector for a sequence = build_feature_vector(row, use_plm=True) + charge_features(row).
- Use the repo venv: ~/Biostuff/.venv/bin/python

## How to score a variant (do exactly this)
For a candidate row, build its feature vector, then for each served property average that
property's assay models' predictions, orienting so HIGHER = WORSE:
  score_prop = mean over assays of ( pred if direction=="higher_bad" else -pred )
A variant is "better" than the lead if its aggregation/polyreactivity/hydrophobicity scores go
DOWN relative to the lead.

## Build: model/design_loop.py
CLI: python -m model.design_loop --vh <SEQ> --vl <SEQ> --out results/designs.csv
Steps:
1. Load platform.pkl.
2. Score the lead (VH+VL) on all three properties = baseline.
3. Generate candidates: all single-point substitutions at surface/CDR-ish positions (keep it
   simple first — you may mutate every position to a small curated set of amino acids, or all
   19; cap total candidates to a few thousand). Keep VL fixed if only VH is optimized, and
   vice versa; support optimizing either chain.
4. Score every candidate; compute delta vs the lead per property and a combined improvement.
5. Output results/designs.csv ranked by combined risk reduction: columns
   mutation, chain, position, wildtype_aa, new_aa, d_aggregation, d_polyreactivity,
   d_hydrophobicity, combined_delta, new_vh, new_vl. Negative deltas = improvement.
6. Print the top 15 and a one-line honest disclaimer.

## Honesty rules
- The predictor is a GUIDE, not truth. Print: "Predicted improvements — must be confirmed by
  assay before trusting." Do not claim any design is validated.
- Do NOT modify the predictor, the platform.pkl, or any file Claude is maintaining
  (model/head_b_*.py, model/head_developability_honest.py, model/head_platform.py). Only ADD
  model/design_loop.py.
- If a feature/import fails, inspect the actual function signatures in the repo and adapt;
  don't guess silently.

## Report back
Tell me: the file you added, an example run on any antibody VH/VL, and the top few proposed
mutations with their predicted deltas.
