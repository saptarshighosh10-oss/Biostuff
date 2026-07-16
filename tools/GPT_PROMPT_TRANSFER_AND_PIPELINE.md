You are building two things in the repo at ~/Biostuff, run locally with
~/Biostuff/.venv/bin/python. Assume I'm a beginner. Write clean, documented code. Be
rigorously honest: report real numbers; if something errors, paste the exact error and stop —
do not hack around it or fabricate results.

## Background you must understand
We predict antibody polyreactivity from sequence. Data lives in
~/Downloads/antibody_sources/polyreactivity_dataset.csv (columns: source, vh, vl, label,
label_type; higher label = more polyreactive). Sources split into:
  - LIBRARY sources (variants of ~1-2 antibodies; EASY): emi, igg, iso, tessier_lib1, tessier_lib2
  - DIVERSE sources (many different antibodies; HARD, and the ones that measure real
    generalization): aims_mouse, aims_flu
The existing model file model/head_c_polyreactivity.py evaluates each source WITHIN itself
(easy). Reuse its featurization: for a row dict {variant_sequence, vh_sequence, vl_sequence},
features = build_feature_vector(row, use_plm=False) + charge_features(row)
  from model.head_b_gbm import build_feature_vector
  from model.head_b_acsins_charge import charge_features
Use a LightGBM classifier (same as head_c: model/head_c_polyreactivity.py _model()).

Key fact we just learned: adding 373k library sequences pushed the MEAN AUC to 0.815 but the
DIVERSE generalization AUC stayed at ~0.617. So the mean is misleading; the number that
matters is performance on the DIVERSE holdout.

## PART A — build model/head_c_transfer.py (transfer test)
Question it answers: does training on library data help predict DIVERSE (unseen-type)
antibodies?
- Train a classifier on a chosen set of TRAIN sources (default: all LIBRARY sources).
- Test on each DIVERSE source (aims_mouse, aims_flu) held out entirely.
- STRICT: remove from training any sequence (by VH) that appears in the test source.
- Report per-diverse-source ROC AUC and the mean.
- CLI: python -m model.head_c_transfer --train emi,tessier_lib1,tessier_lib2 --test aims_mouse,aims_flu
- Print a comparison line: "diverse AUC when trained on library data: X.XXX (baseline within-source diverse was ~0.617)".
Run it once with the default and report the number.

## PART B — build tools/auto_pipeline.py (ingest + quality gate + log)
A safe flywheel step: when a NEW data source is present, decide whether to KEEP it by whether
it helps GENERALIZATION — never by the misleading mean.
- Maintain a small JSON state file tools/pipeline_state.json:
    {"accepted_sources": [...], "quarantined": {source: reason}, "history": [...]}
  Seed accepted_sources with the DIVERSE sources + any library sources already trusted.
- For each source in the dataset CSV NOT yet in accepted or quarantined (a "candidate"):
    1. baseline = mean DIVERSE-holdout AUC using head_c_transfer trained on current
       accepted LIBRARY sources (tested on the diverse holdout, which is NEVER trained on).
    2. trial = same, but ADD the candidate to the training set.
    3. If trial >= baseline - 0.005 (holds or improves): mark ACCEPTED.
       Else: mark QUARANTINED with reason "did not help diverse generalization (baseline X, trial Y)".
    4. Append a timestamped record to history and print the decision.
- NEVER put a diverse holdout source into training. NEVER delete data — quarantine only.
- The gate metric is the DIVERSE-holdout AUC, NOT the within-source mean. This is the whole point.
- CLI: python -m tools.auto_pipeline   (reads the dataset, updates state, prints a report)
Run it once and report which sources it accepted vs quarantined and why.

## Scheduling (leave OFF for now)
Do NOT set up a cron/daemon. Just make auto_pipeline.py runnable on demand. I will decide the
schedule later. The discovery/hunting stays a separate agent that only APPENDS new sources to
the dataset CSV; this pipeline gate-checks them.

## Constraints
- Only ADD files: model/head_c_transfer.py, tools/auto_pipeline.py, tools/pipeline_state.json.
  Do NOT modify model/head_c_polyreactivity.py or anything else under model/.
- Honest reporting only. If the diverse holdout is tiny (aims_flu n=97), say so in the output.
- Report back: Part A's diverse AUC, and Part B's accept/quarantine decisions with the numbers.
