# Next task for Codex — verify CDR-H3 on the ensemble, finish H1/H2, integrate if it holds

CDR-H3 improved the single-model AC-SINS (0.632 -> 0.681, +0.049) but did nothing for
polyreactivity (0.627 -> 0.622), and H1/H2 were never really tested. Two jobs, then a
conditional integration.

## AUTHORIZATION (changed from before)
You ARE now allowed to modify model/head_platform.py — but ONLY for the scoped CDR-feature
integration in step 3, and ONLY if the gate says it helps. Keep the change minimal and
additive (add CDR features to the feature construction; do not alter the existing heads' logic
or the honest-eval code). Everything else stays additive.

## Job 1 — does CDR-H3 survive on the ENSEMBLE (not just the single model)?
The single-model gain may be redundant with the ensemble (charge taught us this: +0.038 single
-> +0.003 ensemble). Test it properly:
- Reuse model/head_b_acsins_ensemble.py (rank-avg ensemble, strict leave-one-study-out, the
  --ab-* A/B pattern). Add CDR-H3 features to its feature vector behind a flag and run
  baseline vs +CDR-H3 in ONE process on the honest AC-SINS holdout.
- Report baseline -> trial solid AUC. If CDR-H3 holds/improves on the ENSEMBLE (>= baseline -
  0.005), it's real. If it's absorbed like charge was, say so.

## Job 2 — actually test the polyreactivity wall with H1+H2+H3
The H1/H2 zero-fill trial was meaningless. Extract H1 and H2 from the RAW VH sequence
(dependency-free, same conserved Cys / framework-motif boundaries you used for H3 — do NOT rely
on supplied columns; compute from the VH string so every row gets real CDRs). Then re-run the
diverse polyreactivity gate (holdout = aims_mouse, aims_flu, shehata2019) with all three heavy
CDRs' descriptors. Report parse rate + baseline -> trial diverse AUC.

## Job 3 — integrate ONLY what passed
- If CDR-H3 held on the AC-SINS ensemble (Job 1): wire those features into the aggregation head
  in head_platform.py and refresh platform.pkl. Minimal, additive change.
- If H1+H2+H3 moved diverse polyreactivity (Job 2): wire them into the polyreactivity head too.
- If something did NOT pass its gate: do NOT integrate it. Log the honest number and leave that
  head unchanged. A null result is a valid outcome.

## Rules (unchanged)
Judge by the held-out generalization AUC, never a padded mean. Never fabricate; never touch
GDPa3; never train on a holdout. Fix mechanical bugs yourself; never fix by lowering the honesty
bar. Commit your own work when done.

## Report back
- Job 1: ensemble baseline -> trial AUC for CDR-H3 (did it survive?).
- Job 2: H1/H2 parse rate + polyreactivity baseline -> trial with all 3 heavy CDRs.
- Job 3: what got integrated into platform.pkl vs left unchanged, and why.
