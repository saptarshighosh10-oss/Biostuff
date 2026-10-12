# Next task for Codex — pivot from data to FEATURES

Great overnight run: HIC generalization went 0.671 -> 0.697 (gate-verified on the Shehata
holdout), and you correctly rejected mirages and a leakage overlap (Boughter). Thank you.

## Stop hunting — the data well is dry (your own digest proves it)
You checked ~55 leads; the honest conclusion is that open, full-sequence, measured antibody
data is exhausted (the rest is figure-only, CDR-only, anonymized, duplicate, or bot-blocked).
Do NOT keep hunting — it's wasted effort now. Only process the 3 items in human_queue.csv IF
the human downloads them. Otherwise, pivot.

## The real remaining lever: CDR-aware features
Diverse polyreactivity is stuck at ~0.57 and AC-SINS at ~0.72 because our current features are
whole-sequence averages that CANNOT see the CDR loops — and polyreactivity/self-association are
driven by the CDRs (especially CDR-H3). More data won't fix a feature blind spot. Better
features might.

### Build this (additive, new file — do NOT modify model/head_*.py or head_platform.py)
1. New module (e.g. model/cdr_features.py): given a VH (and VL) sequence, extract the CDRs
   (use a lightweight scheme — ANARCI if available, else a Kabat/Chothia regex or the
   FR/CDR split already present in some datasets). Compute per-CDR descriptors:
   length, net charge, hydrophobicity (Kyte-Doolittle mean), aromatic fraction, and a
   CDR-H3-specific set (it matters most). ~15-25 new numeric features.
2. A/B TEST honestly, same as always: re-run the diverse polyreactivity gate (holdout =
   aims_mouse, aims_flu, shehata2019) and the AC-SINS gate, WITH vs WITHOUT the CDR features.
   Report baseline -> trial diverse AUC for each.
3. Gate rule unchanged: keep the features only if they hold or improve the held-out
   generalization AUC (>= baseline - 0.005). If they help, wire them into the relevant heads
   and refresh platform.pkl. If they don't, log it honestly and leave the heads as-is.

## Rules (unchanged — obey INGEST_RULES.md)
- Additive only; never modify Claude's model files. Judge by the held-out generalization AUC,
  never a padded mean. Never fabricate. Never touch GDPa3. Fix mechanical bugs yourself and
  keep going; never fix by lowering the honesty bar.

## Also: commit your overnight work cleanly
Your accepted-data + new HIC/PSR heads + platform.pkl refresh should be committed (your own
files) so nothing is lost. Then start the CDR-feature task above.

## Report back
- Whether CDR features moved diverse polyreactivity (past ~0.57?) and/or AC-SINS (past 0.72?).
- Honest baseline->trial numbers. A null result ("CDR features didn't help, here's the number")
  is a valid, useful outcome — report it straight.
