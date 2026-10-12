# Continuous research loop — 72-hour autonomous improvement directive

## Mission
Improve the honest held-out generalization of the three developability heads toward the
preregistered **0.80 macro-source AUC** goal — OR honestly establish the ceiling and *why*.
Current best (development matrix): hydrophobicity 0.757, polyreactivity 0.671, aggregation
0.526 (all-assay matrix). Baseline of record: `python -m tools.benchmark_status`.

## The loop (repeat until 0.80 is reached on a property, or ideas + time are exhausted)
1. **Hypothesize** one concrete experiment (pick the next unused idea from the menu below).
2. **Implement** it additively (new files; you MAY edit head_platform.py ONLY to serve a
   gate-passing win, minimal + reversible).
3. **Evaluate** on the FIXED held-out benchmark — the same sources, never trained on, via the
   existing harness. Report macro-source AUC + per-source.
4. **Gate:** keep only if it improves held-out generalization (>= current best - 0.005). If it
   passes, integrate + refresh platform.pkl. If not, log the honest number and revert.
5. **Log** every experiment (hypothesis, result, decision) to tools/research_log.jsonl and
   commit. Then go to step 1. Do NOT stop while unused ideas or time remain.

## Idea menu (work down it; add your own — all model-improving ideas welcome)
1. PLM fine-tune, frozen head (ESM-2 35M/150M) — the Colab runner is ready.
2. PLM fine-tune, LoRA/adapter (unfreeze a little) — more capacity, less overfit.
3. Antibody-specific PLMs: AbLang2, AntiBERTy, IgBert — likely better than ESM on antibodies.
4. Full fine-tune of a small PLM on the 373k, evaluate diverse transfer.
5. Proper CDR numbering (ANARCI) so H1/H2/H3 parse reliably (H2 was ~50% before) + per-CDR feats.
6. Two-tower (separate VH and VL encoders joined at the interface).
7. Ensembling: blend PLM-finetune + tree + biophysical; stack with out-of-fold predictions.
8. Contrastive / metric learning on polyreactivity pairs.
9. Data strategy: reweight library vs diverse; domain-adaptation; test whether diverse-only
   training (small but on-distribution) beats library-heavy training.
10. Calibration + uncertainty (already partly done) to raise enrichment at fixed screening budget.

## HARD RULES — absolute, never cross even to reach the goal
- Judge ONLY by the fixed held-out generalization AUC. Never the padded mean. Never tune against
  the holdout to chase a number — that is cheating and voids the result.
- **GDPa3 stays frozen** — never inspect, download, count, train, or tune against it. It is the
  single unspent final test.
- Never fabricate sequences/values; never read numbers off figures; molecule-dedup always.
- Reaching 0.80 by lowering the honesty bar is FAILURE, not success. "We hit the honest ceiling
  at X, here's why" is a fully acceptable, valuable end state — report it as the win it is.
- Self-fix mechanical bugs and keep going; never fix by weakening the science.
- Additive; commit often; leave a digest every few hours in tools/RESEARCH_DIGEST.md.

## Stretch goal: push toward 0.85 — but ONLY via non-overfitting gains
You MAY chase 0.85+, but a fixed-holdout number climbing is NOT proof of improvement — repeated
experiments against the same sources overfit them. To count a gain as REAL, it must pass all of:
1. **Nested confirmation source.** Reserve ONE source used ONLY for final confirmation, NEVER
   for choosing/tuning experiments. Rotate which source plays this role. A candidate improvement
   only counts if it also holds on the confirmation source it was NOT selected on.
2. **Seed-stable.** Holds across >=3 seeds (std small), like the 0.809 result did.
3. **Principled, not lottery.** The gain must come from a method with a real reason to
   generalize (better PLM/representations, more diverse data, regularization, variance-reducing
   ensembling) — not from trying many random architectures until one spikes.
4. **Multiple-testing aware.** Log every experiment tried. The more you've tried, the higher the
   bar: a +0.005 dev bump after 100 experiments is noise, not a gain.
If 0.85 cannot be reached while passing 1-4, STOP and report the honest ceiling. A real 0.80 is
worth infinitely more than an overfit 0.85. Never spend effort inflating a fixed-holdout score.

## Report cadence
Every ~3 hours append to tools/RESEARCH_DIGEST.md: experiments run, best held-out AUC per
property, what passed/failed, current distance to 0.80, and the single most promising open lead.
End state (goal reached OR time/ideas exhausted): a final honest summary.
