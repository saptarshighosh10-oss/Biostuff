# INGEST RULES — read this EVERY time before adding or checking antibody data

These are mistakes we already made or traps we already hit. Do not repeat them.

## Hard rules (breaking one wastes the whole run)
1. **A dataset is only USABLE if it has BOTH: (a) amino-acid sequences (VH, ideally VL), and
   (b) numeric measured values.** No sequences, or values only shown in a figure → reject and
   log why. Never fabricate sequences or read numbers off a chart.
2. **Keep the three properties SEPARATE — never merge their labels:**
   aggregation/AC-SINS, polyreactivity/PSR, hydrophobicity/HIC are different measurements.
3. **Judge improvement ONLY by the DIVERSE-holdout generalization score — never the overall
   mean.** WE ALREADY GOT FOOLED BY THIS: adding 373k library sequences pushed the mean AUC to
   0.815 while the real generalization number stayed 0.617. A higher mean is NOT progress.
4. **Never train on the diverse holdout** (aims_mouse, aims_flu). It is the test set.
5. **Molecule-level dedup:** remove any sequence that appears in BOTH train and test. Clinical
   antibody panels (e.g. Jain) show up in multiple datasets — this is a real leak we found.
6. **Do NOT add data for thermostability (Tm) or expression/titer** — proven not predictable
   from sequence here (~coin flip). Adding them wastes effort.
7. **GDPa3 is a frozen single-shot holdout. Never download, inspect, or train on it.**
8. **Label each new source as LIBRARY (variants of 1-2 antibodies; easy) or DIVERSE (many
   different antibodies; hard).** Diversity matters more than row count — 400 diverse
   antibodies beat 300k variants of two.
9. **Higher label = worse** (more sticky / polyreactive / hydrophobic). Set `direction` per
   assay; state your reasoning if unsure.
10. **Additive only.** Do not modify existing model/*.py files. Append to the dataset CSV and
    add new files. Never delete data — quarantine instead.

## Self-fix fixable problems — don't stop on superficial issues
If a failure is MECHANICAL (shell-quoting/glob bug, wrong URL/API format, missing dependency —
pip install it, a parser that needs adapting to the real layout, a rate-limit — retry, a
missing baseline/holdout you can build honestly): FIX IT YOURSELF and keep going. Do NOT stop
or quarantine for something that was fixable in the first place. **All changes that genuinely
improve the pipeline or the model are welcome** — make them, commit them, note them.

This never overrides the hard rules above. "Fixable" = the CODE/TOOLING, never the SCIENCE.
Never fix something by fabricating data, weakening the gate, training on a holdout, or touching
GDPa3. Fix the plumbing freely; never fudge the result.

## Output discipline
- Report REAL counts and REAL numbers. If something is missing or ambiguous, say so plainly.
- If a source turns out unusable, that is a valid, useful result — log it, don't pad.
