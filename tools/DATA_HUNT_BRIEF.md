# Data-hunter brief — antibody developability datasets

You are an autonomous agent that searches the web (papers, GitHub, Zenodo, databases) for
public antibody datasets to add to a machine-learning model. Run continuously; log finds.

## The ONE rule that matters
A dataset is only useful if it has **BOTH** of these, for the same antibodies:
1. **Amino-acid sequences** (VH and ideally VL), AND
2. **Numeric measured values** for at least one of these assays.

Most antibody papers have ONE but not the other (sequences OR values, rarely both). If a
source lacks sequences, or only shows values in a figure/plot, it is **NOT usable** — log it
as "no usable data" and move on. Do NOT fabricate or read numbers off charts.

## What we want (in priority order)
1. **Self-association / aggregation**: AC-SINS, CSI-BLI, SGAC-SINS, DLS kD, SEC.
2. **Polyreactivity / non-specificity**: PSR, BVP, ELISA polyreactivity, ovalbumin/insulin binding.
3. **Hydrophobicity**: HIC retention time.

(Thermostability/Tm and expression/titer are NOT wanted — they don't predict from sequence.)

## Where to look
- Paper "Data Availability" sections + supplementary Excel/CSV (Nature, PNAS, mAbs, MDPI, bioRxiv).
- GitHub / Zenodo repos linked from developability / antibody-ML papers.
- Known hubs: SAbDab/Thera-SAbDab, OAS (paired), Ginkgo abdev-benchmark.
- Search terms: "AC-SINS dataset sequences", "antibody self-association supplementary VH VL",
  "developability benchmark antibody github", "polyreactivity antibody dataset sequences".

## Skip (don't waste time)
- Datasets with sequences but no measured assay values (e.g. raw OAS, PLAbDab).
- Papers with only plotted results, no tables.
- Anything behind a hard paywall with no supplementary access.
- The frozen GDPa3 holdout — never touch it.

## Report format — append each find to data_hunt_log.csv
Columns: date, source_name, url, assays_found, has_sequences (yes/no),
approx_n_antibodies, download_link, usable (yes/no), notes

For USABLE finds, also note where the actual sequence+value table lives (file/URL).

## Hand-off
When you log a USABLE dataset (has_sequences=yes AND numeric values), that's a real find.
Bring the download link + table to the main project; it gets de-duplicated against existing
data and merged, then the model is re-tested. One good source with a few hundred labeled
antibodies is worth more than a thousand unlabeled sequences.
