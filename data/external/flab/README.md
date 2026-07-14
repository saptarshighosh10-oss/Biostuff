# Frozen FLAb snapshot

This directory contains a local snapshot of public FLAb assay data fetched from
the Graylab repository on 2026-07-14. It contains 159 CSV datasets and more
than 118,000 measured assay observations.

The cohorts are intentionally kept separate:

- `aggregation/`: HIC, SEC, AC-SINS, CSI-BLI, SAS, and related aggregation or
  self-association assays. This is the only cohort currently used by the
  aggregation loader.
- `binding/`: 80 binding-affinity assay datasets, retained as a separate
  endpoint family for future multi-task or representation learning.
- `expression/`, `immunogenicity/`, `pharmacokinetics/`, `polyreactivity/`,
  and `thermostability/`: measured developability endpoints for future
  assay-specific or multi-task models. They are not aggregation labels.

`SHA256SUMS` covers every CSV. Do not pool these endpoints into one binary
failure label. Train and evaluate one assay family at a time, preserving the
dataset filename as the study/group key.

Source: <https://github.com/Graylab/FLAb/tree/main/data>
