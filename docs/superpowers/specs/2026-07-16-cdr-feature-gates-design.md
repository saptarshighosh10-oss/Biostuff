# CDR feature gates

## Goal

Test whether lightweight, CDR-aware sequence descriptors improve held-out
polyreactivity and AC-SINS generalization.  The test must not change existing
`model/head_*.py` or `model/head_platform.py`.

## Inputs and extraction

`model/cdr_features.py` will accept the existing row dictionaries.  If a row
has explicitly supplied CDR fields, those sequences take precedence.  The
current polyreactivity CSV has only `source,vh,vl,label,label_type`, so its
rows use the fallback extractor.

The fallback is dependency-free and conservative:

- H3 is the sequence between the last plausible FR3 cysteine and the first
  downstream FR4 `WGxG` motif.
- H1 and H2 are attempted only for the follow-up all-heavy-CDR experiment.
- A failed parse produces zero descriptors and an explicit parse flag; it is
  never replaced with a guessed sequence.

Each parsed CDR contributes length, net charge, Kyte-Doolittle mean,
aromatic fraction, and validity flag.  The module reports per-chain parse
counts and rates.  A rate below 90% is reported as a reliability warning.

## Evaluation

New additive gate scripts will reuse the current fixed source-balanced
polyreactivity protocol and the existing separate AC-SINS holdout protocol.
Every comparison uses the same deterministic rows, strict held-VH removal,
LightGBM configuration, and holdouts as its baseline.

The polyreactivity comparison is baseline features versus baseline plus H3
descriptors on `aims_mouse` and `aims_flu`.  If H3 is within the existing
`baseline - 0.005` gate or better, a second predeclared comparison uses all
heavy CDR descriptors (H1+H2+H3).  AC-SINS receives the analogous comparison
against its existing external holdout.

## Decision and serving

Only a feature set that meets the unchanged held-out gate for the relevant
property is eligible for an additive serving tower and platform refresh.
Otherwise, no existing head or artifact is changed; the exact baseline,
trial, parse rates, and failure reason are logged in the morning digest.

## Verification

The CDR module has a small runnable self-check with representative standard
VH sequences plus a no-motif failure case.  Each gate prints its feature
variant, held-out AUCs, strict training/test sizes, and parse coverage.
