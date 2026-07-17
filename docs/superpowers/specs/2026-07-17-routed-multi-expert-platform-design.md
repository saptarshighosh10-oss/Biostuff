# Routed Multi-Expert Developability Platform

**Date:** 2026-07-17
**Status:** Approved design; implementation not started

## 1. Purpose

Build the strongest honest version of the antibody developability platform by
combining models that are strong on known antibody families with models that
generalize to unseen sources. The platform continues to score three separate
properties:

- aggregation / self-association;
- hydrophobicity;
- polyreactivity.

These targets must never be pooled into one biological label. "One platform"
means one inference contract containing three independently trained heads.

## 2. Why the reported AUCs differ

The high and low numbers measure different tasks and must remain separately
labelled:

- Polyreactivity within-source CV reached 0.948 on `emi`, 0.862 on `igg`,
  0.984 on `tessier_lib1`, and 0.982 on `tessier_lib2`. These are useful
  known-family specialist results.
- The aggregation unified model reached 0.850 on individual GDPa4.1 assays and
  0.756 across solid studies. Its saved rank ensemble reached 0.717 in one run
  and 0.739 in a later controlled baseline.
- The current standardized source-held-out matrix reports development macro
  AUCs of 0.526 for aggregation, 0.757 for hydrophobicity, and 0.671 for
  polyreactivity.

The first group is easier because related variants remain inside the evaluated
source. The standardized matrix holds out whole sources and strips exact and
near-identical training sequences. AUCs from these protocols cannot be averaged
or added to create a legitimate higher score.

## 3. Goals

1. Preserve specialist performance for antibody families represented in the
   training data.
2. Improve or retain source-held-out generalization for unfamiliar antibodies.
3. Choose the appropriate expert from sequence evidence, never a user-supplied
   source label.
4. Reuse the existing registry, feature builders, overlap filter, calibration,
   applicability, uncertainty, and promotion gate.
5. Produce an opt-in experimental artifact first. Keep the current serving
   artifact unchanged unless all promotion requirements pass.
6. Report per-property, per-source, specialist, generalist, routed, enrichment,
   uncertainty, and applicability results without a padded overall mean.

## 4. Non-goals

- No attempt to force every property to 0.80 or 0.90 AUC.
- No use of GDPa3 during development; it remains single-shot final evidence.
- No training on a diverse source while that source is its holdout.
- No new production dependency.
- No automatic addition of CDR-H3, structure, or paired AbLang features. Their
  existing controlled results were neutral or negative.
- No learned neural router. The number of independent sources is too small to
  support one without a high overfitting risk.

## 5. Existing components to reuse

The implementation must route through existing project code rather than create
parallel versions:

- `model/developability_matrix.py` for evidence eligibility, source balancing,
  target assignment, exact/near-VH stripping, source-held-out evaluation,
  calibration, and applicability distances;
- `model/head_b_gbm.py::build_feature_vector` and
  `model/head_b_acsins_charge.py::charge_features` for the serving-compatible
  baseline features;
- the existing frozen-PLM and CDR-aware feature modes, subject to their current
  coverage and parse-rate rules;
- `model/developability_uncertainty.py` for three-seed disagreement and
  out-of-domain abstention;
- `tools/promotion_gate.py` and `tools/benchmark_policy.json` for governance;
- the existing aggregation unified and rank-ensemble implementations when they
  can be evaluated under the identical outer split.

## 6. Architecture

Each property head may contain two expert types.

### 6.1 Known-family specialists

A specialist is trained for a narrow library source only when that source has
enough valid rows and both target classes. Its honest evidence is grouped
within-source CV with identical sequences kept in the same fold. Specialist
scores are reported only as known-family performance. This protocol deliberately
simulates lead optimization around a represented family; it is not evidence of
new-family generalization.

At inference, a specialist is eligible only when the query VH has at least
90% exact positional identity to an equal-length VH in that specialist's
training family. This reuses the matrix's existing near-VH identity definition.
The threshold is fixed before evaluation and is not tuned on a held-out source.

If several specialists qualify, the one with the highest nearest-VH identity is
used. A deterministic specialist name breaks an exact tie. A source name is
never an inference input.

### 6.2 Diverse generalist

The generalist is trained on eligible diverse or source-balanced mixed data.
During evaluation, the held-out source is removed completely, followed by the
existing exact-VH, exact-VL, and 90%-identity VH overlap stripping. If no
specialist qualifies, the generalist is used.

The generalist is also responsible for every property that has no qualifying
library specialist.

The 90%-identity overlap rule means a strict unseen-source holdout cannot also
qualify for a retained library specialist: any such library training sequence
has already been removed. Therefore generalist source-held-out performance and
specialist within-family performance are two separate operating regimes. The
report must not combine them into a single routed macro AUC.

### 6.3 Applicability and abstention

The existing train-fitted feature-distance method supplies `in_domain`,
`borderline`, or `out_of_domain`. Three-seed prediction disagreement remains a
second uncertainty signal.

An out-of-domain query or disagreement above the existing threshold returns the
score with an abstention warning; it is not silently presented as a reliable
prediction. Missing or invalid sequence input is rejected before routing.

## 7. Candidate combination

Only candidate models that can produce predictions for the exact same outer
holdout rows may be combined. Candidate families are:

1. the current biophysical-plus-charge LightGBM baseline;
2. the frozen-PLM LightGBM candidate when real cache coverage is at least 80%
   for both train and test;
3. the current aggregation unified model;
4. the aggregation LightGBM + HistGradientBoosting + Ridge ensemble;
5. assay-context aggregation features where the assay context is known at both
   training and inference time.

Assay-context predictions are diagnostics unless the same context is a required
inference input. They cannot enter the generic sequence-only artifact.

Candidate scores are oriented so a larger value always means worse
developability and are calibrated using training-only data. The combination
search consists only of each single candidate and equal-weight averages of
compatible candidates. This keeps weights nonnegative and avoids fitting a
fragile meta-model to a handful of sources.

The best candidate or average is selected inside an inner leave-one-source-out
loop. The outer held source is not used to choose components, weights,
thresholds, features, or hyperparameters. Ties choose fewer components, then a
stable lexical candidate order.

CDR-H3, structure, and paired AbLang candidates are excluded from the initial
pool. They may be reconsidered only through a preregistered A/B run using the
same nested source-held-out protocol.

## 8. Evaluation data flow

For each property, outer held source, and seed 42/43/44:

1. Load only registry rows eligible for the requested evidence role.
2. Reserve the entire outer source as test data.
3. Remove exact VH, exact VL, and at-least-90%-identical VH overlaps from all
   outer training candidates.
4. Cap each training source/class using the existing deterministic sampler.
5. Generate inner source-held-out predictions for every compatible candidate.
6. Select one candidate or equal-weight average using inner macro source AUC.
7. Refit the selected components on all permitted outer training rows.
8. Fit calibration on training-only rows.
9. Score each outer test antibody once with the selected generalist.
10. Save row-level predictions, component names, calibration,
    overlap counts, applicability, and uncertainty.
11. Compute per-source metrics first and then the unweighted macro source mean.

In a separate known-family run, group exact duplicates, fit each eligible
specialist on the remaining variants from its own source, route by the fixed
90%-identity rule, and report its within-family metrics and coverage. The two
reports share seeds and metric definitions but are never pooled.

Reconstructed rows may train when the registry marks them eligible, but they
remain ineligible as holdouts. Single-chain or CDR-only supervision remains
auxiliary and cannot promote a paired-Fv serving head.

## 9. Metrics and reporting

Every property report must contain:

- source ROC AUC and PR AUC;
- macro source ROC AUC;
- catch rate and enrichment at 10% and 15%;
- three-seed mean and standard deviation;
- source-cluster bootstrap 95% confidence interval;
- calibration method and calibration metrics;
- specialist eligibility, route counts, and within-family coverage;
- applicability bands, nearest training source, and abstention counts;
- exact and near-overlap removal counts;
- frozen-PLM coverage and CDR parse rate when relevant.

Known-family specialist CV and unseen-source generalist evaluation appear in
separate sections. There is no single mean across unrelated properties or
across easy and hard protocols.

## 10. Gates

### 10.1 Experimental component-admission gate

A generalist combination becomes the recorded best experimental generalist for
a property only when, on the identical direct held-out sources:

- macro source ROC AUC improves by at least 0.010 over the strongest comparable
  baseline;
- no held source loses more than 0.030 ROC AUC;
- enrichment at 15% holds or improves;
- seed standard deviation is at most 0.030;
- overlap, calibration, applicability, and uncertainty audits are present.

If it fails, the existing candidate remains best and the negative result is
saved.

A specialist may be included in the experimental artifact when its separate
known-family evaluation passes the existing minimum row/class requirements and
does not underperform the best comparable specialist by more than 0.005 AUC.
This admission does not alter or inflate the generalist metric.

### 10.2 Serving-promotion gate

Experimental admission does not authorize serving changes. The existing strict
promotion policy still applies, including macro AUC at least 0.80, every source
AUC at least 0.65, enrichment at least 2x, three stable seeds, and a genuinely
locked validation source. Because no locked validation source is currently
assigned, this work cannot modify `model/saved/platform.pkl` or claim lab-ready
performance yet.

GDPa3 remains untouched regardless of development performance.

## 11. Inference contract

The opt-in artifact accepts validated VH and optional VL sequences. For each
property it returns:

- higher-is-worse risk score;
- selected expert type and name;
- selected component models;
- nearest-family identity when a specialist is considered;
- applicability and nearest training source;
- three-seed disagreement;
- abstention boolean and reason;
- artifact schema and training-manifest identifiers.

The artifact must contain its feature schema and router threshold so training
and serving cannot silently diverge.

If a selected component requires frozen PLM features, the query embeddings must
already exist in the project cache. A cache miss stops with the exact
precomputation command; it never switches model families or zero-fills features
behind the user's back.

## 12. Error handling

- Reject non-standard or too-short VH input with a clear message.
- Represent missing VL explicitly; never substitute a fabricated chain.
- Skip a PLM candidate when coverage is below 80%; never zero-fill and call it
  a PLM result.
- Skip an evaluation source with too few rows or classes and report the exact
  reason.
- Abstain if no eligible expert can be fitted or if the query is out of domain.
- Stop on artifact/schema mismatch rather than silently truncating features.

## 13. Verification

The minimum implementation checks are:

1. focused unit tests for deterministic routing at, above, and below 90%
   identity, including equal-length enforcement and tie-breaking;
2. a leakage test proving outer-source rows and exact/near matches never enter
   candidate selection, calibration, or fitting;
3. a nested-selection test proving outer labels cannot change the selected
   candidate;
4. a blend test covering compatible candidates, direction normalization, and
   deterministic tie-breaking;
5. artifact train/serve feature-schema parity;
6. regression tests for the existing matrix, uncertainty, promotion gate, and
   GDPa3 untouched guard;
7. the full repository test suite;
8. a fresh three-seed development run and promotion-gate report.

## 14. Completion criteria

The task is complete when the experimental platform artifact and its report are
reproducible from one documented command, all required checks pass, all reported
counts and metrics come from saved artifacts, and the current serving artifact
is either unchanged or promoted through the full existing gate. A null result
is a valid completion: in that case the report identifies the retained baseline
and the routed candidate's exact failure reasons.
