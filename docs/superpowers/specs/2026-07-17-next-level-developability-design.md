# 2026-07-17 — Next-Level Antibody Developability Design

## 0. Status and objective

This design is the approved hybrid path for improving the Biostuff antibody
developability platform using a Mac, free Colab GPUs, and occasional access to
an M5 Max. It covers the three properties that currently have defensible
sequence signal:

- polyreactivity / PSR;
- aggregation / self-association / AC-SINS;
- hydrophobicity / HIC.

The primary success criterion is performance on antibodies from sources that
were excluded entirely from training. Within-source or library-variant scores
are diagnostic only and cannot support a lab-readiness claim.

The desired destination is at least 0.80 source-held-out ROC AUC for each served
property. This is a research target, not a promised outcome. Before new wet-lab
measurements are available, 0.80 is most plausible for HIC, possible but
uncertain for aggregation, and a stretch for polyreactivity.

## 1. Current evidence and starting point

The historical platform scorecard reports approximately 0.72 aggregation,
0.70 polyreactivity, and 0.70 hydrophobicity AUC on its established evaluation.
The more demanding diverse polyreactivity transfer tests are lower: the current
tree baseline is around 0.63, and the two-seed ESM-2 fine-tuning experiment was
around 0.63-0.64. Those harder transfer numbers are the relevant starting point
for the polyreactivity program.

Large narrow libraries are not equivalent to biological diversity. The Tessier
libraries contain hundreds of thousands of variants and are useful for learning
local sequence rules, but they did not solve transfer to AIMS and Shehata-style
diverse antibodies. New development work must therefore balance sources,
scaffolds, germlines, and sequence clusters rather than raw rows.

The following negative results remain binding evidence:

- paired AbLang2 reduced the serving-aligned aggregation ensemble AUC;
- CDR-H3 did not survive the serving-aligned aggregation ensemble gate;
- H1+H2+H3 descriptors did not improve diverse polyreactivity;
- Harvey VHH labels did not improve the fixed paired-Fv polyreactivity gate;
- sparse sequence k-mers performed worse than the biophysical baseline;
- Tm and expression/yield were approximately coin-flip from current inputs.

Rejected experiments may be revisited only when a material premise changes,
such as a new dataset, a corrected endpoint, or a stronger representation.

## 2. Lab-readiness definition

A property becomes lab-ready only when all of the following hold:

1. Source-held-out ROC AUC is at least 0.80 on multiple diverse sources.
2. PR AUC is reported under the real held-out class balance.
3. The top 15% risk bucket catches at least twice as many failures as random
   selection, with top-10% and top-15% results both shown.
4. Results are stable over at least three declared seeds.
5. Source-level or cluster-level bootstrap confidence intervals are reported.
6. Calibration is acceptable at a predeclared assay-specific decision point.
7. No major source, germline, scaffold, or antibody-format subgroup collapses.
8. A locked validation source, not used for experiment selection, confirms the
   direction of the gain.
9. The model can abstain when the input is outside its training domain.

A high within-source mean, a single favorable seed, or a gain on one tiny
source does not satisfy this definition.

## 3. Evaluation hierarchy

Repeated use of a holdout eventually turns it into development data. The new
program therefore uses three levels.

### 3.1 Development evaluation

- Leave one complete source out.
- Remove exact shared VH sequences and conservative sequence-cluster overlaps.
- Tune only inside the remaining training sources.
- Macro-average across held-out sources so large sources do not dominate.
- Use source-balanced and cluster-balanced training weights.

AIMS mouse, AIMS flu, and Shehata can remain in this development suite, but
they are no longer described as pristine final tests because their results have
already informed many decisions.

### 3.2 Locked validation

The next genuinely new, usable diverse source for each assay family is assigned
to locked validation before its labels are used. Predictions, model artifact,
feature manifest, and analysis code are frozen before labels are scored. A
locked source is not recycled into tuning after a disappointing result.

### 3.3 Final single-shot evaluation

GDPa3 remains untouched and governed by the existing one-shot ledger. It is not
downloaded, inspected, or used to select features. Spending GDPa3 requires a
frozen release candidate and an explicit user decision.

## 4. Data architecture

All records enter one provenance-aware registry but retain their molecular and
assay boundaries.

### 4.1 Supervision tiers

1. **Paired Fv supervised:** full VH and VL with a directly measured value.
   These rows can train paired-Fv assay heads.
2. **Single-chain supervised:** complete VH or VHH with a directly measured
   value. These rows train auxiliary heavy-chain or VHH heads only.
3. **CDR supervised:** explicit CDR sequences with a directly measured value.
   These rows train CDR auxiliary tasks only.
4. **Unlabeled antibody sequence:** suitable for representation adaptation and
   out-of-domain modeling, never converted into supervised labels.
5. **Rejected/quarantined:** ambiguous pairing, figure-only values, external
   identity reconstruction, incompatible endpoints, or missing provenance.

Auxiliary tiers may influence a paired-Fv model only through a declared shared
representation or distillation experiment that improves paired-Fv held-out
performance. Missing chains are never represented as fabricated sequences.

The existing paired-Fv ingestion rules remain strict: full paired chains and a
numeric measured value are required for the primary supervised table. The
user-approved VH/VHH, CDR-only, and explicit experimental binary-call formats
live in separate auxiliary tables and cannot silently enter the primary table.

### 4.2 Assay separation

Each measurement retains:

- assay family and exact metric;
- value and units;
- direction (`higher_bad` or `lower_bad`);
- concentration, pH, formulation, format, and protocol when available;
- study, campaign, source, and molecule identity;
- sequence and raw-row hashes;
- library-versus-diverse classification;
- supervision tier and exclusion reasons.

PSR, AC-SINS, and HIC labels are never pooled into one target. Closely related
protocols may share an encoder but keep separate prediction heads or explicit
protocol context.

### 4.3 Identity and leakage control

- Deduplicate exact paired Fv first, exact VH second, and exact VL third.
- Build sequence-similarity components before splitting.
- Quarantine conflicting labels rather than silently choosing one.
- Prevent all variants of a parent antibody from crossing a fold boundary.
- Treat renamed clinical panels as duplicates when their sequences match.
- Preserve append-only conflict, exclusion, and provenance ledgers.

## 5. Data acquisition program

### 5.1 Automated discovery

Search first-party sources through Europe PMC, PubMed Central assets, Crossref,
Zenodo, Figshare, Dryad, GitHub releases, journal source-data endpoints, and
official institutional repositories. Query terminology must cover assay aliases,
including PSR, polyspecificity, polyreactivity, BVP, nonspecific binding,
AC-SINS, CS-SINS, CIC, self-association, HIC, hydrophobic interaction
chromatography, and retention time.

Discovery produces candidates only. It never appends training rows.

### 5.2 Candidate classification

- `OPEN_AUTO`: directly downloadable, parseable public artifact.
- `FREE_MANUAL`: free but browser-mediated artifact.
- `PAYWALLED`: logged and skipped unless the user already has lawful access.
- `REQUEST_ONLY`: logged as a future author-contact candidate.
- `UNUSABLE`: failed a scientific requirement, with the exact reason retained.

### 5.3 Extraction gate

Every candidate must pass:

- direct sequence-to-value row mapping;
- valid amino-acid alphabet and declared molecular format;
- numeric measured endpoint or an explicit experimental binary call;
- assay direction and units;
- molecule-level duplicate and holdout-overlap audit;
- source and license/provenance record;
- real pre-filter and post-filter row counts.

Figure transcription and ambiguous cross-paper sequence joins remain forbidden.

### 5.4 Highest-value acquisition targets

Priority is determined by expected independent biological diversity, not row
count:

1. diverse paired clinical or discovery panels with PSR/AC-SINS/HIC;
2. new laboratories and assay protocols with at least 100 paired antibodies;
3. paired lineage panels spanning several independent parent antibodies;
4. VH/VHH and CDR-only sources for auxiliary representation learning;
5. large narrow libraries for local sequence-rule learning;
6. unlabeled paired repertoires for domain adaptation and OOD estimation.

## 6. Representation and model design

### 6.1 Shared encoder, separate assay heads

The preferred design is a shared antibody-aware sequence representation with
separate heads for PSR, AC-SINS, and HIC. Label losses remain separate. The
shared encoder transfers sequence knowledge without pretending the assays are
interchangeable.

Paired Fv inputs use explicit chain boundaries. VH-only, VHH, and CDR-only
examples use format tokens and compatible auxiliary heads. Light-chain absence
is a real input state, not a zero-filled paired antibody.

### 6.2 Experiment ladder

Experiments stop at the first rung that clears the gate:

1. Existing biophysical and charge baseline.
2. Frozen ESM-2 embeddings with source-balanced LightGBM.
3. Frozen antibody-specific embeddings, tested alone and in a blend.
4. CDR-aware pooling of frozen PLM residue embeddings.
5. Small projection head or MLP on cached embeddings.
6. Parameter-efficient PLM adaptation on Colab.
7. Shared-encoder multitask adaptation with assay-specific heads.
8. Structure-derived features, only after the sequence stack plateaus.

No larger model is attempted merely because it is available. Each rung must
improve the same preregistered held-out sources.

### 6.3 Library-data controls

- Cap or weight each source so Tessier-scale libraries cannot dominate.
- Sample by class, source, parent/scaffold, germline, and sequence cluster.
- Report the number of independent parents and clusters beside row count.
- Strip every VH overlapping a held-out source before training.
- Compare library-only, diverse-only, and mixed training arms.
- Use narrow libraries for local optimization signal, not as evidence of broad
  biological generalization.

### 6.4 Auxiliary tasks

VH/VHH and CDR-only labels are tested through three possible routes:

1. auxiliary classification heads during encoder adaptation;
2. teacher-student distillation into the paired-Fv encoder;
3. ensemble experts invoked only for compatible molecular formats.

The simplest passing route is retained. If none improves paired-Fv transfer,
the auxiliary model remains a separately reported tool.

### 6.5 Ensemble policy

Candidate models generate out-of-fold predictions. Blending uses only those
predictions, never training-fold scores. Prefer a rank average or constrained
nonnegative blend over an unconstrained stacker. A component is included only
if it improves macro held-out AUC or meaningfully reduces source variance.

## 7. Experimental context

Sequence is not the sole determinant of a laboratory measurement. When public
data provide context, encode:

- antibody format and isotype;
- assay concentration;
- buffer pH and ionic strength;
- formulation or excipient category;
- temperature and incubation time;
- laboratory/protocol identifier.

Every context-aware model must be compared with a sequence-only model. Serving
accepts missing context and widens uncertainty. It must not silently substitute
training-set averages and claim assay-specific precision.

## 8. Metrics and statistical discipline

Each experiment declares before execution:

- hypothesis and changed variable;
- training and held-out sources;
- identity filtering rule;
- feature version and model parameters;
- seeds, primary metric, secondary metrics, and promotion threshold;
- expected compute budget;
- artifact and report paths.

Required reports contain:

- per-source ROC AUC and PR AUC;
- macro averages and source-level spread;
- top-10% and top-15% catch-rate and enrichment;
- Brier score or expected calibration error at a declared threshold;
- source/cluster bootstrap 95% confidence intervals;
- independent molecule, parent, cluster, and positive counts;
- subgroup results by germline, format, and OOD distance where supported.

Promotion requires improvement or non-regression within a predeclared tolerance
on the primary metric, no serious subgroup collapse, and reproducibility over
three seeds. Tiny sources such as AIMS flu are shown individually and never
allowed to dominate the conclusion.

## 9. Uncertainty and applicability domain

The platform must distinguish prediction from confidence. Each result includes:

- assay-specific risk score;
- calibrated decision score when a physical threshold exists;
- ensemble disagreement;
- distance to the nearest training sequence/embedding cluster;
- applicability label: `in_domain`, `borderline`, or `out_of_domain`;
- nearest relevant training sources;
- abstention reason when confidence is insufficient.

The first implementation should reuse cached embeddings and ensemble spread.
More elaborate conformal methods are attempted only if this baseline is
miscalibrated under source-held-out testing.

## 10. Compute design

### 10.1 Mac path

- data parsing, manifests, identity graphs, and feature validation;
- cached frozen embeddings from smaller PLMs;
- LightGBM, logistic regression, calibration, and blending;
- source-held-out evaluation and report generation;
- deterministic smoke tests and serving checks.

### 10.2 Free Colab path

- ESM-2 650M or the largest model that fits the assigned GPU;
- parameter-efficient adaptation and multi-seed runs;
- capped sequence length and balanced source sampling;
- checkpoint/resume to Google Drive;
- export embeddings, predictions, manifests, and reports rather than relying on
  notebook output alone.

### 10.3 Occasional M5 Max path

- embedding-cache generation and verification;
- longer local sweeps using already frozen experiment definitions;
- structure work only for a small, informative subset after sequence models
  plateau.

## 11. Delivery phases

### Phase 0 — Benchmark lock and audit

- Freeze current artifacts, datasets, feature schema, and benchmark reports.
- Reconcile historical scorecard numbers with the hard transfer benchmarks.
- Define development, locked-validation, and final-test roles per property.
- Add experiment manifests and a holdout-use ledger.

Exit criterion: every future result has an unambiguous baseline and cannot
silently spend a locked holdout.

### Phase 1 — Unified data registry and expanded formats

- Route paired Fv, VH/VHH, CDR-only, and unlabeled rows through one provenance
  contract.
- Import the new 294-row AIMS CDR dataset as auxiliary data.
- Audit existing Harvey VHH and all paired panels through the same identity
  graph.
- Preserve assay context and rejection reasons.

Exit criterion: deterministic counts, hashes, supervision tiers, conflicts,
and split groups for every source.

### Phase 2 — Reproducible baseline matrix

- Run biophysical, frozen-PLM, CDR-aware, VH-only auxiliary, and source-mixing
  ablations under identical splits.
- Run at least seeds 42, 43, and 44.
- Establish per-property error and subgroup maps.

Exit criterion: one comparable report identifies which representation and data
tier genuinely transfers for each property.

### Phase 3 — Colab representation adaptation

- Rework the current Colab job into a manifest-driven runner.
- Use source/cluster-balanced sampling and explicit chain boundaries.
- Test frozen embeddings before adapters; test adapters before full fine-tuning.
- Test library-only, diverse-only, and mixed arms without touching locked data.

Exit criterion: exported multi-seed predictions and checkpoints reproduce from
one command and either beat the frozen baseline or are rejected.

### Phase 4 — Ensemble, calibration, and OOD

- Build out-of-fold blends from passing components.
- Fit assay-specific calibration inside training folds.
- Add embedding-distance and disagreement-based abstention.
- measure performance at realistic top-10% and top-15% lab budgets.

Exit criterion: each served head has risk, confidence, applicability, and
screening-enrichment outputs with train/serve parity.

### Phase 5 — Locked validation and serving promotion

- Freeze the candidate before scoring a new locked source.
- Promote only properties meeting the lab-readiness definition.
- Keep weaker properties experimental and label their limitations.
- Spend GDPa3 only after an explicit final-release decision.

Exit criterion: a versioned scorecard distinguishes development, locked, and
single-shot evidence and makes no pooled or padded claims.

### Phase 6 — Wet-lab active learning (pinned)

This phase remains designed but inactive until laboratory access exists.

- Select diverse, high-uncertainty antibodies rather than only extreme model
  predictions.
- Measure all three endpoints on the same molecules under standardized context.
- Use 2-4 rounds, retraining after each round.
- Reserve a fixed fraction of every batch as untouched prospective validation.
- Prefer 500-1,500 total diverse measurements for a realistic attempt to move
  polyreactivity beyond the public-data ceiling.

## 12. Stop rules

Stop or defer an experiment when:

- it does not change the effective independent sequence diversity;
- its source overlaps a development or locked holdout;
- it improves a pooled mean but not macro source-held-out performance;
- gains disappear across seeds or source bootstrap samples;
- the model uses unavailable serving-time inputs;
- compute cost increases materially without a preregistered hypothesis;
- structure or fine-tuning fails to beat cached frozen embeddings;
- a property remains near chance after multiple materially different
  representations and data sources.

## 13. Major risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Public diverse labels remain scarce | Polyreactivity plateaus below 0.80 | Maximize source diversity, auxiliary learning, uncertainty, and preserve wet-lab phase |
| Repeated holdout tuning | Inflated generalization claims | Three-level evaluation and append-only holdout ledger |
| Narrow libraries dominate | Learns parent-specific rules | Source/parent/cluster-balanced sampling and reporting |
| Assay heterogeneity | Incompatible labels and poor calibration | Separate heads, context fields, source macro metrics |
| VH/VHH/CDR domain mismatch | Auxiliary data harms paired Fv | Format tokens, auxiliary heads, paired-Fv promotion gate |
| Free Colab interruptions | Lost or irreproducible runs | Drive checkpoints, manifests, resumable jobs, exported reports |
| High AUC but low screening value | Little laboratory benefit | PR AUC, catch-rate, enrichment, calibration, fixed-budget analysis |
| Overconfident OOD predictions | Unsafe candidate ranking | Distance/disagreement checks and abstention |

## 14. Explicit non-goals for this cycle

- No promise of 0.80-0.90 before evidence supports it.
- No pooled universal developability label.
- No promotion based on within-library performance.
- No figure transcription or guessed sequence pairing.
- No full PLM training from scratch.
- No structure-at-all-costs pipeline.
- No serving of Tm or expression/yield until a materially stronger input or
  dataset changes their demonstrated near-chance ceiling.
- No wet-lab execution before access exists.

## 15. First implementation slice after approval

The first slice is Phase 0 only: create a benchmark/holdout ledger and a single
command that reports current per-property baselines, source roles, independent
counts, and metric definitions without retraining. It is deliberately small and
establishes the measurement contract required for every later experiment.
