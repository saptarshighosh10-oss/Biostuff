# 2026-07-18 — Gated Antibody Decision Platform

## Status

Approved direction: separate evidence-gated heads with shared inputs and outputs.
This design does not authorize a universal score or serving an unvalidated head.

## 1. Objective

Extend the existing developability platform into an antibody pre-screen covering:

1. developability;
2. immunogenicity;
3. antigen-aware affinity;
4. off-target/cross-reactivity;
5. pharmacokinetics (PK); and
6. clinical evidence.

Each head answers a different question, uses an endpoint-specific benchmark, and
may abstain. The platform must make missing evidence visible rather than convert
it to a plausible-looking number.

## 2. Non-negotiable decisions

- Keep the current developability platform unchanged while new heads are
  experimental.
- Do not average unrelated endpoints into an overall AUC or universal risk score.
- Do not serve a head merely because it fits its training data.
- Missing mandatory inputs produce `abstain`, never zero-filled predictions.
- Every result identifies its model/data manifest, benchmark role, applicability,
  uncertainty, and limitations.
- GDPa3 remains unspent and outside all development work.
- Clinical success is not a sequence-only endpoint. Until a defensible benchmark
  exists, it is an evidence summary rather than a predictive model.

## 3. Platform boundary

Add a new decision-platform entry point rather than expanding the legacy
`model/head_platform.py` in place. The legacy artifact stays frozen until a new
head passes its own serving gate.

### 3.1 Inputs

The shared request accepts:

- required antibody `vh` and optional/required-by-head `vl`;
- optional antigen sequence and antigen identifier;
- optional Fc/isotype, molecule format, valency, conjugation, and formulation;
- optional clinical context: target, indication, dose, route, and population.

Each head declares which fields are mandatory. Affinity cannot run without an
antigen. PK cannot claim clinical half-life from VH/VL alone.

### 3.2 Output contract

Each head returns:

- `model_status`: `served`, `experimental`, or `unavailable`;
- `prediction_status`: `predicted`, `measured`, `evidence_only`, or `abstain`;
- `score` and explicit direction when a validated score exists;
- uncertainty and applicability/OOD status;
- required and missing inputs;
- inputs actually used;
- benchmark name, sample count, metric, result, and confidence interval;
- model version and data cutoff;
- artifact and data-manifest hashes; and
- a short limitations list.

No absent head contributes a default value to another head.

## 4. Shared scientific controls

### 4.1 Data integrity

- Accept only standard 20-amino-acid sequences of length at least 20.
- Deduplicate by the complete `(VH, VL)` molecule, while retaining an alias and
  endpoint-conflict ledger.
- Remove exact and predeclared near-identity overlap between training and every
  validation cohort before model fitting.
- Preserve source, assay, units, endpoint direction, citation, license, retrieval
  date, and SHA-256 in a run manifest.
- Never synthesize a sequence, endpoint, or missing metadata value.

### 4.2 Evaluation

- Split by molecule family, source/study, and antigen/target family where relevant;
  never use an ungrouped random-row split as evidence of generalization.
- Use at least three declared seeds for fitted heads.
- Report per-source metrics and macro-source summaries rather than a row-weighted
  mean.
- Use source/reference-cluster bootstrap confidence intervals.
- Freeze predictions before reading labels for a genuinely locked confirmation
  cohort.
- Log null results and failed gates; failed experimental code is removable while
  its result manifest remains.

### 4.3 Serving gate

A head is served only when:

1. its endpoint and direction are clinically interpretable;
2. development performance is stable across seeds and sources;
3. a confirmation cohort not used for candidate selection agrees in direction;
4. relevant subgroups do not reveal that the aggregate score is only a confounder;
5. calibration and OOD behavior are measured; and
6. the serving artifact reproduces the evaluated feature path.

Otherwise it reports `model_status` as `experimental` or `unavailable` and uses
`prediction_status=abstain` whenever required evidence or applicability is absent.

## 5. Head designs

### 5.1 Developability

Retain the existing aggregation, hydrophobicity, and polyreactivity heads and
their current research governance. Do not change `model/head_platform.py` or
`model/saved/platform.pkl` merely to expose this new shell.

Thermostability, expression, and other properties remain unserved unless they
pass the existing fixed source-held-out benchmark.

### 5.2 Immunogenicity — first new head

#### Training signal

Use OAS paired human repertoires under its CC-BY 4.0 terms. The OAS human-paired
search inspected on 2026-07-18 exposed 580 data units from 19 studies and
3,003,127 reported unique sequences across units. Units marked with no disease
contained 1,177,822 reported sequences across 236 units. These are source-unit
totals, not a claim of global post-download deduplication.

`Disease=None` does not mean healthy and unexposed: it includes vaccinated donors.
The stricter live query with both `Disease=None` and `Vaccine=None` reported
568,139 sequences. Begin with that subset, audit its metadata rather than naming
it definitively healthy, globally deduplicate exact VH/VL pairs, and preserve
study/subject boundaries. Use donor- and study-balanced sampling so a few large
repertoires cannot dominate. Remove every exact or predeclared near-clone overlap
to a therapeutic validation molecule before training.

#### Model ladder

Stop at the first rung that holds:

1. a local OASis-style one-class repertoire model using VH/VL peptide prevalence
   or normalized sequence likelihood;
2. germline-distance/content features as a transparent comparator; and
3. a frozen or adapted antibody PLM only if it beats the first two under the same
   fixed evaluation.

Do not train a human-versus-mouse classifier as the primary model: it can score
well while learning only species origin. ADA labels are never used to train the
humanness representation.

#### Development clinical benchmark

Use Merck's MIT-licensed BioPhi publication data:

- `immunogenicity.tsv`: 217 unique therapeutic names with numeric reported ADA
  incidence in percent of patients;
- `mabs.fa`: exactly 434 records, providing complete standard-AA VH/VL pairs for
  all 217 names;
- five duplicate `(VH, VL)` groups, leaving 212 unique molecules; and
- 53 raw names at or above 10% ADA and 164 below that threshold.

For repeated molecules, emit one sequence prediction but retain every named
clinical observation, reference, species label, and ADA value. The primary
observation-level metric gives each exact sequence pair total weight one so aliases
cannot create pseudoreplication. Also report a predeclared median-collapsed
sensitivity analysis and every original range. This cohort is development evidence
because its labels and published results are already known.

Primary metrics:

- Spearman correlation between higher risk and reported ADA percent; and
- ROC AUC for ADA incidence at least 10%, matching the publication benchmark.

Required honesty views:

- all molecules;
- human plus humanized molecules;
- humanized molecules alone when both classes are adequate;
- source/reference-cluster confidence intervals; and
- an origin-only baseline to quantify species confounding.

The deposited publication scores reproduce the confounding problem: OASis medium
is about 0.799 AUC over all 217 raw names and about 0.554 within the 95 humanized
names. Hu-mAb is about 0.830 overall but about 0.694 within humanized molecules.
OASis over human plus humanized names is about 0.732. The human-only subset has
only three positives, so its apparently higher AUC is unstable. No aggregate
number is accepted as proof of clinically useful within-origin prediction.

#### Locked confirmation

Use the CC-BY-4.0 Immunogenicity Database Collaborative (IDC) DS V1 as the first
confirmation candidate. Its publication reports 218 therapeutic products, 113
underlying molecules with clinical data, 727 trials, 1,788 patient cohorts, and
4,146 clinical immunogenicity readouts. Of 122 molecules with sequences, only 80
have both sequence and clinical data; the clinical set contains 87 monoclonal
antibodies plus other modalities. Usable antibody counts must be measured after
schema validation, molecule deduplication, modality filtering, and removal of
every BioPhi sequence or alias overlap.

Public aggregate statistics and a schema-level spot check are already known, but
no IDC performance was calculated or used for model selection. Therefore describe
this as an operationally sealed confirmation, not a prospective or completely
unseen clinical validation.

Freeze the development design, model artifact, molecule list, sequence hashes,
and predictions before the first record-level ADA extraction. A single audited
confirmation program may then read the labels once, record that use in the holdout
ledger, and join them to the frozen predictions. If IDC leaves fewer than 30 unique
molecules or fewer than five examples in either ADA class, report insufficient
evidence. Official FDA/EMA product information may fill provenance or metadata
gaps, but it must not be mined iteratively to tune the model after confirmation.

ADA incidence values are assay-, schedule-, indication-, treatment-, and
population-dependent. Preserve patient count, treatment exposure, sampling
schedule, ADA assay type/sensitivity/drug tolerance, neutralizing-versus-binding
status, dose, route, concomitant therapy, and indication when available. Never
treat unlike cohort readouts as repeated measurements of one interchangeable
sequence-only truth.

Report two outputs separately:

1. `intrinsic_immunogenicity`: sequence-only humanness or epitope-liability risk;
2. `contextual_ada`: an optional estimate using the declared clinical context.

For IDC, report sequence-only performance and the incremental value over a
clinical-context-only baseline. Humanness alone is never a calibrated patient ADA
probability.

Serve only as a pre-screening aid if the signal is positive and stable on both
the development subgroup analysis and locked confirmation. Never describe it as
a patient-level safety prediction.

Primary sources:

- OAS: https://opig.stats.ox.ac.uk/webapps/oas/
- BioPhi data and notebooks:
  https://github.com/Merck/BioPhi-2021-publication
- BioPhi paper: https://doi.org/10.1080/19420862.2021.2020203
- IDC release: https://immunogenicitydb.org/data.html
- IDC repository: https://github.com/Immunogenicity-Database-Collaborative/IDC-DB
- FDA ADA assay guidance:
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/immunogenicity-testing-therapeutic-protein-products-developing-and-validating-assays-anti-drug
- EMA therapeutic-protein immunogenicity guidance:
  https://www.ema.europa.eu/en/immunogenicity-assessment-biotechnology-derived-therapeutic-proteins-scientific-guideline

### 5.3 Affinity — antigen-aware, two tasks

Affinity requires both antibody and antigen context. Keep two endpoints separate:

1. binder/non-binder discovery for a specified antigen; and
2. mutation-induced affinity change for lead optimization.

Before implementation, verify that each candidate public source provides the
antibody sequence, antigen sequence or stable identifier, numeric affinity or a
real binding label, units, and license. SAbDab structures without affinity labels
are representation data, not supervised affinity labels. SKEMPI mutation records
are optimization data, not proof of de-novo binder discovery.

AB-Bind contains 1,101 mutants across only 32 complexes and supports mutation
effect work, not broad de-novo affinity. SKEMPI 2.0 contains 7,085 general
protein-protein mutations and is not antibody-specific. SAbDab provides curated
antibody-antigen structures and affinity annotations where available, but usable
rows and assay/unit consistency must be measured from a versioned snapshot.

Validation must hold out entire antigen/complex families and antibody homology
clusters. Random mutant-row splits are forbidden because they leak parent
complexes. Report within-parent mutation ranking, unseen-complex generalization,
unseen-antigen-family generalization, per-antigen metrics, and macro-antigen
performance separately. If only one side of the pair is available, abstain.

Primary sources:

- SAbDab: https://sabdab.opig.stats.ox.ac.uk/about
- AB-Bind: https://pmc.ncbi.nlm.nih.gov/articles/PMC4815335/
- SKEMPI 2.0: https://doi.org/10.1093/bioinformatics/bty635

### 5.4 Off-target/cross-reactivity

Keep two tasks separate:

1. standardized-panel liability for an antibody: any off-target hit or hit count;
2. a specific antibody/counter-target interaction.

For panel liability, Dai et al. (2026) screened 174 therapeutic antibodies against
6,172 human extracellular proteins with REAP. The public supplements contain 130
exact-unique, standard-AA paired VH/VL molecules; 34 have at least one off-target
hit and 96 have none, with counts from 0 to 44. The other 44 antibodies lack
sequences. This supports a small experimental per-antibody classifier or count
model, not a clinical-toxicity claim.

Use nested molecule-held-out evaluation, three seeds, homology-grouped folds, and
a molecule bootstrap confidence interval. Because this is one assay/source with
no independent confirmation, it cannot pass the serving gate by itself. The paper
and supplements do not state a permissive redistribution license; analyze locally
and cite them, but do not commit or redistribute raw files until reuse terms are
resolved.

For a specific counter-target model, a supervised row must include:

- antibody VH/VL;
- intended and tested counter-target identifiers or sequences;
- assay type and conditions; and
- numeric or binary cross-reactivity results.

The published `# off` field is a per-molecule count, not a complete
174-by-6,172 pairwise label matrix. Supplements contain score matrices only for
selected subsets. Therefore the pairwise head remains `unavailable`. Motif alerts
or structural similarity are hypotheses, not labels, and polyreactivity is not
equivalent to specific off-target toxicity.

Pairwise validation must hold out antibody families and target families
simultaneously.

Primary source and data map:

- Paper: https://doi.org/10.1016/j.str.2026.02.012
- REAP antigens: https://ars.els-cdn.com/content/image/1-s2.0-S0969212626000547-mmc2.xlsx
- antibody sequences and hit counts:
  https://ars.els-cdn.com/content/image/1-s2.0-S0969212626000547-mmc3.xlsx
- selected-subset REAP matrices:
  https://ars.els-cdn.com/content/image/1-s2.0-S0969212626000547-mmc4.xlsx and
  https://ars.els-cdn.com/content/image/1-s2.0-S0969212626000547-mmc5.xlsx

### 5.5 Pharmacokinetics

Clinical half-life is affected by FcRn engagement, Fc/isotype, format, target-
mediated disposition, dose, route, indication, and population. The PK head therefore
requires this metadata and keeps human clinical endpoints separate from animal
or in-vitro endpoints.

Candidate rows must include numeric half-life with units plus molecule and study
context. Validation holds out molecules, targets, and studies. Report error and
rank metrics in a single unit after documented conversion. If context is missing,
return `abstain`; never substitute a sequence-only estimate.

A public 2023 survey reports 89 marketed antibody biotherapeutics and 86 half-life
values, but only 45 full-length IgGs form its cleanest approximately linear-PK
comparison after excluding target-mediated and format effects. This is enough for
a source audit or a small experimental context-aware benchmark, not a served
general model. A separate `sequence_clearance_liability` output may be researched,
but it must not be labeled clinical half-life.

Primary source: https://doi.org/10.1080/19420862.2023.2191301

### 5.6 Clinical evidence

Do not build a sequence-only "trial success" classifier. Trial outcomes mix
biology, indication, dose, trial design, sponsor execution, and commercial choices.

The first implementation is an evidence summary that links the other head results
to traceable trial/regulatory facts and explicitly says `not predictive`. A fitted
clinical head requires a separately approved design with a prospective temporal
split, indication-aware labels, and no leakage from post-outcome metadata.

Thera-SAbDab, ClinicalTrials.gov, and the FDA Purple Book can populate this panel,
but they do not form a clean sequence-to-success benchmark. `Completed` does not
mean successful, ongoing molecules are right-censored, termination may be
business-driven, and current status leaks future information. A future transition
model would require historical snapshots, explicit endpoint outcomes, censoring,
and temporal holdouts grouped by molecule, target, and sponsor.

Primary sources:

- Thera-SAbDab: https://doi.org/10.1093/nar/gkz827
- ClinicalTrials.gov: https://clinicaltrials.gov/
- FDA Purple Book: https://purplebooksearch.fda.gov/

## 6. Implementation order

1. Add the shared request/result contract and a non-serving head registry.
2. Ingest and audit the BioPhi ADA development cohort.
3. Download a manifest-driven, study-balanced OAS no-disease/no-vaccine subset;
   audit its metadata, then train and evaluate the local one-class baseline.
4. Build and freeze the operationally sealed IDC confirmation protocol.
5. Attempt a PLM only if the local baseline leaves justified headroom.
6. License-check and audit the Dai REAP supplements; evaluate the 130-molecule
   off-target panel-liability task as experimental only.
7. Audit affinity sources, then implement antigen-aware tasks that pass the data
   contract.
8. Audit the 89-product PK source and keep clinical half-life unavailable unless
   complete context passes the data contract.
9. Add the clinical evidence view without a predictive claim.

No production dependency is added without approval. The first immunogenicity rung
uses existing Python/scikit-learn tooling.

## 7. Error handling and checks

- A downloader fails closed on checksum, schema, HTTP, or license/provenance errors.
- Every ingest prints raw, rejected, duplicate, conflict, and retained counts.
- A leakage check proves no exact or near-identity validation molecule appears in
  training.
- One focused runnable check covers each new parser, split rule, and output contract.
- A full report contains the resolved manifest and per-record frozen predictions.
- Production artifacts change only in a separate promotion commit after all gates
  pass.

## 8. Explicit non-claims

- Humanness is not equivalent to clinical ADA risk.
- An aggregate 0.80 AUC dominated by species separation is not sufficient.
- Clinical ADA incidence is not an intrinsic sequence-only constant.
- Structural antibody data without matched affinity values is not an affinity
  training set.
- REAP panel liability is not proof of off-target clinical toxicity.
- FcRn-related sequence features alone do not determine clinical half-life.
- No sequence model can honestly promise clinical trial success.
