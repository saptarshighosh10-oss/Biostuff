# 2026-07-13 Antibody Developability Rebuild Design (Plan C, Locked)

## 0) Status
- Plan: **Plan C (locked) — no alternates in this cycle**
- Audience: implementer + reviewer workflow for production-grade rebuild
- Date: 2026-07-13
- Branch context: `codex/antibody-perfection`

## 1) Design intent (requirements) + current implementation gap
Plan C is a strict replacement of current baseline behavior so the pipeline predicts **risk** from supervised, assay-specific continuous outcomes while preserving VH/VL pairing and preventing leakage from near-identity relationships.

Core constraints above are Plan C requirements, not implementation claims.
- Current codebase is the baseline in this branch; planned Plan C components are not present yet.

Core constraints:
- No new production dependency.
- Stdlib-first implementations where possible.
- No GDPa3 row download/inspection in training loops or development workflows.
- GDPa3 treated as frozen, unseen holdout for one final external evaluation command only.
- Deterministic, resumable source snapshots.
- Explicit and auditable artifacts for every run.

## 2) Locked high-level architecture (planned requirements)

### Modules (new boundaries)
1. `data/registry.py` (new)
   - Source catalog, source-specific schema declarations, and deterministic fetch plans.
2. `data/cache.py` (new)
   - SHA-256 manifesting, row manifests, exclusion manifests, deterministic local cache writes.
3. `data/sources/*.py` (replace per-source logic)
   - Keep loaders, but each loader only emits raw normalized rows.
4. `data/identity_graph.py` (new)
   - Conservative identity graph, connected-component derivation.
5. `data/pairs.py` (new)
   - VH/VL parsing, canonical pairing, campaign/molecule/study key synthesis.
6. `model/preprocess.py` (new)
   - Endpoint filtering, conflict ledger, split graphing, nested fold orchestration.
7. `model/metrics.py` (new)
   - Spearman, MAE, enrichment, bootstrap, calibration-at-threshold.
8. `model/train.py` / `model/predict.py` / `model/validate.py` (surgical edits)
   - Use the new preprocessing and metrics contracts for train/score/stress entry points.
9. `data/fetch.py` (planned CLI)
   - CLI entrypoint for source fetch orchestration + manifest-only planning.
10. `data/build.py` (planned CLI)
   - CLI entrypoint for source build + source freeze controls.
11. `data/assert_pairs.py` (planned CLI)
   - CLI entrypoint for pair-enforcement validation and quarantines.
12. `model/evaluate_gdpa3.py` (planned CLI)
   - CLI entrypoint for the single external GDPa3 external check.
13. `model/report.py` (planned CLI)
   - CLI entrypoint for stress-test reporting and consolidated summaries.

`requirements` remains unchanged unless strict lockfile updates are required by current runtime context.

All module boundaries and commands in this section are planned requirements only; no new modules are claimed as already implemented.

### 2.1 Implemented today vs Plan C (baseline status)

Legacy and Plan C CLI surfaces are intentionally separated so unimplemented Plan C entrypoints are not presented as currently runnable.

- **Legacy commands/flags (implemented by baseline today):** existing baseline CLI behavior remains outside this Plan C lock and is not claimed below as Plan C-ready.
- **Plan C commands (not yet runnable in current branch):** status is explicit per row.

| Plan C module | Command / flag (planned) | Plan C gate(s) covered | Implemented today |
| --- | --- | --- | --- |
| `data/registry.py` | `python -m data.fetch --source <name> --manifest-only` | 1, 2, 9 | No |
| `data/cache.py` | `python -m data.fetch ...`, `python -m data.build ...` | 1, 2, 9 | No |
| `data/sources/*.py` | `python -m data.build --source-filter all --freeze` | 2, 9 | No |
| `data/identity_graph.py` | `python -m data.assert_pairs --enforce-vh-vl` | 5, 10 | No |
| `data/pairs.py` | `python -m data.assert_pairs --enforce-vh-vl` | 5, 10 | No |
| `model/preprocess.py` | `python -m model.train --scheme plan-c ...`, `python -m model.validate --baseline ...` | 4, 5 | No |
| `model/metrics.py` | `python -m model.validate --baseline ...`, `python -m model.report --stress arras` | 1, 6, 10, 11 | No |
| `model/train.py` | `python -m model.train --scheme plan-c --cv-strategy leave_study_out --outer-k 5 --inner-k 5 --seed 42` | 4, 5 | No |
| `model/predict.py` | `python -m model.predict --input <candidates.json> --out <out.json>` | 1, 7 | No |
| `model/evaluate_gdpa3.py` | `python -m model.evaluate_gdpa3 --run-once --manifest <path> --out results/gdpa3_report.json --expected-data-hash <sha256>` | 8, 9 | No |
| `model/report.py` | `python -m model.report --stress arras`, `python -m model.report --stress tnp` | 6, 10, 11 | No |
| `data/fetch.py` | `python -m data.fetch --source <name> --manifest-only` | 1, 2, 9 | No |
| `data/build.py` | `python -m data.build --source-filter all --freeze` | 1, 2, 9 | No |
| `data/assert_pairs.py` | `python -m data.assert_pairs --enforce-vh-vl` | 5, 10 | No |

## 3) Explicit data schemas

### 3.1 Normalized supervised row schema (single source-agnostic row)
```
record_id: str
source: "flab" | "proteingym" | "canya" | "figshare_agg" | "abdev" | "antiref" | "sabdab" | "anchors"
study_id: str
campaign_id: str
molecule_id: str
pair_id: str | null          # Stable antibody pairing key when VH/VL both available
chain_id: str | null         # "vh" | "vl" | null
sequence: str                # A-Z amino acids only
vh_sequence: str | null
vl_sequence: str | null
assay_id: str                # assay/platform specific protocol key
assay_metric: str            # e.g., "HIC_RT", "SEC_monomer_pct", "AC-SINS", "TAP_score"
endpoint_direction: str      # "higher_bad" | "lower_bad"
endpoint_value: float | null  # continuous assay output; required when supervision_status == "supervised"
endpoint_unit: str | null    # e.g., ms, ug/mL, deltaG
target_value: float | null   # normalized regression target; required when supervision_status == "supervised"
supervision_status: "supervised" | "auxiliary" | "background_ood" | "unlabeled"
binary_label: 0|1 | null    # only when supervision_status == "supervised" and threshold exists
label_threshold: float | null # predeclared physical threshold used to derive binary_label
group_study: str
group_molecule: str
group_campaign: str
sequence_hash: str           # sha256(sequence)
exclusions: list[str]        # reason codes
source_url: str
source_row_hash: str         # sha256(raw row canonical json)
feature_flags: list[str]     # includes "single_chain_only" when pair_id is null
```

### 3.2 Training row schema (model-ready)
```
features: dict[str, float]
label: 0|1 | null
split_group_id: str          # study + campaign + component fold token
identity_component_id: str
conflict_reason: "" | "label_flip" | "identity_overlap" | "chain_mismatch" | null
```

### 3.3 Conflict-preserving outputs
- `conflicts.tsv` (always emitted)
  - same `sequence_hash` with conflicting labels and provenance.
- `exclusions.tsv`
  - removed rows and deterministic reason codes.

Both outputs are mandatory and must be finalized before feature extraction; feature rows from conflicting/excluded records are never materialized.

## 4) Assay-specific continuous endpoints
- All binary labels must derive from declared continuous assay endpoints (source-specific, experimentally measured).
- No pure proxy labels without an endpoint.
- Endpoint mapping table (minimum required):
  - FLAb: assay-specific metric columns.
  - ProteinGym: `dms_score` extremes (fitness) with explicit assay direction; endpoint-specific non-antibody auxiliary only, never pooled with antibody targets.
  - CANYA: peptide-nucleation proxy columns only; auxiliary only after metadata validation.
  - Figshare A3D: computed/circular A3D score only when source context is explicit; never treated as supervised target.
  - AbDev/AntiRef/SAbDab/Anchors: unlabeled OOD/background references only.

## 5) VH/VL preserved and paired
- Every antibody must preserve molecule and chain identity with `molecule_id` and `chain_id`.
- `pair_id` is required only when both chains are recoverable and attributable to one molecule; if only one chain is available, `pair_id` is `null` and `feature_flags += ["single_chain_only"]`.
- Scoring/exclusion checks run on true pair keys when present. Pair-only models must explicitly exclude `pair_id = null` rows and route them to a separately validated chain model.

## 6) Deterministic caching and manifests

### 6.1 Source cache layout
- `data/cache/raw/<source>/<dataset_sha>/rows.jsonl.gz`
- `data/cache/raw/<source>/<dataset_sha>/dataset.json`

### 6.2 Manifests
- `data/cache/manifests/source_manifest.json` with:
  - source, canonical URL list, request headers, raw bytes hash, row count, sample schema hash, fetch timestamp UTC.
- `data/cache/manifests/exclusion_manifest.json` with deterministic row-level filters and reasons.
- `data/cache/manifests/feature_manifest.json` with feature-versioned definitions and hashes.

- Manifest files are serialized deterministically (sorted keys, fixed decimal formatting), and content hashes exclude volatile timestamps.

### 6.3 Cache governance
- If source content hash changes, previous cohorts are retained in place and new run is marked as non-frozen.
- No run may mix rows from different source manifests without an explicit migration note.
- Source and cache failures are fail-closed: unless an exact matching frozen manifest and cache exist, the pipeline must abort rather than continue.

## 7) Identity and connected-component quarantine

### 7.1 Conservative deterministic identity algorithm (no difflib)
- Use Needleman-Wunsch global alignment with fixed scoring:
  - match = +1, mismatch = -1, gap_open = -2, gap_extend = -1.
- Percent identity = `matches / alignment_length`.
- `is_similar(chain_a, chain_b)` is true when:
  - exact identity OR
  - percent identity >= 0.90 **AND** length ratio in [0.85, 1.15].
- If either chain in a pair satisfies exact or >=90% condition, the pair receives a quarantine edge.

### 7.2 Connected components
- Build graph on `pair_id` when present; single-chain rows use `chain_id` + `molecule_id` identity keys to preserve chain provenance without synthetic pairing keys.
- Create components.
- Add `identity_component_id` = component representative.
- Enforce component-level split groups (`outer_group_id = identity_component_id` at minimum).

## 8) Grouping and fold strategy

### 8.1 Mandatory split keys
- `split_study = study_id`
- `split_campaign = campaign_id`
- `split_component = identity_component_id`
- `split_molecule = molecule_id` for chain-safe audit trails
- `split_chain = chain_id` for single-chain routing checks

### 8.2 Nested grouped CV
- **Outer loop (evaluation)**: grouped K folds on composite key `(split_study, split_campaign, split_component)` so study and leakage groups are both honored.
- **Inner loop (tuning)**: the same composite groups are reused for tuning inside each outer training block; no alternate ungrouped fallback.
- All preprocessing, feature selection, calibration fitting, and model tuning must be fold-local inside each nested loop.
- Outer fold count is set to 5; it may only be reduced deterministically when composite groups are insufficient, and must not fall back to ungrouped splitting.

### 8.3 Leave-study-out is required baseline
- Final comparisons must honor study-aware composite splits as above, with no fallback to ungrouped or standard KFold.
- Source-level leakage checks must run before fold creation and can only reduce feasible folds.

## 9) Metric plan
1. **Spearman ρ** per assay endpoint + study, then macro aggregate across endpoints and studies only when metric definitions are comparable.
2. **MAE** only on endpoint predictions that share the same physical unit and assay family; no pooled MAE across unlike units.
   - **Macro-average only** across comparable assay endpoints when endpoints are directly commensurate.
3. **Group-bootstrap confidence intervals**:
  - bootstrap over outer groups (studies/campaigns/components).
  - compute CI on Spearman and MAE with 95% percentile intervals.
4. **Screening enrichment**:
  - enrichment of failures in top-K risk bins (K = 1%, 5%, 10%).
5. **Calibration only on predeclared physical thresholds**:
  - choose assay-specific clinically relevant thresholds before fit.
  - compute calibration diagnostics only around those binary decisions, not model-global continuous class probabilities.
  - abstain if effective group count or effective sample count drops below declared minimums.

## 10) Risk outputs
- Model outputs become `risk_score` (monotonic, calibrated rank score) and `calibrated_decision_score` at physical threshold, not universal failure probabilities.
- `risk_score` fields:
  - `risk_score_global`
  - `assay_name`
  - `risk_decision_margin`
  - `pair_id`, `identity_component_id`, `split_group`

## 11) GDPa3 handling
- GDPa3 remains **frozen** throughout all CV and development.
- One and only one external command for final external check:
  - `python -m model.evaluate_gdpa3 --run-once --manifest <path> --out results/gdpa3_report.json --expected-data-hash <sha256>`
- GDPa3 rows must never be downloaded, inspected, or logged during development.
- Command requires model/data/code/environment hashes and refuses to run if attempt ledger already contains completed run.
- Labels are loaded only after prediction inputs and model are frozen and serialized.
- Re-run behavior is fail-closed: no result-driven re-executions.
- Use append-only attempt ledger (`results/gdpa3_attempts.jsonl`) with state transitions and completion status.

## 12) VHH format-shift stress tests
- `Arras` and `TNP` are mandatory standalone stress-test sets.
- Calibration for Arras/TNP is **abstain-by-default**.
- They are reported as VHH-format stress tests only when predeclared thresholds and sufficient independent group coverage are available.
- Both are reported as VHH-format stress tests with:
  - separate fold summaries
  - separate calibration-at-threshold
  - separate enrichment tables
- Do not blend into antibody fold metrics.

## 13) CLI contracts (v1.0)
Planned modules and commands are not yet implemented in the current baseline; these are the intended executable boundaries for Plan C.

### Planned module-to-command map
1. `data/fetch.py` → `python -m data.fetch --source <name> --manifest-only`
2. `data/build.py` → `python -m data.build --source-filter all --freeze`
3. `data/assert_pairs.py` → `python -m data.assert_pairs --enforce-vh-vl`
4. `model/train.py` → `python -m model.train --scheme plan-c --cv-strategy leave_study_out --outer-k 5 --inner-k 5 --seed 42`
5. `model/validate.py` → `python -m model.validate --baseline --bootstrap 1000 --endpoint-thresholds config/thresholds.json`
6. `model/predict.py` → `python -m model.predict --input <candidates.json> --out <out.json>`
7. `model/evaluate_gdpa3.py` → `python -m model.evaluate_gdpa3 --run-once --manifest <path> --out results/gdpa3_report.json --expected-data-hash <sha256>`
8. `model/report.py` → `python -m model.report --stress arras` and `python -m model.report --stress tnp`

## 14) Migration and deletion plan
### 14.1 Migration
1. Add `data/cache`, `data/registry`, `data/identity_graph`, `data/pairs`, `data/schema` modules and contract tests.
2. Add one-time backfill CLI to produce cached manifests from current sources.
3. Introduce split/identity-aware train path behind Plan C flag.
4. Add validation and stress-test reports.

### 14.2 Deletion/compat
- Deprecate direct single-source negative import path and remove compatibility shims.
- Remove legacy assumptions about unlabeled single-chain `variant_sequence` after migration complete.
- Remove proxy label pathways instead of retaining compatibility shims for deprecated imports.

## 15) Test matrix
- Unit:
  - schema validation, identity identity engine deterministic outputs, manifest hashing, conflict ledger.
- Integration:
  - fetch + cache + parse each supported source; split integrity checks; nested CV runbook.
- Reproducibility:
  - two identical runs produce same row manifest and row hashes.
- Statistical:
  - bootstrap CI, threshold calibration only, enrichment calculations.
- Regression:
  - old vs new metrics on fixed frozen development slice.
- Negative tests:
  - malformed FASTA/CSV rows, missing VH/VL, conflicting labels.

## 16) Acceptance gates
0. Every documented module entrypoint must pass `python -m <module> --help` once implemented (`--help` smoke check is an implementation gate, not a current-state claim).
1. Deterministic manifests and cache hashes for every run; manifest serialization excludes volatile timestamps.
2. No row enters train/val/test without `sequence_hash`, `source_row_hash`, source manifest link.
3. Zero dropped-conflict rows without recorded `exclusions.tsv` reason; exclusions + conflicts ledgers are consumed before feature extraction.
4. Source-only and format-only baselines are run and compared only with stratified, group-respecting permutation exchangeability checks.
5. Leave-study-out baseline and nested grouped CV are enforced in report generation; no ungrouped fallback.
6. Calibration is abstain-by-default and only emitted when predeclared thresholds and minimum effective group/sample counts are met.
7. No universal failure probability claim; all inference returns `risk_score` + threshold decision scores.
8. GDPa3 command requires expected dataset/model/data/code/config/environment hashes and append-only attempt ledger; must abort on repeated completed attempt.
9. Source cache behavior is fail-closed.
10. OOD identity and feature-distance summaries are reported for external baselines and stress tests.
11. Arras and TNP outputs include abstention reasons when they are below threshold coverage.

## 17) Failure behavior
- Missing endpoint: row marked `exclusions` + not used for primary target; remains in audit artifact.
- Identity graph collision ambiguity: falls back to deterministic chain ordering + hash and logs `identity_ambiguous`.
- Source unavailable: use cached snapshot only when it exactly matches frozen manifest and cache; otherwise the run must abort.
- Split infeasible for a study/campaign: fallback logged and excluded with manual review marker.
- GDPa3 row download/inspection in development is treated as hard failure.

## 18) Reproducibility artifacts
- Required outputs per run:
  - `results/run_meta.json`
  - `results/source_manifests.json`
  - `results/row_manifest.jsonl`
  - `results/conflicts.tsv`
  - `results/exclusions.tsv`
  - `results/metrics.json` (including CIs)
  - `results/folds.json`
  - `results/risk_schema.json`

## 19) CPU/Colab constraints
- Identity DP is O(L²) per pair; for full antibody sets use blocking:
  - exact-match bucket, length banding, and hash bucketing before DP.
- Keep remote fetches serialized where required by API limits; prefer cached reruns.
- Colab baseline target: run in under 12GB RAM with streaming parsers and batch feature extraction.

## 20) Non-goals
- No synthetic label creation.
- No external wet-lab endpoint expansion in this release.
- No change to external output UIs unless contract requires `risk_score` schema migration.

## 21) Staged Spark implementer/reviewer workflow
1. **Implementer-0 (build/cache):** build registry + caches + manifests + pair graph.
2. **Reviewer-0 (contract):** verify schemas, conflict logs, split keys, and manifest determinism.
3. **Implementer-1 (model):** plug nested CV and metrics, risk outputs.
4. **Reviewer-1 (audit):** run evidence checklist from this design and diff manifest integrity.
5. **Implementer-2 (stress):** add Arras/TNP path, finalize GDPa3 command.
6. **Reviewer-2 (freeze):** sign off acceptance gates and freeze Plan C.

## 22) Claim language (only claims that pass evidence checks)
- "Observed behavior from baseline code suggests leakage risk and cohort drift; Plan C is therefore a corrective reconstruction, not a minor tuning pass." (Supported)
- "Risk score is assay-contextual and threshold-conditioned; universal failure probability claims are disabled." (Planned)
- "Modeling results are comparable only within assay-context and outer study leave-out folds." (Planned)
- "GDPa3 performance is externally reported and single-shot only." (Planned)

## 23) External references
- Ginkgo antibody dataset framing and competition structure: `https://datapoints.ginkgo.bio/dataset-access`, `https://github.com/ginkgobioworks/abdev-benchmark`
- VHH format-shift context: `https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2023.1249247/full`
- Developability context (supportive literature URLs):
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC12928636/`
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC12767642/`
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC12963540/`
