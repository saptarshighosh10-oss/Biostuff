# 2026-07-13 Baseline Audit — Antibody Failure Pipeline

## Scope and context
- Branch: `codex/antibody-perfection`
- Commit: `44743828b3ed7fd7692c1ece4678fd7e2581b53e`
- Audited files:
  - `model/failure_model.py`
  - `model/train.py`
  - `model/calibration_plot.py`
  - `model/significance.py`
  - `model/validate.py`
  - `model/features.py`
  - `data/flab.py`
  - `data/abdev.py`
  - `data/antiref.py`
  - `data/sabdab.py`
  - `data/anchor_negatives.py`
  - `data/proteingym.py`
  - `data/canya.py`
  - `data/figshare_agg.py`
  - `predictors/multi_predictor.py`
  - `requirements.txt`
  - `.gitignore`
- Supporting context from README and `README.md` baseline description was inspected to cross-check claims.

## Baseline evidence snapshot
1. Baseline run command in README uses `python -m model.train --flab --abdev --anchors --proteingym --sabdab` with AUC claims and grouped CV language.
2. Runtime behavior is live-network dependent for many data loaders (`requests` fetches against `api.github.com`, raw GitHub URLs, HuggingFace/Zenodo, etc.; see data loaders above).
3. No `results/experiment_log.jsonl` exists in current checkout (no persisted baseline run history to diff against in-repo).

## Findings by severity

### 1) High — major reproducibility/lineage risk: no deterministic source snapshotting
- `data/flab.py` builds the FLAb file list from remote API at runtime (`GITHUB_API_URL`) and only falls back to local static names (`KNOWN_FLAB_FILES`) on error, without writing any snapshot manifest/manifest hash for each run (`data/flab.py:19-27`, `data/flab.py:52-67`).
- Equivalent behavior exists in other sources (`data/canya.py` GitHub + raw URLs + Zenodo fallback, `data/abdev.py` API + fallback URLs, `data/antiref.py` head probes, `data/figshare_agg.py` API listing + file downloads).
- No SHA-256 or row-exclusion manifests are persisted in the pipeline for any source. `requirements.txt` and `.gitignore` do not include any cache artifact paths for canonical source snapshots.
- Impact: identical command invocations can ingest different cohorts as upstream files change, including the FLAb cohort. This directly affects reproducibility and any downstream claims.

### 2) High — source conflict handling is destructive and undocumented
- `model/train.py` deduplicates by sequence globally across failures then workings using a single `seen` set (`model/train.py:265-277`) and silently drops second occurrences regardless of label conflicts.
- This means the same sequence cannot exist as both failure and working; conflicts are not surfaced, not triaged, and not traced.
- Impact: unresolvable contradictions are erased; leakage into label confidence and inflated signal trust can result.

### 3) High — grouped leakage can silently degrade to non-grouped splits
- In `model/failure_model.py`, grouped CV is only used when `n_unique_groups >= n_splits` (`failure_model.py:95-103`), where `n_splits=min(5, n_failures)`.
- For valid datasets with multiple related variants but insufficient distinct groups, it silently falls back to `StratifiedKFold` (`failure_model.py:104-108`), allowing sibling variants to split across folds.
- `build_training_data` also assigns broad groups only by source-provided `group_id` or `dataset` or sequence hash fallback (`model/train.py:87-101`), so many rows can map to weakly informative groups.
- Impact: leakage and optimistic CV estimates are possible even when users interpret output as grouped/strain-safe.

### 4) High — assay-level label assumptions are mixed and insufficient for endpoint-specific supervision
- `model/train.py` mixes endpoint-supervised and non-endpoint rows in one target pool, including `abdev`, `antiref`, `sabdab`, and `anchors` as proxy negatives, and applies inconsistent assumptions for `figshare_agg`, `canya`, and `proteingym`.
- This conflicts with a strict “assay-specific continuous endpoint + supervised labels only” strategy; current training objective can mix incomparable semantics under one binary target (`model/train.py:225-267`, `data/abdev.py:121-179`, `data/antiref.py:105-139`, `data/sabdab.py:59-95`, `data/anchor_negatives.py:21-58`, `data/canya.py`, `data/figshare_agg.py`, `data/proteingym.py`).
- Impact: baseline model blends assay supervision with unrelated source priors and proxy negatives, producing optimistic but weakly interpretable labels.

### 5) High — no antibody chain pairing or molecule-level structure preserved
- FLAb loader keeps a single selected sequence column (`load_dataset`) rather than guaranteed VH/VL paired structure; mutation loaders return only `variant_sequence` and `group_id` (`data/flab.py:84-99`, `data/flab.py:195-218`).
- Anchor/Negative loaders and non-antibody datasets similarly output single `variant_sequence` strings (`data/anchor_negatives.py:40-53`, `data/abdev.py:174-179`, `data/sabdab.py:59-95`, etc.).
- No graph construction or chain-level linked identity quarantine is implemented.
- Impact: VH/VL identity confounds, heavy/light interchange, and chain-combination leakage are not modeled or mitigated.

### 6) Medium — no deterministic quarantine using identity-derived connected components
- No component-based quarantine exists anywhere in the stack; no pairwise identity thresholds are applied to create anti-leak components.
- `model/train.py` can still group only by `group_id`, while `model/failure_model.py` groups folds by those IDs but does not enforce linked exclusions when near-duplicate chains overlap across studies/groups.
- Impact: near-duplicate antibodies with one-to-one relationship may cross training/test splits.

### 7) Medium — calibration claims exceed implementation guarantees
- `model/calibration_plot.py` and `model/failure_model.py` both compute/use calibration internals, but no mandatory threshold-specific calibration policy exists.
- `model/failure_model.predict_proba` exposes averaged calibrated probabilities regardless of assay type (`failure_model.py:135-147`); no assay-conditioned safety thresholds or abstention boundaries.
- `model/calibration_plot.py` computes model-wide MACE (`model/calibration_plot.py:137-151`) and publishes calibration plot regardless of endpoint relevance.
- Impact: calibration is treated as a global score and can be over-interpreted as actionable probability for all endpoints.

### 8) Medium — validation and significance do not enforce leave-study-out or endpoint-level separability constraints
- Significance/validation use the same global dedup-and-featurize pool and do not enforce strict study-wise exclusion beyond default group splitting (`model/significance.py:27-63`, `model/validate.py:20-31`).
- Impact: reported AUC and ranking significance may mix assay domains and overstate generalization.

## Unsupported claims and mismatches with implementation
1. README text says grouped CV prevents leakage in general; implementation falls back to standard folds when group count is insufficient (`failure_model.py:95-108`) and also assigns low-information group IDs (`model/train.py:97-101`).
2. Baseline feature count narratives (~23+) are imprecise but not numerically false; explicit `FEATURE_NAMES` length is exactly 27 (`model/features.py:18-50`).
3. “Purely experimental” framing is undermined by multiple fixed-negative sequence resources used as working labels without assay endpoints (`data/abdev.py`, `data/antiref.py`, `data/sabdab.py`, `data/anchor_negatives.py`).

## Source-by-source label validity matrix (binary mapping from current loaders)
| Source | Current binary mapping | Plan C classification | Label rule quality | Location |
|---|---|---|---|---|
| FLAb | top/bottom percentile by assay score, with assay-type aware sign (`low_is_bad`) | Supervised antibody endpoint candidates (contested in baseline implementation) | Strong if assay metadata is correct; depends on robust column detection | `data/flab.py:133-138`, `data/flab.py:159-170` |
| ProteinGym | score bottom/highest percentile by DMS fitness, then sampled by seed or extremes | Endpoint-specific **non-antibody auxiliary only**; never pooled with antibody targets | Medium: useful mutational trend signal, but not antibody-target direct supervision in Plan C | `data/proteingym.py:150-176`, `data/proteingym.py:169-173` |
| CANYA | label heuristics/nucleator parsing plus fallback threshold | **Peptide nucleation auxiliary only** | Weak for direct antibody failure supervision; parser depends on schema variability | `data/canya.py:106-134`, `data/canya.py:77-146` |
| Figshare A3D | score threshold (`> 0.0`) from inferred column | **Computed/circular auxiliary only**; never supervised | Medium: score column auto-detected heuristically and is not a direct experimental endpoint | `data/figshare_agg.py:53-87`, `data/figshare_agg.py:105-111` |
| AbDev | all loaded sequences marked working | **Unlabeled background/OOD reference only** | Weak for outcome supervision; no assay endpoints | `data/abdev.py:121-179` |
| AntiRef | all loaded sequences marked working | **Unlabeled background/OOD reference only** | Weak; germline corpus only as proxy non-aggregation | `data/antiref.py:81-139`, `data/antiref.py:134-139` |
| SAbDab | all extracted sequences marked working | **Unlabeled background/OOD reference only** | Weak; structural deposition not direct assay outcome | `data/sabdab.py:67-94` |
| anchors (PDB anchors) | all phase1 anchor sequences marked working | **Unlabeled background/OOD reference only** | Weak; crystallization-only context | `data/anchor_negatives.py:21-58` |

## FLAb cohort drift fact
- `data/flab.py` is live-network first: `list_flab_datasets()` calls GitHub contents at runtime and only falls back to `KNOWN_FLAB_FILES` on failure (`data/flab.py:52-67`).
- Combined with dedup and per-file parsing behavior (`data/flab.py:223-255`), any upstream file set/update can alter cohort composition and sequence set between two runs.
- Therefore, the FLAb cohort is not fixed without snapshot caching and can materially change across run time.

## Data/model leakage map (evidence highlights)
- Dedupe order makes class precedence flow failure-first, with working labels dropped when sequence collisions occur (`model/train.py:265-277`).
- Group assignment depends on source-provided metadata; otherwise sequence hash fallback can create singleton groups that do not prevent related split leakage (`model/train.py:97-101`).
- `failure_model` and `significance` reuse this grouping; when groups are insufficient, they explicitly use ungrouped splits (`failure_model.py:95-108`, `model/significance.py:55-63`).

## Recommended risk posture before redesign
- Treat all current baseline AUC/calibration/significance outputs as **non-final for publication** unless source manifests, cohort manifests, identity quarantine, and assay-stratified evaluation are added.
- Keep this audit artifacts-only; no code changes were made in this pass.

## External references (for design constraints and dataset context)
- Ginkgo datapoints portal references antibody datasets (`GDPa1`, `GDPa2.1`, `GDPa3`) and competition framing in site docs and search snippets: `https://datapoints.ginkgo.bio/dataset-access`.
- Ginkgo antibody benchmark baseline conventions and evaluation structure (`GDPa1`, Spearman/top-10% recall) are visible in the official repo: `https://github.com/ginkgobioworks/abdev-benchmark`.
- Frontiers work on VHH optimization and format-shift implications supports separation of VHH stress testing from canonical antibody pipelines: `https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2023.1249247/full`.
- Additional cited primary context URLs (if needed for claim language grounding): `https://pmc.ncbi.nlm.nih.gov/articles/PMC12928636/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC12767642/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC12963540/`.
