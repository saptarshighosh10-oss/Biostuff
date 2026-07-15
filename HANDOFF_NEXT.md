# Handoff — Antibody Aggregation Pipeline (PLM scale-up)

_Last updated: 2026-07-14. Branch: `codex/antibody-perfection`._

## TL;DR
The pipeline was rebuilt from a 28-feature logistic-regression toy into a
PLM-backed, leakage-controlled, multi-source antibody developability model with
a clean two-head split. The current Head B artifact has been retrained after
data-integrity and feature-shape fixes; GDPa3 remains frozen.

## Latest integrated run — 2026-07-14
Public aggregation snapshots are now wired into Head B. After canonical
sequence/assay dedupe, the antibody cohort contains 1,828 aggregation rows
before GDPa profile additions, including 597 DOTAD rows and 939 paired
observations. All 624 canonical sequences now have complete ESM2 + AbLang2
embeddings. The autonomous lab promoted the PLM-backed IgG champion to
`model/saved/champion_head_b_aggregation.pkl`.

- IgG: mean Spearman `0.147718` across 17 evaluated assays; 13 serving assays after four negative assays were quarantined.
- GDPa4 mAb: mean Spearman `0.06165` across 12 evaluated assays; 7 serving assays after five negative assays were quarantined.
- GDPa4 bispecific: mean Spearman `0.082925` across 12 evaluated assays; 8 serving assays after four negative assays were quarantined.
- VHH: mean Spearman `-0.278375`; quarantined and not promoted for serving.
- GDPa3 remained frozen and excluded from development.

Refinements in this run: public FLAb rows now use the canonical VH+VL
sequence and existing assay IDs, molecule groups use SHA-256 identities, SEC
direction metadata is corrected, FcRn endpoints are excluded from aggregation,
all PLM embeddings are precomputed, and negative grouped-CV assays remain in
reports but are removed from serving artifacts.

Run artifacts are under `results/autonomous_lab/`; the aggregate report is
`results/autonomous_lab_report.json`.

The improvement loop is also populated: `results/head_b_calibration.json`
contains 28 cross-fit empirical-tail calibration diagnostics, and
`results/active_learning_candidates.json` contains the top 25 candidates from a
606-sequence pool ranked by PLM/biophysical disagreement plus modeled risk.
Calibration ECE remains high, so these scores are diagnostic and must not be
reported as physical failure probabilities. GDPa3 remains the locked external
validation gate and must not be rerun.

Design doc: [docs/superpowers/specs/2026-07-14-plm-scaleup-design.md](docs/superpowers/specs/2026-07-14-plm-scaleup-design.md).
Architecture gate that started this: the 2026-07-14 review at commit `5dc5c4b`.

## Two-head architecture (keep them separate — separate metrics, artifacts, reports)
- **Head A** — general ProteinGym mutation-fitness. `model/pretrain_proteingym_fitness.py`.
  Trains/serves over `HEAD_A_FEATURE_NAMES` (14 sequence-intrinsic features).
  Protein-disjoint split. Feeds Head B only via the derived `proteingym_fitness_score`.
- **Head B** — antibody aggregation. `model/head_b_gbm.py`. Per-assay regression on
  `[ PLM(ESM-2 + AbLang2) | biophysical(28) ]`, leave-homology-cluster-out grouped CV,
  per-assay Spearman/MAE/enrichment + group-bootstrap CIs. It accepts only
  an explicit aggregation metric allowlist (AC-SINS/HIC/SEC/PSR/SAS/SMAC and
  related aggregation assays) and uses
  `molecule_group_id` when available. Identical metrics from different studies
  are evaluated separately, with target/specificity context reported but never
  treated as an aggregation label. Trains ONLY on the supervised bucket —
  background/auxiliary never contaminate the target.

## File map (what each piece does)
**Features / models**
- `model/features.py` — biophysical features + `HEAD_A_FEATURE_NAMES`, `feature_schema_hash`, shared `general_fitness_score` (kills train/serve skew).
- `features/plm.py` — ESM-2 + AbLang2 cache-first embeddings (`data/cache/embeddings/`), guarded heavy imports, `--precompute` CLI.
- `model/head_b_gbm.py` — Head B driver (`train_head_b`, `evaluate_assay`, `build_feature_vector`); provenance block in the report.
- `model/failure_model.py` — legacy ensemble; now schema-hash-guarded on load.
- `model/risk.py` — Phase 4 risk outputs (`risk_score_global`, `risk_decision_margin`, calibrated decision score, abstain). **No universal failure probability.**
- `model/metrics.py` — Spearman / MAE / top-K enrichment / group-bootstrap CI / calibration-at-threshold (pure stdlib).
- `model/plan_c_cv.py` — leave-cluster-out nested grouped folds (reuses `data/identity_graph.py`).
- `model/evaluate_gdpa3.py` — GDPa3 single-shot, fail-closed, hash-gated, append-only ledger.

**Data layer**
- `data/sources_registry.py` — central source catalog with roles; add a source = one `SourceSpec`.
- `data/cohort.py` — `build_antibody_cohort` (FLAb+public aggregation+GDPa), `build_full_cohort` (all buckets), `emit_ledgers`.
- `data/public_aggregation.py` — dependency-free loaders for staged public FLAb aggregation CSVs and measured DOTAD XLSX endpoints; exact observation dedupe.
- `data/contract.py` — preserves `source_version` and structured `assay_conditions` through normalization and ledgers.
- `data/flab.py` — `load_all_developability_data` (ALL FLAb assay subdirs, continuous endpoints).
- `data/gdpa.py` — GDPa1/2 loader (real verified schema); `load_gdpa3` = frozen tripwire.
- `data/oas.py` — Observed Antibody Space natural repertoires (background_ood).
- `data/thera_sabdab.py` — clinical-stage therapeutic mAbs (background_ood).
- `data/normalize.py` — adapter onto `data/contract.py::NormalizedRow`.
- `data/pairs.py` — VH/VL chain typing + deterministic pairing (heuristic — see caveats).

## Data sources & roles
| Source | Role | State |
|---|---|---|
| FLAb (all assays; Head B filters to aggregation family) | supervised | ✅ local, 9,202 rows loading |
| Public FLAb aggregation snapshot | supervised | ✅ 853 sequence-linked measured rows |
| DOTAD measured aggregation endpoints | supervised | ✅ 597 sequence-linked measured rows; predicted/negated columns excluded |
| GDPa1 / GDPa2 | supervised | loader ready — **drop CSV in `data/external/gdpa/`** |
| OAS | background_ood | loader ready — drop units in `data/external/oas/` |
| Thera-SAbDab | background_ood | loader ready — drop CSV in `data/external/thera_sabdab/` |
| AbDev / AntiRef / SAbDab | background_ood | network loaders (need `requests`) |
| CANYA / Figshare-A3D | auxiliary | network loaders (need `requests`) |

Supervised = trains Head B. background_ood = OOD/abstention reference. auxiliary = derived features.

## How to get to a trained model
```bash
make install                              # torch, transformers, ablang2, lightgbm, sklearn, numpy, requests
 make test                                 # confirm 173+ green after install

# Head A (optional derived feature):
python -m data.build_proteingym_partition # build protein-disjoint cohort (needs network)
make head-a                               # train Head A

# Head B:
make cohort                               # dry-run: see source composition
make embeddings SEQS=<seqs.txt>           # fill PLM cache (ESM-2 + AbLang2), one-time
make head-b-plm                           # per-assay grouped-CV eval, PLM features
#   or: make head-b                       # biophysical-only (runs without PLM cache)
#   add PROGRESS_EVERY=1 or 2 for row-level live progress output

# External holdout (once, after model frozen):
make gdpa3 FILE=data/external/gdpa/GDPa3.csv HASH=<sha256>
```

## Known limitations / must-do before any real-world claim
1. **Training dependencies remain environment-specific** — run the autonomous lab; use `make install` first if PLM/GBM packages are absent.
2. **`classify_chain` in `data/pairs.py` is a heuristic** (FR4 motif). Swap for ANARCI/HMMER before a production gate; misclassifies truncated/VHH/engineered chains.
3. **GDPa2 columns assumed** (no public CSV to verify); GDPa1 verified real. Reconcile headers when the real file lands.
4. **`normalize` injects `endpoint_unit='unknown'`** stub — replace with real assay units when loaders provide them.
5. **`risk.py` needs per-assay calibrators fitted at predeclared physical thresholds** for `calibrated_decision_score` to be non-null.
6. **`evaluate_gdpa3` ships no default scorer** by design — wire a frozen serialized model into `predict_fn` before the single-shot run.
7. **Paired rows must clear `chain_id` before `normalize_legacy_row`** (contract forbids pair_id + chain_id together) — `emit_ledgers` already does this; any new path must too.
8. **Auxiliary sources (CANYA/Figshare-A3D) are wired but not yet used as features** — next modeling step is an auxiliary head → derived feature, like `proteingym_fitness_score`.

## What must NOT be claimed (from the architecture gate)
- Protein-disjoint ≠ leakage-free (homologs) — only true after leave-cluster-out CV runs.
- Pooled AUC over mixed endpoints is meaningless — per-assay only.
- ProteinGym transfer ≠ antibody skill — the PLM is the real transfer.
- No universal failure probability — assay-contextual risk only.

## Next steps (priority order)
1. `make autonomous-lab` (or `make install` first if dependencies are absent) → profile-specific Head B candidates with the public aggregation rows.
2. `make embeddings` + `make head-b-plm` → the PLM lift over baseline.
3. Fit per-assay calibrators; wire into `risk.py`.
4. Swap `classify_chain` for ANARCI; verify GDPa2 headers.
5. Freeze a model; run the single-shot GDPa3 eval.
6. Add auxiliary (CANYA/A3D) derived-feature head.
```
```
