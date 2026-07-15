# 2026-07-14 — Antibody Aggregation Scale-Up Design (PLM features + full data + Plan C activation)

## 0) Status & intent
- Goal: maximize **real-world antibody-aggregation prediction** quality, not benchmark-gaming a broken split.
- Owner workflow: implementer (Claude/Codex) + reviewer. M4/M5 Mac, CPU/MPS only, no cloud GPU.
- Non-negotiable framing from the architect gate (`2026-07-14` review of `5dc5c4b`): fix the serve-time blockers first, then change the *feature backbone*, then activate the already-written Plan C leakage controls. ProteinGym volume is transfer signal, not the objective.

### The one-paragraph thesis
The current model is a 28-feature logistic-regression + random-forest ensemble on hand-crafted biophysical scores (`model/features.py`). That is the ceiling. No quantity of ProteinGym rows moves antibody-aggregation accuracy meaningfully because ProteinGym is non-antibody. The real levers, in order of real-world impact: **(1) replace the feature backbone with protein language model embeddings (general ESM-2 + antibody-specific AbLang2/AntiBERTy)**, **(2) use every antibody developability dataset in the repo (and GDPa1/2) with proper VH/VL pairing and continuous endpoints**, **(3) evaluate with leave-homology-cluster-out CV using the Plan C modules that already exist but are wired to nothing**. ProteinGym stays, correctly scoped, as an auxiliary transfer feature.

---

## 1) Phase 0 — Fix the serve-time blockers (prerequisite, ~half a day)
No training run is worth doing until these land; today a `--proteingym` model is wrong the moment it scores anything.

| # | Blocker | Fix | Files |
|---|---------|-----|-------|
| 0.1 | `proteingym_fitness_score` is real at train time, always 0.0 at inference (train/serve skew) | Compute the derived feature on the **same code path** for train and serve. Make `extract_features` lazy-load Head A once (module-level cache) and populate the score when a sequence is present; remove the ad-hoc loop in `train.py`. | `model/features.py`, `model/train.py:255-266` |
| 0.2 | `score_general_fitness` hardcodes `mutations=[]`, but Head A trains on real mutation lists | Pass real mutations through; or drop the 4 mutation-delta features from Head A's vector so train==serve by construction. | `model/pretrain_proteingym_fitness.py:44-46` |
| 0.3 | No feature-schema version check on load; a stale `model.pkl` (27-col, pre-commit) is on disk now | Store `sha256(FEATURE_NAMES)` in every saved model; assert on `load()`. | `model/failure_model.py:237-245`, `model/pretrain_proteingym_fitness.py:95-100` |
| 0.4 | Add a train/serve parity self-check | One `demo()`/`test_` that featurizes a fixed sequence through the training path and `model/predict.py`, asserts vectors identical. | `tests/test_train_serve_parity.py` (new) |

**Acceptance:** parity test green; loading a mismatched-schema pickle raises; `--proteingym` score is nonzero and identical at train and serve for the same sequence.

---

## 2) Phase 1 — Data expansion (the *right* data)

### 2.1 Use ALL of FLAb, not just `aggregation/`
`data/flab.py` hardcodes the `aggregation/` subdir (31 CSVs). The repo already has, on disk under `data/external/flab/`:
- `aggregation/` — HIC, AC-SINS, SEC, CGE, SGAC-SINS, CIC (the direct targets)
- `thermostability/` — Tm, DSF, DLS (aggregation-correlated)
- `polyreactivity/` — PSR, BVP, ELISA, CIC-RT (self-association proxies)
- `pharmacokinetics/`, `expression/`, `binding/` — developability context / auxiliary

**Change:** generalize the loader to walk every subdir, tag each row with `assay_family` + `assay_metric` + `endpoint_direction`, and keep them as *separate endpoints* (not one pooled binary). Aggregation assays are the supervised target; the rest are auxiliary features or separate heads.

### 2.2 Add the gold-standard antibody developability sets
These are what make the model publishable / real-world credible:
- **Ginkgo GDPa1 + GDPa2** (`datapoints.ginkgo.bio`, `github.com/ginkgobioworks/abdev-benchmark`) — thousands of antibodies, paired VH/VL, multiple developability assays incl. AC-SINS/HIC/Tm/PR. This is *the* benchmark the design docs already reference. **GDPa3 stays frozen** (design §11) — single-shot holdout, never downloaded in dev loops.
- **Jain 2017 (137 clinical mAbs) + Jain 2024 assessment** — already partially in FLAb; ensure full 12-assay panel is ingested with continuous values.
- **Thera-SAbDab / SAbDab** — therapeutic + structural antibodies (already referenced) for OOD/background and structure.
- **Boughter 2020 / Harvey 2022 polyreactivity** — self-association labels.

### 2.3 Endpoints as regression, not just top/bottom-quartile binary
Current labeling throws away most of the signal by keeping only the extreme 25% tails and binarizing. Keep the **continuous endpoint value** (AC-SINS Δλmax, HIC retention time, SEC monomer %, Tagg) as the primary regression target per assay, with a predeclared physical threshold for the binary decision (matches design §4, §10). This alone typically lifts Spearman materially over tail-binarization.

### 2.4 ProteinGym — keep, but scope honestly
- Full 2.7M is fine for **Head A** if we keep the two-head design. But the derived scalar is a weak feature for antibodies.
- **Better use:** use ProteinGym (and general protein corpora) only through the PLM, which is already pretrained on hundreds of millions of sequences. That subsumes ProteinGym transfer for free. Recommendation: **demote Head A to optional**; the PLM embedding is the superior transfer path.

**Acceptance:** one normalized cohort table (`NormalizedRow` schema from `data/contract.py`) covering all FLAb subdirs + GDPa1/2 + Jain, with per-assay endpoint values, VH/VL pairing where recoverable, source manifests + SHA-256 (design §6).

---

## 3) Phase 2 — Model architecture upgrade (the real leverage)

### 3.1 Feature backbone: PLM embeddings
Replace the 28-number bottleneck with a rich, learned representation, **concatenated** with the existing biophysical features (which remain useful inductive bias — keep them, don't delete):

1. **General PLM:** ESM-2 `t33_650M` — mean-pooled + CLS embedding (1280-d). Runs on M4 via MPS or CPU; ~230-residue antibody Fv is cheap.
2. **Antibody-specific PLM:** **AbLang2** (or AntiBERTy) — trained on OAS antibody repertoires, captures VH/VL germline + CDR context that ESM-2 misses. Small, fast, CPU-friendly.
3. **Existing biophysical block:** the 28 features stay as an explicit, interpretable channel (TANGO/AGGRESCAN/CamSol/charge/hydrophobic-patch). These give the explanation layer something mechanistic to point at.
4. **Optional structural block (Phase 2b):** **IgFold** (antibody-specific, far lighter than ESMFold on M4) → SAP score, developability index, CDR SASA, per-residue pLDDT.

Final feature vector = `[ESM2_mean | AbLang2_mean | biophysical_28 | structural_k]`, cached per unique sequence (Phase 0 memoization makes 1.26M-scale featurization tractable).

### 3.2 Model head — stay lazy on top of rich features
Frozen embeddings + a strong tabular head is the M4 sweet spot (no GPU, hours not days):
- **Primary:** LightGBM / gradient-boosted trees per assay endpoint (regression) — handles high-dim embeddings, gives feature attribution, trains in minutes on CPU.
- **Alternative:** small 2-layer MLP with dropout on MPS.
- **Stretch (only if data supports):** LoRA fine-tune AbLang2 on the aggregation endpoints — M4-feasible with `peft`, but frozen-embeddings first; fine-tuning is diminishing returns until data volume justifies it.

Keep the ensemble + calibration machinery from `model/failure_model.py`; swap what feeds it.

### 3.3 Two-head boundary preserved
- **Head A** (`pretrain_proteingym_fitness.py`): general fitness, protein-disjoint, **per-endpoint-family metrics only**. Optional after PLM lands.
- **Head B** (antibody aggregation): PLM-backed, antibody data only, Plan C CV. Separate metrics/reports (unchanged rule).

**Acceptance:** Head B trained on PLM features beats the 28-feature baseline on leave-cluster-out Spearman/PR-AUC by a reported margin, on the same frozen split.

---

## 4) Phase 3 — Activate Plan C evaluation (wire in what's already built)
`data/identity_graph.py`, `model/preprocess.py`, `data/contract.py::NormalizedRow` are written + tested with **zero non-test importers**. Wire them into training:
- **Homology quarantine:** identity-graph connected components (Needleman-Wunsch ≥90% id, length ratio 0.85–1.15) → `identity_component_id`. Protein-disjoint by UniProt ID is *not enough*; homologs/paralogs still leak.
- **Leave-study-out + leave-cluster-out nested CV** (design §8): outer groups = `(study, campaign, identity_component)`. No ungrouped fallback.
- **Conflict/exclusion ledgers** (design §3.3) replace the current destructive global dedup that silently drops the losing label.
- **Metrics** (design §9): per-assay Spearman ρ, MAE (same-unit only), top-K enrichment (K=1/5/10%), group-bootstrap 95% CIs, calibration only at predeclared thresholds.

**Acceptance:** every reported number carries an effective-independent-group count; no fold splits a homology component; conflicts.tsv + exclusions.tsv emitted per run.

---

## 5) Phase 4 — Real-world outputs & honesty
- Output `risk_score` (calibrated rank) + `calibrated_decision_score` at physical threshold per assay, not a universal failure probability (design §10).
- Abstain when ensemble disagreement or OOD distance exceeds ceiling (machinery exists in `predict.py`).
- Explanations stay evidence-bounded (`model/explanations.py`) — mechanistic hypotheses flagged as hypotheses, plus nearest-neighbor "resembles known antibody X".
- GDPa3 single-shot command (design §11) for the one external claim.

---

## 6) Compute budget on M4/M5 (feasibility)
| Step | Cost | Notes |
|------|------|-------|
| ESM-2 650M embed, ~10k antibodies | minutes–1hr | MPS batch; cache to disk keyed by seq hash |
| ESM-2 650M embed, 1.26M ProteinGym | hours (one-time) | only if keeping Head A on PLM; else skip |
| AbLang2 embed, all antibodies | minutes | small model, CPU fine |
| IgFold structures, ~10k Fv | hours (one-time, cached) | Phase 2b, optional |
| LightGBM per-endpoint train | minutes | CPU |
| Full nested CV | tens of minutes | embeddings cached |
Everything fits M4; PLM weights are the only large download (ESM-2 650M ≈ 2.5 GB).

---

## 7) Local models to provision (need go-ahead)
1. **ESM-2** `esm2_t33_650M_UR50D` — `fair-esm` or HF `facebook/esm2_t33_650M_UR50D`. General PLM backbone.
2. **AbLang2** (`ablang2`) or **AntiBERTy** (`antiberty`) — antibody-specific PLM.
3. **ANARCI** (IMGT/Chothia numbering; needs HMMER) — CDR-aware features, VH/VL split.
4. **IgFold** — antibody structure (Phase 2b; lighter than ESMFold).
5. **LightGBM** — CPU tabular head.
New deps: `torch`, `fair-esm`/`transformers`, `ablang2`, `anarci`/`hmmer`, `lightgbm`, optional `igfold`. All M4-runnable.

---

## 8) Sequencing / ownership
- **P0 blockers** — must land first (small, mechanical). Claude or Codex.
- **P1 data** — FLAb full walk + GDPa ingest + NormalizedRow mapping. Highest data leverage.
- **P2 PLM** — embedding cache + LightGBM head. Highest model leverage.
- **P3 Plan C wiring** — homology CV + ledgers + metrics. Highest credibility leverage.
- **P4 outputs** — risk schema + GDPa3. Ship.

## 9) What must NOT be claimed (carried from the gate)
- Protein-disjoint ≠ leakage-free (homologs). Fixed only after Phase 3.
- Pooled AUC over mixed endpoints is meaningless — per-assay only.
- ProteinGym transfer ≈ antibody skill — it doesn't; the PLM is the real transfer.
- No universal failure probability; assay-contextual risk only.
