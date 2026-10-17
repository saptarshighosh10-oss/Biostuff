# Pre-registration: leakage susceptibility, confirmatory phase

Written 2026-10-16, before any fresh result was computed. The git commit that adds this file, on its own, is the timestamp.
Every confirm script quotes that commit hash in its docstring. The full reasoning is in `PLAN.md` (commit 26ef104). This file is the binding part.

**Change rule.** After this commit, any change to a rule below goes in **Deviations** at the bottom, with the date and reason, committed *before* the affected run. A change made after its result is seen makes that result exploratory.

**Freeze.** No new predictor variants after **2026-12-20**. Anything run later is labelled exploratory.

---

## 1. Data: burned vs fresh

**Burned (already looked at, exploratory only):**
- all 15 cross-lab antibody cells (HIC, AC-SINS, SMAC × 5 models) and all 45 within-GDPa1 cells;
- EuroSAT seeds 0–2;
- LUCAS salinity seeds 0–5, and log-EC as a target on any seed already run.

**Fresh (no DiD ever computed; the only confirmatory data):**
- four Jain2017 → GDPa1 assay pairs (Section 4, C1/C2);
- five LUCAS 2018 soil properties on seeds 6–8 (C3);
- LUCAS log-EC on seeds 6–11, EuroSAT seeds 3–8, GDPa4.1 (pre-specified checks outside Holm).

---

## 2. Metric

**Skill** S = 1 − Σ|pred − obs| / Σ|obs|, summed over the cells of a test.
- S > 0: beats "predict 0 leak". S = 1: perfect.
- pred per cell = mean over planting seeds; obs = the DiD point estimate.
- Every S states its **scope** (all models, or knn_geo only).
- Inside-CI counts and correlations are reported but are **descriptive only**.

---

## 3. Null ladder (reported beside every magnitude result)

| | null | role |
|---|---|---|
| a | predict 0 | the null for C1 and C3 |
| b | oracle constant: the observed mean leak of the table | in-sample reference line only |
| c | **LOAO model mean**: mean over HIC, AC-SINS, SMAC of LS(model) from `out/leakage_susceptibility.csv` (rank-matched), × f_near of the new pair. Ignores label agreement | the null for C2. On the old 15 cells it scored S .76, above the predictor of record (.70) |
| d | agreement-scaled model mean: null c × agree_new / 0.923 (0.923 = mean of the old agreements .968, .890, .912) | secondary comparison for C2 |
| e | in-sample β-rescaled exact copy | ceiling reference |

Salinity also reports a **ρ = 0 arm**: the same planting, with labels and soil values uncorrelated with the test point.

---

## 4. Confirmatory tests (Holm, α = .05, one-sided)

Three tests. Each p-value is from 2000 paired bootstrap replicates. Holm cut-offs: smallest p < .0167, next < .025, last < .05.

### C1 and C2: antibody, fresh assay pairs

**Pairs** (one Jain2017 assay per GDPa1 target, matched by what it measures; agreements from planning, recomputed in stage 1):

| Jain2017 file (FLAb) | GDPa1 column | Spearman agreement | n shared |
|---|---|---|---|
| `polyreactivity/jain2017biophysical_PSR.csv` | PR_CHO | .65 | 108 |
| `polyreactivity/jain2017biophysical_ELISA.csv` | PR_Ova | .43 | 108 |
| `expression/jain2017biophysical_HEK.csv` | Titer | .33 | 129 |
| `thermostability/jain2017biophysical_Tm.csv` | Tm1 | .09 | 129 |

Signs are used as they come (all four agreements are positive). If stage 1 recomputes an agreement more than .01 away from this table, the stage-1 value is used and the gap is logged.

**Models:** NN-copy, kmer-ridge, physchem-RF, esm2-ridge, ablang2-ridge (as in `audit.py`). 4 pairs × 5 models = **20 cells**.

**Predictor of record** (the `did_antibody_rank.py` recipe, unchanged):
- f_near = share of GDPa1 antibodies (with the target measured) whose exact VH+VL is in the Jain file (with its value measured).
- Plant f_near of each GDPa1 hierarchical-cluster test fold into training, with Gaussian-copula labels at the measured Spearman agreement, mapped through the target's own values.
- pred = Spearman(score with planting) − Spearman(score without), mean of planting seeds 0, 1, 2.

**Observed leak** (the `did_antibody.py` part A recipe, unchanged): train on the Jain file, test on GDPa1. near = GDPa1 antibodies with an exact VH+VL copy in the Jain file. 20 half-dedups (drop the copies of a random half of near antibodies). Per-item loss in Spearman units. obs = f_near × [mean Δloss(near) − mean Δloss(far)].

**Bootstrap replicate:** resample near and far antibodies (with replacement, separately, as in `did_antibody.py`) and recompute obs per cell; resample the 3 planting seeds with replacement and recompute pred per cell.

- **C1:** pooled S over all 20 cells (the Holm test always uses all 20) vs predict-0. p = share of replicates with S ≤ 0. **Pass:** Holm-significant.
- **C2:** MAE(pred) < MAE(null c) over the 20 cells. p = share of replicates where MAE(pred) − MAE(null c) ≥ 0. **Pass:** Holm-significant.

**Two stages.** Stage 1 (agreements, predictions, nulls c and d → `out/fresh_antibody_pred.csv`) is committed by **2026-11-01**, before any stage-2 DiD is run.

**Size check.** Per cell, far-item drop ÷ near-item drop. Over one half → "size-sensitive". C1 is reported with and without size-sensitive cells. If the verdict differs, antibody magnitude is called "assumption-dependent".

**Construct check.** Fab Tm (DSF) vs whole-IgG Tm1, and HEK titer vs GDPa1 Titer, may be different quantities. The cells stay as fixed; the write-up calls them "labels that don't match", never "labs disagree".

**Sensitivity rows (outside C1/C2):** ELISA → PR_CHO and BVP ELISA → PR_CHO. CSIBLI → AC-SINS_pH7.4 is reported separately as semi-fresh (its agreement .75 was seen).

### C3: salinity, held-out soil properties

**Targets:** pH_H2O (raw); OC, CaCO3, N, K (log1p). Rows marked "< LOD" or "<0.0" are dropped per target (OC 34, CaCO3 9, N 14, K 39 of 18,975). CaCO3 is measured at 11,211 points only, so its geometry differs; this is accepted. P is not a target (4,976 censored) but stays a feature.

**Features:** the current soil set (pH_H2O, OC, CaCO3, P, N, K, Elev) minus the target. EC is **not** added.

**Seeds** 6, 7, 8. **Radii** 5, 10, 15 km. **Models:** knn_geo (scored) and ridge_soil (negative control). RF is skipped for compute.

**Predictor of record:** `salinity/did_multi.py` `nearest` arm, unchanged except:
- calibration **arm B is primary**: the copula agreement for each target and distance bin comes from train–train 5-NN pairs only (current recipe uses all points). Arm A is reported beside it.

**Observed leak:** the did_multi DiD: obs = f_near × [mean Δloss(near) − mean Δloss(far)], Δloss = (sqerr_dedup − sqerr_full) / var(y_test).

**Bootstrap replicate:** per seed, resample test points once and apply the same draw to all five properties (they share locations and are correlated; a point missing a property is skipped for it); recompute obs and pred from saved per-point vectors (paired).

- **C3:** pooled knn_geo S over 5 properties × 3 radii × 3 seeds = 45 cells, vs predict-0. p = share of replicates with S ≤ 0. **Pass:** Holm-significant.
- **Fallback if behind schedule:** pH only (3 radii × 3 seeds = 9 cells), same rule. Using it is logged under Deviations before the run.

---

## 5. Pre-specified checks outside Holm

**Ranking verdict R.** Model groups are fixed by type:
- memorisers: NN-copy, kmer-ridge, physchem-RF, 1-NN, 5-NN, knn_geo, rf_soil, rf_soil_geo;
- smooth: esm2-ridge, ablang2-ridge, EuroSAT ridge, ridge_soil.

R passes only if all hold:
1. antibody PR_CHO: mean DiD of memorisers > mean DiD of smooth models, CI excluding 0;
2. EuroSAT seeds 3–8: knn1 and knn5 > ridge at t = .80 and .85;
3. salinity (fresh EC seeds and C3 targets): knn_geo > ridge_soil at 10 and 15 km;
4. "a copy only leaks if the model can see it": rf_soil DiD below both knn_geo and rf_soil_geo at 15 km (fresh EC seeds).

R makes no claim about order within memorisers.

**Salinity log-EC, seeds 6–11** (`did_multi.py` nearest, arm A, all 4 models):
- knn_geo pooled S, paired-bootstrap lower bound > 0 → "replicated"; else "did not replicate on fresh splits".
- Matched vs ρ = 0: lower bound of 1 − MAE_matched / MAE_ρ0 ≥ 0.20 → agreement matters for soil; else drop the agreement wording for soil.
- "Unbiased" only if knn_geo pred/obs is in [0.67, 1.5] at all three radii (the 5 km ratio is expected to stay high: 2.52 on seeds 0–5).
- Negative control: ridge_soil mean DiD > 25% of knn_geo's at 10 or 15 km → flag a geometry or size artifact.
- Known predictor failures reported as measured: ridge_soil fake leak (18/18 cells outside CI on seeds 0–5), rf_soil_geo predicted ≈ 0 vs real .027 at 15 km.

**EuroSAT, seeds 3–8** (`sat/` scripts with `range(3, 9)`):
- pred_old: pooled S > 0 **and** S > 0 in at least 5 of 6 seeds → kept; else pred_old magnitude is dropped.
- Redundancy: in at least 5 of 6 seeds, 1-NN leak at t = .95 < 0.5 pts **and** at t = .80 ≥ 2 pts → replicated; else "seen on 3 seeds".
- Expected miss: knn5 under-predicted at t = .80, reported with per-seed CIs.

**GDPa4.1 noise model** (real re-measured copies vs copula copies, 71 mAbs, GDPa1 hierarchical-cluster folds, 5 fold seeds, f ∈ {0, .1, .2}, 20 copula draws). Column map (agreements already seen in planning):

| GDPa1 | GDPa4.1 | agreement | n |
|---|---|---|---|
| HIC | hihplc_normretentiontime | .99 | 69 |
| SMAC | smachplc_retentiontime | .97 | 69 |
| AC-SINS_pH7.4 | acsins_deltaLmax_1X PBS | .91 | 71 |
| PR_CHO | pr_score_CHO | .90 | 70 |
| PR_Ova | pr_score_Ovalbumin | .86 | 70 |
| Tm1 | thermostability_tm1 | .44 | 71 |
| Tm2 | thermostability_tm2 | .28 | 70 |

Memorisers' pooled LS_copula / LS_real, bootstrap over antibodies:
- 95% CI inside [0.67, 1.5] → copula copies are realistic;
- CI entirely > 1.5 → magnitude predictions are upper bounds; entirely < 0.67 → lower bounds;
- otherwise inconclusive (likely, with n = 71).

Smooth models are reported as LS_copula − LS_real with a CI. The SMAC row does **not** test the cross-lab skew idea (GDPa4 is within one lab).

---

## 6. What each outcome means for the headline

| result | claim |
|---|---|
| C1 passes | planting predicts antibody leak size better than "no leak" on new assays |
| C1 fails | no antibody magnitude claim |
| C2 fails | "LS is a model property; label agreement adds nothing" |
| null d ≥ planting | "scale LS by agreement" replaces "plant per assay" |
| C3 fails | soil magnitude is "log-EC only, exploratory" |
| R fails in a domain | no ranking claim in that domain |

**Fallback headline** (used if C1 fails; written now so it is not post-hoc):

> Copy planting separates memorising models from smooth ones in all three domains. Exact copies overstate near-duplicate leaks. A copy only leaks to a model that can see it is a copy. Dedup at a high similarity cutoff gives false reassurance. So the real leak must be measured with a near-vs-far DiD.

**Cut order if behind:** LUCAS 2015 → C3 beyond pH → EuroSAT pred_old (keep redundancy) → ρ = 0 arm → GDPa4 smooth-model arm. Never cut C1, C2 or the null comparisons.

---

## 7. Disclosures

- `salinity/did_multi.py` says its design was fixed before running, but it was untracked until commit 26ef104, so git cannot show the order. Its `nearest` arm was chosen as predictor of record **after** seeing seeds 0–5, and is not the best knn_geo scorer on old data (additive pred_dist_red scored higher, with over-informative labels).
- `results/_killcheck_crosslab.py` (file date 2026-10-14) computed Spearman agreement for PSR and ELISA vs both PR_CHO and PR_Ova, plus HIC, AC-SINS and SMAC.
- Planning on 2026-10-15/16 computed every other candidate agreement: PSR → PR_Ova .58, ELISA → PR_CHO .43, BVP ELISA → PR_CHO .47 and → PR_Ova .44, Fab Tm → Tm2 −.06 and → Tonset −.06, CSIBLI → AC-SINS .75, and all GDPa4.1 agreements above. Planning also counted censored rows in the LUCAS soil columns. **No fresh DiD was computed.**
- Commits from 2026-07 tested PSR rank transfer and polyreactivity domain adaptation (a different question, same assays).
- The antibody predictor of record (rank-matched copula) was adopted after seeing the SMAC miss on the old 15 cells.
- Naive-minus-honest "15/15, corr .92" is demoted to exploratory: `audit.py` E2 scores the naive model on all of GDPa1 but the honest one on the unseen subset only (range restriction).

## 8. Licence

LUCAS 2018 raw data stays in the gitignored `data/external/lucas2018/`. Per-point vectors go only to the gitignored `salinity/_local/`. Committed CSVs and figures hold aggregates only, with no coordinates or point maps. Acknowledge EU/JRC/ESDAC; cite Fernandez-Ugalde et al. 2022 and Orgiazzi et al. 2018.

---

## Deviations

**2026-10-16, before any stage-2 antibody DiD was run** (stage 1 committed in c9fa415 and pushed). These clarify two cases §4 leaves undefined. No rule changes.
1. **Bootstrap draws.** Within each replicate, near and far antibodies are resampled once per pair, and that draw is shared by the pair's 5 models. The 3 planting seeds are resampled once per replicate, shared by all cells.
2. **Size check when the near drop is ≤ 0.** In that case the ratio is undefined. Such a cell is flagged size-sensitive (the conservative choice).

**2026-10-16, before any C3 run.** These are implementation details for C3, written in `salinity/confirm_c3.py`. No rule changes. Base and censored counts were checked and match §4.
3. **Split.** Each seed's 80/20 split is drawn on all 18,975 points, as in did_multi. Points missing the target are then dropped from both train and test. This lets one test-point draw be shared across the five properties.
4. **Arm B pairs.** These are 5-NN pairs among that seed's training points. A pair with a missing value is skipped, as arm A already does.
5. **Extra arms and intervals.** The ρ = 0 arm (§3) is included. Per-cell CIs use 300 replicates, as in did_multi. The C3 p-value uses 2000.
