# Final plan: leakage susceptibility, ISEF 2027

**Base.** The rigor plan, which two of the three judges rated best. Grafted in:
- from the falsify plan: the leave-one-assay-out (LOAO) model-mean null, a paired bootstrap, per-radius bias, and held-out soil properties;
- from the lean plan: fair logistics, "own the method", and the three-null scoreboard;
- from the impact plan: the "same overlap, different agreement" test, the model-relative overlap finding, compliance, and the pitch.

**Numbers checked read-only today (Oct 16)** from `out/did_antibody_rank.csv`, `out/did_antibody_real.csv`, `out/leakage_susceptibility.csv`, `salinity/did.csv`, `salinity/did_seeds3to5.csv`, `salinity/did_rank_seeds*.csv`, `salinity/did_multi_seeds*.csv`, `salinity/audit.log`, `sat/did.csv` and `sat/real_planted.csv`, plus the FLAb, GDPa1 and GDPa4.1 files. The GDPa4 agreements were recomputed today with `python -I` and are no longer the judges' figures.

**Skill metric used everywhere:** S = 1 − Σ|pred − obs| / Σ|obs|.
- S > 0 means the predictor beats "predict 0 leak".
- S = 1 is perfect.
- Every S row says its **scope** (all models or knn_geo only), because the two can differ in sign.

**Burned vs fresh data.** Burned (already seen): all 15 cross-lab antibody cells, all 45 within-GDPa1 cells, EuroSAT seeds 0–2, salinity seeds 0–5, and log-EC as a salinity target. The only confirmatory data are targets whose leak (DiD) has never been computed:
- four new Jain2017 → GDPa1 assay pairs;
- five new soil properties, on fresh seeds 6–8.

GDPa4 (a noise-model check) and LUCAS 2015 (conditional) are also new data.

**Progress (update as steps finish).**
- 2026-10-16: Step 0 done (commit A 26ef104, PREREG b9076c3). Step 3 done: `score.py` reproduces all 40 §3 values to ±.01.
- 2026-10-16: Step 4 stage 1 done. `confirm_antibody.py` → `out/fresh_antibody_pred.csv`. All 7 agreements are within .01 of PREREG. f_near is .54–.55, the same as the old cross-lab f_near (.54). The sensitivity and semi-fresh rows are predicted too.
- Construct check (sources: GDPa4.1 Definitions sheet and the FLAb column headers):

  | pair | Jain 2017 assay | GDPa1 assay | construct |
  |---|---|---|---|
  | PSR → PR_CHO | PSR SMP score, CHO membrane reagent | polyreactivity vs CHO cell lysate | same construct |
  | ELISA → PR_Ova | multi-antigen ELISA | polyreactivity vs ovalbumin | related; antigen differs |
  | Tm → Tm1 | Fab Tm by DSF | whole-IgG first transition, PTS-IF | **labels that don't match**; IgG Tm1 need not be the Fab |
  | HEK → Titer | HEK titer, mg/L | not defined in GDPa4 | check GDPa1 docs |

  Cells stay as fixed.
- 2026-10-16: Step 4 stage 2 done (`confirm_antibody_did.py`; Deviations entry b97f07a committed before the run).
  - C1 fails: S .05, p .42.
  - C2 passes: p < .001.
  - null d beats planting: S .22.
  - Readings are in RESULTS §0b.
- 2026-10-16: Step 7 done (`salinity/confirm_c3.py`; Deviations 3–5 in 0e4961f, committed before the run).
  - C3 fails: S −.15, p .72. Planting over-predicts the knn_geo leak 2–3× on pH, OC, N and K; CaCO3 is under-predicted.
  - Code check: on log-EC it reproduces `did_multi.py` exactly.
  - Final Holm: C2 passes, C1 and C3 fail. Soil magnitude is "log-EC only, exploratory", and the fallback headline stands. Details in RESULTS §0c.
- Next: Step 5 (GDPa4) and Step 6 (salinity log-EC seeds 6–11, needed for the rest of R). Step 2 prior-art reading is still open. Student: ESDAC application, fair dates by Oct 23.

---

## 1. Headline claim

**Headline.** Planting copies of test items into training does three things in antibody, satellite and soil benchmarks:
1. **It separates models an overlap can inflate (memorisers) from models it cannot (smooth ones),** as long as the overlap is visible in the model's own features. It does **not** order models within the memoriser group:
   - salinity rf_soil has the highest exact-copy inflation (R² .28 at f = 1) but a small real leak (.014 at 15 km), against knn_geo at .077;
   - antibody physchem-RF is over-predicted.
2. **It predicts how much real overlap inflates the score better than assuming no leak,** when the planted copies' labels agree with the test labels only as well as the real overlapping labels do. This is tested on assay pairs and soil properties whose leak has never been computed.
   - Stated up front: on the old 15 cross-lab cells, an agreement-blind model-mean null (LOAO) already beats per-cell planting (S .76 vs .70).
   - So the open question is whether label agreement matters on new assays. Test C2 asks exactly that.
3. **Near-duplicate leaks must be measured directly** with a near-vs-far difference-in-differences (DiD), because exact copies overstate them and high-threshold dedup leaves them in place.

**Fallback claim.** The riskiest steps are Step 4 (fresh antibody assays) and Step 5 (GDPa4 real copies). If they fail, the claim becomes:

> Copy planting separates memorising models from smooth ones in all three domains. Exact copies overstate near-duplicate leaks. A copy only leaks to a model that can see it is a copy. Dedup at a high similarity cutoff gives false reassurance. So the real leak must be measured with a near-vs-far DiD.

This fallback is written into PREREG.md now, so falling back is not a post-hoc move.

---

## 2. Ordered steps

**Confirmatory tests (Holm-adjusted, α = .05).** There are exactly three planned magnitude tests. Each p-value is one-sided and comes from a paired bootstrap that resamples items **and** planting seeds. Inside-CI counts and correlations are descriptive only.
- **C1:** antibody fresh assays. Pooled S over 20 cells (4 pairs × 5 models) vs predict-0. p = share of bootstrap replicates with S ≤ 0.
- **C2:** antibody fresh assays. MAE(planting) < MAE(null c, agreement-blind LOAO). p = share of replicates where the MAE gap is ≤ 0. This tests whether agreement matters.
- **C3:** salinity held-out properties. Pooled knn_geo S, scored with the frozen did_multi `nearest` recipe and arm-B calibration (see Step 7). Test points are resampled jointly across properties, because the properties share the same geometry and are correlated.

**Pre-specified but outside Holm:**
- the ranking verdict (R);
- null (d), the agreement-scaled model mean, as a secondary comparison for C2;
- the GDPa4 noise-model verdict;
- the size-effect checks;
- the split-robustness runs.

Fresh split seeds are **not** independent data. They are never t-tested as if they were.

| # | step | domain | hypothesis | pre-specified success | kill criterion | hours | depends on |
|---|---|---|---|---|---|---|---|
| 0 | Freeze state and write PREREG.md | all | n/a (bookkeeping) | Commit A today: did_multi files, RESULTS diff, `_killcheck_crosslab.py`, plus `.gitignore` entries for `.serena/` and the local per-point folder. Commit B (PREREG.md alone) before any fresh run | A held-out set turns out to be already scored → relabel it exploratory before commit B | 3 | – |
| 1 | Fair dates, forms, licence, ESDAC | admin | n/a | Fair and deadline dates confirmed with the sponsor by Oct 23. Forms drafted. ESDAC application **submitted** Oct 16, checking whether a minor needs a sponsor or institutional co-signer | Fair before Feb 14 → shift Steps 11–13 earlier and use PREREG's pH-only C3 version | 2 | – |
| 2 | Prior art: read arXiv 2605.24818 in full | writing | No prior work tests planting against a causally measured leak from near-duplicate, imperfect-label overlap in scientific benchmarks, or reports redundancy | A checkable 3–5 sentence differentiation. Authors verified before naming. Table of 8 or more works | A paper already does this → drop "new method"; frame as an independent test plus a near-duplicate stress test | 3 | 0 |
| 3 | Retrospective scoreboard, null ladder, forking-path table | all | Each predictor of record beats predict-0 on old data. Exact copy fails on neighbours | `score.py` reproduces §3 to ±.01, with a scope column. `out/forking_paths.csv` lists every variant with a status | A predictor of record has S ≤ 0 in its stated scope → that domain claims ranking only | 4 | 0 |
| 4 | Antibody fresh cross-lab assays (C1, C2), two-stage, with a size check | antibody | Real DiD leak scales with label agreement. Copula-planted I_M(f) predicts it | C1: S lower bound > 0. C2: MAE(pred) < MAE(null c). Both Holm | C1 fails → no antibody magnitude claim. C2 fails → "LS is a model property; agreement adds nothing". C1 verdict flips when size-sensitive cells are dropped → antibody magnitude is called "assumption-dependent" | 8 | 0, 3 |
| 5 | GDPa4 real re-measured copies vs copula copies | antibody | A copula copy at the measured agreement leaks as much as a real re-measurement | The memorisers' pooled LS_C/LS_R 95% CI lies inside [0.67, 1.5] | CI entirely > 1.5 → magnitude predictions become upper bounds. CI entirely < 0.67 → they become lower bounds. CI straddles a bound → "inconclusive" (the likely result, given n = 71) | 6 | 0 |
| 6 | Salinity fresh seeds 6–11: split check, ρ = 0 arm, size check | salinity | Nearest-only planting predicts the knn_geo leak on unseen splits. Matched agreement, not geometry, does the work | knn_geo pooled S, paired-bootstrap lower bound > 0. Matched arm's MAE is at least 20% below the ρ = 0 arm's (paired-bootstrap lower bound) | S lower bound ≤ 0 → "did not replicate on fresh splits". Matched arm does not beat ρ = 0 by 20% → drop the agreement wording for soil | 5 | 0, 3 |
| 7 | Salinity held-out soil properties (C3), seeds 6–8 | salinity | The frozen recipe predicts knn_geo leaks for other properties without retuning | C3: pooled knn_geo S over 5 properties × 3 radii × 3 seeds, lower bound > 0 (Holm) | C3 fails → soil magnitude is "EC only, exploratory" | 4 | 6 |
| 8 | EuroSAT seeds 3–8: split check, redundancy replication, ranking at t = .80 and .85 | EuroSAT | pred_old beats predict-0 on new splits. The redundancy pattern repeats | pred_old pooled S > 0, and > 0 in at least 5 of 6 seeds. Redundancy holds when, in at least 5 of 6 seeds, the 1-NN leak at t = .95 is < 0.5 pts **and** the leak at t = .80 is ≥ 2 pts | Either rule fails → pred_old magnitude dropped, or redundancy demoted to "seen on 3 seeds". No middle zone | 4 | 0 |
| 9 | Confirmatory scoring, ranking verdict, RESULTS/DEFINITION rewrite, **freeze Dec 20** | all | Most of C1–C3 pass. R passes in all 3 domains | `score.py --confirm` prints raw and Holm p for C1–C3, plus R, null (d), GDPa4, size-check and split verdicts, from fresh CSVs only | No new predictor variants after this. Any later run is labelled exploratory | 4 | 4–8 |
| 10 | Own the method | student | n/a | Hand-computed DiD on 6 toy items. Copula in one sentence. Walk-throughs of 3 scripts. Recomputes 5 headline numbers in pandas | A script the student cannot explain after 2 sessions → moves to an appendix | 6 | 3 |
| 11 | Figures (3–4) | writing | n/a | Every value traceable to a CSV column. Hallway test passed by 3 classmates. Licence checklist signed for every figure | A figure fails the hallway test twice → cut | 7 | 9 |
| 12 | Paper, abstract, signed forms | writing | n/a | Abstract ≤ 250 words. Every claim cites a scoreboard row. Mentor-reviewed | Late → shorten the discussion; never cut the null comparisons | 12 | 9, 11 |
| 13 | Board, hostile-question bank, mock judgings | writing | n/a | 2-, 5- and 10-minute talks. 20 questions answered in under 45 s each. 2 mock judgings | Any number the student cannot explain is cut from the board | 14 | 10–12 |
| 14 | (Conditional) LUCAS 2015 replication with the frozen script | salinity | The Step 6/7 criteria hold on independent data | Same thresholds as Steps 6/7, with agreement re-measured on 2015 data | No data by Jan 10 → run after the regional fair (by Apr 15) or list as future work | 5 | 7, ESDAC |

**Total:** about 82 h core (about 5 h/week, with more in Thanksgiving week and winter break), plus 5 h conditional.

**Cut order if behind:**
1. Step 14.
2. Step 7 beyond pH. PREREG states this pH-only C3 version in advance.
3. Step 8 pred_old (keep the redundancy replication).
4. The ρ = 0 arm.
5. Step 5 smooth-model arm (keep memorisers).

Never cut Steps 0, 3 or 4, or the null comparisons.

---

## 3. Per-step detail

### Step 0: Freeze and pre-register

**Commit A, today:**
- `salinity/did_multi.py`, `did_multi_seeds0to2.csv`, `did_multi_seeds3to5.csv` and the RESULTS diff.
  - All three are untracked, so git cannot show that did_multi's "fixed before running" docstring came before the run. PREREG says so.
- `results/_killcheck_crosslab.py` (file date Oct 14). Committing it records what was already looked at.
- Add `.serena/` and `leakage_audit/salinity/_local/` (per-point error vectors) to `.gitignore`.
  - Already ignored, checked today: `data/external/lucas2018/`, `data/external/gdpa/` and `sat/eurosat.npz`.
- Before every commit, run `git status --porcelain`. Confirm that no salinity file holds coordinates or per-point values. The current salinity CSVs are per seed × radius × model only.

**Commit B (by Oct 18): `leakage_audit/PREREG.md`, committed on its own.** It states:
- **Metric.** S, with its scope, for every variant.
- **Null ladder for every magnitude claim:**
  - (a) predict-0;
  - (b) oracle constant: the observed mean leak of that table. It is in-sample, so it is a reference line, not a confirmatory null;
  - (c) LOAO model-mean I_M(f): the model's mean from the other assays. It ignores agreement. On old data it scores S .76, above the predictor of record (.70);
  - (d) agreement-scaled model mean: (c) × agreement_new / agreement_old. It shows what per-assay planting adds beyond simple scaling;
  - (e) in-sample β-rescaled exact copy, as a ceiling.
  - Salinity adds a ρ = 0 (geometry-only) arm.
- **Predictors of record:**
  - Antibody: copula-planted I_M(f) at the measured agreement (DEFINITION step 4). LS × f is reported beside it.
  - Salinity: did_multi `nearest`, rank-matched labels, knn_geo only.
    - It was chosen after seeing seeds 0–5. It is not the top knn_geo scorer on old data (§3).
    - Known failures: on ridge_soil it predicts a fake leak in 18/18 cells; on rf_soil_geo it predicts ≈ 0 against a real .027 at 15 km.
    - Only fresh data can decide it.
  - EuroSAT: pred_old.
- **Memoriser and smooth labels, fixed by model type.** They match the old measured LS split in every domain:
  - antibody: lowest memoriser .58 vs highest smooth .31;
  - EuroSAT: 5-NN 5.4 vs ridge 2.0;
  - salinity: .25–.28 vs .002.
  - Memorisers: NN-copy, kmer-ridge, physchem-RF, 1-NN, 5-NN, knn_geo, rf_soil, rf_soil_geo. Smooth: esm2-ridge, ablang2-ridge, EuroSAT ridge, ridge_soil.
  - "Sees the overlap" rule for salinity: the overlap is geographic, so only models with location features see it (knn_geo, rf_soil_geo). rf_soil does not.
- **Fixed in advance:** held-out sets, the C1–C3 rules, C3's arm and scope, the pH-only fallback, every kill rule, the fallback headline and the freeze date (2026-12-20).
- **Disclosures:**
  - `_killcheck_crosslab.py` (Oct 14) computed PSR and ELISA vs both PR_CHO and PR_Ova, plus HIC, AC-SINS and SMAC.
  - Planning on Oct 15–16 computed every other candidate agreement and all GDPa4 agreements (§3, Steps 4–5).
  - No fresh DiD was computed.
  - The July commits tested PSR rank transfer and polyreactivity domain adaptation.
  - Burned: salinity seeds 0–5 and EuroSAT seeds 0–2.

Every new confirm script quotes commit B's hash in its docstring.

### Step 1: Logistics

- **ESDAC:** the student applies personally today (free; the licence requires it). Approval is not instant. If it asks for an adult or institutional signatory, the sponsor handles it this week. If there is no approval by Jan 10, Step 14 becomes future work.
- Adult sponsor confirms fair dates and registration/abstract deadlines by Oct 23.
- Draft Forms 1, 1A, 1B and the Research Plan.
- **Form 7 check:** git history starts 2026-07-01 with the antibody pipeline; the first leakage commit is a229165 on 2026-10-15.
- One-page AI-use disclosure.
- Data-provenance and licence table: GDPa1 v1.2, GDPa4.1, FLAb, EuroSAT, LUCAS 2018. LUCAS gets aggregates only, with no raw values and no point maps.
- Category: Computational Biology & Bioinformatics, with EAEV as the alternative.

### Step 2: Prior art

- Read 2605.24818 ("Correcting test set contamination by spiking the training data", 2026) in full.
- The judges disagreed on its authors and on whether it tests against nulls. Verify both, and **do not claim it lacks nulls**.
- Likely real differences:
  - near-duplicate and neighbour overlap;
  - imperfect label agreement (copula);
  - non-LLM scientific benchmarks;
  - DiD on natural overlap;
  - redundancy.
- Extend `prior_art.py` (it calls OpenAlex over HTTP; the openalex MCP is down) with cited_by walks from 2605.24818.
- Table:
  - Already in `out/prior_art.txt`: DataSAIL 2025, Elangovan 2021, Rosenblatt 2024, and the Geoderma 2025 leave-profile-out paper.
  - To add, each verified before citing: Wadoux 2021, Roberts 2017, Linnenbrink 2024, Barz & Denzler 2020, Feldman & Zhang 2020, Carlini 2019.
- Spatial-CV context: knn_geo R² .273 on random splits vs .017 with 4° blocks (`salinity/audit.log`).
- Rename the salinity estimand "overlap-attributable inflation relative to a buffered evaluation". This answers Wadoux.

### Step 3: Retrospective scoreboard

New `leakage_audit/score.py` writes `out/scoreboard.csv` and `out/forking_paths.csv`.
- The scoreboard has S for each null, per-seed S and a scope column.
- The forking-path table lists, for every variant: domain, commit, whether it was fixed before its DiD was seen, S per scope, and status (record / exploratory / demoted).

It must reproduce these numbers (checked today):

| setting | variant | scope | S vs predict-0 | note |
|---|---|---|---|---|
| antibody cross-lab (15 cells) | pred_old, SD noise, pre-specified | all | .65 (MAE .096) | – |
| | **I_M(.54), copula (record), post-hoc** | all | .70 (MAE .082) | corr .76, 10/15. SMAC kmer-ridge .25 vs .49 |
| | LS × .54 | all | .72 (MAE .076) | SMAC kmer-ridge .28 vs .49 |
| | **LOAO model-mean (null c)** | all | **.76 (MAE .065)** | corr .83, 12/15: beats the record |
| | oracle constant (null b) | all | .51 (MAE .133) | predict-0 MAE .273 |
| within-GDPa1 (45 cells) | exact copy / realistic / rms | all | −1.42 / .01 / .02 | ties the null (MAE .048, 31/45) |
| EuroSAT (36 rows) | **pred_old (record)** | all | .39 (seeds .35/.36/.45) | knn5 at t = .80: pred 2.07 vs obs 4.06, outside every per-seed CI (4.2–5.6, 2.3–3.5, 3.8–5.1). knn1 at .80 over-predicted (6.47 vs 4.52) |
| | pred_red (easiness-confounded) / real_planted pred_old / pred_real_red | all | .55 / .36 / .01 | – |
| salinity, seeds 0–2 / 3–5 | exact copy (pred_old) | all · knn | −5.98 / −5.44 · −2.22 / −1.79 | – |
| | additive pred_dist | all · knn | −.64 / −.52 · −.59 / −.30 | – |
| | additive pred_dist_red (original) | all · knn | .43 / .38 · **.71 / .59** | Higher knn S than the record, but its labels are over-informative (Pearson .67 vs real .40): two errors cancelling. Demoted |
| | rank pred_dist | all · knn | −.11 / −.16 · .65 / .52 | – |
| | rank pred_dist_red | all · knn | .34 / .27 · .55 / .41 | – |
| | **did_multi nearest (record)** | all · knn | .14 / .23 · .58 / .57 | knn per-radius pred/obs 2.52 / 1.27 / 0.85 at 5 / 10 / 15 km |
| | did_multi all | all · knn | −.37 / −.25 · −.04 / .01 | knn 2.6 / 2.3 / 1.8× over |

- **Demoted:** naive-minus-honest 15/15 and corr .92. `audit.py` E2 scores naive on all of GDPa1 but honest on the unseen subset only, so range restriction confounds it. The rank fix was also post-hoc.
- **DEFINITION fix:** its ".28 vs .49" is LS × .54. RESULTS' ".25" is I_M(.54). Label which predictor each number belongs to.
- RESULTS change: the scoreboard replaces the inside-CI columns as the primary evidence.

### Step 4: Antibody fresh assays (C1, C2)

**Script.** New `leakage_audit/confirm_antibody.py`.
- It execs `audit.py`, `synthetic.py` (rank mode) and `did_antibody.py` part A, using the existing `.replace` + assert pattern, and extends ASSAYS locally.
- One `.replace` makes it also save the mean near-item and far-item drop per cell. `did_antibody.py` computes these but does not keep them.
- Run from `~/Biostuff` with `python -I`.

**Pairing rule:** one Jain assay per GDPa1 target, matched by what it measures. PSR uses CHO membrane reagent, so it pairs with PR_CHO. Agreements, checked today:

| Jain2017 → GDPa1 | agreement (Spearman) | n |
|---|---|---|
| PSR → PR_CHO | .65 | 108 |
| ELISA → PR_Ova | .43 | 108 |
| HEK titer → Titer | .33 | 129 |
| Fab Tm (DSF) → Tm1 | .09 | 129 |

- **All other pairings already seen, disclosed:** PSR → PR_Ova .58, ELISA → PR_CHO .43, BVP ELISA → PR_CHO .47 and → PR_Ova .44, Fab Tm → Tm2 −.06 (n = 104) and → Tonset −.06, CSIBLI → AC-SINS_pH7.4 .75 (n = 131).
- ELISA → PR_CHO and BVP → PR_CHO are reported as a sensitivity row outside C1/C2.
- CSIBLI → AC-SINS is reported separately as semi-fresh.
- **Construct check (stage 1).** Record what each column measures, from Jain 2017 and GDPa4's "Definitions" sheet.
  - Fab Tm by DSF vs whole-IgG Tm1, and HEK titer vs GDPa1 Titer, may be different quantities, not lab disagreement.
  - The cells stay as fixed, and the measured agreement is used either way.
  - The pitch and figures call such pairs "labels that don't match", never "labs disagree".

**Stage 1 (commit by Nov 1).**
- Compute agreement, the copula I_M(f) (f ≤ .5, 3 planting seeds) and the predictions.
- Compute nulls (c) and (d) from the old `leakage_susceptibility.csv`.
- Write `out/fresh_antibody_pred.csv` (one row per planting seed) and commit it.

**Stage 2.**
- 20 half-dedups, saving per-item drops → `out/fresh_antibody_items.csv`.
- DiD with 2000 bootstraps over antibodies → `out/fresh_antibody_did.csv`.

**Size check (replaces the kept-half placebo).** The kept-half comparison is mechanically biased: for memorisers, the copy that stays shields those items. So it is dropped.
- The DiD assumes that near items, once their copy is gone, lose as much from the smaller training set as far items do.
- This cannot be tested directly, because only 6 Jain antibodies are non-copies.
- So report, per cell, the far-item drop as a share of the near-item drop. Cells where it is over one half are flagged "size-sensitive".
- C1 is reported with and without flagged cells. If the verdict differs, antibody magnitude is "assumption-dependent".

**Scoring.**
- C1 = pooled S over the 20 cells vs predict-0.
- C2 = MAE(pred) vs MAE(null c).
- Secondary, with pre-stated readings:
  - vs null (d): if (d) is at least as good, the headline says "scale LS by agreement" rather than "plant per assay";
  - vs null (b) as a reference;
  - Tm1 memoriser cells with |pred| < .05.

### Step 5: GDPa4 noise-model test

**Script.** New `leakage_audit/gdpa4_copies.py`.
- It reads `data/external/gdpa/GDPa4.1_20260616.xlsx` (sheets "mAb Sequences" and "Assay Data - median") with `python -I`.
- `openpyxl` is not installed, so use a stdlib zip/XML reader kept in the scripts folder.
- Checked today: all 71 GDPa4 mAbs have VH+VL identical to GDPa1, so their embeddings are cached. The script asserts this.

**Setup.**
- PREREG fixes the column mapping, using the "Definitions" sheet. It also discloses that these agreements were already seen today:

| GDPa1 column | GDPa4 column | agreement | n |
|---|---|---|---|
| HIC | hihplc_normretentiontime | .99 | 69 |
| SMAC | smachplc_retentiontime | .97 | 69 |
| AC-SINS_pH7.4 | acsins_deltaLmax_1X PBS | .91 | 71 |
| PR_CHO | pr_score_CHO | .90 | 70 |
| PR_Ova | pr_score_Ovalbumin | .86 | 70 |
| Tm1 | thermostability_tm1 | .44 | 71 |
| Tm2 | thermostability_tm2 | .28 | 70 |

- Use 5 fold seeds of the GDPa1 hierarchical_cluster_fold.
- Plant f ∈ {0, .1, .2}, using only GDPa4 test items, with the same items in both arms:
  - arm R: the real GDPa4 value, quantile-mapped onto the GDPa1 marginal;
  - arm C: 20 copula draws at the measured agreement.

**Scoring.**
- The verdict uses the memorisers' pooled LS_C/LS_R, with a bootstrap over antibodies.
- Smooth models have near-zero LS, so a ratio is meaningless for them. They are reported as LS_C − LS_R with a CI.
- **The SMAC row is not a test of the cross-lab skew idea.** GDPa4 re-measures inside one lab, with agreement .97. The skew hypothesis stays open (§4).
- **Writes** `out/gdpa4_copies.csv`.

### Step 6: Salinity split check and validity

**Script.** New `salinity/confirm_salinity.py`. It execs `did_multi.py` with `.replace` to:
1. save per-test-point error vectors to the gitignored `salinity/_local/`, for a paired bootstrap of (obs − pred). Pred and obs share the dedup baseline error, so an unpaired CI is miscalibrated;
2. add a ρ = 0 arm, with planted labels and soil values uncorrelated with the test point;
3. save the far-item drop.

**No removal placebo.** Removing train points far from any test point repeats the clean-control design that RESULTS §2 rejected. Those points sit in sparse areas, so removing them mostly hits far items. The triple difference is dropped with it. Validity checks:
- **Negative control:** ridge_soil's observed DiD. On old data it is ≤ 9% of knn_geo's at 10 and 15 km. If its mean on fresh seeds goes over 25% of knn_geo's at 10 or 15 km, flag a geometry or size artifact.
- **Size check:** the far-item drop as a share of the near-item drop, per radius, as in Step 4.

**Runs.** Seeds 6–11, r = 5/10/15 km, RF in the background. Removed train points are about 1.27k / 4.3k / 8.0k at 5 / 10 / 15 km, leaving about 13.9k / 10.9k / 7.2k.

**Scoring.**
- Magnitude is scored on knn_geo only, since it is the only material leaker.
- Report per-radius pred/obs. Say "unbiased" only if all three radii fall in [0.67, 1.5]; the 5 km ratio is expected to stay high (2.52 on seeds 0–5).
- Report the predictor's ridge_soil failure as measured: on old data, pred_nearest falls outside the observed CI in 18/18 ridge_soil cells (pred_all 17/18), predicting .013 / .026 at 10 / 15 km against .001 / .007 observed.
- Report the other models' S.
- No t-test across seeds.

**Writes** `salinity/confirm_seeds6to11.csv` (aggregates only).

### Step 7: Salinity held-out properties (C3)

**Script.** New `salinity/fresh_targets.py`, which execs the Step 6 setup with `.replace` on the target and feature list.

**Recipe, frozen in PREREG.**
- Seeds 6–8, which are fresh. Seeds 0–2 are burned.
- Targets: pH_H2O raw; OC, CaCO3, N and K as log1p.
  - P is excluded as a target because of below-detection-limit censoring, but it stays as a feature.
  - Features = the current soil set minus the target. EC is **not** added as a feature, so the model set stays fixed.
- Predictor: did_multi `nearest`, rank-matched, knn_geo-only S.
- **Calibration arm B is primary:** agreement per target and distance bin from train–train pairs only. The current recipe uses 5-NN pairs over all points, test points included.
  - Arm A (the current recipe) is reported beside it.
  - This removes any test-information circularity, and there is no gap zone to interpret.
- Models: knn_geo (scored) and ridge (negative control). RF is skipped for compute, and this is declared.
- The properties are correlated and share the same geometry, so C3 is one pooled test, not five.

**Writes** `salinity/fresh_targets.csv` (aggregates only).

### Step 8: EuroSAT seeds 3–8

**Script.** New `sat/confirm_sat.py`.
- `sat/did.py`, `predict.py`, `planted.py` and `real_planted.py` all hard-code `range(3)`, and `did.py`'s pred_old depends on `predict.py`/`planted.py`.
- So the wrapper execs all four with `.replace("for s in range(3):", "for s in range(3, 9):")` and new output names, following the `did_rank.py` pattern.
- It reads `sat/eurosat.npz`.

**Ranking.** Tested only at t = .80 and .85, because every observed leak at .95 is about 0. Old seed means (points):

| t | knn1 | knn5 | ridge |
|---|---|---|---|
| .80 | 4.52 | 4.06 | 0.17 |
| .85 | 3.96 | 1.22 | 0.01 |

**Expectation.** The knn5 under-prediction at t = .80 is expected to recur in every seed, and is reported as a miss with per-seed CIs.

### Step 9: Confirmatory scoring and freeze

**Script.** `score.py --confirm` reads only the fresh CSVs from Steps 4–8 and prints the C1–C3 p-values and the Holm verdicts.

**Holm cut-offs:** smallest p < .0167, next < .025, last < .05.

**Ranking verdict R** (pre-specified, outside Holm). It passes only if all of these hold:
- antibody PR_CHO: the memorisers' mean DiD > the smooth models' mean DiD, with the CI excluding 0;
- EuroSAT: knn1 and knn5 > ridge at t = .80 and .85;
- salinity (fresh EC seeds and Step 7 targets): knn_geo > ridge at 10 and 15 km;
- salinity "sees the overlap" prediction: rf_soil's DiD is below both knn_geo's and rf_soil_geo's at 15 km.
  - On old data: .014 vs .077 and .027.
  - Exact-copy inflation predicts the opposite order (.25 vs .23).

R does **not** claim an order within memorisers.

**RESULTS change.** A new top section, "Pre-registered tests", listing pass/fail including failures. The 15/15 result and the inside-CI counts move to "Exploratory / superseded".

**DEFINITION.md changes.**
- Step 4's check becomes the DiD.
- "corr .92, 15/15" is replaced with S values, and every number names its predictor.
- Add: "a copy only counts as a copy in the model's own feature space".

### Step 10: Own the method

- Hand-compute one DiD on 6 toy items.
- Draw the copula as a picture: mix normal scores, then map them back through the assay's own values.
- Line-by-line walkthroughs with the mentor of `did_antibody.py`, `salinity/did_multi.py` and `sat/did.py`.
- Recompute 5 headline numbers in pandas.
- Write one paragraph per script in the student's own words.

### Step 11: Licence check on figures

Every figure gets a one-line sign-off:
- no LUCAS coordinates or per-point values;
- buffer schematics use made-up points;
- the Step 7 plots show aggregates only.

### Step 14: Conditional LUCAS 2015

- Commit a PREREG addendum before opening the data.
- Run `confirm_salinity.py` and `fresh_targets.py` with a data-path `.replace`.
- Keep the data in a gitignored `data/external/lucas2015/` and run with `python -I`.
- Commit an aggregate CSV only.

---

## 4. Dropped or deferred

**Mapping of the "Not done" list in RESULTS:**
1. Realistic EuroSAT predictor → dropped (row 3 below).
2. Salinity shared-regional planting and dropping soil features → deferred (rows 4 and 12).
3. SMAC skew test → deferred, still untested (row 5).
4. LUCAS 2015 → Step 14.

| item | reason |
|---|---|
| Naive-minus-honest 15/15 (corr .92) as the headline | Range restriction: `audit.py` E2 scores different test subsets. The rank fix was post-hoc. Kept as exploratory |
| Inside-CI counts and correlation as evidence | Predict-0 alone scores 23–29/36 in salinity and 31/45 within GDPa1. Correlation rewards the right order at the wrong size (salinity exact copy: decent order, S = −6) |
| New EuroSAT magnitude predictors | 8 variants already tried on the same 36 rows. Blends are too strong and reserve tiles too weak (pred_real_red S = .01) |
| Salinity one-factor shared-regional planting | It would be salinity variant 8 or more and would use up the fresh data. The mechanism is reported as measured (neighbour–neighbour Spearman .34 real vs .13 planted). Future work |
| SMAC kmer-ridge skew test | GDPa4 cannot test it, because those copies are within one lab. Reported as an unexplained miss (I_M .25 vs DiD .49), with the skew idea (GDPa1 5.7 vs Jain 3.4) stated as untested. Future work |
| Within-GDPa1 magnitude | The null ties (realistic S .01, predict-0 MAE .048). Kept as a boundary condition: 80–90% identity neighbours barely agree in label |
| Kept-half antibody placebo | Mechanically biased: a kept copy shields memorisers from the size drop. Replaced by the far-drop size check |
| Salinity removal placebo and triple difference | The same flaw as the clean control rejected in RESULTS §2. Non-leak points sit in sparse areas, near far items. Replaced by the ridge_soil negative control and the size check |
| The ρ\* criterion in an agreement sweep | Already known to fail (per-radius ratios 2.52 / 1.27 / 0.85). Only the ρ = 0 arm is kept |
| Memorisation-knob dose-response, MLP model class | High prior chance of passing, and each is a new fork. Post-ISEF future work |
| FLAb other sources | Mostly the same antibodies (jain2017 vs GDPa1 135/137). Embedding cache limits |
| New or mutated sequences, new embeddings, second EuroSAT backbone | Embeddings are cache-only, and laptop compute is limited |
| 25 and 50 km radii | Degenerate: about 2,500 and 150 training points left |
| SGACSINS → AC-SINS_pH6.0 (−.52); CICRT → HAC | Sign-convention ambiguity: flipping the sign after the fact is a fork. CICRT → HAC has n = 51 |
| Dropping soil features in did_multi to remove the ridge artifact | This changes the model set after seeing results. The ridge failure is reported instead |
| Salinity maps; a packaged tool | Licence (locations must stay hidden) and no evidence value. A buffer schematic and a 4-line recipe box are used instead |
| Removal-control analyses (calibrate, headroom, clean_control) | Superseded by DiD. One appendix line |

---

## 5. Poster figures and the 60-second explanation

**Figures.** Each value maps to a CSV column. Each has a licence sign-off. No LUCAS point locations anywhere.
1. **Method cartoon (hand-drawn).** Left: plant copies, and the score rises; the slope is LS. Right: dedup, then compare near and far test items (DiD). Far items absorb the "smaller training set" effect, and the size check shows how big that effect is.
2. **Scoreboard and moderator.** Skill vs predict-0 for every setting and variant, with the LOAO null drawn as a bar. Ordered by how well the overlapping labels agree:
   - Tm1 .09 → Titer .33 → ELISA/PR_Ova .43 → salinity neighbours ≈ .40 → PR_CHO .65 → cross-lab .89–.97 → EuroSAT .92–1.0.
   - Tm1 and Titer are marked "may measure a different quantity".
   - Fresh points are filled and exploratory ones hollow. Failures are shown.
   - Labelled descriptive, because the points are not independent.
3. **Ranking.** LS vs real DiD per model, one panel per domain, with memorisers and smooth models marked. The salinity panel shows rf_soil (exact-copy .25 vs real .014) next to knn_geo (.23 vs .077): "a copy only leaks to a model that can see it".
4. **Redundancy.** EuroSAT 1-NN leak removed vs dedup threshold: 0.00 at .95, 0.28 at .90, 3.96 at .85, 4.52 at .80. Closeness after dedup goes from .94 to .77. Seeds 0–8.

Beside the figures sits an **honesty panel** (text): variants tried per domain, the null ladder (including that LOAO beat planting on old data), and pass/fail for C1–C3, R and GDPa4.

**60-second explanation (student voice):**

> When we test an AI model, some test questions may have leaked into its study material. It might be the same antibody measured twice, or soil samples a few kilometres apart. The score then looks better than it is. I built a copy-paste stress test: I copy test items into the training data on purpose and watch how much the score jumps. To get the true leak, I compare test items that had a near-copy in training with items that didn't, before and after removing the copies.
>
> I found three things across antibodies, satellite images and soil. One: models that memorise, like nearest-neighbour, jump a lot, while smooth models like ridge on embeddings barely move. But a copy only counts if the model can see it: a soil model that doesn't use location barely leaks from nearby samples. Two: overlap only inflates scores when the copied labels really match. For stickiness, the old and new datasets agree, so that overlap leaks. For melting temperature, the two numbers don't line up, so the same overlap leaks almost nothing. I wrote that prediction down before I measured it. Three: the usual fix, deleting training items more than 95% similar, removed zero inflation, because a 94% look-alike is still there.
>
> So: run a copy test before trusting a leaderboard, check that your model can actually see the overlap, and dedup at a lower threshold.

If Step 4 fails, replace the "Two" sentences with: "Exact copies give an upper bound; real neighbours leak much less, so measure them directly."

---

## 6. Timeline (2026-10-16 to ISEF)

| dates | work |
|---|---|
| **Fri Oct 16** | Step 0 commit A. **Student submits the ESDAC application** and checks whether a co-signer is needed. |
| Oct 17–18 | PREREG.md written and committed on its own (commit B). |
| Oct 19–25 | Step 1 (fair dates by Oct 23; sponsor co-signs ESDAC if needed), Step 2 (framing decided Oct 25), Step 3 scoreboard. |
| Oct 26–Nov 1 | Step 4 stage 1: construct check, predictions and nulls **committed by Sun Nov 1**. |
| Nov 2–15 | Step 4 stage 2 (DiD and size check); Step 5 GDPa4. |
| Nov 16–29 | Step 6 (RF in the background), Step 7 (Thanksgiving week carries the bulk). |
| Nov 30–Dec 13 | Step 8. Step 10 starts and runs to Jan 10. |
| Dec 14–20 | Step 9 scoring and rewrite. **Experiment freeze Sun Dec 20.** |
| Dec 21–Jan 3 | Winter break: Step 11 figures, Step 12 paper draft. |
| Jan 4–17 | Mentor review, abstract, forms signed (earlier if Step 1 found an earlier deadline). **Jan 10: LUCAS 2015 go/no-go.** If the data is here, Step 14 runs Jan 11–17 with the frozen script. |
| Jan 18–Feb 7 | Step 13: board final Jan 31, printed by Feb 5; mock judgings Jan 31 and Feb 7; Q&A bank. |
| Feb–Apr 2027 | Regional and state fairs. Clarity edits only. |
| By Apr 15 (if advancing) | Step 14 if not yet run (addendum first; frozen script only). ISEF abstract and board refresh. |
| May 2027 | ISEF. |

**If the regional fair is before Feb 14,** move Steps 11–13 earlier by the difference, use the pH-only C3, and skip Step 14 until after the fair.

---

## 7. Risks and handling

| risk | handling |
|---|---|
| LOAO already beats per-cell planting on old data (S .76 vs .70) | Said openly in the headline. C2 tests whether agreement matters on new assays. Null (d) tests whether planting adds anything beyond scaling, and has a pre-stated reading |
| Fresh antibody leaks are small on low-agreement assays, so C1 rests on about 5 PR_CHO cells | C1 is pooled. C2 is the stronger test: null (c) predicts old-size leaks on Tm1 and Titer, where planting predicts about 0. If agreement does not matter, (c) wins. The fallback headline is pre-registered |
| Low agreement is a different measurement, not lab noise (Tm1, Titer) | Stage 1 construct check. The cells stay, but the wording changes to "labels that don't match" |
| DiD size assumption (far items absorb the size effect) | It cannot be tested directly in cross-lab antibodies (only 6 non-copies). A far-drop share flag and a C1 with/without check handle it. Salinity uses the ridge_soil negative control. Both are pre-specified |
| The copula itself is wrong | Step 5 tests it with real re-measured copies, memorisers pooled, with both upper- and lower-bound branches. "Inconclusive" is the expected outcome at n = 71 |
| Planting-seed noise understated | The bootstrap resamples planting seeds as well as items |
| Pseudo-replication (seeds share the same data) | Seeds are labelled split robustness and never t-tested. The confirmatory tests use new assay pairs and new soil properties on fresh seeds |
| Forking paths and overclaiming under judging | PREREG commit hash, forking-path table with scopes, disclosed pairings and agreements, Holm over 3 tests, a null beside every number, freeze on Dec 20 |
| Prior art (2605.24818) pre-empts the idea | Settled by Oct 25, before any framing. The fallback framing is an independent test plus a near-duplicate stress test |
| Student cannot defend the AI-assisted code | Step 10, the AI-use disclosure, the hostile-question bank, and 2 mock judgings. The bank covers: variant counts, why predict-0, why LOAO wins on old data, the spiking paper, spatial CV, rf_soil, and "why predict when you can measure the DiD?" (to rank models before training, and to audit benchmarks that have no dedup) |
| LUCAS licence | Aggregates only. Per-point error vectors stay in the gitignored `salinity/_local/`. Per-figure licence sign-off. `git status` before every commit. `python -I`, with scripts kept outside data directories |
| Compute or time overrun | RF runs in the background. Cut order is in §2. Steps 0, 3, 4 and the nulls are never cut |
| ESDAC access is late or needs an adult signatory | Applied Oct 16, with the sponsor ready to co-sign. Step 14 is conditional, runs only the frozen script, and has an addendum committed first |

---

**Critic points rejected (checked against the files):**
- **"C2 uses the weak, agreement-scaled null (c)."** Null (c) is agreement-blind; (d) is the agreement-scaled one. C2 stays against (c), and (d) is now a secondary with a stated reading. The oracle constant needs the observed mean, so it cannot be a confirmatory null.
- **"DEFINITION's SMAC .28 is wrong."** .28 is LS × .54 (the `predicted` column of `leakage_susceptibility.csv`). .25 is I_M(.54) (`pred_rank` in `did_antibody_rank.csv`). Both are right for different predictors. DEFINITION gets a label, not a correction.
- **"Commit A bundles RESULTS edits with the PREREG."** The plan already made PREREG a separate commit B. The valid part, that did_multi has no git timestamp, is accepted and disclosed.
- **"All agreements, including CSIBLI and Tm2, were computed in `_killcheck_crosslab.py`."** That script covers only PSR/ELISA × PR_CHO/PR_Ova plus HIC, AC-SINS and SMAC. The other agreements came from planning. Both sources are disclosed.
- **"EuroSAT knn1 CI 3.41–4.72."** That interval was for knn5, not knn1. The substance (averaged bounds are not a CI) is accepted and fixed.
- **"No predict-0 for Step 7 or GDPa4."** C3 was already scored against predict-0. GDPa4 compares two planted curves rather than predicting a leak, so its readout is the ratio CI.
- **"Memoriser labels conflict with measured LS."** The type labels match the old measured LS split in every domain. rf_soil's small real leak is a DiD finding, now a pre-registered prediction in R.
- **"GDPa4 embeddings may be missing."** Checked: all 71 GDPa4 mAbs have sequences identical to GDPa1, so their embeddings are cached.
- **"No held-out set is named."** The fresh assay pairs and soil properties have never had a DiD computed; they are now named explicitly as the only confirmatory data. The point that salinity seeds 0–2 were burned is accepted: Step 7 now uses seeds 6–8.
