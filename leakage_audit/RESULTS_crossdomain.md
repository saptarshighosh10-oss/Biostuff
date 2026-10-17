# Cross-domain leakage results (antibody DiD, EuroSAT, LUCAS salinity)

Status as of 2026-10-15. Every predictor variant tried is listed, including the failures. On EuroSAT, eight
variants were tested against the same 36 rows, so treat any single EuroSAT "win" as exploratory. The salinity
redundancy predictor was fixed in the script docstring before it was run.

## 1. Susceptibility ranking (robust, all three domains)

Planted exact-copy inflation at f = 1 (all test items copied into train):

| domain | high | low |
|---|---|---|
| antibody (see `out/leakage_susceptibility.csv`) | memorising models | linear |
| EuroSAT, accuracy points | 1-NN +18.1, 5-NN +5.4 | ridge +2.0 |
| LUCAS log-EC, R² | RF 0.28, coords-kNN 0.25 | ridge 0.002 |

## 2. Measuring the real drop: use difference-in-differences, not a removal control

Train-side dedup at threshold t removes leak neighbours, but it also shrinks the training set.
- **Size-matched random removal** (`sat/predict.py`) also deletes leak neighbours, which dilutes the measured effect.
- **Clean control** (`sat/clean_control.py`) only removes non-neighbours, which handicaps the control on far tiles.
  The two controls gave nearly the same answer, so neither isolates the leak.
- **DiD** (`sat/did.py`, `salinity/did.py`): leak = f_near × [drop on near items − drop on far items]. Far items
  absorb the training-set-size effect. This gives tight CIs, and it is the estimand used from here on.

## 3. Redundancy: threshold dedup barely removes inflation at high thresholds

After removing every train tile with cosine ≥ t to a test tile, the near test tiles still have a neighbour just
below t (mean closeness after dedup: 0.94 at t = .95, 0.87 at t = .90). Real 1-NN leak removed (DiD):
0.00 points at t = .95, 0.28 at .90, 3.96 at .85, 4.52 at .80. **Practical point: dedup at a high similarity cutoff
gives false reassurance.**

## 4. Magnitude prediction: depends on the domain

### LUCAS salinity, r = 5/10/15 km buffers, 36 rows (`salinity/did.py`)
| predictor | corr | inside CI | MAE (R²) | signed |
|---|---|---|---|---|
| exact-copy planted curve | 0.57 | 7/36 | 0.108 | +0.108 |
| realistic neighbours (label/feature noise matched to real LUCAS pairs at that distance) | 0.79 | 15/36 | 0.025 | +0.004 |
| **realistic + redundancy, g(d1) − g(d2)** (fixed before running) | **0.84** | **26/36** | **0.009** | **−0.002** |

Replication on fresh split seeds 3–5 (`salinity/did_seeds3to5.csv`): realistic + redundancy corr 0.79, 27/36 inside the CI,
MAE 0.010, signed −0.002. Exact copy: corr 0.47, 5/36, +0.107.

Why exact copies over-predict here: real soil samples 1–55 km apart differ by rms 0.76 in log EC, against 0.97 for
unrelated points. A neighbour is a weak copy.
Radii of 25 and 50 km are excluded because buffering leaves about 2,500 and 150 training points (degenerate).

**Noise-form check and null baseline (2026-10-16, `salinity/did_rank.py`, fixed before running).** The realistic planting added
N(0, rms_pair) to the test label. That keeps the planted label centred on the true one, so it agrees at Pearson 0.67, while real 5-NN
pairs agree at only 0.40 (a real neighbour's label regresses to the mean). Soil features were roughly matched (Spearman 0.51 planted vs 0.59 real).
The rerun used rank-matched (copula) labels and features at the measured pair agreement, the same recipe as the antibodies.
| seeds | predictor | corr | inside CI | MAE | signed | knn_geo only: inside / MAE / signed |
|---|---|---|---|---|---|---|
| 0–2 | additive noise + redundancy (original) | 0.84 | 26/36 | 0.009 | −0.002 | 9/9 / .011 / +.009 |
| 0–2 | rank-matched + redundancy | 0.84 | 31/36 | 0.010 | −0.010 | 9/9 / .017 / −.017 |
| 0–2 | **null: predict 0** | – | 23/36 | 0.016 | – | 3/9 / .037 / −.037 |
| 3–5 | additive noise + redundancy (original) | 0.79 | 27/36 | 0.010 | −0.002 | 9/9 / .017 / +.004 |
| 3–5 | rank-matched + redundancy | 0.82 | 27/36 | 0.012 | −0.011 | 7/9 / .024 / −.024 |
| 3–5 | **null: predict 0** | – | 28/36 | 0.017 | – | 5/9 / .041 / −.041 |

- **The inside-CI count is a weak test here.** Only knn_geo leaks materially. Most of the 36 cells are ≈ 0 with CIs covering 0, so
  predicting 0 scores 23/36 and 28/36. The predictor's real advantage is on knn_geo (MAE about 2–3× lower than the null) and in correlation.
- Rank-matched labels shrink the knn_geo prediction by half and under-predict the r = 15 km leak (.038 vs .070/.085). The original
  fit was partly two errors cancelling.
- A likely reason, post-hoc and untested: each test point gets one planted neighbour, but dedup removes several real ones (mean 1.3 train
  points within 5 km of a near test point, 2.1 within 10 km, 3.8 within 15 km). The under-prediction grows with r, as this predicts. The over-informative
  additive label made up for the missing neighbours.

**Multi-neighbour planting (`salinity/did_multi.py`, fixed before running): the count explanation is refuted.** This keeps the real geometry and
replaces every train point that dedup removes with a planted point at its own location. The planted label and soil values are drawn from the
nearest test point (rank-matched, as above). With the real values, this reproduces the observed leak exactly, so any gap comes from the values alone.
The ablation `nearest` replaces only the nearest removed point of each near test point.
| seeds | predictor | corr | inside CI | MAE | knn_geo: inside / MAE / signed |
|---|---|---|---|---|---|
| 0–2 | all removed neighbours | 0.72 | 15/36 | 0.021 | 3/9 / .039 / +.039 |
| 0–2 | nearest only | 0.63 | 19/36 | 0.013 | 7/9 / .015 / +.002 |
| 0–2 | null: predict 0 | – | 24/36 | 0.016 | 4/9 / .037 / −.037 |
| 3–5 | all removed neighbours | 0.80 | 13/36 | 0.021 | 2/9 / .041 / +.041 |
| 3–5 | nearest only | 0.72 | 19/36 | 0.013 | 8/9 / .018 / +.004 |
| 3–5 | null: predict 0 | – | 29/36 | 0.017 | 6/9 / .041 / −.041 |

- The pre-registered expectation was that `all` would fit and `nearest` would under-predict. The opposite happened: `all` **over-predicts** the knn_geo leak
  about 2× (.13–.15 vs .07–.09 at 15 km), and `nearest` is about unbiased.
- **Why (checked after the run):** real neighbours of the same test point agree with *each other* (Spearman 0.34) about as much as with the
  test point (0.35–0.39). They mostly share a regional signal, so the extra neighbours add little. Independent planted draws agree with each other
  at only 0.13, so each one adds new information about the test label, and k of them overstate the leak.
- **A second artifact:** ridge_soil gets a fake leak (.02–.04 vs real ≈ .005), because half the training set becomes synthetic points derived from
  near test points, which shifts the training distribution toward them. This also drags the inside-CI counts below the null.
- Takeaway: one realistic neighbour per test point, at the real nearest distance, predicts the knn_geo leak (15/18 inside the CI, MAE about 2.5× below the null).
  Planting more neighbours needs them to be correlated with each other the way real ones are.

### EuroSAT, t = .80–.95, 36 rows (DiD estimand)
| predictor | corr | inside CI | MAE (pts) | signed |
|---|---|---|---|---|
| exact-copy planted curve (`did.py`) | 0.85 | 5/36 | 0.74 | +0.30 |
| same, replicated on a 60/20/20 split (`real_planted.py`) | 0.84 | 5/36 | 0.80 | +0.28 |
| correlational P(correct given closeness) + redundancy (`did.py`) | 0.89 | 9/36 | 0.54 | −0.28 |
| blended soft copies + redundancy (`did_red.py`) | 0.26 | 8/36 | 1.19 | −1.17 |
| real reserve tiles planted + redundancy (`real_planted.py`) | 0.37 | 17/36 | 1.23 | −1.23 |

- Label noise is not the EuroSAT problem. Test tiles share their nearest train tile's class 100% of the time at cos ≥ .90, and 92–97% at .80–.90,
  so planted copies with true labels are close to realistic. The mismatch is in the features.
- The exact-copy curve is nearly **unbiased in aggregate** (+0.3 points, replicated), but it mis-allocates across
  thresholds. It over-predicts at high t (redundancy) and under-predicts 5-NN at t = .80.
- Realistic image neighbours could not be synthesised. Blends are too strong: a 1-NN gain of about 17% even at cosine .73.
  Real reserve tiles are too weak: they almost never beat the tile's existing nearest train neighbour, so the gain is about 1%.
- The correlational curve fits best, but it is confounded by tile easiness: it predicts ridge losses that do not happen.

### Earlier EuroSAT attempts (removal-control estimand, superseded by §2)
Hard-threshold curve 13/36 inside CI (corr .81); closeness-calibrated 12/36 (`calibrate.py`); headroom-weighted 26/36,
discarded because its logic was backwards (`headroom.py`). A direct check (`headroom_check.py`) refuted the headroom idea:
near tiles without their neighbour are as hard as average.

### Antibody, DiD estimand (`did_antibody.py`, `out/did_antibody.csv`, design fixed before running)
Per-item loss L_i = (z(rank y) − z(rank pred))²/2 gives Spearman rho = 1 − mean(L) exactly, so the DiD is in rho units.

**A. Cross-lab Jain2017 → GDPa1 (the headline setting, f = 0.54).** Removing every copy would leave about 6 training
antibodies, so this uses half-dedup: drop the copies of a random half of the near antibodies (20 halvings, train 137 → 72).
| | corr | inside CI | MAE (rho) | signed |
|---|---|---|---|---|
| DiD vs old naive-minus-honest observed | 0.87 | – | – | mean 0.27 vs 0.29 |
| planted prediction vs DiD | 0.55 | 8/15 | 0.096 | −0.041 |

Mean by model (pred / DiD): NN-copy .29/.44, kmer-ridge .33/.44, physchem-RF .27/.18, esm2-ridge .15/.15, ablang2-ridge .12/.15.
- The old observed drop holds up in aggregate. physchem-RF was the exception: its old value of .29 is really .18 under the DiD.
- The cell-level fit weakens from 12/15 to 8/15. Memorisers leak more than predicted, physchem-RF less. Ranking holds: memorisers > PLM ridge.
- Caveat: half-dedup halves the training set. The DiD assumes the size effect hits near and far antibodies equally.

**B. Within GDPa1 random folds, identity dedup at t = .90/.85/.80** (t = .95 skipped, only 2 near antibodies).
| t | f_near | closeness after dedup s2 | exact-copy pred | DiD |
|---|---|---|---|---|
| .90 | .06 | .84 | .027 | .010 |
| .85 | .27 | .83 | .117 | .050 |
| .80 | .64 | .78 | .279 | .041 |

The exact-copy curve over-predicts: corr 0.41, 19/45 inside the CI, signed +0.107, the same pattern as salinity.
- GDPa1 has almost no near-duplicates. Its "near" antibodies are 80–90% identity neighbours, which are weak copies, and they keep a .78–.84 neighbour after dedup (redundancy).
- The assay matters: AC-SINS real leak ≈ 0 at every t, HIC is the largest (NN-copy .27 at t = .80).

**Realistic-neighbour predictor** (`did_antibody_real.py`, `out/did_antibody_real.csv`, fixed before running). Planted neighbours
are exact sequence copies whose labels agree with the test label only as much as real pairs do. Sequences cannot be mutated
because the ESM and AbLang embeddings are cache-only.
| setting | predictor | corr | inside CI | MAE | signed |
|---|---|---|---|---|---|
| A cross-lab | exact copy, 0.9 agreement | 0.55 | 8/15 | 0.096 | −0.041 |
| A cross-lab | measured agreement (HIC .97, AC-SINS .89, SMAC .91) | 0.57 | 7/15 | 0.098 | −0.017 |
| B within GDPa1 | exact copy | 0.41 | 19/45 | 0.116 | +0.107 |
| B within GDPa1 | realistic labels + redundancy | 0.22 | 32/45 | 0.047 | −0.031 |
| B within GDPa1 | same, salinity-style additive noise | 0.28 | 30/45 | 0.047 | −0.033 |
| B within GDPa1 | **null: predict 0 leak** | – | **31/45** | **0.048** | – |

- **The realistic predictor is no better than predicting zero.** Real GDPa1 pairs at 70–90% identity barely agree in label
  (rho ≈ 0–0.25), so it predicts almost no leak. That removes the bias, but it also misses the cells that do leak
  (HIC/SMAC memorisers at t = .85: about .14–.18 real vs ≈ .01 predicted). Calibrating on 5-nearest-neighbour pairs instead
  of all pairs gives similar or noisier rho, so that variant was not run.
- **Cross-lab:** the measured agreement is close to the assumed 0.9, so label realism does not explain why the memorisers' SMAC leak is
  under-predicted (.16 vs .42–.49). The next subsection explains it.

**Why SMAC was under-predicted: the planted noise was sized wrong for skewed labels** (`did_antibody_rank.py`, post-hoc).
The planted labels used Gaussian noise sized by the SD (Pearson agreement 0.9).
- SMAC is very skewed (skew 5.7; 90% of values ≤ 3.2, max 10). The few outliers inflate the SD, so that noise scrambles the tightly packed bulk.
- As a result, planted copies agreed with the true label at Spearman 0.37, against 0.91 for real cross-lab copies. HIC (0.84 vs 0.97)
  and AC-SINS (0.78 vs 0.89) had the same problem, but milder.
- The fix is rank-matched noise (Gaussian copula). It hits the measured Spearman exactly and keeps each assay's own value distribution.

| predictor vs DiD | corr | inside CI | MAE | signed |
|---|---|---|---|---|
| SD-sized noise (original) | 0.55 | 8/15 | 0.096 | −0.041 |
| rank-matched noise | **0.76** | **10/15** | 0.082 | +0.029 |

- SMAC NN-copy goes from .14 to .49 predicted, against .42 real (now inside the CI).
- SMAC kmer-ridge is still under-predicted (.25 vs .49). A likely reason, not directly tested: GDPa1 SMAC is far more skewed than Jain's
  (5.7 vs 3.4). Ridge on skewed labels wastes the copies; ridge on ranked labels jumps from 0.44 to 0.64 cross-lab.
- physchem-RF is still over-predicted on HIC and AC-SINS.
- Post-hoc: the hypothesis came after seeing the miss. The noise level comes from the measured agreement, not from fitting the DiD.
- **LS table recomputed** with rank-matched noise (`synthetic.py`, `out/leakage_susceptibility.csv`). The SD-noise LS reproduced exactly.
  - Mean LS by model: NN-copy 0.83, kmer-ridge 0.73, physchem-RF 0.58, esm2-ridge 0.31, ablang2-ridge 0.23 (was .56/.61/.51/.29/.21).
  - NN-copy is now the ceiling.
  - Prediction = LS × 0.54. Against the naive-minus-honest drop: corr 0.92, 15/15 inside the CI (was 0.79, 12/15). Against the DiD: corr 0.76, 10/15.

## 5. Defensible claim

1. Planted-leak susceptibility **ranks** models correctly in three domains.
2. Real leak size must be measured by DiD. Threshold dedup leaves near-threshold neighbours (redundancy).
3. **Magnitude** is approximately predicted when planted neighbours resemble real ones and redundancy is modelled (salinity, 19k points):
   for the one model that leaks (knn_geo), MAE is 2–3× lower than a predict-0 null. It is not a clean win, though. The original fit relied on
   over-informative labels offsetting single-neighbour planting, and with matched labels the large-radius leak is under-predicted by about half.
   For images, the exact-copy curve is right on average but not per threshold.
   Antibodies: there is no magnitude prediction better than a null. With 242 antibodies the real leaks are small and the CIs wide.

## Not done
- A realistic-neighbour predictor for EuroSAT that is not tuned on these 36 rows.
- Salinity: plant several neighbours that share a regional component (a one-factor model fitted to the neighbour–test 0.37 and neighbour–neighbour
  0.34 agreement), and drop the soil features to avoid the ridge artifact. Pre-specify it and judge it against the predict-0 null.
- Directly test the skew explanation for SMAC kmer-ridge.
- LUCAS 2015 as an independent replication of the salinity result.
