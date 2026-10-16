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

### EuroSAT, t = .80–.95, 36 rows (DiD estimand)
| predictor | corr | inside CI | MAE (pts) | signed |
|---|---|---|---|---|
| exact-copy planted curve (`did.py`) | 0.85 | 5/36 | 0.74 | +0.30 |
| same, replicated on a 60/20/20 split (`real_planted.py`) | 0.84 | 5/36 | 0.80 | +0.28 |
| correlational P(correct given closeness) + redundancy (`did.py`) | 0.89 | 9/36 | 0.54 | −0.28 |
| blended soft copies + redundancy (`did_red.py`) | 0.26 | 8/36 | 1.19 | −1.17 |
| real reserve tiles planted + redundancy (`real_planted.py`) | 0.37 | 17/36 | 1.23 | −1.23 |

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

## 5. Defensible claim

1. Planted-leak susceptibility **ranks** models correctly in three domains.
2. Real leak size must be measured by DiD. Threshold dedup leaves near-threshold neighbours (redundancy).
3. **Magnitude** is predicted when planted neighbours match the real neighbour statistics and redundancy is
   modelled (salinity, pre-specified). For images, the exact-copy curve is right on average but not per threshold.
   Building realistic synthetic image neighbours is open.

## Not done
- A realistic-neighbour predictor for EuroSAT that is not tuned on these 36 rows.
- An antibody realistic-neighbour predictor: label noise matched to real pairs at each identity, plus redundancy
  (the salinity recipe). Feature realism is limited because the ESM and AbLang embeddings are cache-only.
- LUCAS 2015 as an independent replication of the salinity result.
