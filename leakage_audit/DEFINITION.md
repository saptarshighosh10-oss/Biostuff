# Leakage susceptibility (domain-agnostic)

Given a benchmark with a train/test split, a model M, and an overlap rule O (what counts as "the same item": exact match, identity >= t, same tile/location, ...):

1. **Overlap fraction** f = share of test items that O flags as present in train.
2. **Inflation curve** I_M(f) = score(M, f of the test set planted into train, labels noised to the benchmark's label-agreement level) - score(M, f = 0). Built by planting, so it needs no extra data.
3. **Leakage susceptibility** LS_M = slope of I_M(f) for f <= 0.5 (score points per unit overlap). Higher = the model memorizes more. NN-copy is the ceiling reference.
4. **Predicted inflation** of a reported benchmark = I_M(f_observed). **Check** = measured naive-vs-honest drop with a paired bootstrap CI.

Label noise must match agreement on ranks, not SD. Use a Gaussian copula: normal scores mixed to the measured Spearman agreement,
mapped back through the assay's own quantiles. SD-sized Gaussian noise made planted SMAC copies agree at only Spearman 0.37 (real 0.91),
because SMAC is skewed (5.7), and that under-stated LS on skewed assays. Recomputed by `synthetic.py`; the table is `out/leakage_susceptibility.csv`.

Antibody evidence (Jain2017 -> GDPa1, f = 131/242 = 0.54, 15 model x assay cells; measured agreement HIC .97, AC-SINS .89, SMAC .91):
- vs naive-minus-honest drop: corr 0.92, 15/15 inside the paired 95% CI (SD-sized noise gave 0.79, 12/15).
- vs the DiD leak (`did_antibody.py`): corr 0.76, 10/15 inside the CI.
  Misses: physchem-RF over-predicted (HIC, AC-SINS), SMAC kmer-ridge under-predicted (.28 vs .49), and AC-SINS NN-copy and esm2 just outside the CI.

LS by model (rank-matched labels, mean of 3 assays): NN-copy 0.83, kmer-ridge 0.73, physchem-RF 0.58, esm2-ridge 0.31, ablang2-ridge 0.23.
(SD-sized noise gave 0.56 / 0.61 / 0.51 / 0.29 / 0.21. NN-copy is now the ceiling, as it should be.)

Second domain: pick a benchmark, define O (e.g. same scene / spatial buffer for satellite tiles), measure f, build I_M(f) by planting, compare to the honest-split drop. Same four steps.
