# Leakage susceptibility (domain-agnostic)

Given a benchmark with a train/test split, a model M, and an overlap rule O (what counts as "the same item": exact match, identity >= t, same tile/location, ...):

1. **Overlap fraction** f = share of test items that O flags as present in train.
2. **Inflation curve** I_M(f) = score(M, f of the test set planted into train, labels noised to the benchmark's label-agreement level) - score(M, f = 0). Built by planting, so it needs no extra data.
3. **Leakage susceptibility** LS_M = slope of I_M(f) for f <= 0.5 (score points per unit overlap). Higher = the model memorizes more. NN-copy is the ceiling reference.
4. **Predicted inflation** of a reported benchmark = I_M(f_observed). **Check** = measured naive-vs-honest drop with a paired bootstrap CI.

Antibody evidence (Jain2017 -> GDPa1, f = 131/242 = 0.54, 15 model x assay cells):
predicted vs observed drop correlation 0.79; predicted value inside the paired 95% CI in 12 of 15 cells.
The 3 misses (HIC kmer-ridge, SMAC NN-copy, SMAC kmer-ridge) all underpredict: real cross-lab copies carry the lab's labels, planted ones come from the same lab.

LS by model (noisy labels, mean of 3 assays): NN-copy 0.56, kmer-ridge 0.61, physchem-RF 0.51, esm2-ridge 0.29, ablang2-ridge 0.21.

Second domain: pick a benchmark, define O (e.g. same scene / spatial buffer for satellite tiles), measure f, build I_M(f) by planting, compare to the honest-split drop. Same four steps.
