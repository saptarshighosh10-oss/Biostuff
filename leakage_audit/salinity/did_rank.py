"""Salinity planted-neighbour realism check (fixed 2026-10-16 before running). calibrate_dist.planted() adds N(0, rms_pair) to the test label/features.
That keeps the planted label centred on the test label, so it agrees with it at Pearson ~0.67, while real 5-NN pairs agree at ~0.40 (a real neighbour
regresses to the mean). Fix, same recipe as antibodies: Gaussian copula per distance bin at the measured 5-NN pair Spearman, for the label and each
soil feature, mapped back through that variable's own quantiles (NaN stays NaN). Everything else is did.py unchanged. Expectation: weaker planted
neighbours -> smaller g(d) -> pred_dist_red falls below the 26/36 (seeds 0-2) and 27/36 (seeds 3-5) fits, i.e. the old fit partly relied on
over-informative labels. Usage (from ~/Biostuff): python leakage_audit/salinity/did_rank.py 0   (or 3 for seeds 3-5)."""
import sys
s0 = int(sys.argv[1]); src = open("/Users/saptarshighosh/Biostuff/leakage_audit/salinity/did.py").read()
patch = '''
from scipy.stats import norm, spearmanr
nq = lambda v: norm.ppf(pd.Series(v).rank().values / (np.sum(~np.isnan(v)) + 1))
def rho(a, b):
    k = ~(np.isnan(a) | np.isnan(b)); return 2 * np.sin(np.pi * spearmanr(a[k], b[k])[0] / 6)
qy, qs = nq(y), np.column_stack([nq(c) for c in soil.T])
ry = {b: rho(y[I][bins == b], y[J][bins == b]) for b in range(len(B) - 1)}
rs = {b: [rho(soil[I][bins == b, j], soil[J][bins == b, j]) for j in range(soil.shape[1])] for b in range(len(B) - 1)}
cop = lambda q, r, v: np.nanquantile(v, norm.cdf(np.nan_to_num(r * q + np.sqrt(1 - r ** 2) * rng.normal(0, 1, len(q)))))
def planted(tr, te, dkm, b):
    th = rng.uniform(0, 2 * np.pi, len(te)); la = lat[te] + dkm * np.cos(th) / 6371; lo = lon[te] + dkm * np.sin(th) / (6371 * np.cos(lat[te]))
    xyz2 = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    s2 = np.column_stack([np.where(np.isnan(soil[te, j]), np.nan, cop(qs[te, j], rs[b][j], soil[:, j])) for j in range(soil.shape[1])])
    return {"ridge_soil": s2, "rf_soil": s2, "knn_geo": xyz2, "rf_soil_geo": np.c_[s2, xyz2]}, cop(qy[te], ry[b], y)
_, chk = planted(idx, idx, 0, 0); print("label spearman real", round(spearmanr(y[I][bins == 0], y[J][bins == 0])[0], 2), "planted", round(spearmanr(y, chk)[0], 2))
'''
src = src.replace('[0])  # data, models, planted(), g bins', '[0])  # data, models, planted(), g bins' + patch)
src = src.replace("for s in range(3):", f"for s in range({s0}, {s0 + 3}):").replace('O + "did.csv"', f'O + "did_rank_seeds{s0}to{s0 + 2}.csv"')
assert "did_rank_seeds" in src and "def rho" in src
exec(src)
