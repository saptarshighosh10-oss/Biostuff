"""Distance-aware prediction. g_M(d): R2 gain when every test point gets a planted neighbour at distance d whose label/features differ
by the natural rms difference of real LUCAS neighbour pairs at that distance (so a neighbour is not an exact copy). Predicted dedup drop at radius r = mean_i g_M(d_i)[d_i<=r],
d_i = km to nearest train point. Observed = loss vs a control removing the same count of train points in the same spatial pattern (neighbourhoods of random train points), which cancels the coverage confound."""
import numpy as np, pandas as pd, sys
sys.argv = ["x"]; src = open("/Users/saptarshighosh/Biostuff/leakage_audit/salinity/audit.py").read().split("rows = []; idx")[0]; exec(src)  # reuse data/model setup only
idx = np.arange(len(y)); rng = np.random.default_rng(0); B = [0, 3, 7, 15, 30, 55]
dk, nb = BallTree(np.c_[lat, lon], metric="haversine").query(np.c_[lat, lon], k=6); dk, nb = dk[:, 1:] * 6371, nb[:, 1:]
bins = np.digitize(dk.ravel(), B) - 1; I = np.repeat(idx, 5); J = nb.ravel()
rms = lambda v, b: np.sqrt(np.nanmean(v[bins == b] ** 2, axis=0))
dy = {b: rms((y[I] - y[J])[:, None], b)[0] for b in range(len(B) - 1)}; ds = {b: rms(soil[I] - soil[J], b) for b in range(len(B) - 1)}
mid = [(B[i] + B[i + 1]) / 2 for i in range(len(B) - 1)]; print("natural pair rms dLogEC by distance bin (km):", {m: round(dy[i], 2) for i, m in enumerate(mid)})
def planted(tr, te, dkm, b):
    th = rng.uniform(0, 2 * np.pi, len(te)); la = lat[te] + dkm * np.cos(th) / 6371; lo = lon[te] + dkm * np.sin(th) / (6371 * np.cos(lat[te]))
    xyz2 = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]; s2 = soil[te] + rng.normal(0, 1, soil[te].shape) * ds[b]
    return {"ridge_soil": s2, "rf_soil": s2, "knn_geo": xyz2, "rf_soil_geo": np.c_[s2, xyz2]}, y[te] + rng.normal(0, dy[b], len(te))
out = []
for s in range(3):
    tr, te = train_test_split(idx, test_size=.2, random_state=s); base = {n: r2(n, tr, te) for n in MK}
    d_i = BallTree(np.c_[lat[tr], lon[tr]], metric="haversine").query(np.c_[lat[te], lon[te]], k=1)[0][:, 0] * 6371
    g = {n: [] for n in MK}
    for b in range(len(B) - 1):
        Xa, ya = planted(tr, te, mid[b], b)
        for n in MK: g[n].append(r2(n, tr, te, Xa=Xa[n], ya=ya) - base[n])
    Ttr = BallTree(np.c_[lat[tr], lon[tr]], metric="haversine")
    for r in [5, 10, 25, 50]:
        bad = np.unique(np.concatenate(Ttr.query_radius(np.c_[lat[te], lon[te]], r / 6371))); keep = np.setdiff1d(np.arange(len(tr)), bad)
        nbr = Ttr.query_radius(np.c_[lat[tr], lon[tr]], r / 6371); rem = np.zeros(len(tr), bool); P = rng.permutation(len(tr))  # control: same removal count, same spatial pattern (neighbourhoods of random train points), not around test points
        for p_ in P:
            if rem.sum() >= len(bad): break
            if not rem[p_]: rem[nbr[p_]] = True; rem[p_] = False
        ctrl = np.where(~rem)[0]
        for n in MK:
            ek = (y[te] - pred(n, tr[keep], te)) ** 2; ec = (y[te] - pred(n, tr[ctrl], te)) ** 2; v = y[te].var()
            bs = [(lambda i: (ek[i].mean() - ec[i].mean()) / v)(rng.integers(0, len(te), len(te))) for _ in range(200)]
            out.append(dict(seed=s, r=r, model=n, pred_dist=(np.interp(d_i, mid, g[n]) * (d_i <= r)).mean(), obs=(ek.mean() - ec.mean()) / v,
                            lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n_keep=len(keep), n_ctrl=len(ctrl)))
D = pd.DataFrame(out); D.to_csv("leakage_audit/salinity/calibrated.csv", index=False)
print((D.groupby(["model", "r"])[["pred_dist", "obs", "lo", "hi"]].mean()).round(3).to_string())
print("corr", round(np.corrcoef(D.pred_dist, D.obs)[0, 1], 2), "inside CI", int(((D.pred_dist >= D.lo) & (D.pred_dist <= D.hi)).sum()), "/", len(D), "MAE", round((D.pred_dist - D.obs).abs().mean(), 3))
