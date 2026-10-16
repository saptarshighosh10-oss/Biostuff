"""Out-of-domain check of the EuroSAT DiD finding (predictors fixed before running). Observed leak = f_near*[loss_near - loss_far], loss = (sqerr_dedup - sqerr_full)/var(y_test),
near = test point with a train point within r km. Predictors: pred_old = exact-copy planted curve at f_near; pred_dist = sum of distance-aware gain g(d1);
pred_dist_red = DiD form of g(d1) - g(d2), d2 = km to nearest train point left after dedup (redundancy-aware, causal g)."""
import numpy as np, pandas as pd
exec(open("/Users/saptarshighosh/Biostuff/leakage_audit/salinity/calibrate_dist.py").read().split("\nout = []")[0])  # data, models, planted(), g bins
Rp = pd.read_csv(O + "results.csv"); Rp = Rp[Rp.kind == "planted"].assign(f=lambda x: x.setting.astype(float)).groupby(["model", "f"]).r2.mean().unstack()
curve = lambda n, f: np.interp(f, [0] + list(Rp.columns), [0] + list(Rp.loc[n]))
did = lambda d, near: near.mean() * (d[near].mean() - d[~near].mean()); out = []
for s in range(3):
    tr, te = train_test_split(idx, test_size=.2, random_state=s); base = {n: r2(n, tr, te) for n in MK}
    Ttr = BallTree(np.c_[lat[tr], lon[tr]], metric="haversine"); d1 = Ttr.query(np.c_[lat[te], lon[te]], k=1)[0][:, 0] * 6371
    g = {n: [] for n in MK}
    for b in range(len(B) - 1):
        Xa, ya = planted(tr, te, mid[b], b)
        for n in MK: g[n].append(r2(n, tr, te, Xa=Xa[n], ya=ya) - base[n])
    ef = {n: (y[te] - pred(n, tr, te)) ** 2 for n in MK}; v = y[te].var()
    for r in [5, 10, 15]:
        bad = np.unique(np.concatenate(Ttr.query_radius(np.c_[lat[te], lon[te]], r / 6371))); keep = np.setdiff1d(np.arange(len(tr)), bad)
        d2 = BallTree(np.c_[lat[tr[keep]], lon[tr[keep]]], metric="haversine").query(np.c_[lat[te], lon[te]], k=1)[0][:, 0] * 6371; near = d1 <= r
        for n in MK:
            d = ((y[te] - pred(n, tr[keep], te)) ** 2 - ef[n]) / v; gi = lambda dd: np.interp(dd, mid, g[n])
            bs = [(lambda i: did(d[i], near[i]))(rng.integers(0, len(d), len(d))) for _ in range(300)]
            out.append(dict(seed=s, r=r, model=n, f_near=near.mean(), n_keep=len(keep), pred_old=curve(n, near.mean()), pred_dist=(gi(d1) * near).mean(),
                            pred_dist_red=did(gi(d1) - gi(d2), near), obs=did(d, near), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
D = pd.DataFrame(out); D.to_csv(O + "did.csv", index=False)
print(D.groupby(["model", "r"])[["f_near", "n_keep", "pred_old", "pred_dist", "pred_dist_red", "obs", "lo", "hi"]].mean().round(3).to_string())
for c in ["pred_old", "pred_dist", "pred_dist_red"]: print(c, "corr", round(np.corrcoef(D[c], D.obs)[0, 1], 2), "inside CI", int(((D[c] >= D.lo) & (D[c] <= D.hi)).sum()), "/", len(D), "MAE", round((D[c] - D.obs).abs().mean(), 3), "signed", round((D[c] - D.obs).mean(), 3))
