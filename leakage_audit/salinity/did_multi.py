"""Multi-neighbour planting (fixed 2026-10-16 before running). did_rank.py plants ONE neighbour per test point, but dedup at r removes several
(mean 1.3 / 2.1 / 3.8 train points within 5 / 10 / 15 km of a near test point), and it under-predicts knn_geo at 15 km by ~half.
Design: keep the real geometry, synthesise the values. Every train point that dedup removes is replaced by a planted point at ITS OWN location,
with label and soil features drawn from its nearest test point by the rank-matched copula at the measured 5-NN pair agreement for that distance
bin (did_rank.py). Predicted per-item gain = (sqerr(dedup) - sqerr(dedup + planted)) / var; pred = same DiD as obs. With the real values this
reproduces obs exactly, so any gap is down to value realism alone. Ablation: 'nearest' plants only the nearest removed point of each near test
point (one neighbour, same geometry). Pre-registered reading: if the count explanation is right, 'all' removes the knn_geo under-prediction
(signed ~0) and 'nearest' keeps it. Both are judged against the predict-0 null. Usage (from ~/Biostuff): python .../did_multi.py 0  (or 3)."""
import sys
s0 = int(sys.argv[1]); H = "/Users/saptarshighosh/Biostuff/leakage_audit/salinity/"
exec(open(H + "calibrate_dist.py").read().split("\nout = []")[0] + open(H + "did_rank.py").read().split("'''")[1])  # data, models, copula
did = lambda d, near: near.mean() * (d[near].mean() - d[~near].mean()); out = []
def plant(j, it, dj):  # j: train rows to replace, it: nearest test row of each, dj: km between them
    b = np.clip(np.digitize(dj, B) - 1, 0, len(B) - 2); ya, sa = np.empty(len(j)), np.full((len(j), soil.shape[1]), np.nan)
    for bb in np.unique(b):
        m = b == bb; ya[m] = cop(qy[it[m]], ry[bb], y)
        for c in range(soil.shape[1]): sa[m, c] = np.where(np.isnan(soil[it[m], c]), np.nan, cop(qs[it[m], c], rs[bb][c], soil[:, c]))
    return {"ridge_soil": sa, "rf_soil": sa, "knn_geo": xyz[j], "rf_soil_geo": np.c_[sa, xyz[j]]}, ya
for s in range(s0, s0 + 3):
    tr, te = train_test_split(idx, test_size=.2, random_state=s); P = lambda a: np.c_[lat[a], lon[a]]
    Ttr, Tte = BallTree(P(tr), metric="haversine"), BallTree(P(te), metric="haversine")
    d1, n1 = (a[:, 0] for a in Ttr.query(P(te), k=1)); d1 *= 6371
    ef = {n: (y[te] - pred(n, tr, te)) ** 2 for n in MK}; v = y[te].var()
    for r in [5, 10, 15]:
        bad = np.unique(np.concatenate(Ttr.query_radius(P(te), r / 6371))); keep = np.setdiff1d(np.arange(len(tr)), bad); near = d1 <= r
        dj, it = (a[:, 0] for a in Tte.query(P(tr[bad]), k=1)); dj *= 6371
        one = np.isin(bad, np.unique(n1[near]))  # nearest removed point of each near test point
        V = {"all": plant(tr[bad], te[it], dj), "nearest": plant(tr[bad][one], te[it[one]], dj[one])}
        for n in MK:
            ek = (y[te] - pred(n, tr[keep], te)) ** 2; d = (ek - ef[n]) / v
            bs = [(lambda i: did(d[i], near[i]))(rng.integers(0, len(d), len(d))) for _ in range(300)]
            row = dict(seed=s, r=r, model=n, f_near=near.mean(), n_removed=len(bad), n_nearest=int(one.sum()), obs=did(d, near),
                       lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
            for k, (Xa, ya) in V.items(): row["pred_" + k] = did((ek - (y[te] - pred(n, tr[keep], te, Xa=Xa[n], ya=ya)) ** 2) / v, near)
            out.append(row)
D = pd.DataFrame(out); D.to_csv(O + f"did_multi_seeds{s0}to{s0 + 2}.csv", index=False); D["null"] = 0.0
print(D.groupby(["model", "r"])[["f_near", "n_removed", "n_nearest", "pred_nearest", "pred_all", "obs", "lo", "hi"]].mean().round(3).to_string())
for c in ["pred_nearest", "pred_all", "null"]:
    k = D[D.model == "knn_geo"]; ins = lambda x: int(((x[c] >= x.lo) & (x[c] <= x.hi)).sum())
    print(f"{c:13s} corr {np.corrcoef(D[c], D.obs)[0, 1] if D[c].std() else np.nan:.2f} inside {ins(D)}/{len(D)} MAE {(D[c] - D.obs).abs().mean():.3f} "
          f"signed {(D[c] - D.obs).mean():+.3f} | knn_geo inside {ins(k)}/{len(k)} MAE {(k[c] - k.obs).abs().mean():.3f} signed {(k[c] - k.obs).mean():+.3f}")
