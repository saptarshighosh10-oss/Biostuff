"""LUCAS 2018 EC: random vs spatial-block split gap, planted-leak curve, and buffer-dedup drop predicted from the curve. Metric: R2 on log1p(EC)."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, BallTree
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score
D = "/Users/saptarshighosh/Biostuff/data/external/lucas2018/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv"; O = "/Users/saptarshighosh/Biostuff/leakage_audit/salinity/"
d = pd.read_csv(D, low_memory=False).dropna(subset=["EC"]); d = d[d.EC >= 0].reset_index(drop=True)
y = np.log1p(d.EC.values); lat, lon = np.radians(d.TH_LAT.values), np.radians(d.TH_LONG.values)
xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
soil = d[["pH_H2O", "OC", "CaCO3", "P", "N", "K", "Elev"]].apply(pd.to_numeric, errors="coerce").values
F = {"ridge_soil": soil, "rf_soil": soil, "knn_geo": xyz, "rf_soil_geo": np.c_[soil, xyz]}
MK = {"ridge_soil": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(1.0)),
      "rf_soil": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(100, min_samples_leaf=3, n_jobs=-1, random_state=0)),
      "knn_geo": lambda: KNeighborsRegressor(5), "rf_soil_geo": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(100, min_samples_leaf=3, n_jobs=-1, random_state=0))}
def pred(n, tr, te, Xa=None, ya=None):  # Xa/ya: extra training rows (planted copies)
    X = F[n]; Xt = X[tr] if Xa is None else np.vstack([X[tr], Xa]); yt = y[tr] if ya is None else np.concatenate([y[tr], ya])
    return MK[n]().fit(Xt, yt).predict(X[te])
r2 = lambda n, tr, te, **k: r2_score(y[te], pred(n, tr, te, **k))
rows = []; idx = np.arange(len(y))
# 1) random vs spatial-block gap
for s in range(3):
    tr, te = train_test_split(idx, test_size=.2, random_state=s)
    for n in MK: rows.append(dict(kind="split", setting="random", seed=s, model=n, r2=r2(n, tr, te)))
for deg in [0.5, 1, 2, 4]:
    g = (np.floor(d.TH_LAT / deg).astype(int) * 1000 + np.floor(d.TH_LONG / deg).astype(int)).values
    for k, (tr, te) in enumerate(GroupKFold(5).split(idx, groups=g)):
        for n in MK: rows.append(dict(kind="split", setting=f"block{deg}deg", seed=k, model=n, r2=r2(n, tr, te)))
# 2) planted-leak curve (random split) and 3) buffer-dedup vs predicted
tree_all = BallTree(np.c_[lat, lon], metric="haversine"); rng = np.random.default_rng(0)
for s in range(3):
    tr, te = train_test_split(idx, test_size=.2, random_state=s); base = {n: r2(n, tr, te) for n in MK}
    for f in [.1, .25, .5, 1.0]:
        pick = rng.choice(te, int(f * len(te)), replace=False)
        for n in MK: rows.append(dict(kind="planted", setting=f, seed=s, model=n, r2=r2(n, tr, te, Xa=F[n][pick], ya=y[pick]) - base[n]))
    dist = BallTree(np.c_[lat[tr], lon[tr]], metric="haversine").query(np.c_[lat[te], lon[te]], k=1)[0][:, 0] * 6371
    for r in [5, 10, 25, 50]:
        near = BallTree(np.c_[lat[tr], lon[tr]], metric="haversine").query_radius(np.c_[lat[te], lon[te]], r / 6371)
        bad = np.unique(np.concatenate(near)); keep = np.setdiff1d(np.arange(len(tr)), bad)
        rnd = rng.choice(len(tr), len(keep), replace=False)
        for n in MK:
            ek = (y[te] - pred(n, tr[keep], te)) ** 2; er = (y[te] - pred(n, tr[rnd], te)) ** 2
            bs = [(lambda i: (ek[i].mean() - er[i].mean()) / y[te][i].var())(rng.integers(0, len(te), len(te))) for _ in range(200)]
            rows.append(dict(kind="dedup", setting=r, seed=s, model=n, f_obs=(dist <= r).mean(), obs=(ek.mean() - er.mean()) / y[te].var(),
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
R = pd.DataFrame(rows); R.to_csv(O + "results.csv", index=False)
S = R[R.kind == "split"].groupby(["model", "setting"]).r2.mean().unstack().round(3); print("R2 by split\n", S[["random", "block0.5deg", "block1deg", "block2deg", "block4deg"]].to_string())
P = R[R.kind == "planted"].groupby(["model", "setting"]).r2.mean().unstack().round(3); print("\nplanted R2 inflation by f\n", P.to_string())
Dd = R[R.kind == "dedup"].copy(); curve = lambda n, f: np.interp(f, [0] + list(P.columns), [0] + list(P.loc[n]))
Dd["pred"] = [curve(m, f) for m, f in zip(Dd.model, Dd.f_obs)]
# obs = R2 loss from buffer-dedup vs size-matched random removal = (mse_keep - mse_rnd)/var(y_test)
M = Dd.groupby(["model", "setting"])[["f_obs", "pred", "obs", "lo", "hi"]].mean().round(3); print("\ndedup: predicted vs observed R2 drop\n", M.to_string())
print("corr", round(np.corrcoef(Dd.pred, Dd.obs)[0, 1], 2), "inside CI", int(((Dd.pred >= Dd.lo) & (Dd.pred <= Dd.hi)).sum()), "/", len(Dd))
