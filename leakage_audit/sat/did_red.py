"""EuroSAT version of the salinity winner (same method in both domains): causal closeness-gain curve g_M(c) from planted soft copies (calibrate.py recipe),
prediction = DiD form of g(s1) - g(s2); s1/s2 = closeness to nearest train tile before/after dedup. Observed = DiD from did.csv."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
z = np.load("leakage_audit/sat/eurosat.npz"); X, y = z["emb"], z["y"]
MK = {"ridge": lambda: RidgeClassifier(alpha=10), "knn1": lambda: KNeighborsClassifier(1), "knn5": lambda: KNeighborsClassifier(5)}
def correct(Xtr, ytr, te, n):
    sc = StandardScaler().fit(Xtr); return MK[n]().fit(sc.transform(Xtr), ytr).predict(sc.transform(X[te])) == y[te]
nz = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True); Dd = pd.read_csv("leakage_audit/sat/did.csv"); rng = np.random.default_rng(1); out = []
did = lambda d, near: near.mean() * (d[near].mean() - d[~near].mean())
for s in range(3):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
    S = np.vstack([nz(X[te][i:i + 1500]) @ nz(X[tr]).T for i in range(0, len(te), 1500)]); s1 = S.max(1)
    G = {n: [] for n in MK}; base = {n: correct(X[tr], y[tr], te, n) for n in MK}
    for a in [.4, .55, .7, .8, .9, .95, 1.0]:
        other = np.array([rng.choice(tr[y[tr] == y[i]]) for i in te]); Xp = a * X[te] + (1 - a) * X[other]; c = (nz(Xp) * nz(X[te])).sum(1)
        for n in MK: G[n].append(np.c_[c, correct(np.vstack([X[tr], Xp]), np.concatenate([y[tr], y[te]]), te, n).astype(float) - base[n]])
    for n in MK:
        g = np.vstack(G[n]); q = np.quantile(g[:, 0], np.linspace(0, 1, 11)); b = np.clip(np.digitize(g[:, 0], q[1:-1]), 0, 9)
        cx = np.array([g[b == i, 0].mean() for i in range(10)]); gy = np.maximum.accumulate(np.array([g[b == i, 1].mean() for i in range(10)])); gi = lambda v: np.interp(v, cx, gy)
        for t in [.95, .90, .85, .80]:
            bad = (S >= t).any(0); s2 = S[:, ~bad].max(1); near = s1 >= t; o = Dd[(Dd.seed == s) & (Dd.t == t) & (Dd.model == n)].iloc[0]
            out.append(dict(seed=s, t=t, model=n, pred_old=o.pred_old, pred_red=o.pred_red, pred_cal_red=did(gi(s1) - gi(s2), near), obs=o.obs, lo=o.lo, hi=o.hi))
D = pd.DataFrame(out); D.to_csv("leakage_audit/sat/did_red.csv", index=False)
print((D.groupby(["model", "t"])[["pred_old", "pred_red", "pred_cal_red", "obs", "lo", "hi"]].mean() * 100).round(2).to_string())
for c in ["pred_old", "pred_red", "pred_cal_red"]: print(c, "corr", round(np.corrcoef(D[c], D.obs)[0, 1], 2), "inside CI", int(((D[c] >= D.lo) & (D[c] <= D.hi)).sum()), "/", len(D), "MAE(pts)", round((D[c] - D.obs).abs().mean() * 100, 2), "signed(pts)", round((D[c] - D.obs).mean() * 100, 2))
