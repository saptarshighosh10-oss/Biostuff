"""Does the planted-leak curve predict the real drop on EuroSAT? Train-side dedup at cosine threshold t vs a size-matched random removal (same test set -> paired)."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
z = np.load("leakage_audit/sat/eurosat.npz"); X, y = z["emb"], z["y"]
P = pd.read_csv("leakage_audit/sat/planted.csv").groupby(["model", "f"]).acc.mean().unstack(0)
curve = lambda m, f: np.interp(f, P.index, P[m] - P[m].loc[0])
MK = {"ridge": lambda: RidgeClassifier(alpha=10), "knn1": lambda: KNeighborsClassifier(1), "knn5": lambda: KNeighborsClassifier(5)}
def correct(tr, te, n):
    sc = StandardScaler().fit(X[tr]); return MK[n]().fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te])) == y[te]
rows = []; rng = np.random.default_rng(0)
for s in range(3):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
    N = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True); A, B = N(X[tr]), N(X[te])
    S = np.vstack([B[i:i + 1500] @ A.T for i in range(0, len(te), 1500)])  # test x train cosine
    for t in [.95, .90, .85, .80]:
        near = (S >= t); f_obs = near.any(1).mean(); drop_tr = near.any(0)
        keep = tr[~drop_tr]; rnd = rng.choice(tr, len(keep), replace=False)
        for n in MK:
            a0 = correct(tr, te, n); ad = correct(keep, te, n); ar = correct(rnd, te, n)
            d = ar.astype(float) - ad  # paired per-test-item: size-matched random minus deduped = leakage effect
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]
            rows.append(dict(seed=s, t=t, model=n, f_obs=f_obs, n_removed=int(drop_tr.sum()), pred=curve(n, f_obs), obs=d.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
R = pd.DataFrame(rows); R.to_csv("leakage_audit/sat/predict.csv", index=False)
G = R.groupby(["model", "t"])[["f_obs", "n_removed", "pred", "obs", "lo", "hi"]].mean().round(4); print((G.assign(**{c: G[c] * 100 for c in ["pred", "obs", "lo", "hi"]})).round(2).to_string())
print("corr(pred,obs) all rows:", round(np.corrcoef(R.pred, R.obs)[0, 1], 2), " inside CI:", int(((R.pred >= R.lo) & (R.pred <= R.hi)).sum()), "/", len(R))
