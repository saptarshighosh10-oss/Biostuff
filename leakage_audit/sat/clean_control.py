"""Clean control: the size-matched random removal in predict.py also deletes near neighbours of test tiles (diluting the observed drop).
Here the control removes the same number of train tiles drawn ONLY from tiles that are no test tile's neighbour (cos < t to every test tile).
obs_clean = acc(clean control) - acc(dedup), paired per test tile. Predictions reused from predict.csv (hard) and calibrated.csv (closeness)."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
z = np.load("leakage_audit/sat/eurosat.npz"); X, y = z["emb"], z["y"]
MK = {"ridge": lambda: RidgeClassifier(alpha=10), "knn1": lambda: KNeighborsClassifier(1), "knn5": lambda: KNeighborsClassifier(5)}
def correct(tr, te, n):
    sc = StandardScaler().fit(X[tr]); return MK[n]().fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te])) == y[te]
Pr = pd.read_csv("leakage_audit/sat/predict.csv"); Ca = pd.read_csv("leakage_audit/sat/calibrated.csv"); rows = []; rng = np.random.default_rng(0)
for s in range(3):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
    N = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True); A, B = N(X[tr]), N(X[te])
    S = np.vstack([B[i:i + 1500] @ A.T for i in range(0, len(te), 1500)])
    for t in [.95, .90, .85, .80]:
        bad = (S >= t).any(0); keep = tr[~bad]
        ctrl = np.setdiff1d(tr, rng.choice(tr[~bad], bad.sum(), replace=False))  # same count removed, no test neighbour among them
        assert len(ctrl) == len(keep) and not np.isin(tr[bad], np.setdiff1d(tr, ctrl)).any()
        for n in MK:
            d = correct(ctrl, te, n).astype(float) - correct(keep, te, n)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]
            k = (Pr.seed == s) & (Pr.t == t) & (Pr.model == n); kc = (Ca.seed == s) & (Ca.t == t) & (Ca.model == n)
            rows.append(dict(seed=s, t=t, model=n, pred_old=Pr[k].pred.iloc[0], pred_cal=Ca[kc].pred_cal.iloc[0], obs_old=Pr[k].obs.iloc[0], obs=d.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
D = pd.DataFrame(rows); D.to_csv("leakage_audit/sat/clean_control.csv", index=False)
print((D.groupby(["model", "t"])[["pred_old", "pred_cal", "obs_old", "obs", "lo", "hi"]].mean() * 100).round(2).to_string())
for c in ["pred_old", "pred_cal"]: print(c, "vs clean obs: corr", round(np.corrcoef(D[c], D.obs)[0, 1], 2), "inside CI", int(((D[c] >= D.lo) & (D[c] <= D.hi)).sum()), "/", len(D), "MAE(pts)", round((D[c] - D.obs).abs().mean() * 100, 2), "mean signed err(pts)", round((D[c] - D.obs).mean() * 100, 2))
