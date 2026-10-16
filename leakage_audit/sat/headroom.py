"""Headroom-aware prediction: only tiles the model currently gets wrong can gain. fix_M = planted gain(f=1)/baseline error. pred_h = mean_i near_i*wrong_i*fix_M; pred_hl also needs the nearest train tile to share the test label."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
z = np.load("leakage_audit/sat/eurosat.npz"); X, y = z["emb"], z["y"]
MK = {"ridge": lambda: RidgeClassifier(alpha=10), "knn1": lambda: KNeighborsClassifier(1), "knn5": lambda: KNeighborsClassifier(5)}
P = pd.read_csv("leakage_audit/sat/planted.csv").groupby(["model", "f"]).acc.mean().unstack(0)
fix = {m: (P[m].loc[1.0] - P[m].loc[0]) / (1 - P[m].loc[0]) for m in MK}; print("fix rate (share of wrong tiles fixed by a true-label copy):", {m: round(v, 3) for m, v in fix.items()})
nz = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True); R = pd.read_csv("leakage_audit/sat/predict.csv"); out = []
for s in range(3):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
    S = np.vstack([nz(X[te][i:i + 1500]) @ nz(X[tr]).T for i in range(0, len(te), 1500)]); mx, am = S.max(1), S.argmax(1); same = y[tr][am] == y[te]
    sc = StandardScaler().fit(X[tr])
    for n in MK:
        wrong = MK[n]().fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te])) != y[te]
        for t in [.95, .90, .85, .80]:
            o = R[(R.seed == s) & (R.t == t) & (R.model == n)].iloc[0]; near = mx >= t
            out.append(dict(seed=s, t=t, model=n, pred_old=o.pred, pred_h=(near * wrong).mean() * fix[n], pred_hl=(near * wrong * same).mean() * fix[n], obs=o.obs, lo=o.lo, hi=o.hi))
D = pd.DataFrame(out); D.to_csv("leakage_audit/sat/headroom.csv", index=False); print((D.groupby(["model", "t"])[["pred_old", "pred_h", "pred_hl", "obs", "lo", "hi"]].mean() * 100).round(2).to_string())
for c in ["pred_old", "pred_h", "pred_hl"]: print(c, "corr", round(np.corrcoef(D[c], D.obs)[0, 1], 2), "inside CI", int(((D[c] >= D.lo) & (D[c] <= D.hi)).sum()), "/", len(D), "MAE(pts)", round((D[c] - D.obs).abs().mean() * 100, 2))
