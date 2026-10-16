"""Closeness-calibrated prediction. g_M(c): accuracy gain from a planted neighbour at realised cosine c (soft copies = a*x + (1-a)*x_same_class_other).
Predicted dedup drop at threshold t = mean over ALL test tiles of g_M(s_i) * [s_i >= t], s_i = natural max cosine to train. Compared with observed paired drop (predict.csv)."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
z = np.load("leakage_audit/sat/eurosat.npz"); X, y = z["emb"], z["y"]
MK = {"ridge": lambda: RidgeClassifier(alpha=10), "knn1": lambda: KNeighborsClassifier(1), "knn5": lambda: KNeighborsClassifier(5)}
def correct(Xtr, ytr, te, n):
    sc = StandardScaler().fit(Xtr); return MK[n]().fit(sc.transform(Xtr), ytr).predict(sc.transform(X[te])) == y[te]
nz = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
R = pd.read_csv("leakage_audit/sat/predict.csv"); rng = np.random.default_rng(1); out = []
for s in range(3):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
    S = np.concatenate([(nz(X[te][i:i + 1500]) @ nz(X[tr]).T).max(1) for i in range(0, len(te), 1500)])
    # gain-vs-cosine curve from soft copies (no use of the dedup outcome)
    G = {n: [] for n in MK}; base = {n: correct(X[tr], y[tr], te, n) for n in MK}
    for a in [.4, .55, .7, .8, .9, .95, 1.0]:
        other = np.array([rng.choice(tr[y[tr] == y[i]]) for i in te])
        Xp = a * X[te] + (1 - a) * X[other]; c = (nz(Xp) * nz(X[te])).sum(1)
        for n in MK: G[n].append(np.c_[c, correct(np.vstack([X[tr], Xp]), np.concatenate([y[tr], y[te]]), te, n).astype(float) - base[n]])
    for n in MK:
        g = np.vstack(G[n]); bins = np.quantile(g[:, 0], np.linspace(0, 1, 11)); idx = np.clip(np.digitize(g[:, 0], bins[1:-1]), 0, 9)
        cx = np.array([g[idx == b, 0].mean() for b in range(10)]); gy = np.maximum.accumulate(np.array([g[idx == b, 1].mean() for b in range(10)]))  # monotone in closeness
        gi = np.interp(S, cx, gy)
        for t in [.95, .90, .85, .80]:
            o = R[(R.seed == s) & (R.t == t) & (R.model == n)].iloc[0]
            out.append(dict(seed=s, t=t, model=n, pred_old=o.pred, pred_cal=(gi * (S >= t)).mean(), obs=o.obs, lo=o.lo, hi=o.hi))
D = pd.DataFrame(out); D.to_csv("leakage_audit/sat/calibrated.csv", index=False)
M = D.groupby(["model", "t"])[["pred_old", "pred_cal", "obs", "lo", "hi"]].mean() * 100; print(M.round(2).to_string())
for c in ["pred_old", "pred_cal"]:
    print(c, "corr", round(np.corrcoef(D[c], D.obs)[0, 1], 2), "inside CI", int(((D[c] >= D.lo) & (D[c] <= D.hi)).sum()), "/", len(D), "MAE(pts)", round((D[c] - D.obs).abs().mean() * 100, 2))
