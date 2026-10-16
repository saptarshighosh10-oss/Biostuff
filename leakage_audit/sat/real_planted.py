"""EuroSAT analog of the salinity winner, fixed before running. Blended soft copies were unrealistic (gain flat ~17% for 1-NN even at cos .73), so plant REAL tiles:
split 60% train / 20% reserve / 20% test; for each test tile plant its k-th nearest reserve tile with that tile's OWN label (k=1,2,5,20,100 gives a spread of closeness).
g_M(c) = per-tile gain vs realised closeness c. Prediction = DiD form of g(s1)-g(s2) around threshold dedup; observed = DiD of the real dedup drop. Also exact-copy curve (pred_old)."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
z = np.load("leakage_audit/sat/eurosat.npz"); X, y = z["emb"], z["y"]
MK = {"ridge": lambda: RidgeClassifier(alpha=10), "knn1": lambda: KNeighborsClassifier(1), "knn5": lambda: KNeighborsClassifier(5)}
def correct(Xtr, ytr, te, n):
    sc = StandardScaler().fit(Xtr); return MK[n]().fit(sc.transform(Xtr), ytr).predict(sc.transform(X[te])) == y[te]
nz = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True); cos = lambda A, Bm: np.vstack([nz(X[A][i:i + 1500]) @ nz(X[Bm]).T for i in range(0, len(A), 1500)])
did = lambda d, near: near.mean() * (d[near].mean() - d[~near].mean()); rng = np.random.default_rng(0); out = []
for s in range(3):
    rest, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s); tr, rv = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=s)
    S = cos(te, tr); s1 = S.max(1); Sr = cos(te, rv); order = np.argsort(-Sr, 1)
    base = {n: correct(X[tr], y[tr], te, n) for n in MK}; G = {n: [] for n in MK}; E = {n: [] for n in MK}
    for k in [1, 2, 5, 20, 100]:
        pick = rv[order[:, k - 1]]; c = Sr[np.arange(len(te)), order[:, k - 1]]
        for n in MK: G[n].append(np.c_[c, correct(np.vstack([X[tr], X[pick]]), np.concatenate([y[tr], y[pick]]), te, n).astype(float) - base[n]])
    for f in [.25, .5, 1.0]:  # exact-copy curve on this split
        p = rng.choice(te, int(f * len(te)), replace=False)
        for n in MK: E[n].append(correct(np.vstack([X[tr], X[p]]), np.concatenate([y[tr], y[p]]), te, n).mean() - base[n].mean())
    for n in MK:
        g = np.vstack(G[n]); q = np.quantile(g[:, 0], np.linspace(0, 1, 13)); b = np.clip(np.digitize(g[:, 0], q[1:-1]), 0, 11)
        cx = np.array([g[b == i, 0].mean() for i in range(12)]); gy = np.maximum.accumulate(np.array([g[b == i, 1].mean() for i in range(12)])); gi = lambda v, cx=cx, gy=gy: np.interp(v, cx, gy)
        if s == 0: print(n, "g(c):", dict(zip(cx.round(2), (gy * 100).round(1))))
        for t in [.95, .90, .85, .80]:
            bad = (S >= t).any(0); s2 = S[:, ~bad].max(1); near = s1 >= t
            d = base[n].astype(float) - correct(X[tr[~bad]], y[tr[~bad]], te, n)
            bs = [(lambda i: did(d[i], near[i]))(rng.integers(0, len(d), len(d))) for _ in range(300)]
            out.append(dict(seed=s, t=t, model=n, f_near=near.mean(), pred_old=np.interp(near.mean(), [0, .25, .5, 1], [0] + E[n]), pred_real_red=did(gi(s1) - gi(s2), near),
                            obs=did(d, near), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
D = pd.DataFrame(out); D.to_csv("leakage_audit/sat/real_planted.csv", index=False)
print((D.groupby(["model", "t"])[["f_near", "pred_old", "pred_real_red", "obs", "lo", "hi"]].mean() * [1, 100, 100, 100, 100, 100]).round(2).to_string())
for c in ["pred_old", "pred_real_red"]: print(c, "corr", round(np.corrcoef(D[c], D.obs)[0, 1], 2), "inside CI", int(((D[c] >= D.lo) & (D[c] <= D.hi)).sum()), "/", len(D), "MAE(pts)", round((D[c] - D.obs).abs().mean() * 100, 2), "signed(pts)", round((D[c] - D.obs).mean() * 100, 2))
