"""Two fixes, no removal control at all.
Observed (difference-in-differences): dedup changes accuracy on near tiles (leak lost + smaller train) and far tiles (smaller train only).
  leak = f_near * [drop_near - drop_far], drop = acc(full train) - acc(dedup train), bootstrap over test tiles.
Predicted (redundancy-aware): a tile only loses what its closeness drops by: s1 = max cos to full train, s2 = max cos to dedup train.
  a_M(s) = P(correct | closeness s), fitted on the OTHER seeds' full-train runs. pred_red = same DiD formula applied to a(s1)-a(s2)."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
z = np.load("leakage_audit/sat/eurosat.npz"); X, y = z["emb"], z["y"]
MK = {"ridge": lambda: RidgeClassifier(alpha=10), "knn1": lambda: KNeighborsClassifier(1), "knn5": lambda: KNeighborsClassifier(5)}
def correct(tr, te, n):
    sc = StandardScaler().fit(X[tr]); return MK[n]().fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te])) == y[te]
N = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True); Pr = pd.read_csv("leakage_audit/sat/predict.csv"); T = [.95, .90, .85, .80]
runs = {}
for s in range(3):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
    S = np.vstack([N(X[te][i:i + 1500]) @ N(X[tr]).T for i in range(0, len(te), 1500)]); s1 = S.max(1)
    full = {n: correct(tr, te, n) for n in MK}; runs[s] = dict(s1=s1, full=full, t={})
    for t in T:
        bad = (S >= t).any(0); runs[s]["t"][t] = dict(s2=S[:, ~bad].max(1), ck={n: correct(tr[~bad], te, n) for n in MK})
def a_curve(n, s_out):  # P(correct | closeness) from other seeds, 20 quantile bins
    s1 = np.concatenate([runs[k]["s1"] for k in runs if k != s_out]); c = np.concatenate([runs[k]["full"][n] for k in runs if k != s_out])
    q = np.quantile(s1, np.linspace(0, 1, 21)); b = np.clip(np.digitize(s1, q[1:-1]), 0, 19)
    return lambda v: np.interp(v, [s1[b == i].mean() for i in range(20)], [c[b == i].mean() for i in range(20)])
did = lambda d, near: near.mean() * (d[near].mean() - d[~near].mean())
rng = np.random.default_rng(0); rows = []
for s, R in runs.items():
    for t, Q in R["t"].items():
        near = R["s1"] >= t
        for n in MK:
            d = R["full"][n].astype(float) - Q["ck"][n]; a = a_curve(n, s); dp = a(R["s1"]) - a(Q["s2"])
            bs = [(lambda i: did(d[i], near[i]))(rng.integers(0, len(d), len(d))) for _ in range(500)]
            k = (Pr.seed == s) & (Pr.t == t) & (Pr.model == n)
            rows.append(dict(seed=s, t=t, model=n, pred_old=Pr[k].pred.iloc[0], pred_red=did(dp, near), obs=did(d, near), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             s2_near_mean=Q["s2"][near].mean()))
D = pd.DataFrame(rows); D.to_csv("leakage_audit/sat/did.csv", index=False)
G = D.groupby(["model", "t"])[["pred_old", "pred_red", "obs", "lo", "hi", "s2_near_mean"]].mean(); G[["pred_old", "pred_red", "obs", "lo", "hi"]] *= 100; print(G.round(2).to_string())
for c in ["pred_old", "pred_red"]: print(c, "corr", round(np.corrcoef(D[c], D.obs)[0, 1], 2), "inside CI", int(((D[c] >= D.lo) & (D[c] <= D.hi)).sum()), "/", len(D), "MAE(pts)", round((D[c] - D.obs).abs().mean() * 100, 2), "signed(pts)", round((D[c] - D.obs).mean() * 100, 2))
