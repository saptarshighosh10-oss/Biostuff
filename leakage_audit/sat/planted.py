"""EuroSAT planted-leak curve + natural near-neighbour analysis (frozen ResNet-50 embeddings)."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
z = np.load("leakage_audit/sat/eurosat.npz"); X, y = z["emb"], z["y"]
MK = {"ridge": lambda: RidgeClassifier(alpha=10), "knn1": lambda: KNeighborsClassifier(1), "knn5": lambda: KNeighborsClassifier(5)}
def fit_acc(Xtr, ytr, Xte, yte, name):
    sc = StandardScaler().fit(Xtr); m = MK[name]().fit(sc.transform(Xtr), ytr); return (m.predict(sc.transform(Xte)) == yte)
rows, nat = [], []
for s in range(3):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s); rng = np.random.default_rng(s)
    for f in [0, .1, .25, .5, 1.0]:
        pl = rng.choice(te, int(f * len(te)), replace=False); Xtr = np.vstack([X[tr], X[pl]]); ytr = np.concatenate([y[tr], y[pl]])
        for n in MK: rows.append(dict(seed=s, f=f, model=n, acc=fit_acc(Xtr, ytr, X[te], y[te], n).mean()))
    # natural: cosine similarity of each test tile to its nearest train tile, accuracy by similarity band
    A = X[tr] / np.linalg.norm(X[tr], axis=1, keepdims=True); B = X[te] / np.linalg.norm(X[te], axis=1, keepdims=True)
    sim = np.concatenate([(B[i:i + 2000] @ A.T).max(1) for i in range(0, len(te), 2000)])
    for n in MK:
        ok = fit_acc(X[tr], y[tr], X[te], y[te], n)
        for q, (lo, hi) in enumerate(zip(np.quantile(sim, [0, .25, .5, .75]), np.quantile(sim, [.25, .5, .75, 1]))):
            msk = (sim >= lo) & (sim <= hi); nat.append(dict(seed=s, model=n, quartile=q + 1, sim_lo=lo, acc=ok[msk].mean()))
D = pd.DataFrame(rows); D.to_csv("leakage_audit/sat/planted.csv", index=False); N = pd.DataFrame(nat); N.to_csv("leakage_audit/sat/natural_sim.csv", index=False)
T = D.groupby(["model", "f"]).acc.mean().unstack(0); T = T - T.loc[0]; print("inflation vs f (acc points over f=0):\n", (T * 100).round(2).to_string())
print("\nnatural: accuracy by nearest-train-similarity quartile (Q1 = most novel)\n", N.groupby(["model", "quartile"])[["sim_lo", "acc"]].mean().round(3).unstack(0).to_string())
