"""EuroSAT learning curve: accuracy vs training-set size, frozen ResNet-50 embeddings. 5 seeds, fixed 20% stratified test split per seed."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
z = np.load("leakage_audit/sat/eurosat.npz"); X, y = z["emb"], z["y"]
N = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 21600]; rows = []
for s in range(5):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
    rng = np.random.default_rng(s)
    for n in N:
        n = min(n, len(tr)); idx = rng.permutation(tr)[:n]
        sc = StandardScaler().fit(X[idx]); A, B = sc.transform(X[idx]), sc.transform(X[te])
        for name, m in [("ridge", RidgeClassifier(alpha=10)), ("knn1", KNeighborsClassifier(1)), ("knn5", KNeighborsClassifier(min(5, n)))]:
            rows.append(dict(seed=s, n=n, model=name, acc=(m.fit(A, y[idx]).predict(B) == y[te]).mean()))
D = pd.DataFrame(rows); D.to_csv("leakage_audit/sat/learning_curve.csv", index=False)
T = D.groupby(["model", "n"]).acc.agg(["mean", "std"]).round(4).unstack(0); print(T.to_string())
