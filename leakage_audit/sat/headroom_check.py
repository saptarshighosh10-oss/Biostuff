"""Diagnostic: are near-neighbour test tiles easier than average once their neighbours are removed? err on near tiles (neighbour-free train) vs global err (full train)."""
import numpy as np
exec(open("leakage_audit/sat/headroom.py").read().split("P = pd.read_csv")[0])
nz = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True); rows = []
for s in range(3):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
    S = np.vstack([nz(X[te][i:i + 1500]) @ nz(X[tr]).T for i in range(0, len(te), 1500)])
    for t in [.85, .80]:
        near = S.max(1) >= t; keep = ~(S >= t).any(0)
        for n in MK:
            fit = lambda idx: (lambda sc: MK[n]().fit(sc.transform(X[idx]), y[idx]).predict(sc.transform(X[te])))(StandardScaler().fit(X[idx])) != y[te]
            rows.append((n, t, near.mean(), fit(tr).mean(), fit(tr[keep])[near].mean(), fit(tr)[near].mean()))
import pandas as pd; print(pd.DataFrame(rows, columns=["model", "t", "frac_near", "err_global", "err_near_no_nbr", "err_near_with_nbr"]).groupby(["model", "t"]).mean().round(3).to_string())
