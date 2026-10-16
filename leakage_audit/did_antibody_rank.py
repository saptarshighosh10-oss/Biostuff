"""Why is the cross-lab SMAC leak under-predicted? Hypothesis (fixed 2026-10-15 before running): planted labels used Gaussian noise sized
by the SD (Pearson agreement 0.9). SMAC is skewed (5.7), so that noise scrambles the packed bulk: planted Spearman agreement 0.37 vs real
cross-lab 0.91. Fix: rank-matched noise (Gaussian copula) - normal scores q, q' = r*q + sqrt(1-r^2)*e with r = 2 sin(pi*rs/6) so
Spearman(y, label) = measured cross-lab rs, then map q' back through y's own quantiles (same marginal, same ties).
Prediction = planted inflation at f = f_near in GDPa1 cluster folds, 3 seeds, compared with the DiD in did_antibody.csv. Run from ~/Biostuff."""
import os
from scipy.stats import rankdata, norm
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.py")).read()
exec(src.split("# ---- experiments")[0].replace("__file__", repr(os.path.abspath("leakage_audit/audit.py"))))
D0 = pd.read_csv(f"{OUT}/did_antibody.csv"); D0 = D0[D0.part == "A_crosslab"].set_index(["assay", "model"])
out = []
for aname, (gcol, j17, _) in ASSAYS.items():
    if not j17 or (aname, "NN-copy") not in D0.index: continue
    G = g[["vh", "vl", gcol, "hierarchical_cluster_fold"]].rename(columns={gcol: "y"}).dropna().reset_index(drop=True); y = G.y.values
    J = jain(j17).dropna(); jy = dict(zip(zip(J.vh, J.vl), J.y))
    sh = [(jy[k], v) for k, v in zip(zip(G.vh, G.vl), y) if k in jy]; rs = spearmanr(*zip(*sh))[0]; fn = len(sh) / len(G)
    q, r_ = norm.ppf(rankdata(y) / (len(y) + 1)), 2 * np.sin(np.pi * rs / 6)
    lab = lambda idx, r: np.quantile(y, norm.cdf(r_ * q[idx] + np.sqrt(1 - r_ ** 2) * r.normal(0, 1, len(idx))))
    got = np.mean([spearmanr(y, lab(np.arange(len(y)), np.random.default_rng(s)))[0] for s in range(20)])
    print(f"{aname}: real cross-lab spearman {rs:.2f}, planted rank-matched spearman {got:.2f}")
    cf = sorted(G.hierarchical_cluster_fold.unique())
    for m in MODELS:
        base = np.zeros(len(G))
        for k in cf:
            te, tr = G[G.hierarchical_cluster_fold == k], G[G.hierarchical_cluster_fold != k]; base[te.index] = fit_predict(m, tr, te)
        gains = []
        for s in range(3):
            r, p = np.random.default_rng(s), np.zeros(len(G))
            for k in cf:
                te, tr = G[G.hierarchical_cluster_fold == k], G[G.hierarchical_cluster_fold != k]
                L = te.sample(frac=fn, random_state=s).copy(); L["y"] = lab(L.index.values, r)
                p[te.index] = fit_predict(m, pd.concat([tr, L]), te)
            gains.append(spearmanr(y, p)[0] - spearmanr(y, base)[0])
        d = D0.loc[(aname, m)]
        out.append(dict(assay=aname, model=m, agree=rs, pred_old=d.pred, pred_rank=np.mean(gains), did=d.did, lo=d.lo, hi=d.hi))
O = pd.DataFrame(out); O.to_csv(f"{OUT}/did_antibody_rank.csv", index=False)
pd.set_option("display.width", 200); print(O.round(3).to_string(index=False))
for c in ["pred_old", "pred_rank"]:
    print(f"{c}: corr={np.corrcoef(O[c], O.did)[0, 1]:.2f} inside={((O[c] >= O.lo) & (O[c] <= O.hi)).sum()}/{len(O)} "
          f"MAE={np.abs(O[c] - O.did).mean():.3f} signed={(O[c] - O.did).mean():+.3f}")
