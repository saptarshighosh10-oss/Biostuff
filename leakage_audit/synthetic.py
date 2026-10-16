"""Synthetic replicate: inside GDPa1 cluster folds, plant a fraction f of each test fold's antibodies (exact copies) into train,
with true labels, labels noised to ~0.9 Pearson agreement (SD-sized; too noisy for skewed assays), or rank-matched labels (Gaussian copula,
Spearman = measured Jain2017-vs-GDPa1 agreement, same value distribution). Measures score inflation vs leak fraction, then writes the LS table
(LS = OLS slope of inflation over f <= 0.5, rank mode). Run from ~/Biostuff."""
import os
from scipy.stats import rankdata, norm
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.py")).read()
exec(src.split("# ---- experiments")[0].replace("__file__", repr(os.path.abspath("leakage_audit/audit.py"))))
FR, SEEDS = [0, .1, .25, .5, 1.0], 3
out, AGREE, FN = [], {}, {}
for aname in ["HIC", "AC-SINS", "SMAC"]:
    G = g[["vh", "vl", ASSAYS[aname][0], "hierarchical_cluster_fold"]].rename(columns={ASSAYS[aname][0]: "y"}).dropna().reset_index(drop=True)
    sd = G.y.std() * np.sqrt(1 / 0.9**2 - 1)
    J = jain(ASSAYS[aname][1]).dropna(); jy = dict(zip(zip(J.vh, J.vl), J.y))
    sh = [(jy[k], v) for k, v in zip(zip(G.vh, G.vl), G.y) if k in jy]; AGREE[aname], FN[aname] = spearmanr(*zip(*sh))[0], len(sh) / len(G)
    q, rr = norm.ppf(rankdata(G.y) / (len(G) + 1)), 2 * np.sin(np.pi * AGREE[aname] / 6)
    for m in MODELS:
        for mode in ["exact", "noisy", "rank"]:
            for f in FR:
                rs = []
                for seed in range(SEEDS if f not in (0,) else 1):
                    r = np.random.default_rng(seed); pred = np.zeros(len(G))
                    for k in sorted(G.hierarchical_cluster_fold.unique()):
                        te, tr = G[G.hierarchical_cluster_fold == k], G[G.hierarchical_cluster_fold != k]
                        L = te.sample(frac=f, random_state=seed).copy() if f else te.iloc[:0].copy()
                        if mode == "noisy": L["y"] = L.y + r.normal(0, sd, len(L))
                        if mode == "rank": L["y"] = np.quantile(G.y, norm.cdf(rr * q[L.index.values] + np.sqrt(1 - rr**2) * r.normal(0, 1, len(L))))
                        pred[te.index] = fit_predict(m, pd.concat([tr, L]), te)
                    rs.append(spearmanr(G.y, pred)[0])
                out.append(dict(assay=aname, model=m, labels=mode, leak_frac=f, rho=np.mean(rs), sd=np.std(rs)))
O = pd.DataFrame(out); O.to_csv(f"{OUT}/synthetic.csv", index=False)
pd.set_option("display.width", 200)
print(O.pivot_table(index=["assay", "model", "labels"], columns="leak_frac", values="rho").round(2).to_string())

# LS table: slope over f <= 0.5; predicted = LS * f_near; checked against naive-honest (paired.csv) and the DiD (did_antibody.csv)
P = pd.read_csv(f"{OUT}/paired.csv").query("comparison == 'E2 naive-honest'").set_index(["assay", "model"])
D = pd.read_csv(f"{OUT}/did_antibody.csv").query("part == 'A_crosslab'").set_index(["assay", "model"])
slope = lambda d: np.polyfit(d.leak_frac, d.rho - d.rho[d.leak_frac == 0].iloc[0], 1)[0]
T = []
for (a, m, lab), d in O[O.leak_frac <= 0.5].groupby(["assay", "model", "labels"]):
    if lab != "rank": continue
    ls, p, dd = slope(d), P.loc[(a, m)], D.loc[(a, m)]; pr = ls * FN[a]
    T.append(dict(assay=a, model=m, agree=round(AGREE[a], 2), LS=round(ls, 2), LS_sd_noise=round(slope(O[(O.assay == a) & (O.model == m) & (O.labels == "noisy") & (O.leak_frac <= 0.5)]), 2),
                  predicted=round(pr, 2), observed=round(p["diff"], 2), ci=f"{p.lo:.2f}-{p.hi:.2f}", pred_in_ci=p.lo <= pr <= p.hi,
                  did=round(dd.did, 2), did_ci=f"{dd.lo:.2f}-{dd.hi:.2f}", pred_in_did_ci=dd.lo <= pr <= dd.hi))
T = pd.DataFrame(T); T.to_csv(f"{OUT}/leakage_susceptibility.csv", index=False); print(T.to_string(index=False))
print(T.groupby("model")[["LS", "LS_sd_noise"]].mean().round(2).sort_values("LS", ascending=False).to_string())
for c, k in [("observed", "pred_in_ci"), ("did", "pred_in_did_ci")]:
    print(f"vs {c}: corr={np.corrcoef(T.predicted, T[c])[0, 1]:.2f} inside={T[k].sum()}/{len(T)} MAE={np.abs(T.predicted - T[c]).mean():.3f} signed={(T.predicted - T[c]).mean():+.3f}")
