"""Synthetic replicate: inside GDPa1 cluster folds, plant a fraction f of each test fold's antibodies (exact copies) into train,
with true labels or labels noised to ~0.9 cross-lab agreement. Measures score inflation vs leak fraction. Run from ~/Biostuff."""
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.py")).read()
exec(src.split("# ---- experiments")[0].replace("__file__", repr(os.path.abspath("leakage_audit/audit.py"))))
FR, SEEDS = [0, .1, .25, .5, 1.0], 3
out = []
for aname in ["HIC", "AC-SINS", "SMAC"]:
    G = g[["vh", "vl", ASSAYS[aname][0], "hierarchical_cluster_fold"]].rename(columns={ASSAYS[aname][0]: "y"}).dropna().reset_index(drop=True)
    sd = G.y.std() * np.sqrt(1 / 0.9**2 - 1)
    for m in MODELS:
        for mode in ["exact", "noisy"]:
            for f in FR:
                rs = []
                for seed in range(SEEDS if f not in (0,) else 1):
                    r = np.random.default_rng(seed); pred = np.zeros(len(G))
                    for k in sorted(G.hierarchical_cluster_fold.unique()):
                        te, tr = G[G.hierarchical_cluster_fold == k], G[G.hierarchical_cluster_fold != k]
                        L = te.sample(frac=f, random_state=seed).copy() if f else te.iloc[:0].copy()
                        if mode == "noisy": L["y"] = L.y + r.normal(0, sd, len(L))
                        pred[te.index] = fit_predict(m, pd.concat([tr, L]), te)
                    rs.append(spearmanr(G.y, pred)[0])
                out.append(dict(assay=aname, model=m, labels=mode, leak_frac=f, rho=np.mean(rs), sd=np.std(rs)))
O = pd.DataFrame(out); O.to_csv(f"{OUT}/synthetic.csv", index=False)
pd.set_option("display.width", 200)
print(O.pivot_table(index=["assay", "model", "labels"], columns="leak_frac", values="rho").round(2).to_string())
