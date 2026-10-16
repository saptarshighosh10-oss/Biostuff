"""Realistic-neighbour predictor for the antibody DiD (did_antibody.py). Fixed 2026-10-15 before running. Run from ~/Biostuff.
Planted neighbour = exact VH/VL sequence copy (ESM/AbLang embeddings are cache-only, so sequences cannot be mutated) whose label
agrees with the test label as much as real pairs do. Label in z units: rho*z + sqrt(1 - rho^2)*noise (same rms gap and marginal as real pairs).
A) Cross-lab: rho = Spearman agreement of Jain2017 vs GDPa1 labels on shared antibodies (replaces the assumed 0.9).
   Prediction = planted inflation at f = f_near in GDPa1 cluster folds, 3 seeds.
B) Within GDPa1: rho_b = 1 - mean(dz^2)/2 over real GDPa1 pairs in identity bin b, clipped to [0, 1]. g_M(b) = rho gain when every
   test antibody gets such a neighbour (actual random-fold split, 3 seeds). Prediction = DiD form of g(s1) - g(s2), s1/s2 = nearest-train
   identity before/after dedup (salinity recipe). Secondary, for comparison with salinity only: label = y + N(0, rms dy_b).
Expected: closer for NN-copy than for feature models, because planted features are exact copies."""
import os
from scipy.stats import rankdata
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.py")).read()
exec(src.split("# ---- experiments")[0].replace("__file__", repr(os.path.abspath("leakage_audit/audit.py"))))
D0 = pd.read_csv(f"{OUT}/did_antibody.csv"); LSC = pd.read_csv(f"{OUT}/leakage_susceptibility.csv").set_index(["assay", "model"])
z = lambda v: (lambda r: (r - r.mean()) / r.std())(rankdata(v))
loss = lambda y, p: (z(y) - z(p)) ** 2 / 2
did = lambda d, nr: nr.mean() * (d[nr].mean() - d[~nr].mean())
BINS, SEEDS = [0, .6, .7, .75, .8, .85, .9, 1.01], range(3)

def plant(L, mu, sd, rho, r, rms=None):  # copy of L with realistic labels
    L = L.copy(); e = r.normal(0, 1, len(L))
    L["y"] = L.y + rms * e if rms is not None else mu + sd * (rho * (L.y - mu) / sd + np.sqrt(1 - rho ** 2) * e)
    return L

out = []
for aname, (gcol, j17, _) in ASSAYS.items():
    if (aname, "NN-copy") not in LSC.index: continue
    G = g[["vh", "vl", gcol, "random_fold", "hierarchical_cluster_fold"]].rename(columns={gcol: "y"}).dropna().reset_index(drop=True)
    y, mu, sd = G.y.values, G.y.mean(), G.y.std()
    # A) cross-lab agreement-calibrated planted curve
    J = jain(j17).dropna(); jy = dict(zip(zip(J.vh, J.vl), J.y))
    sh = [(jy[k], v) for k, v in zip(zip(G.vh, G.vl), y) if k in jy]; rx = spearmanr(*zip(*sh))[0]; fn = len(sh) / len(G)
    cf = sorted(G.hierarchical_cluster_fold.unique())
    for m in MODELS:
        base, gains = np.zeros(len(G)), []
        for k in cf:
            te, tr = G[G.hierarchical_cluster_fold == k], G[G.hierarchical_cluster_fold != k]; base[te.index] = fit_predict(m, tr, te)
        for s in SEEDS:
            r, p = np.random.default_rng(s), np.zeros(len(G))
            for k in cf:
                te, tr = G[G.hierarchical_cluster_fold == k], G[G.hierarchical_cluster_fold != k]
                p[te.index] = fit_predict(m, pd.concat([tr, plant(te.sample(frac=fn, random_state=s), mu, sd, rx, r)]), te)
            gains.append(spearmanr(y, p)[0] - spearmanr(y, base)[0])
        d = D0[(D0.part == "A_crosslab") & (D0.assay == aname) & (D0.model == m)].iloc[0]
        out.append(dict(part="A_crosslab", assay=aname, model=m, t=1.0, agree=rx, pred_old=d.pred, pred_real=np.mean(gains), pred_rms=np.nan, did=d.did, lo=d.lo, hi=d.hi))
    # B) within-GDPa1 identity-calibrated g(s), redundancy-aware
    S = jacc_matrix(G, G); iu = np.triu_indices(len(G), 1); sp = S[iu]; zy = (y - mu) / sd
    dz2, dy2 = (zy[iu[0]] - zy[iu[1]]) ** 2, (y[iu[0]] - y[iu[1]]) ** 2
    b_of = np.digitize(sp, BINS) - 1; nb = len(BINS) - 1
    xb = np.array([sp[b_of == b].mean() for b in range(nb)])
    rho_b = np.clip([1 - dz2[b_of == b].mean() / 2 for b in range(nb)], 0, 1); rms_b = [np.sqrt(dy2[b_of == b].mean()) for b in range(nb)]
    print(aname, "cross-lab agreement", round(rx, 2), "| identity bins", xb.round(2), "rho", np.round(rho_b, 2), "pairs", np.bincount(b_of, minlength=nb))
    rf = sorted(G.random_fold.unique())
    for m in MODELS:
        base = np.zeros(len(G))
        for k in rf:
            te, tr = G[G.random_fold == k], G[G.random_fold != k]; base[te.index] = fit_predict(m, tr, te)
        L0 = loss(y, base); gb, gr = np.zeros(nb), np.zeros(nb)
        for b in range(nb):
            for s in SEEDS:
                for acc, kw in ((gb, {}), (gr, {"rms": rms_b[b]})):
                    r, p = np.random.default_rng(s), np.zeros(len(G))
                    for k in rf:
                        te, tr = G[G.random_fold == k], G[G.random_fold != k]
                        p[te.index] = fit_predict(m, pd.concat([tr, plant(te, mu, sd, rho_b[b], r, **kw)]), te)
                    acc[b] += (L0 - loss(y, p)).mean() / len(SEEDS)
        for t in (0.90, 0.85, 0.80):
            s1, s2 = np.zeros(len(G)), np.zeros(len(G))
            for k in rf:
                te, tr = G[G.random_fold == k], G[G.random_fold != k]
                s1[te.index] = jacc_matrix(te, tr).max(1); kp = tr[jacc_matrix(tr, te).max(1) < t]; s2[te.index] = jacc_matrix(te, kp).max(1)
            nr, gi = s1 >= t, lambda v, gg: np.interp(v, xb, gg)
            d = D0[(D0.part == "B_random") & (D0.assay == aname) & (D0.model == m) & np.isclose(D0.t, t)].iloc[0]
            out.append(dict(part="B_random", assay=aname, model=m, t=t, agree=np.nan, pred_old=d.pred, pred_real=did(gi(s1, gb) - gi(s2, gb), nr),
                            pred_rms=did(gi(s1, gr) - gi(s2, gr), nr), did=d.did, lo=d.lo, hi=d.hi))
O = pd.DataFrame(out); O.to_csv(f"{OUT}/did_antibody_real.csv", index=False)
pd.set_option("display.width", 220); print(O.round(3).to_string(index=False))
for p, d in O.groupby("part"):
    for c in ["pred_old", "pred_real", "pred_rms"]:
        if d[c].notna().all():
            print(f"{p} {c}: corr={np.corrcoef(d[c], d.did)[0, 1]:.2f} inside={((d[c] >= d.lo) & (d[c] <= d.hi)).sum()}/{len(d)} "
                  f"MAE={np.abs(d[c] - d.did).mean():.3f} signed={(d[c] - d.did).mean():+.3f}")
