"""Antibody DiD. Design and predictions fixed 2026-10-15 before running. Run from ~/Biostuff.
Per-item loss L_i = (z(rank y)_i - z(rank pred)_i)^2 / 2 gives rho = 1 - mean(L) exactly (ties included), so leak is in rho units.
leak = f_near * [mean dL(near) - mean dL(far)], dL = L(deduped train) - L(full train). Caveat: ranks couple items.
A) Cross-lab Jain2017 -> GDPa1 (E2). near = exact VH+VL seen in Jain2017. Dropping all copies leaves ~6 train antibodies,
   so half-dedup: drop the Jain copies of a random half of near antibodies, 20 halvings. Prediction: leakage_susceptibility.csv 'predicted'.
B) GDPa1 random folds. near = nearest-train identity >= t; dedup drops train >= t to any test antibody of the fold.
   Prediction: LS * f_near(t). Expected to over-predict where closeness after dedup (s2) stays high (redundancy)."""
import os
from scipy.stats import rankdata
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.py")).read()
exec(src.split("# ---- experiments")[0].replace("__file__", repr(os.path.abspath("leakage_audit/audit.py"))))
LSC = pd.read_csv(f"{OUT}/leakage_susceptibility.csv").set_index(["assay", "model"])
z = lambda v: (lambda r: (r - r.mean()) / r.std())(rankdata(v))
loss = lambda y, p: (z(y) - z(p)) ** 2 / 2
rng = np.random.default_rng(0)

def did(dl, near, far, f, n=2000):
    est = f * (dl[near].mean() - dl[far].mean())
    bs = [f * (dl[rng.choice(near, len(near))].mean() - dl[rng.choice(far, len(far))].mean()) for _ in range(n)]
    return est, *np.percentile(bs, [2.5, 97.5])

out = []
for aname, (gcol, j17, _) in ASSAYS.items():
    if (aname, "NN-copy") not in LSC.index: continue
    G = g[["vh", "vl", gcol, "random_fold"]].rename(columns={gcol: "y"}).dropna().reset_index(drop=True)
    y = G.y.values
    # A) cross-lab half-dedup
    J = jain(j17).dropna()
    jk = list(zip(J.vh, J.vl)); seen = np.array([k in set(jk) for k in zip(G.vh, G.vl)])
    near, far = np.where(seen)[0], np.where(~seen)[0]
    halves = [rng.choice(near, len(near) // 2, replace=False) for _ in range(20)]
    for m in MODELS:
        L0, s, c = loss(y, fit_predict(m, J, G)), np.zeros(len(G)), np.zeros(len(G))
        for A in halves:
            drop = set(zip(G.vh.values[A], G.vl.values[A]))
            d = loss(y, fit_predict(m, J[[k not in drop for k in jk]], G)) - L0
            s[A] += d[A]; c[A] += 1; s[far] += d[far]; c[far] += 1
        est, lo, hi = did(s / np.maximum(c, 1), near, far, seen.mean())
        r = LSC.loc[(aname, m)]
        out.append(dict(part="A_crosslab", assay=aname, model=m, t=1.0, f_near=seen.mean(), n_train=len(J) - len(near) // 2,
                        s2=np.nan, pred=r.predicted, did=est, lo=lo, hi=hi, old_obs=r.observed, old_ci=r.ci))
    # B) within-GDPa1 random folds, threshold dedup
    folds = sorted(G.random_fold.unique())
    for t in (0.95, 0.90, 0.85, 0.80):
        nn1, nn2, keep = np.zeros(len(G)), np.zeros(len(G)), {}
        for k in folds:
            te, tr = G[G.random_fold == k], G[G.random_fold != k]
            nn1[te.index] = jacc_matrix(te, tr).max(1)
            keep[k] = tr[jacc_matrix(tr, te).max(1) < t]; nn2[te.index] = jacc_matrix(te, keep[k]).max(1)
        nr = nn1 >= t; near, far = np.where(nr)[0], np.where(~nr)[0]
        if len(near) < 10: print(f"{aname} t={t}: only {len(near)} near, skipped"); continue
        for m in MODELS:
            p0, p1 = np.zeros(len(G)), np.zeros(len(G))
            for k in folds:
                te, tr = G[G.random_fold == k], G[G.random_fold != k]
                p0[te.index] = fit_predict(m, tr, te); p1[te.index] = fit_predict(m, keep[k], te)
            est, lo, hi = did(loss(y, p1) - loss(y, p0), near, far, nr.mean())
            out.append(dict(part="B_random", assay=aname, model=m, t=t, f_near=nr.mean(), n_train=int(np.mean([len(v) for v in keep.values()])),
                            s2=nn2[nr].mean(), pred=LSC.loc[(aname, m)].LS * nr.mean(), did=est, lo=lo, hi=hi, old_obs=np.nan, old_ci=""))
O = pd.DataFrame(out); O.to_csv(f"{OUT}/did_antibody.csv", index=False)
pd.set_option("display.width", 220)
print(O.round(3).to_string(index=False))
for p, d in O.groupby("part"):
    inside = ((d.pred >= d.lo) & (d.pred <= d.hi)).sum()
    print(f"{p}: corr(pred,did)={np.corrcoef(d.pred, d.did)[0, 1]:.2f} inside={inside}/{len(d)} MAE={np.abs(d.pred - d.did).mean():.3f} signed={(d.pred - d.did).mean():+.3f}")
