"""Step 4 stage 2 (PREREG.md commit b9076c3, C1/C2; Deviations b97f07a). Run after stage-1 predictions were committed (c9fa415) and pushed.
Observed leak = did_antibody.py part A recipe, unchanged (its loss, z, rng and did() are exec'd from that file): train on the Jain2017 file,
test on GDPa1; near = GDPa1 antibodies with an exact VH+VL copy in the Jain file; 20 half-dedups; obs = f_near x [mean dL(near) - mean dL(far)].
Size check: far-item drop / near-item drop per cell; > 1/2, or near drop <= 0, = size-sensitive.
C1: pooled S over the 20 C1C2 cells vs predict-0. C2: MAE(pred) - MAE(null c). 2000 paired bootstrap replicates (draws as in Deviations 1).
Holm needs C3 too: p < .0167 passes and p >= .05 fails whatever C3 gives.
Writes out/fresh_antibody_items.csv, out/fresh_antibody_did.csv. Run from ~/Biostuff with python -I."""
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "did_antibody.py")).read().split("out = []")[0])
P = pd.read_csv(f"{OUT}/fresh_antibody_pred.csv")
cells, items = [], []
for (role, pair, j17, gcol), Pp in P.groupby(["role", "pair", "jain_file", "gdpa_col"], sort=False):
    G = g[["vh", "vl", gcol, "hierarchical_cluster_fold"]].rename(columns={gcol: "y"}).dropna().reset_index(drop=True); y = G.y.values
    J = jain(j17).dropna(); jk = list(zip(J.vh, J.vl)); seen = np.array([k in set(jk) for k in zip(G.vh, G.vl)])
    assert np.isclose(seen.mean(), Pp.f_near.iloc[0]), pair
    near, far = np.where(seen)[0], np.where(~seen)[0]
    halves = [rng.choice(near, len(near) // 2, replace=False) for _ in range(20)]
    for m in MODELS:
        L0, s, c = loss(y, fit_predict(m, J, G)), np.zeros(len(G)), np.zeros(len(G))
        for A in halves:
            drop = set(zip(G.vh.values[A], G.vl.values[A]))
            d = loss(y, fit_predict(m, J[[k not in drop for k in jk]], G)) - L0
            s[A] += d[A]; c[A] += 1; s[far] += d[far]; c[far] += 1
        dl = s / np.maximum(c, 1); est, lo, hi = did(dl, near, far, seen.mean()); Pm = Pp[Pp.model == m]
        cells.append(dict(role=role, pair=pair, model=m, f_near=seen.mean(), n_near=len(near), n_far=len(far), pred=Pm.gain.mean(),
                          null_c=Pm.null_c.iloc[0], null_d=Pm.null_d.iloc[0], obs=est, lo=lo, hi=hi, drop_near=dl[near].mean(), drop_far=dl[far].mean()))
        items += [dict(role=role, pair=pair, model=m, item=i, near=bool(seen[i]), dloss=v) for i, v in enumerate(dl)]
C, I = pd.DataFrame(cells), pd.DataFrame(items)
C["far_over_near"] = C.drop_far / C.drop_near; C["size_sensitive"] = (C.drop_near <= 0) | (C.far_over_near > .5)
C.to_csv(f"{OUT}/fresh_antibody_did.csv", index=False); I.to_csv(f"{OUT}/fresh_antibody_items.csv", index=False)
pd.set_option("display.width", 220); print(C.drop(columns=["drop_near", "drop_far"]).round(3).to_string(index=False))

S = lambda p, o: 1 - np.abs(p - o).sum() / np.abs(o).sum()
K = C[C.role == "C1C2"].reset_index(drop=True); key = list(zip(K.pair, K.model)); ok = ~K.size_sensitive.values
DL = {k: I[(I.role == "C1C2") & (I.pair == k[0]) & (I.model == k[1])] for k in key}
SD = {k: P[(P.role == "C1C2") & (P.pair == k[0]) & (P.model == k[1])].sort_values("seed").gain.values for k in key}
br, b1, b1ok, b2 = np.random.default_rng(1), [], [], []
for _ in range(2000):
    sd = br.integers(0, 3, 3); draw = {}
    for p in K.pair.unique():
        nr = DL[(p, MODELS[0])].near.values; ni, fi = np.where(nr)[0], np.where(~nr)[0]
        draw[p] = br.choice(ni, len(ni)), br.choice(fi, len(fi))
    o = np.array([f * (DL[k].dloss.values[draw[k[0]][0]].mean() - DL[k].dloss.values[draw[k[0]][1]].mean()) for k, f in zip(key, K.f_near)])
    pr = np.array([SD[k][sd].mean() for k in key])
    b1.append(S(pr, o)); b1ok.append(S(pr[ok], o[ok])); b2.append(np.abs(pr - o).mean() - np.abs(K.null_c.values - o).mean())
p_ = lambda b, bad: np.mean(np.array(b) >= 0 if bad == "ge" else np.array(b) <= 0)
print(f"\nC1  S(pred) all 20 cells = {S(K.pred, K.obs):.3f}  p = {p_(b1, 'le'):.4f}")
print(f"C1  S(pred) without {(~ok).sum()} size-sensitive cells = {S(K.pred[ok], K.obs[ok]):.3f}  p = {p_(b1ok, 'le'):.4f}" if ok.any() else "C1  all cells size-sensitive")
print(f"C2  MAE(pred) {np.abs(K.pred - K.obs).mean():.3f} vs MAE(null c) {np.abs(K.null_c - K.obs).mean():.3f}  p = {p_(b2, 'ge'):.4f}")
print("Holm (C3 pending): p < .0167 passes regardless, p >= .05 fails regardless.")
print(f"secondary: S(null c) {S(K.null_c, K.obs):.3f}  S(null d) {S(K.null_d, K.obs):.3f}  S(null b, oracle const) {S(K.obs.mean() + 0 * K.obs, K.obs):.3f}  "
      f"corr(pred,obs) {np.corrcoef(K.pred, K.obs)[0, 1]:.2f}  inside CI {((K.pred >= K.lo) & (K.pred <= K.hi)).sum()}/20 (descriptive)")
for r, d in C[C.role != "C1C2"].groupby("role"): print(f"{r}: S(pred) {S(d.pred, d.obs):.3f}  S(null c) {S(d.null_c, d.obs):.3f}  n={len(d)}")
