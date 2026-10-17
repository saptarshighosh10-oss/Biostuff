"""Step 7, C3 (PREREG.md commit b9076c3 §4; Deviations 3-5 in 0e4961f). Held-out soil properties, LUCAS 2018, seeds 6-8, radii 5/10/15 km.
Predictor of record = did_multi.py 'nearest' arm, unchanged except the copula agreement per target and distance bin comes from train-train
5-NN pairs (arm B, primary). Arm A (all-point pairs, the old recipe) and a rho = 0 arm (same planting, agreement 0) are reported beside it.
Targets pH_H2O (raw); OC, CaCO3, N, K (log1p); censored/missing rows dropped per target; features = soil set minus target (EC not added).
Models knn_geo (scored) and ridge_soil (negative control). C3 = pooled knn_geo S over 45 cells vs predict-0; p = share of 2000 paired
bootstrap replicates (per seed one test-point draw shared by all properties) with S <= 0. Per-point vectors -> salinity/_local/ (gitignored,
LUCAS licence); the committed CSV holds aggregates only. Usage (from ~/Biostuff): python -I leakage_audit/salinity/confirm_c3.py"""
import os; H = "/Users/saptarshighosh/Biostuff/leakage_audit/salinity/"
exec(open(H + "calibrate_dist.py").read().split("\nout = []")[0] + open(H + "did_rank.py").read().split("'''")[1])  # data, models, copula
did = lambda dd, near: near.mean() * (dd[near].mean() - dd[~near].mean())
S = lambda p, o: 1 - np.abs(p - o).sum() / np.abs(o).sum()
COLS = ["pH_H2O", "OC", "CaCO3", "P", "N", "K", "Elev"]; TARGETS = {"pH_H2O": False, "OC": True, "CaCO3": True, "N": True, "K": True}
CENS = {"pH_H2O": 0, "OC": 34, "CaCO3": 9, "N": 14, "K": 39}; MODELS3 = ["knn_geo", "ridge_soil"]; soil0 = soil.copy()
assert len(y) == 18975 and all((d[t].notna() & np.isnan(soil0[:, COLS.index(t)])).sum() == CENS[t] for t in TARGETS)

def calib(pts):  # copula agreement per distance bin from 5-NN pairs among pts (rho() skips pairs with a missing value)
    dk_, nb_ = BallTree(np.c_[lat[pts], lon[pts]], metric="haversine").query(np.c_[lat[pts], lon[pts]], k=6)
    bn = np.digitize((dk_[:, 1:] * 6371).ravel(), B) - 1; Ii, Jj = np.repeat(pts, 5), pts[nb_[:, 1:].ravel()]
    return ({b: rho(y[Ii][bn == b], y[Jj][bn == b]) for b in range(len(B) - 1)},
            {b: [rho(soil[Ii][bn == b, k], soil[Jj][bn == b, k]) for k in range(soil.shape[1])] for b in range(len(B) - 1)})

def plant(j, it, dj, ry_, rs_):  # did_multi.plant with the agreement tables passed in
    b = np.clip(np.digitize(dj, B) - 1, 0, len(B) - 2); ya, sa = np.empty(len(j)), np.full((len(j), soil.shape[1]), np.nan)
    for bb in np.unique(b):
        m = b == bb; ya[m] = cop(qy[it[m]], ry_[bb], y)
        for k in range(soil.shape[1]): sa[m, k] = np.where(np.isnan(soil[it[m], k]), np.nan, cop(qs[it[m], k], rs_[bb][k], soil[:, k]))
    return {"ridge_soil": sa, "knn_geo": xyz[j]}, ya

rows, VEC, TES = [], {}, {}
for s in (6, 7, 8):
    TR, TE = train_test_split(idx, test_size=.2, random_state=s); TES[s] = len(TE); P = lambda a: np.c_[lat[a], lon[a]]
    for tname, logt in TARGETS.items():
        c = COLS.index(tname); y = np.log1p(soil0[:, c]) if logt else soil0[:, c].copy(); soil = np.delete(soil0, c, 1)
        F["ridge_soil"] = soil; qy, qs = nq(y), np.column_stack([nq(v_) for v_ in soil.T]); has = ~np.isnan(y)
        z0 = {b: 0.0 for b in range(len(B) - 1)}
        arms = {"B": calib(TR), "A": calib(idx), "rho0": (z0, {b: [0.0] * soil.shape[1] for b in z0})}
        tr, te, pos = TR[has[TR]], TE[has[TE]], np.where(has[TE])[0]
        Ttr, Tte = BallTree(P(tr), metric="haversine"), BallTree(P(te), metric="haversine")
        d1, n1 = (a[:, 0] for a in Ttr.query(P(te), k=1)); d1 *= 6371
        ef = {n: (y[te] - pred(n, tr, te)) ** 2 for n in MODELS3}; v = y[te].var()
        for r in [5, 10, 15]:
            bad = np.unique(np.concatenate(Ttr.query_radius(P(te), r / 6371))); keep = np.setdiff1d(np.arange(len(tr)), bad); near = d1 <= r
            dj, it = (a[:, 0] for a in Tte.query(P(tr[bad]), k=1)); dj *= 6371
            one = np.isin(bad, np.unique(n1[near]))  # nearest removed point of each near test point
            V = {k: plant(tr[bad][one], te[it[one]], dj[one], *a) for k, a in arms.items()}
            for n in MODELS3:
                ek = (y[te] - pred(n, tr[keep], te)) ** 2; dobs = (ek - ef[n]) / v
                bs = [(lambda i: did(dobs[i], near[i]))(rng.integers(0, len(dobs), len(dobs))) for _ in range(300)]
                row = dict(seed=s, target=tname, r=r, model=n, n_test=len(te), f_near=near.mean(), n_removed=len(bad), n_nearest=int(one.sum()),
                           obs=did(dobs, near), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
                for k, (Xa, ya) in V.items():
                    dp = (ek - (y[te] - pred(n, tr[keep], te, Xa=Xa[n], ya=ya)) ** 2) / v; row["pred_" + k] = did(dp, near)
                    if k == "B": VEC[(s, tname, r, n)] = (pos, near, dobs, dp)
                rows.append(row)
        print(f"seed {s} {tname}: n_test {len(te)}, agreement bin0 label A {arms['A'][0][0]:.2f} B {arms['B'][0][0]:.2f}", flush=True)
D = pd.DataFrame(rows); D.to_csv(H + "confirm_c3.csv", index=False); os.makedirs(H + "_local", exist_ok=True)
np.savez_compressed(H + "_local/c3_vectors.npz", **{f"{s}|{t}|{r}|{n}|{w}": x for (s, t, r, n), V4 in VEC.items() for w, x in zip("pndq", V4)})

K = D[D.model == "knn_geo"].reset_index(drop=True); key = list(zip(K.seed, K.target, K.r))
br, b3 = np.random.default_rng(2), []
for _ in range(2000):
    dr = {s: br.integers(0, TES[s], TES[s]) for s in TES}; o, p = [], []
    for s, t, r in key:
        pos, near, dobs, dp = VEC[(s, t, r, "knn_geo")]; at = np.full(TES[s], -1); at[pos] = np.arange(len(pos))
        i = at[dr[s]]; i = i[i >= 0]; o.append(did(dobs[i], near[i])); p.append(did(dp[i], near[i]))
    b3.append(S(np.array(p), np.array(o)))
p3 = np.mean(np.array(b3) <= 0)
pd.set_option("display.width", 220)
print(D.groupby(["target", "model", "r"])[["f_near", "n_nearest", "obs", "lo", "hi", "pred_B", "pred_A", "pred_rho0"]].mean().round(3).to_string())
print(f"\nC3  knn_geo S(arm B) over {len(K)} cells = {S(K.pred_B, K.obs):.3f}  p = {p3:.4f}")
print(f"    beside: S(arm A) {S(K.pred_A, K.obs):.3f}  S(rho0) {S(K.pred_rho0, K.obs):.3f}  S(null b, oracle const) {S(K.obs.mean() + 0 * K.obs, K.obs):.3f}  "
      f"inside CI {((K.pred_B >= K.lo) & (K.pred_B <= K.hi)).sum()}/{len(K)} (descriptive)")
for t, x in K.groupby("target", sort=False): print(f"    {t:7s} S(B) {S(x.pred_B, x.obs):+.3f}  mean obs {x.obs.mean():.3f}  mean pred_B {x.pred_B.mean():.3f}")
R = D[D.model == "ridge_soil"]; print(f"    ridge_soil control: mean obs {R.obs.mean():+.4f}, mean pred_B {R.pred_B.mean():+.4f}")
PV = {"C1": .4215, "C2": 0.0, "C3": p3}  # C1/C2 from out/confirm_antibody_did.log (11186c2)
for j, (k, pv) in enumerate(sorted(PV.items(), key=lambda kv: kv[1])):
    cut = .05 / (3 - j); ok = pv < cut and all(PV[k2] < .05 / (3 - j2) for j2, (k2, _) in enumerate(sorted(PV.items(), key=lambda kv: kv[1])) if j2 < j)
    print(f"Holm {k}: p {pv:.4f} vs {cut:.4f} -> {'PASS' if ok else 'fail'}")
