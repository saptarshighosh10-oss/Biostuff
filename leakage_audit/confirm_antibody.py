"""Step 4 stage 1 (PREREG.md commit b9076c3, C1/C2): predictions for the fresh Jain2017 -> GDPa1 pairs, made before any stage-2 DiD.
Predictor of record = the did_antibody_rank.py recipe, unchanged: f_near = share of GDPa1 antibodies (target measured) whose exact VH+VL is
in the Jain file (value measured); plant f_near of each hierarchical-cluster test fold into training with Gaussian-copula labels at the measured
Spearman agreement, mapped through the target's own values; gain = Spearman(with planting) - Spearman(without), planting seeds 0, 1, 2.
Nulls: c = mean LS(model) over HIC, AC-SINS, SMAC (out/leakage_susceptibility.csv, rank-matched) x f_near; d = c x agree / 0.923.
Writes out/fresh_antibody_pred.csv (one row per planting seed). Stage 2 (DiD) is not run here. Run from ~/Biostuff with python -I."""
import os
from scipy.stats import rankdata, norm
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.py")).read()
exec(src.split("# ---- experiments")[0].replace("__file__", repr(os.path.abspath("leakage_audit/audit.py"))))
PAIRS = [  # (role, Jain2017 file, GDPa1 column, planning agreement); C1/C2 rows from PREREG §4, then its sensitivity and semi-fresh rows
    ("C1C2", "polyreactivity/jain2017biophysical_PSR.csv", "PR_CHO", .65),
    ("C1C2", "polyreactivity/jain2017biophysical_ELISA.csv", "PR_Ova", .43),
    ("C1C2", "expression/jain2017biophysical_HEK.csv", "Titer", .33),
    ("C1C2", "thermostability/jain2017biophysical_Tm.csv", "Tm1", .09),
    ("sensitivity", "polyreactivity/jain2017biophysical_ELISA.csv", "PR_CHO", .43),
    ("sensitivity", "polyreactivity/jain2017biophysical_BVPELISA.csv", "PR_CHO", .47),
    ("semi-fresh", "aggregation/jain2017biophysical_CSIBLI.csv", "AC-SINS_pH7.4", .75)]
LS = pd.read_csv(f"{OUT}/leakage_susceptibility.csv"); assert set(LS.assay) == {"HIC", "AC-SINS", "SMAC"}
LSm = LS.groupby("model").LS.mean()
out = []
for role, j17, gcol, plan in PAIRS:
    G = g[["vh", "vl", gcol, "hierarchical_cluster_fold"]].rename(columns={gcol: "y"}).dropna().reset_index(drop=True); y = G.y.values
    J = jain(j17).dropna(); jy = dict(zip(zip(J.vh, J.vl), J.y))
    sh = [(jy[k], v) for k, v in zip(zip(G.vh, G.vl), y) if k in jy]; rs = spearmanr(*zip(*sh))[0]; fn = len(sh) / len(G)
    q, r_ = norm.ppf(rankdata(y) / (len(y) + 1)), 2 * np.sin(np.pi * rs / 6)
    lab = lambda idx, r: np.quantile(y, norm.cdf(r_ * q[idx] + np.sqrt(1 - r_ ** 2) * r.normal(0, 1, len(idx))))
    got = np.mean([spearmanr(y, lab(np.arange(len(y)), np.random.default_rng(s)))[0] for s in range(20)])
    pair = f"{j17.split('_')[-1][:-4]}->{gcol}"
    print(f"{pair}: agreement {rs:.3f} (plan {plan}, n {len(sh)}), f_near {fn:.3f} of {len(G)}, planted spearman {got:.2f}"
          + ("  ** agreement gap > .01, stage-1 value used **" if abs(rs - plan) > .01 else ""))
    cf = sorted(G.hierarchical_cluster_fold.unique())
    for m in MODELS:
        base = np.zeros(len(G))
        for k in cf:
            te, tr = G[G.hierarchical_cluster_fold == k], G[G.hierarchical_cluster_fold != k]; base[te.index] = fit_predict(m, tr, te)
        for s in range(3):
            r, p = np.random.default_rng(s), np.zeros(len(G))
            for k in cf:
                te, tr = G[G.hierarchical_cluster_fold == k], G[G.hierarchical_cluster_fold != k]
                L = te.sample(frac=fn, random_state=s).copy(); L["y"] = lab(L.index.values, r)
                p[te.index] = fit_predict(m, pd.concat([tr, L]), te)
            out.append(dict(role=role, pair=pair, jain_file=j17, gdpa_col=gcol, model=m, seed=s, agree=rs, agree_plan=plan, n_shared=len(sh),
                            n_target=len(G), f_near=fn, planted_spearman=got, rho_base=spearmanr(y, base)[0], rho_planted=spearmanr(y, p)[0],
                            gain=spearmanr(y, p)[0] - spearmanr(y, base)[0], null_c=LSm[m] * fn, null_d=LSm[m] * fn * rs / 0.923))
O = pd.DataFrame(out); O.to_csv(f"{OUT}/fresh_antibody_pred.csv", index=False)
pd.set_option("display.width", 200)
print(O.groupby(["role", "pair", "model"], sort=False)[["f_near", "gain", "null_c", "null_d"]].mean().round(3).to_string())
