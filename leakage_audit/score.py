"""Step 3 retrospective scoreboard (PLAN.md §3, PREREG.md §2-3). Skill S = 1 - sum|pred - obs| / sum|obs|: S > 0 beats "predict 0 leak".
Scores every predictor variant ever tried, plus nulls a (predict 0) and b (oracle constant), per scope (all models / knn_geo only), per seed
group, with per-seed S. Each variant is scored against the obs of its own run. Old data only, so every number here is exploratory.
Writes out/scoreboard.csv and out/forking_paths.csv; exits 1 if a PLAN §3 value is not reproduced to +-.01.
Run from ~/Biostuff/leakage_audit with python -I."""
import subprocess, sys, numpy as np, pandas as pd
rd = pd.read_csv
S = lambda p, o: 1 - np.abs(p - o).sum() / np.abs(o).sum()

ab = rd("out/did_antibody_rank.csv"); L = rd("out/leakage_susceptibility.csv"); key = list(zip(ab.assay, ab.model))
fn = rd("out/did_antibody.csv").query("part == 'A_crosslab'").set_index(["assay", "model"]).f_near
ab["LSxf"] = L.set_index(["assay", "model"]).loc[key, "predicted"].values
ab["null_c"] = [L[(L.model == m) & (L.assay != a)].LS.mean() * fn[(a, m)] for a, m in key]  # LOAO model mean, agreement-blind
sal = lambda f: pd.concat([rd("salinity/did.csv" if f == "did_" else f"salinity/{f}seeds0to2.csv").assign(grp="0-2"),
                           rd(f"salinity/{f}seeds3to5.csv").assign(grp="3-5")])
# (setting, frame, obs column, [(pred column, variant, script, fixed before its DiD was seen, status)])
RUNS = [
    ("antibody cross-lab", ab, "did", [
        ("pred_old", "SD-noise planting", "did_antibody.py", "yes", "exploratory"),
        ("pred_rank", "I_M(f) copula planting", "did_antibody_rank.py", "no: adopted after the SMAC miss", "record"),
        ("LSxf", "LS x f (copula)", "synthetic.py", "no", "exploratory"),
        ("null_c", "null c: LOAO model mean", "score.py", "-", "null")]),
    ("antibody cross-lab vs naive-minus-honest", L, "observed", [
        ("predicted", "LS x f (copula)", "synthetic.py", "no: rank fix post-hoc", "demoted: range restriction in audit.py E2")]),
    ("within-GDPa1", rd("out/did_antibody_real.csv").query("part == 'B_random'"), "did", [
        ("pred_old", "exact copy", "did_antibody.py", "yes", "exploratory"),
        ("pred_real", "realistic neighbour", "did_antibody_real.py", "yes", "exploratory"),
        ("pred_rms", "rms-noise neighbour", "did_antibody_real.py", "yes", "exploratory")]),
    ("EuroSAT", rd("sat/did.csv"), "obs", [
        ("pred_old", "exact-copy curve", "sat/did.py", "not stated", "record"),
        ("pred_red", "P(correct|closeness) + redundancy", "sat/did.py", "not stated", "exploratory: easiness-confounded")]),
    ("EuroSAT", rd("sat/did_red.csv"), "obs", [
        ("pred_cal_red", "blended soft copies + redundancy", "sat/did_red.py", "not stated", "exploratory")]),
    ("EuroSAT 60/20/20 split", rd("sat/real_planted.csv"), "obs", [
        ("pred_old", "exact-copy curve (replicate)", "sat/real_planted.py", "yes", "exploratory"),
        ("pred_real_red", "real reserve tiles + redundancy", "sat/real_planted.py", "yes", "exploratory")]),
    ("EuroSAT removal control (superseded)", rd("sat/calibrated.csv"), "obs", [
        ("pred_old", "hard-threshold curve", "sat/calibrate.py", "not stated", "superseded estimand"),
        ("pred_cal", "closeness-calibrated curve", "sat/calibrate.py", "not stated", "superseded estimand")]),
    ("EuroSAT removal control (superseded)", rd("sat/headroom.csv"), "obs", [
        ("pred_h", "headroom-weighted", "sat/headroom.py", "not stated", "superseded: logic backwards"),
        ("pred_hl", "headroom-weighted (variant)", "sat/headroom.py", "not stated", "superseded: logic backwards")]),
    ("salinity", sal("did_"), "obs", [
        ("pred_old", "exact copy", "salinity/did.py", "yes", "exploratory"),
        ("pred_dist", "additive pred_dist", "salinity/did.py", "yes", "exploratory"),
        ("pred_dist_red", "additive pred_dist_red", "salinity/did.py", "yes", "demoted: labels over-informative")]),
    ("salinity", sal("did_rank_"), "obs", [
        ("pred_dist", "rank pred_dist", "salinity/did_rank.py", "yes", "exploratory"),
        ("pred_dist_red", "rank pred_dist_red", "salinity/did_rank.py", "yes", "exploratory")]),
    ("salinity", sal("did_multi_"), "obs", [
        ("pred_nearest", "did_multi nearest", "salinity/did_multi.py", "design yes (untracked until 26ef104); chosen as record after seeing seeds 0-5", "record"),
        ("pred_all", "did_multi all", "salinity/did_multi.py", "design yes (untracked until 26ef104)", "exploratory: pre-registered reading refuted")]),
    ("salinity removal control (superseded)", rd("salinity/calibrated.csv").assign(grp="0-2"), "obs", [
        ("pred_dist", "additive pred_dist (removal control)", "salinity/calibrate_dist.py", "not stated", "superseded estimand")]),
]

def score(setting, D, oc, variant, p, status):
    out = []
    for g, Dg in D.groupby(D.get("grp", pd.Series("-", D.index))):
        for scope, m in [("all", np.ones(len(Dg), bool)), ("knn", (Dg.model == "knn_geo").values)]:
            if not m.any() or (scope == "knn" and m.all()): continue
            d, o = Dg[m], Dg[m][oc]; pr = p(d)
            seeds = "/".join(f"{S(p(x), x[oc]):.2f}" for _, x in d.groupby("seed")) if "seed" in d else ""
            out.append(dict(setting=setting, variant=variant, scope=scope, seeds=g, n_cells=len(d), S=S(pr, o), MAE=np.abs(pr - o).mean(),
                            MAE_predict0=np.abs(o).mean(), S_per_seed=seeds, status=status))
    return out

rows, meta = [], {}
for setting, D, oc, V in RUNS:
    for col, variant, script, fixed, status in V:
        rows += score(setting, D, oc, variant, lambda d, c=col: d[c], status); meta[(setting, variant)] = (script, fixed, status)
    if not any(r["setting"] == setting and r["variant"].startswith("null b") for r in rows):
        rows += score(setting, D, oc, "null b: oracle constant", lambda d, c=oc: pd.Series(d[c].mean(), d.index), "null (in-sample reference)")
B = pd.DataFrame(rows); B.to_csv("out/scoreboard.csv", index=False)

added = lambda f: (subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", f], capture_output=True, text=True).stdout.split() or ["untracked"])[-1]
F = B[~B.variant.str.startswith("null b")].pivot_table(index=["setting", "variant"], columns="scope", values="S", aggfunc=lambda s: "/".join(f"{v:.2f}" for v in s))
F = F.add_prefix("S_").reset_index()
F[["script", "fixed_before_did", "status"]] = [meta[k] for k in zip(F.setting, F.variant)]
F["commit"] = F.script.map(added); F.to_csv("out/forking_paths.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(B.drop(columns="status").round(3).to_string(index=False))
EXP = {  # PLAN §3, checked 2026-10-16: (setting, variant, scope, seeds) -> S
    ("antibody cross-lab", "SD-noise planting", "all", "-"): .65, ("antibody cross-lab", "I_M(f) copula planting", "all", "-"): .70,
    ("antibody cross-lab", "LS x f (copula)", "all", "-"): .72, ("antibody cross-lab", "null c: LOAO model mean", "all", "-"): .76,
    ("antibody cross-lab", "null b: oracle constant", "all", "-"): .51,
    ("within-GDPa1", "exact copy", "all", "-"): -1.42, ("within-GDPa1", "realistic neighbour", "all", "-"): .01,
    ("within-GDPa1", "rms-noise neighbour", "all", "-"): .02,
    ("EuroSAT", "exact-copy curve", "all", "-"): .39, ("EuroSAT", "P(correct|closeness) + redundancy", "all", "-"): .55,
    ("EuroSAT 60/20/20 split", "exact-copy curve (replicate)", "all", "-"): .36,
    ("EuroSAT 60/20/20 split", "real reserve tiles + redundancy", "all", "-"): .01}
for v, vals in {"exact copy": (-5.98, -5.44, -2.22, -1.79), "additive pred_dist": (-.64, -.52, -.59, -.30), "additive pred_dist_red": (.43, .38, .71, .59),
                "rank pred_dist": (-.11, -.16, .65, .52), "rank pred_dist_red": (.34, .27, .55, .41), "did_multi nearest": (.14, .23, .58, .57),
                "did_multi all": (-.37, -.25, -.04, .01)}.items():
    for (sc, g), x in zip([("all", "0-2"), ("all", "3-5"), ("knn", "0-2"), ("knn", "3-5")], vals): EXP[("salinity", v, sc, g)] = x
got = B.set_index(["setting", "variant", "scope", "seeds"]).S
bad = [(k, x, round(got.get(k, np.nan), 3)) for k, x in EXP.items() if not abs(got.get(k, np.nan) - x) <= .01]
print(f"\nPLAN §3 check: {len(EXP) - len(bad)}/{len(EXP)} reproduced to +-.01")
for b in bad: print("  MISMATCH", *b)
sys.exit(1 if bad else 0)
