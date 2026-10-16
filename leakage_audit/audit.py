"""Leakage audit: how much do antibody-developability scores drop under honest splits?

Run: cd ~/Biostuff && /opt/anaconda3/bin/python leakage_audit/audit.py
Never touches GDPa3. Outputs: leakage_audit/out/{summary.csv,items.csv}
"""
import os, re, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import FeatureHasher

ROOT = ROOT_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, ROOT); from features import plm
FL = f"{ROOT}/data/external/flab/"
OUT = f"{ROOT}/leakage_audit/out"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)

# ---- data -----------------------------------------------------------------
g = pd.read_csv(f"{ROOT}/data/external/gdpa/GDPa1_v1.2_20250814.csv")
g["vh"], g["vl"] = g.vh_protein_sequence.str.strip(), g.vl_protein_sequence.str.strip()
ASSAYS = {  # name -> (GDPa1 column, Jain2017 file, Jain2024 file)
    "HIC": ("HIC", "aggregation/jain2017biophyscial_HICRT.csv", "aggregation/jain2024assessment_HIC.csv"),
    "AC-SINS": ("AC-SINS_pH7.4", "aggregation/jain2017biophysical_ACSINS.csv", "aggregation/jain2024assessment_ACSINS.csv"),
    "SMAC": ("SMAC", "polyreactivity/jain2017biophysical_SMACRT.csv", None),
    "SEC%mono": ("SEC %Monomer", None, "aggregation/jain2024assessment_SEC.csv"),
}

def jain(path):
    d = pd.read_csv(FL + path)
    return pd.DataFrame({"vh": d.heavy.str.strip(), "vl": d.light.str.strip(), "y": d.iloc[:, 2]})

# ---- features -------------------------------------------------------------
KD = dict(zip("ACDEFGHIKLMNPQRSTVWY", [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
def h3(vh):
    m = re.search(r"YY?C[A-Z]{0,2}(.{3,30}?)WG.G", vh)
    return m.group(1) if m else ""
def phys(s):
    if not s: return [0, 0, 0, 0]
    ch = sum(s.count(a) for a in "KR") - sum(s.count(a) for a in "DE") + 0.1 * s.count("H")
    return [ch, np.mean([KD.get(a, 0) for a in s]), sum(s.count(a) for a in "FWY") / len(s), len(s)]
def physfeat(df):
    return np.array([phys(a) + phys(b) + phys(h3(a)) for a, b in zip(df.vh, df.vl)])
def kmers(s, k=3): return {s[i:i + k] for i in range(len(s) - k + 1)}
def kmerfeat(df):
    fh = FeatureHasher(n_features=2048, input_type="string")
    return fh.transform([[f"H{k}" for k in kmers(a)] + [f"L{k}" for k in kmers(b)] for a, b in zip(df.vh, df.vl)]).toarray()
import subprocess, tempfile
def _chain_identity():
    """All-vs-all exhaustive global-ish identity (matches / longer chain) for every VH/VL chain, via mmseqs2."""
    chains = set(g.vh) | set(g.vl)
    for f in {a[1] for a in ASSAYS.values() if a[1]} | {a[2] for a in ASSAYS.values() if a[2]}:
        d = pd.read_csv(FL + f); chains |= set(d.heavy.str.strip()) | set(d.light.str.strip())
    chains = sorted(chains); idx = {c: i for i, c in enumerate(chains)}
    with tempfile.TemporaryDirectory() as t:
        open(f"{t}/c.fa", "w").write("".join(f">{i}\n{c}\n" for i, c in enumerate(chains)))
        subprocess.run(["mmseqs", "easy-search", f"{t}/c.fa", f"{t}/c.fa", f"{t}/o.m8", f"{t}/tmp", "--exhaustive-search", "1",
                        "-e", "1e9", "--max-seqs", "100000", "--min-seq-id", "0", "--format-output", "query,target,fident,alnlen,qlen,tlen"],
                       check=True, capture_output=True)
        M = np.zeros((len(chains), len(chains)))
        for q, tg, fi, al, ql, tl in (l.split("\t") for l in open(f"{t}/o.m8")):
            M[int(q), int(tg)] = max(M[int(q), int(tg)], float(fi) * int(al) / max(int(ql), int(tl)))
    return M, idx
IDM, IDX = _chain_identity()
assert all(abs(IDM[i, i] - 1) < 1e-9 for i in range(len(IDX))), "self-identity != 1"
def jacc_matrix(te, tr):  # name kept: now TRUE identity, mean of VH and VL identity
    r = lambda ser: np.array([IDX[x] for x in ser])
    return (IDM[np.ix_(r(te.vh), r(tr.vh))] + IDM[np.ix_(r(te.vl), r(tr.vl))]) / 2

# ---- models (fit on tr, predict te) ---------------------------------------
def fit_predict(name, tr, te):
    y = tr.y.values
    if name == "NN-copy":   # pure memorisation baseline
        return y[jacc_matrix(te, tr).argmax(1)]
    if name == "kmer-ridge":
        return Ridge(alpha=10).fit(kmerfeat(tr), y).predict(kmerfeat(te))
    if name == "esm2-ridge":
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 9)))
        return mdl.fit(esmfeat(tr), y).predict(esmfeat(te))
    if name == "ablang2-ridge":
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 9)))
        return mdl.fit(ablfeat(tr), y).predict(ablfeat(te))
    if name == "physchem-RF":
        return RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(physfeat(tr), y).predict(physfeat(te))
MODELS = ["NN-copy", "kmer-ridge", "physchem-RF", "esm2-ridge", "ablang2-ridge"]
def ablfeat(df):  # published antibody LM AbLang2, paired VH+VL embedding (cache-only; run embed_ablang.py first)
    return np.array([plm.load_cached(plm.MODEL_TAG_ABLANG2_PAIRED, plm.ablang2_pair_cache_key(a, b)) for a, b in zip(df.vh, df.vl)])

def esmfeat(df):  # mean-pooled ESM-2 650M, VH|VL concatenated (cache-only; run embed.py first)
    get = lambda q: plm.load_cached(plm.MODEL_TAG_ESM2, q)
    return np.array([get(a) + get(b) for a, b in zip(df.vh, df.vl)])

def rho_ci(y, p, n=500):
    r = spearmanr(y, p)[0]
    idx = RNG.integers(0, len(y), (n, len(y)))
    bs = [spearmanr(y[i], p[i])[0] for i in idx]
    return r, *np.nanpercentile(bs, [2.5, 97.5])

rows, items, PAIR, OOF = [], [], [], {}
def record(exp, assay, model, te, pred, sim):
    r, lo, hi = rho_ci(te.y.values, pred)
    rows.append(dict(experiment=exp, assay=assay, model=model, n_test=len(te), rho=r, lo=lo, hi=hi, mean_nn_sim=float(np.mean(sim))))
    items.extend(dict(experiment=exp, assay=assay, model=model, nn_sim=s, y=a, pred=b) for s, a, b in zip(sim, te.y, pred))

# ---- experiments ----------------------------------------------------------
for aname, (gcol, j17, j24) in ASSAYS.items():
    G = g[["vh", "vl", gcol, "random_fold", "hierarchical_cluster_fold"]].rename(columns={gcol: "y"}).dropna().reset_index(drop=True)
    # E1: within GDPa1, random vs cluster 5-fold (out-of-fold, pooled)
    for split in ["random_fold", "hierarchical_cluster_fold"]:
        for m in MODELS:
            pred, sim = np.zeros(len(G)), np.zeros(len(G))
            for f in sorted(G[split].unique()):
                te, tr = G[G[split] == f], G[G[split] != f]
                pred[te.index] = fit_predict(m, tr, te)
                sim[te.index] = jacc_matrix(te, tr).max(1)
            record(f"E1_{split}", aname, m, G, pred, sim)
            OOF[(aname, m, split)] = (G.y.values, pred)
    # E4: cluster folds, but also drop from train anything >= t identity to ANY test antibody of that fold
    for t in (0.90, 0.85, 0.80):
        for m in MODELS:
            pred, sim, ntr = np.zeros(len(G)), np.zeros(len(G)), []
            for f in sorted(G.hierarchical_cluster_fold.unique()):
                te, tr = G[G.hierarchical_cluster_fold == f], G[G.hierarchical_cluster_fold != f]
                S_ = jacc_matrix(tr, te).max(1); tr = tr[S_ < t]; ntr.append(len(tr))
                pred[te.index] = fit_predict(m, tr, te); sim[te.index] = jacc_matrix(te, tr).max(1)
            record(f"E4_strict_{t:.2f}", aname, m, G, pred, sim)
            _, pb = OOF[(aname, m, "hierarchical_cluster_fold")]
            PAIR.append((f"E4 cluster-strict{t:.2f} (avg train {int(np.mean(ntr))})", aname, m, G.y.values, pb, np.ones(len(G), bool), pred, np.ones(len(G), bool)))
    # E2: train Jain2017 -> test GDPa1; naive (all) vs honest (drop exact VH+VL seen in training)
    if j17:
        J = jain(j17).dropna()
        key = set(zip(J.vh, J.vl))
        seen = np.array([(a, b) in key for a, b in zip(G.vh, G.vl)])
        simall = jacc_matrix(G, J).max(1)
        for m in MODELS:
            pr = fit_predict(m, J, G)
            for label, mask in [("E2_naive_all", np.ones(len(G), bool)), ("E2_honest_unseen", ~seen)]:
                record(label, aname, m, G[mask], pr[mask], simall[mask])
            for t in (0.99, 0.95, 0.90):
                k = simall < t
                if k.sum() >= 20:
                    record(f"E5_lt_{t:.2f}", aname, m, G[k], pr[k], simall[k])
                    PAIR.append((f"E5 naive-vs-id<{t:.2f} (n={k.sum()})", aname, m, G.y.values, pr, np.ones(len(G), bool), pr, k))
            PAIR.append(("E2 naive-honest", aname, m, G.y.values, pr, np.ones(len(G), bool), pr, ~seen))
    # E3: train Jain2017 -> test Jain2024 (disjoint panels, small n)
    if j17 and j24:
        J, K = jain(j17).dropna(), jain(j24).dropna()
        for m in MODELS:
            record("E3_Jain17_to_Jain24", aname, m, K, fit_predict(m, J, K), jacc_matrix(K, J).max(1))

for aname in ASSAYS:
    for m in MODELS:
        if (aname, m, "random_fold") in OOF:
            y, pa = OOF[(aname, m, "random_fold")]; _, pb = OOF[(aname, m, "hierarchical_cluster_fold")]
            PAIR.append(("E1 random-cluster", aname, m, y, pa, np.ones(len(y), bool), pb, np.ones(len(y), bool)))

def paired(y, pa, ma, pb, mb, n=2000):
    """rho_A - rho_B, resampling the SAME antibodies for both (A on ma, B on mb restricted to resample)."""
    d = []
    for i in RNG.integers(0, len(y), (n, len(y))):
        a, b = i[ma[i]], i[mb[i]]
        d.append(spearmanr(y[a], pa[a])[0] - spearmanr(y[b], pb[b])[0])
    d = np.array(d)
    pt = spearmanr(y[ma], pa[ma])[0] - spearmanr(y[mb], pb[mb])[0]
    return pt, *np.nanpercentile(d, [2.5, 97.5]), float(np.mean(d <= 0))
PR = pd.DataFrame([dict(comparison=c, assay=a, model=m, diff=r[0], lo=r[1], hi=r[2], p_le0=r[3])
                   for c, a, m, y, pa, ma, pb, mb in PAIR for r in [paired(y, pa, ma, pb, mb)]])
PR.to_csv(f"{OUT}/paired.csv", index=False)
print("\nPAIRED bootstrap (diff = score on leaky split minus score on honest split; CI excluding 0 => real inflation)")
print(PR.round(2).to_string(index=False))

I = pd.DataFrame(items)
E2 = I[I.experiment == "E2_naive_all"].copy()
E2["id_bin"] = pd.cut(E2.nn_sim, [0, .7, .9, 1.01], labels=["<0.70", "0.70-0.90", ">=0.90"], right=False)
B = E2.groupby(["assay", "model", "id_bin"], observed=True).apply(lambda d: pd.Series(dict(n=len(d), rho=spearmanr(d.y, d.pred)[0]))).reset_index()
B.to_csv(f"{OUT}/rho_by_identity.csv", index=False)
print("\nE2 naive: rho by nearest-train TRUE identity")
print(B.pivot_table(index=["assay", "model"], columns="id_bin", values="rho", observed=True).round(2).to_string())
print(B.pivot_table(index=["assay"], columns="id_bin", values="n", observed=True, aggfunc="first").to_string())

S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/summary.csv", index=False)
pd.DataFrame(items).to_csv(f"{OUT}/items.csv", index=False)
pd.set_option("display.width", 200)
print(S.pivot_table(index=["assay", "model"], columns="experiment", values="rho").round(2).to_string())
print(S.pivot_table(index=["assay"], columns="experiment", values="mean_nn_sim").round(2).to_string())
