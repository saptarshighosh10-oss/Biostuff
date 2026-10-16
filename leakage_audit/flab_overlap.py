"""Overlap between every pair of FLAb datasets (+GDPa1): exact VH+VL pairs and >=0.95 identity pairs. No models."""
import glob, os, subprocess, tempfile, numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FL = f"{R}/data/external/flab/"
D = {}
for f in sorted(glob.glob(FL + "*/*.csv")):
    d = pd.read_csv(f)
    if not {"heavy", "light"} <= set(d.columns): continue
    d = d.dropna(subset=["heavy", "light"]); d = d[(d.heavy.str.len() > 50) & (d.light.str.len() > 50)]
    if len(d): D[os.path.relpath(f, FL)[:-4]] = list(zip(d.heavy.str.strip(), d.light.str.strip()))
g = pd.read_csv(f"{R}/data/external/gdpa/GDPa1_v1.2_20250814.csv")
D["GDPa1"] = list(zip(g.vh_protein_sequence.str.strip(), g.vl_protein_sequence.str.strip()))
D = {k: sorted(set(v)) for k, v in D.items()}
print(len(D), "paired datasets; sizes:", sum(map(len, D.values())))
ch = sorted({c for v in D.values() for p in v for c in p}); ix = {c: i for i, c in enumerate(ch)}
with tempfile.TemporaryDirectory() as t:  # cluster chains at 95% identity, 90% coverage; near-dup pair = same (VH cluster, VL cluster)
    open(f"{t}/c.fa", "w").write("".join(f">{i}\n{c}\n" for i, c in enumerate(ch)))
    subprocess.run(["mmseqs", "easy-cluster", f"{t}/c.fa", f"{t}/o", f"{t}/tmp", "--min-seq-id", "0.95", "-c", "0.9", "--threads", "4"], check=True, capture_output=True)
    rep = {int(b_): int(a_) for a_, b_ in (l.split() for l in open(f"{t}/o_cluster.tsv"))}
key = lambda p: (rep[ix[p[0]]], rep[ix[p[1]]])
K = {n: {key(p) for p in v} for n, v in D.items()}; E = {n: set(v) for n, v in D.items()}
names = list(D); rows = []
for i, a in enumerate(names):
    for b in names[i + 1:]:
        ex = len(E[a] & E[b]); near = len(K[a] & K[b])
        if ex or near: rows.append(dict(a=a, b=b, n_a=len(D[a]), n_b=len(D[b]), exact=ex, near95=near, frac_of_smaller=round(near / min(len(D[a]), len(D[b])), 2)))
O = pd.DataFrame(rows).sort_values("near95", ascending=False); O.to_csv(f"{R}/leakage_audit/out/flab_overlap.csv", index=False)
print(len(O), "dataset pairs with any overlap of", len(names) * (len(names) - 1) // 2)
pd.set_option("display.width", 220); print(O.head(40).to_string(index=False))
