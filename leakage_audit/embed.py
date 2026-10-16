"""Precompute ESM-2 (650M) embeddings for every chain the audit uses (cache-first)."""
import sys, pandas as pd
sys.path.insert(0, "."); from features import plm
FL = "data/external/flab/"
g = pd.read_csv("data/external/gdpa/GDPa1_v1.2_20250814.csv")
seqs = set(g.vh_protein_sequence.str.strip()) | set(g.vl_protein_sequence.str.strip())
for f in ["aggregation/jain2017biophyscial_HICRT.csv", "aggregation/jain2024assessment_HIC.csv"]:
    d = pd.read_csv(FL + f); seqs |= set(d.heavy.str.strip()) | set(d.light.str.strip())
seqs = sorted(seqs)
plm.embed_esm2(seqs)
print("done", len(seqs), sum(plm.load_cached(plm.MODEL_TAG_ESM2, s) is not None for s in seqs), "cached")
