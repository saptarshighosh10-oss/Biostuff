import sys, csv; sys.path.insert(0, ".")
from features import plm
P = set()
for f, h, l in [("data/external/gdpa/GDPa1_v1.2_20250814.csv", "vh_protein_sequence", "vl_protein_sequence"),
                ("data/external/flab/aggregation/jain2017biophyscial_HICRT.csv", "heavy", "light"),
                ("data/external/flab/aggregation/jain2024assessment_HIC.csv", "heavy", "light")]:
    for r in csv.DictReader(open(f)): P.add((r[h].strip(), r[l].strip()))
P = sorted(P); v = plm.embed_ablang2_pairs(P, device="cpu"); print("done", len(P), len(v), len(v[0]))
