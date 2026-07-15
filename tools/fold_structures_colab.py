"""COLAB folding + structural aggregation features for AC-SINS antibodies.

Run this on Google Colab with a GPU runtime (Runtime -> Change runtime type -> T4 GPU).
It folds every sequence in acsins_sequences_to_fold.csv with ESMFold (via HuggingFace,
no openfold install needed) and computes structural features, saving structure_features.csv.

WHY these features: self-association (AC-SINS) is driven by hydrophobic patches on the
antibody SURFACE. You can only see "surface" in 3D, which is why folding adds signal the
sequence-only model can't get. The key output is `sap_score` / `max_hydrophobic_patch`
(a Spatial-Aggregation-Propensity proxy) plus structure-confidence (pLDDT) features.

=== STEPS (paste each block into a Colab cell) ===

# CELL 1 — install
!pip -q install transformers accelerate biotite numpy

# CELL 2 — upload the sequences file
from google.colab import files
files.upload()   # choose acsins_sequences_to_fold.csv

# CELL 3 — paste EVERYTHING below and run. It's resumable: rerun to continue.

Outputs: structure_features.csv  (download it, then locally place at
data/external/structures/structure_features.csv and tell Claude to merge it in).
"""
import csv
import os
import numpy as np

IN_CSV = "acsins_sequences_to_fold.csv"
OUT_CSV = "structure_features.csv"

# Kyte-Doolittle hydrophobicity (positive = hydrophobic)
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def load_model():
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()  # speed
    return tok, model, torch


def fold_one(seq, tok, model, torch):
    """Return (ca_coords Nx3, plddt N, sequence) for one sequence."""
    with torch.no_grad():
        inp = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
        out = model(inp)
    # positions: [8, B, L, 14, 3]; atom 1 = CA. plddt in out["plddt"] [B, L, 14]
    ca = out["positions"][-1, 0, :, 1, :].cpu().numpy()
    plddt = out["plddt"][0, :, 1].cpu().numpy()  # 0..100
    return ca, plddt, seq


def structural_features(ca, plddt, seq, radius=10.0, burial_k=18):
    n = len(seq)
    # pairwise CA distances
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    neighbors = (d < radius).sum(axis=1) - 1              # residues within radius
    exposure = np.clip(1.0 - neighbors / burial_k, 0, 1)  # 1 = fully exposed, 0 = buried
    kd = np.array([KD.get(a, 0.0) for a in seq])
    hydrophobic = np.clip(kd, 0, None)                    # only hydrophobic residues contribute
    # SAP proxy: total surface-exposed hydrophobicity
    sap = float((exposure * hydrophobic).sum())
    # biggest local surface hydrophobic patch: sum over spatial neighbourhood
    surf_h = exposure * hydrophobic
    patch = float(max((surf_h[(d[i] < radius)].sum()) for i in range(n))) if n else 0.0
    return {
        "plddt_mean": float(plddt.mean()),
        "plddt_variance": float(plddt.var()),
        "low_confidence_fraction": float((plddt < 70).mean()),
        "sap_score": sap,
        "sap_per_residue": sap / max(n, 1),
        "max_hydrophobic_patch": patch,
        "surface_hydrophobic_fraction": float((surf_h > 0).mean()),
        "n_res": n,
    }


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    done = set()
    if os.path.exists(OUT_CSV):
        done = {r["seq_id"] for r in csv.DictReader(open(OUT_CSV))}
    tok, model, torch = load_model()
    fields = ["seq_id", "plddt_mean", "plddt_variance", "low_confidence_fraction",
              "sap_score", "sap_per_residue", "max_hydrophobic_patch",
              "surface_hydrophobic_fraction", "n_res"]
    new_file = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for i, r in enumerate(rows, 1):
            if r["seq_id"] in done:
                continue
            try:
                ca, plddt, seq = fold_one(r["sequence"], tok, model, torch)
                feats = structural_features(ca, plddt, seq)
                w.writerow({"seq_id": r["seq_id"], **feats})
                f.flush()
                print(f"[{i}/{len(rows)}] {r['seq_id']} folded  sap={feats['sap_score']:.1f}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(rows)}] {r['seq_id']} FAILED: {e}", flush=True)
    print("done ->", OUT_CSV)


if __name__ == "__main__":
    main()
