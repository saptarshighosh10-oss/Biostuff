You are helping me run a protein-folding job on Google Colab. Assume I am a beginner and
walk me through it. Here is everything you need — do not ask me for more files.

## Context (so you understand the job)
I have a machine-learning project that predicts how "sticky" (self-associating /
aggregation-prone) an antibody is, from its amino-acid sequence. Sticky antibodies fail as
drugs. Stickiness happens on the antibody's 3D SURFACE, so I need to fold 606 antibodies
into 3D structures and extract surface-hydrophobicity features. My own computer can't do
this (no GPU, only 16 GB RAM, and Python 3.14 breaks the folding libraries), so we do it on
Colab's free GPU. Your only job: get the folding to run and produce one clean output file.

## Rules (important)
- Use ESMFold via HuggingFace ONLY: `transformers.EsmForProteinFolding`,
  model `facebook/esmfold_v1`. Do NOT install `openfold` or `igfold` — they break.
- Do NOT change the feature formulas or the output column names in the code below.
- Do NOT split the sequences into chains — fold each sequence exactly as given.
- Be honest: if folds fail, tell me the count; don't fake results.

## What I have
A file `acsins_sequences_to_fold.csv` with two columns: `seq_id, sequence` (606 rows).
I will upload it to Colab.

## What I need you to do
Give me the exact Colab cells to run, in order, using the code below. Tell me to first set
Runtime → Change runtime type → T4 GPU. The script is resumable (it appends and skips
finished ids), so if Colab disconnects I just rerun the last cell. Expected time ~2–4 hours.

### Cell 1 — install
```
!pip -q install transformers accelerate numpy
```

### Cell 2 — upload the CSV
```
from google.colab import files
files.upload()   # choose acsins_sequences_to_fold.csv
```

### Cell 3 — fold + compute features (paste all of this)
```python
import csv, os
import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

IN_CSV, OUT_CSV = "acsins_sequences_to_fold.csv", "structure_features.csv"
KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,"H":-3.2,
      "I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,
      "Y":-1.3,"V":4.2}

tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
model = model.cuda().eval()
model.esm = model.esm.half()
model.trunk.set_chunk_size(64)   # lower memory

def fold_one(seq):
    with torch.no_grad():
        ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
        out = model(ids)
    ca = out["positions"][-1, 0, :, 1, :].cpu().numpy()   # CA coords, L x 3
    plddt = out["plddt"][0, :, 1].cpu().numpy()           # per-residue 0..100
    return ca, plddt

def feats(ca, plddt, seq, radius=10.0, burial_k=18):
    n = len(seq)
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    neighbors = (d < radius).sum(axis=1) - 1
    exposure = np.clip(1.0 - neighbors / burial_k, 0, 1)   # 1 exposed, 0 buried
    hydro = np.clip(np.array([KD.get(a, 0.0) for a in seq]), 0, None)
    surf_h = exposure * hydro
    sap = float(surf_h.sum())
    patch = float(max((surf_h[(d[i] < radius)].sum()) for i in range(n))) if n else 0.0
    return {"plddt_mean": float(plddt.mean()), "plddt_variance": float(plddt.var()),
            "low_confidence_fraction": float((plddt < 70).mean()), "sap_score": sap,
            "sap_per_residue": sap/max(n,1), "max_hydrophobic_patch": patch,
            "surface_hydrophobic_fraction": float((surf_h > 0).mean()), "n_res": n}

rows = list(csv.DictReader(open(IN_CSV)))
done = {r["seq_id"] for r in csv.DictReader(open(OUT_CSV))} if os.path.exists(OUT_CSV) else set()
cols = ["seq_id","plddt_mean","plddt_variance","low_confidence_fraction","sap_score",
        "sap_per_residue","max_hydrophobic_patch","surface_hydrophobic_fraction","n_res"]
new = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    if new: w.writeheader()
    ok = fail = 0
    for i, r in enumerate(rows, 1):
        if r["seq_id"] in done: continue
        try:
            ca, plddt = fold_one(r["sequence"])
            w.writerow({"seq_id": r["seq_id"], **feats(ca, plddt, r["sequence"])}); f.flush(); ok += 1
            print(f"[{i}/{len(rows)}] {r['seq_id']} ok", flush=True)
        except Exception as e:
            fail += 1; print(f"[{i}/{len(rows)}] {r['seq_id']} FAILED: {e}", flush=True)
print(f"DONE: {ok} folded, {fail} failed -> {OUT_CSV}")
```

### Cell 4 — download the result
```
from google.colab import files
files.download("structure_features.csv")
```

## When done
Tell me how many of the 606 folded and any that failed. I will place
`structure_features.csv` at `Biostuff/data/external/structures/` on my computer, where the
rest of the pipeline picks it up. If anything errors, explain it simply and suggest a fix —
don't silently change the approach.
