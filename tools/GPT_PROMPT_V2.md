You are helping me run a second, more accurate protein-folding job on Google Colab. Assume
I am a beginner and walk me through it. Everything you need is here — don't ask for files.

## Context
I predict antibody "stickiness" (aggregation) from sequence. Stickiness happens on the
antibody's 3D SURFACE. My first attempt folded each antibody as ONE chain with ESMFold and
used a rough surface estimate — it slightly HURT the model. This v2 fixes that: fold the
heavy and light chains as a real PAIR with IgFold (correct 3D shape), then compute the REAL
surface area with freesasa and a proper SAP (Spatial Aggregation Propensity) score. Your job:
get it running and produce one output file. Be honest — if folds fail, report the count.

## Rules
- Use IgFold for folding and freesasa for surface area. Fold with do_refine=False (no
  pyrosetta). Do NOT change the feature formulas or output column names in the code below.
- The input CSV has columns seq_id, mode, H, L. `mode=paired` -> fold {"H":H,"L":L};
  `mode=single` -> fold {"H":H}. The code already handles this — don't alter it.

## What I have
`paired_chains_to_fold.csv` (607 rows: seq_id, mode, H, L). I will upload it.

## What to give me
The exact Colab cells, in order. Tell me to use a GPU runtime. The script is resumable
(appends, skips done ids). ~10–30 s per antibody.

### Cell 1 — install
```
!pip -q install igfold freesasa numpy
!pip -q install "setuptools<81"
```

### Cell 2 — upload
```
from google.colab import files
files.upload()   # choose paired_chains_to_fold.csv
```

### Cell 3 — fold + real surface features (paste all of this)
```python
import csv, os
import numpy as np
from igfold import IgFoldRunner
import freesasa

IN_CSV, OUT_CSV = "paired_chains_to_fold.csv", "structure_features_v2.csv"
KD = {"ALA":1.8,"ARG":-4.5,"ASN":-3.5,"ASP":-3.5,"CYS":2.5,"GLN":-3.5,"GLU":-3.5,"GLY":-0.4,
      "HIS":-3.2,"ILE":4.5,"LEU":3.8,"LYS":-3.9,"MET":1.9,"PHE":2.8,"PRO":-1.6,"SER":-0.8,
      "THR":-0.7,"TRP":-0.9,"TYR":-1.3,"VAL":4.2}

def parse_ca(p):
    out=[]
    for l in open(p):
        if l.startswith("ATOM") and l[12:16].strip()=="CA":
            out.append((l[21],int(l[22:26]),l[17:20].strip(),
                        float(l[30:38]),float(l[38:46]),float(l[46:54]),float(l[60:66])))
    return out

def feats(pdb, radius=10.0):
    ca=parse_ca(pdb); n=len(ca)
    if n==0: raise ValueError("no CA")
    areas=freesasa.calc(freesasa.Structure(pdb)).residueAreas()
    rel=np.zeros(n); kd=np.zeros(n)
    coords=np.array([[a[3],a[4],a[5]] for a in ca]); bf=np.array([a[6] for a in ca])
    for i,(ch,num,rn,*_) in enumerate(ca):
        try: rel[i]=min(areas[ch][str(num)].relativeTotal,1.0)
        except: rel[i]=0.0
        kd[i]=KD.get(rn,0.0)
    surf_h=rel*np.clip(kd,0,None); sap=float(surf_h.sum())
    d=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=-1)
    patch=float(max((surf_h[(d[i]<radius)].sum()) for i in range(n)))
    return {"sap_score":sap,"sap_per_residue":sap/n,"max_hydrophobic_patch":patch,
            "hydrophobic_surface_fraction":float((surf_h>0.1).mean()),
            "mean_rel_sasa":float(rel.mean()),"mean_prmsd":float(bf.mean()),"n_res":n}

igfold=IgFoldRunner()
rows=list(csv.DictReader(open(IN_CSV)))
done={r["seq_id"] for r in csv.DictReader(open(OUT_CSV))} if os.path.exists(OUT_CSV) else set()
cols=["seq_id","sap_score","sap_per_residue","max_hydrophobic_patch",
      "hydrophobic_surface_fraction","mean_rel_sasa","mean_prmsd","n_res"]
new=not os.path.exists(OUT_CSV); ok=fail=0
with open(OUT_CSV,"a",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols)
    if new: w.writeheader()
    for i,r in enumerate(rows,1):
        if r["seq_id"] in done: continue
        seqs={"H":r["H"]} if (r["mode"]=="single" or not r["L"]) else {"H":r["H"],"L":r["L"]}
        pdb=f"/tmp/{r['seq_id']}.pdb"
        try:
            igfold.fold(pdb,sequences=seqs,do_refine=False,do_renum=False)
            fe=feats(pdb); w.writerow({"seq_id":r["seq_id"],**fe}); f.flush(); ok+=1
            print(f"[{i}/{len(rows)}] {r['seq_id']} ok sap={fe['sap_score']:.1f}",flush=True)
        except Exception as e:
            fail+=1; print(f"[{i}/{len(rows)}] {r['seq_id']} FAILED: {e}",flush=True)
        finally:
            if os.path.exists(pdb): os.remove(pdb)
print(f"DONE: {ok} folded, {fail} failed -> {OUT_CSV}")
```

### Cell 4 — download
```
from google.colab import files
files.download("structure_features_v2.csv")
```

## When done
Tell me how many of the 607 folded and any failures. I will place `structure_features_v2.csv`
at `Biostuff/data/external/structures/`. If something errors (IgFold install, freesasa),
explain it simply and suggest a fix — don't silently swap methods.
